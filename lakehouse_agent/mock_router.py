"""Rule-based stand-in for the LLM used in offline demo mode.

It speaks the same tool-calling protocol as a real model, so the LangGraph loop, guardrails
and executor all run for real. It only understands a fixed set of question patterns:

* catalog questions: list tables (optionally per layer), describe a table / its keys, lineage;
* a handful of keyword-matched analytic questions with hand-written SQL templates.

Anything else gets an honest "offline mode can't answer this" reply instead of a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from lakehouse_agent.catalog import LAYERS, load_catalog
from lakehouse_agent.llm import LLMResponse, Message, ToolCall

OFFLINE_CANNOT_ANSWER = (
    "Offline demo mode can't answer this question: the rule-based router only knows a fixed set of "
    "question patterns (list tables, describe a table, lineage, and a few common sales/delivery/review "
    "metrics). Set ANTHROPIC_API_KEY (or another LiteLLM provider key) to let a real model plan the SQL."
)

NON_SALE = "order_status NOT IN ('canceled', 'unavailable')"


@dataclass(frozen=True)
class Template:
    """A keyword-matched analytic question with fixed SQL."""

    name: str
    all_of: tuple[str, ...]      # regexes that must all match the question
    table: str                   # table to describe first (shows discovery)
    sql: Callable[[str], str]    # question -> SQL
    blurb: str                   # one-line description of the method used
    none_of: tuple[str, ...] = ()


def _top_n(question: str, default: int) -> int:
    m = re.search(r"\btop\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:customer\s+|product\s+)?(?:states|sellers|categories|products)\b", question)
    if m:
        return max(1, min(int(m.group(1) or m.group(2)), 50))
    return default


def _year(question: str) -> str | None:
    m = re.search(r"\b(2016|2017|2018)\b", question)
    return m.group(1) if m else None


TEMPLATES: tuple[Template, ...] = (
    Template(
        "category_revenue", (r"categor", r"revenue|sales"), "gold.fact_order_items",
        lambda q: f"SELECT category_name, sum(price) AS revenue FROM gold.fact_order_items "
                  f"WHERE {NON_SALE} GROUP BY category_name ORDER BY revenue DESC LIMIT {_top_n(q, 10)}",
        "summed item price per category from gold.fact_order_items, excluding canceled/unavailable orders",
    ),
    Template(
        "category_items", (r"categor", r"items|units|sold|volume"), "gold.fact_order_items",
        lambda q: f"SELECT category_name, count(*) AS items_sold FROM gold.fact_order_items "
                  f"WHERE {NON_SALE} GROUP BY category_name ORDER BY items_sold DESC LIMIT {_top_n(q, 10)}",
        "counted item rows per category in gold.fact_order_items, excluding canceled/unavailable orders",
    ),
    Template(
        "monthly_revenue", (r"month", r"revenue|sales|gmv"), "gold.agg_daily_sales",
        lambda q: "SELECT date_trunc('month', order_date) AS month, sum(revenue) AS revenue, sum(orders) AS orders "
                  "FROM gold.agg_daily_sales "
                  + (f"WHERE year(order_date) = {_year(q)} " if _year(q) else "")
                  + "GROUP BY month ORDER BY month",
        "rolled the daily sales mart gold.agg_daily_sales up to months",
    ),
    Template(
        "review_late_vs_ontime", (r"review|score|satisf", r"late"), "gold.fact_orders",
        lambda q: "SELECT is_late, avg(review_score) AS avg_review_score, count(*) AS orders "
                  "FROM gold.fact_orders WHERE is_late IS NOT NULL GROUP BY is_late ORDER BY is_late",
        "compared average review_score of late vs on-time delivered orders in gold.fact_orders",
    ),
    Template(
        "late_rate", (r"late", r"rate|share|percent|%|how often|fraction"), "gold.fact_orders",
        lambda q: "SELECT avg(CASE WHEN is_late THEN 1.0 ELSE 0.0 END) AS late_rate, count(*) AS delivered_orders "
                  "FROM gold.fact_orders WHERE is_late IS NOT NULL",
        "averaged the is_late flag over delivered orders in gold.fact_orders",
        none_of=(r"\bstate", r"seller"),
    ),
    Template(
        "delivery_by_state", (r"deliver", r"time|days|slow|long|fast", r"\bstates?\b"), "gold.fact_orders",
        lambda q: "SELECT c.state, avg(f.delivery_days) AS avg_delivery_days, count(*) AS delivered_orders "
                  "FROM gold.fact_orders f JOIN gold.dim_customers c ON f.customer_id = c.customer_id "
                  "WHERE f.order_status = 'delivered' AND f.delivery_days IS NOT NULL "
                  f"GROUP BY c.state ORDER BY avg_delivery_days DESC LIMIT {_top_n(q, 27)}",
        "joined gold.fact_orders to gold.dim_customers on customer_id and averaged delivery_days per state",
    ),
    Template(
        "orders_by_state", (r"\bstates?\b", r"orders|customers|most|top"), "gold.dim_customers",
        lambda q: "SELECT c.state, count(*) AS orders FROM gold.fact_orders f "
                  "JOIN gold.dim_customers c ON f.customer_id = c.customer_id "
                  f"GROUP BY c.state ORDER BY orders DESC LIMIT {_top_n(q, 10)}",
        "joined gold.fact_orders to gold.dim_customers on customer_id and counted orders per state",
        none_of=(r"seller",),
    ),
    Template(
        "top_sellers", (r"seller", r"revenue|top|best|biggest|sales"), "gold.agg_seller_performance",
        lambda q: "SELECT seller_id, seller_state, revenue, orders, avg_review_score, late_delivery_rate "
                  f"FROM gold.agg_seller_performance ORDER BY revenue DESC LIMIT {_top_n(q, 10)}",
        "ranked the seller mart gold.agg_seller_performance by revenue",
    ),
    Template(
        "orders_by_status", (r"status",), "gold.fact_orders",
        lambda q: "SELECT order_status, count(*) AS orders FROM gold.fact_orders GROUP BY order_status ORDER BY orders DESC",
        "counted gold.fact_orders per order_status",
    ),
    Template(
        "payment_types", (r"payment", r"type|method"), "silver.order_payments",
        lambda q: "SELECT payment_type, count(*) AS payments, sum(payment_value) AS total_paid "
                  "FROM silver.order_payments GROUP BY payment_type ORDER BY total_paid DESC",
        "grouped silver.order_payments by payment_type (gold keeps only the primary method per order)",
    ),
    Template(
        "avg_review", (r"review|score|rating", r"average|avg|mean|overall"), "silver.order_reviews",
        lambda q: "SELECT avg(review_score) AS avg_review_score, count(*) AS reviews FROM silver.order_reviews",
        "averaged review_score over all reviews in silver.order_reviews",
    ),
    Template(
        "busiest_day", (r"\bday\b", r"most|highest|busiest|peak"), "gold.agg_daily_sales",
        lambda q: "SELECT order_date, orders, revenue FROM gold.agg_daily_sales ORDER BY orders DESC LIMIT 1",
        "took the top day by order count from gold.agg_daily_sales",
    ),
    Template(
        "avg_order_value", (r"average order value|aov",), "gold.fact_orders",
        lambda q: "SELECT avg(order_value) AS avg_order_value FROM gold.fact_orders WHERE order_status = 'delivered'",
        "averaged order_value (items + freight) over delivered orders in gold.fact_orders",
    ),
    Template(
        "unique_customers", (r"how many|number of|count", r"unique customers|customers"), "gold.dim_customers",
        lambda q: "SELECT count(DISTINCT customer_unique_id) AS customers FROM gold.dim_customers",
        "counted distinct customer_unique_id in gold.dim_customers (customer_id is per order)",
        none_of=(r"state", r"repeat|more than one"),
    ),
    Template(
        "total_revenue", (r"total|overall", r"revenue|sales"), "gold.agg_daily_sales",
        lambda q: "SELECT sum(revenue) AS revenue, sum(orders) AS orders FROM gold.agg_daily_sales",
        "summed revenue in gold.agg_daily_sales (item prices, canceled/unavailable orders excluded)",
    ),
    Template(
        "total_orders", (r"how many orders|total (number of )?orders|number of orders",), "gold.fact_orders",
        lambda q: "SELECT count(*) AS orders FROM gold.fact_orders",
        "counted rows in gold.fact_orders (one row per order, all statuses)",
        none_of=(r"state", r"status", r"day", r"month"),
    ),
)


# ------------------------------------------------------------------ helpers
def _question(messages: list[Message]) -> str:
    return next((m["content"] for m in messages if m["role"] == "user"), "")


def _tool_results(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "tool":
            try:
                out.append(json.loads(m["content"]))
            except (TypeError, json.JSONDecodeError):
                out.append({"error": str(m["content"])})
    return out


def _call(name: str, args: dict[str, Any], n: int) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"mock_call_{n}", name=name, arguments=args)])


def _mentioned_table(question: str) -> str | None:
    """Find a catalog table named in the question (qualified name first, then short name)."""
    catalog = load_catalog()
    q = question.lower()
    for name in catalog.tables:
        if re.search(rf"\b{re.escape(name)}\b", q):
            return name
    preference = {"gold": 0, "silver": 1, "bronze": 2}
    hits = [t for t in catalog.tables.values() if re.search(rf"\b{re.escape(t.short_name)}\b", q)]
    hits.sort(key=lambda t: (-len(t.short_name), preference[t.layer]))
    layer = next((lyr for lyr in LAYERS if re.search(rf"\b{lyr}\b", q)), None)
    if layer:
        in_layer = [t for t in hits if t.layer == layer]
        hits = in_layer or hits
    return hits[0].name if hits else None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _summarise_rows(result: dict) -> str:
    cols, rows = result["columns"], result.get("rows_preview", [])
    if result["row_count"] == 0:
        return "The query returned no rows."
    if result["row_count"] == 1:
        pairs = ", ".join(f"{c} = **{_fmt(v)}**" for c, v in zip(cols, rows[0]))
        return f"Result: {pairs}."
    shown = "; ".join(
        f"{_fmt(r[0])} ({', '.join(f'{c} {_fmt(v)}' for c, v in zip(cols[1:], r[1:]))})" for r in rows[:5]
    )
    more = f" and {result['row_count'] - 5} more" if result["row_count"] > 5 else ""
    return f"The query returned {result['row_count']} rows: {shown}{more}."


# ---------------------------------------------------------- catalog answers
def _answer_list(result: dict) -> str:
    if "error" in result:
        return result["error"]
    lines = [f"There are {len(result['tables'])} tables:"]
    lines += [f"- `{t['name']}` ({t['row_count']:,} rows): {t['description']}" for t in result["tables"]]
    return "\n".join(lines)


def _answer_describe(result: dict) -> str:
    if "error" in result:
        return result["error"]
    fks = "; ".join(f"({', '.join(f['columns'])}) -> {f['references']}" for f in result["foreign_keys"]) or "none"
    refs = "; ".join(f"{r['table']}({', '.join(r['columns'])})" for r in result["referenced_by"]) or "none"
    cols = "\n".join(f"- `{c['name']}` {c['type']}: {c['description']}" for c in result["columns"])
    return (
        f"`{result['name']}` ({result['layer']} layer, {result['row_count']:,} rows): {result['description']}\n\n"
        f"Primary key: {', '.join(result['primary_key']) or 'none (raw layer, keys not enforced)'}. "
        f"Foreign keys: {fks}. Referenced by: {refs}.\n"
        f"Upstream: {', '.join(result['upstream']) or (result['source_file'] or 'none')}. "
        f"Downstream: {', '.join(result['downstream']) or 'none'}.\n\nColumns:\n{cols}"
    )


def _answer_lineage(result: dict) -> str:
    if "error" in result:
        return result["error"]
    return (
        f"Lineage of `{result['table']}` ({result['layer']}):\n"
        f"- Built directly from: {', '.join(result['direct_upstream']) or 'raw file (bronze)'}\n"
        f"- Comes from (all upstream): {', '.join(result['all_upstream']) or 'none'}\n"
        f"- Raw source files: {', '.join(result['raw_sources']) or 'none'}\n"
        f"- Feeds (all downstream): {', '.join(result['all_downstream']) or 'none'}"
    )


# ----------------------------------------------------------------- responder
def mock_responder(messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
    question = _question(messages)
    q = question.lower()
    done = _tool_results(messages)
    step = len(done)

    # 1) Lineage questions.
    if re.search(r"lineage|upstream|downstream|come from|comes from|derived|feeds|depends on|built from", q):
        table = _mentioned_table(q)
        if table:
            return _call("get_lineage", {"name": table}, step) if step == 0 else LLMResponse(_answer_lineage(done[-1]))

    # 2) Describe / schema / keys questions.
    if re.search(r"describe|schema of|columns|primary key|foreign key|\bkeys?\b|what is in|what's in|explain the", q):
        table = _mentioned_table(q)
        if table:
            return _call("describe_table", {"name": table}, step) if step == 0 else LLMResponse(_answer_describe(done[-1]))

    # 3) List tables.
    if re.search(r"(list|show|what|which).*(tables|datasets)|tables (are|do)", q):
        layer = next((lyr for lyr in LAYERS if re.search(rf"\b{lyr}\b", q)), None)
        args = {"layer": layer} if layer else {}
        return _call("list_tables", args, step) if step == 0 else LLMResponse(_answer_list(done[-1]))

    # 4) Keyword-matched analytics: describe the main table, run SQL, summarise.
    for t in TEMPLATES:
        if all(re.search(p, q) for p in t.all_of) and not any(re.search(p, q) for p in t.none_of):
            if step == 0:
                return _call("describe_table", {"name": t.table}, step)
            if step == 1:
                return _call("run_sql", {"sql": t.sql(q)}, step)
            last = done[-1]
            if not last.get("ok", False):
                return LLMResponse(f"The query failed and offline mode does not self-correct: {last.get('error')}")
            return LLMResponse(
                f"{_summarise_rows(last)} Method: {t.blurb}. "
                f"Layer used: {', '.join(last.get('layers', []))}."
            )

    return LLMResponse(OFFLINE_CANNOT_ANSWER)
