"""LangGraph agent: plan -> tool calls -> validate SQL -> execute -> (self-correct) -> answer.

Graph:

    plan ──(tool calls)──> tools ──(run_sql)──> validate ──ok──> execute ──ok──> plan
      │                      │                     │                 │
      │                      └──(discovery only)──> plan             │
      │                                            └──error──┬───────┘
      │                                                      ├─ retries left ──> plan (self-correct)
      └──(final text)──> answer <────── retries exhausted ───┘
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from lakehouse_agent.config import MAX_SQL_RETRIES
from lakehouse_agent.guardrails import GuardrailError, validate_sql
from lakehouse_agent.llm import LLMClient, Message, ToolCall, Usage, get_llm
from lakehouse_agent.mock_router import mock_responder
from lakehouse_agent.tools import TOOL_SCHEMAS, LakehouseTools

MAX_PLAN_STEPS = 10
RESULT_PREVIEW_ROWS = 25


def build_system_prompt(tools: LakehouseTools) -> str:
    cat = tools.catalog
    lines = [
        "You are a data discovery and analytics agent for an e-commerce lakehouse in DuckDB.",
        "Layers:",
        *[f"- {layer}: {desc}" for layer, desc in cat.layer_descriptions.items()],
        "Tables:",
        *[f"- {t.name}: {t.description}" for t in cat.tables.values()],
        "",
        "Rules:",
        "1. Prefer the gold layer for business questions; use silver for detail gold lacks; bronze only for raw/audit questions.",
        "2. Call describe_table before writing SQL against a table you have not described yet.",
        "3. Use run_sql with ONE DuckDB SELECT/WITH query and schema-qualified names (gold.fact_orders).",
        "4. Join only on documented primary/foreign keys. Revenue = item prices excluding freight;",
        "   exclude order_status 'canceled' and 'unavailable' when the question is about sales.",
        "5. If run_sql returns an error, read it, fix the SQL and try again.",
        "6. For schema, key or lineage questions, answer from list_tables/describe_table/get_lineage without SQL.",
        "7. Final answer: 2-5 sentences of plain English with the key numbers, then say which layer,",
        "   tables, keys and joins you used. Do not paste the full result table; the UI shows it.",
    ]
    return "\n".join(lines)


class AgentState(TypedDict, total=False):
    question: str
    messages: list[Message]
    pending: list[dict]          # tool calls from the last LLM turn not yet answered
    sql_call: dict | None        # {"id", "sql"} awaiting validation/execution
    validated_sql: str | None
    last_result: dict | None     # last successful run_sql result
    last_error: str | None
    sql_attempts: int
    sql_failures: int
    tool_calls: int
    steps: int
    final_text: str
    usage: dict
    trace: list[dict]


@dataclass
class AgentResult:
    question: str
    answer: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    explanation: str = ""
    tool_calls: int = 0
    sql_attempts: int = 0
    retries: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    mode: str = "mock"
    model: str = "mock"
    trace: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


def _tool_message(call_id: str, payload: dict) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload, default=str)}


class LakehouseAgent:
    """Tool-calling SQL agent over the lakehouse. Works with a real LLM or the offline MockLLM."""

    def __init__(self, tools: LakehouseTools | None = None, llm: LLMClient | None = None):
        self.tools = tools or LakehouseTools()
        self.llm = llm or get_llm(mock_responder)
        self.system_prompt = build_system_prompt(self.tools)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def _plan(self, state: AgentState) -> AgentState:
        steps = state.get("steps", 0) + 1
        if steps > MAX_PLAN_STEPS:
            return {"steps": steps, "pending": [], "final_text": "",
                    "last_error": f"Stopped after {MAX_PLAN_STEPS} planning steps without a final answer."}
        response = self.llm.complete(state["messages"], TOOL_SCHEMAS)
        usage = Usage(**state["usage"])
        usage.add(response.usage)
        return {
            "steps": steps,
            "messages": [*state["messages"], response.as_message()],
            "pending": [asdict(tc) for tc in response.tool_calls],
            "final_text": response.content if not response.tool_calls else "",
            "usage": asdict(usage),
        }

    def _run_tools(self, state: AgentState) -> AgentState:
        messages = list(state["messages"])
        trace = list(state.get("trace", []))
        sql_call: dict | None = None
        n_calls = state.get("tool_calls", 0)
        for raw in state.get("pending", []):
            call = ToolCall(**raw)
            n_calls += 1
            if call.name == "run_sql":
                if sql_call is None:
                    sql_call = {"id": call.id, "sql": str(call.arguments.get("sql", ""))}
                    continue
                messages.append(_tool_message(call.id, {"error": "Skipped: run one query at a time."}))
                continue
            if call.name == "list_tables":
                result = self.tools.list_tables(call.arguments.get("layer"))
            elif call.name == "describe_table":
                result = self.tools.describe_table(str(call.arguments.get("name", "")))
            elif call.name == "get_lineage":
                result = self.tools.get_lineage(str(call.arguments.get("name", "")))
            else:
                result = {"error": f"Unknown tool '{call.name}'."}
            trace.append({"step": "tool", "tool": call.name, "args": call.arguments, "ok": "error" not in result})
            messages.append(_tool_message(call.id, result))
        return {"messages": messages, "trace": trace, "sql_call": sql_call, "pending": [], "tool_calls": n_calls}

    def _fail_sql(self, state: AgentState, stage: str, error: str, sql: str) -> AgentState:
        failures = state.get("sql_failures", 0) + 1
        left = MAX_SQL_RETRIES + 1 - failures
        hint = f" You have {left} attempt(s) left; fix the SQL and call run_sql again." if left > 0 else ""
        msg = _tool_message(state["sql_call"]["id"], {"ok": False, "stage": stage, "error": error + hint})
        return {
            "messages": [*state["messages"], msg],
            "trace": [*state.get("trace", []), {"step": stage, "tool": "run_sql", "sql": sql, "ok": False, "error": error}],
            "sql_failures": failures,
            "last_error": error,
            "sql_call": None,
        }

    def _validate(self, state: AgentState) -> AgentState:
        sql = state["sql_call"]["sql"]
        attempts = state.get("sql_attempts", 0) + 1
        try:
            checked = validate_sql(sql, self.tools.catalog, self.tools.max_rows)
        except GuardrailError as exc:
            return {**self._fail_sql(state, "validate", str(exc), sql), "sql_attempts": attempts}
        return {"validated_sql": checked.sql, "sql_attempts": attempts,
                "trace": [*state.get("trace", []), {"step": "validate", "tool": "run_sql", "sql": checked.sql, "ok": True}]}

    def _execute(self, state: AgentState) -> AgentState:
        result = self.tools.run_sql(state["validated_sql"])
        if not result["ok"]:
            return self._fail_sql(state, "execute", result["error"], state["validated_sql"])
        preview = {k: v for k, v in result.items() if k != "rows"}
        preview["rows_preview"] = result["rows"][:RESULT_PREVIEW_ROWS]
        return {
            "messages": [*state["messages"], _tool_message(state["sql_call"]["id"], preview)],
            "trace": [*state.get("trace", []), {"step": "execute", "tool": "run_sql", "ok": True,
                                                "rows": result["row_count"], "ms": result["elapsed_ms"]}],
            "last_result": result,
            "last_error": None,
            "sql_call": None,
        }

    # ---------------------------------------------------------------- routing
    @staticmethod
    def _after_plan(state: AgentState) -> Literal["tools", "answer"]:
        return "tools" if state.get("pending") else "answer"

    @staticmethod
    def _after_tools(state: AgentState) -> Literal["validate", "plan"]:
        return "validate" if state.get("sql_call") else "plan"

    @staticmethod
    def _after_validate(state: AgentState) -> Literal["execute", "plan", "answer"]:
        if state.get("sql_call"):
            return "execute"
        return "plan" if state.get("sql_failures", 0) <= MAX_SQL_RETRIES else "answer"

    @staticmethod
    def _after_execute(state: AgentState) -> Literal["plan", "answer"]:
        if state.get("last_error") and state.get("sql_failures", 0) > MAX_SQL_RETRIES:
            return "answer"
        return "plan"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("tools", self._run_tools)
        g.add_node("validate", self._validate)
        g.add_node("execute", self._execute)
        g.add_node("answer", lambda state: {})
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self._after_plan)
        g.add_conditional_edges("tools", self._after_tools)
        g.add_conditional_edges("validate", self._after_validate)
        g.add_conditional_edges("execute", self._after_execute)
        g.add_edge("answer", END)
        return g.compile()

    # -------------------------------------------------------------------- run
    def explain(self, result: dict | None, trace: list[dict] | None = None) -> str:
        """Deterministic explanation of layers, keys and joins used by the executed SQL."""
        if not result:
            used = sorted({t["tool"] for t in trace or [] if t.get("step") == "tool"})
            return f"Answered from the data catalog via {', '.join(used)} (no SQL needed)." if used else ""
        anatomy = self.tools.explain_sql(result["sql"])
        parts = []
        layers = result.get("layers") or sorted({t["layer"] for t in anatomy["tables"]})
        parts.append(f"Layer: {', '.join(layers)}.")
        tables = [f"{t['name']} (PK {', '.join(t['primary_key']) or 'none'})" for t in anatomy["tables"]]
        if tables:
            parts.append(f"Tables: {'; '.join(tables)}.")
        if anatomy["joins"]:
            joins = [f"{j['left']} = {j['right']} [{j['relationship']}]" for j in anatomy["joins"]]
            parts.append(f"Joins: {'; '.join(joins)}.")
        else:
            parts.append("No joins.")
        if result.get("truncated"):
            parts.append(f"Results capped at {self.tools.max_rows} rows.")
        return " ".join(parts)

    def ask(self, question: str) -> AgentResult:
        started = time.perf_counter()
        initial: AgentState = {
            "question": question,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "usage": asdict(Usage()),
            "trace": [],
            "sql_attempts": 0,
            "sql_failures": 0,
            "tool_calls": 0,
            "steps": 0,
        }
        state: AgentState = self.graph.invoke(initial, {"recursion_limit": 60})
        last = state.get("last_result")
        usage = Usage(**state["usage"])
        final_text = (state.get("final_text") or "").strip()
        error = None
        if not final_text:
            error = state.get("last_error") or "The agent did not produce an answer."
            final_text = f"I could not answer this reliably: {error}"
        return AgentResult(
            question=question,
            answer=final_text,
            sql=last["sql"] if last else None,
            columns=last["columns"] if last else [],
            rows=last["rows"] if last else [],
            truncated=bool(last and last.get("truncated")),
            explanation=self.explain(last, state.get("trace")),
            tool_calls=state.get("tool_calls", 0),
            sql_attempts=state.get("sql_attempts", 0),
            retries=min(state.get("sql_failures", 0), max(0, state.get("sql_attempts", 0) - 1)),
            latency_s=round(time.perf_counter() - started, 3),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=round(usage.cost_usd, 6),
            mode="mock" if self.llm.is_mock else "llm",
            model=self.llm.model,
            trace=state.get("trace", []),
            error=error,
        )
