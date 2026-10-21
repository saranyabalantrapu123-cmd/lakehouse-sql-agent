"""Command-line entry point.

    python -m lakehouse_agent.cli "Which 5 states have the slowest deliveries?"
    python -m lakehouse_agent.cli --demo      # run the sample questions and save a transcript
"""

from __future__ import annotations

import argparse
import sys

from tabulate import tabulate

from lakehouse_agent.agent import AgentResult, LakehouseAgent
from lakehouse_agent.config import DATA_DIR, DB_PATH
from lakehouse_agent.llm import OFFLINE_BANNER

DEMO_QUESTIONS = [
    "What tables are in the gold layer?",
    "What are the primary key and foreign keys of fact_order_items?",
    "What is the lineage of gold.agg_seller_performance?",
    "What are the top 5 product categories by revenue?",
    "Which 5 customer states have the longest average delivery time?",
    "What is the average review score of late versus on-time deliveries?",
]


def render(res: AgentResult, max_rows: int = 15) -> str:
    out = [f"Q: {res.question}", "", res.answer, ""]
    if res.sql:
        out += ["SQL:", res.sql, ""]
        table = tabulate(res.rows[:max_rows], headers=res.columns, tablefmt="github", floatfmt=",.2f")
        out += [table]
        if len(res.rows) > max_rows:
            out.append(f"... {len(res.rows) - max_rows} more rows")
        out.append("")
    if res.explanation:
        out += [f"How: {res.explanation}"]
    meta = f"[{res.mode}:{res.model}] tool calls={res.tool_calls} sql attempts={res.sql_attempts} latency={res.latency_s:.2f}s"
    if res.mode != "mock":
        meta += f" tokens={res.prompt_tokens + res.completion_tokens:,} cost=${res.cost_usd:.4f}"
    out.append(meta)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the lakehouse SQL agent a question.")
    parser.add_argument("question", nargs="*", help="natural-language question")
    parser.add_argument("--demo", action="store_true", help="run the built-in sample questions")
    args = parser.parse_args()
    if not DB_PATH.exists():
        print(f"Lakehouse not found at {DB_PATH}. Run `make data build` first.", file=sys.stderr)
        return 1
    questions = DEMO_QUESTIONS if args.demo or not args.question else [" ".join(args.question)]

    agent = LakehouseAgent()
    if agent.llm.is_mock:
        print(f"*** {OFFLINE_BANNER} ***\n")
    blocks = []
    for q in questions:
        block = render(agent.ask(q))
        blocks.append(block)
        print(block)
        print("\n" + "=" * 90 + "\n")
    if args.demo:
        path = DATA_DIR / "demo_output.md"
        path.write_text("\n\n---\n\n".join(f"```\n{b}\n```" for b in blocks))
        print(f"Transcript saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
