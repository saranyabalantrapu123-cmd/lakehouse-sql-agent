"""Execution-accuracy eval: run the agent on evals/questions.yaml and compare result sets.

Usage:
    python -m evals.run_eval                 # run the agent (mock if no key) and score it
    python -m evals.run_eval --check-gold    # only verify every gold SQL runs on the built data

Reports execution accuracy, average tool calls, latency, tokens and cost. In offline mode the
report is clearly labelled MOCK: it checks the pipeline end-to-end, not model quality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import duckdb
import yaml

from evals.compare import results_match
from lakehouse_agent.agent import LakehouseAgent
from lakehouse_agent.config import DATA_DIR, DB_PATH
from lakehouse_agent.llm import OFFLINE_BANNER
from lakehouse_agent.tools import to_jsonable

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_gold(sql: str, db_path: Path = DB_PATH) -> list[list]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return [[to_jsonable(v) for v in row] for row in con.execute(sql).fetchall()]


def check_gold(questions: list[dict], db_path: Path = DB_PATH) -> int:
    failures = 0
    for q in questions:
        try:
            rows = run_gold(q["sql"], db_path)
            print(f"  OK   {q['id']}  {len(rows):>3} row(s)  e.g. {rows[0] if rows else '-'}")
        except duckdb.Error as exc:
            failures += 1
            print(f"  FAIL {q['id']}  {exc}")
    print(f"{len(questions) - failures}/{len(questions)} gold queries executed successfully.")
    return failures


def evaluate(questions: list[dict], agent: LakehouseAgent, db_path: Path = DB_PATH) -> dict:
    records = []
    for q in questions:
        gold = run_gold(q["sql"], db_path)
        res = agent.ask(q["question"])
        correct = res.sql is not None and results_match(gold, res.rows)
        records.append({
            "id": q["id"],
            "question": q["question"],
            "correct": correct,
            "answered_with_sql": res.sql is not None,
            "tool_calls": res.tool_calls,
            "sql_attempts": res.sql_attempts,
            "latency_s": res.latency_s,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "cost_usd": res.cost_usd,
            "agent_sql": res.sql,
            "error": res.error,
        })
        mark = "PASS" if correct else ("FAIL" if res.sql else "SKIP")
        print(f"  {mark}  {q['id']}  tools={res.tool_calls}  {res.latency_s:.2f}s  {q['question'][:70]}")

    n = len(records)
    summary = {
        "mode": "mock" if agent.llm.is_mock else "llm",
        "model": agent.llm.model,
        "questions": n,
        "execution_accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "answered_with_sql": sum(r["answered_with_sql"] for r in records),
        "avg_tool_calls": statistics.mean(r["tool_calls"] for r in records) if n else 0.0,
        "avg_latency_s": statistics.mean(r["latency_s"] for r in records) if n else 0.0,
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in records),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 6),
    }
    return {"summary": summary, "records": records}


def print_summary(summary: dict) -> None:
    label = "MOCK (rule-based router, no LLM)" if summary["mode"] == "mock" else f"LLM: {summary['model']}"
    print("\nEVAL SUMMARY —", label)
    print("-" * 64)
    print(f"Execution accuracy : {summary['execution_accuracy']:.0%} "
          f"({round(summary['execution_accuracy'] * summary['questions'])}/{summary['questions']})")
    print(f"Answered with SQL  : {summary['answered_with_sql']}/{summary['questions']}")
    print(f"Avg tool calls     : {summary['avg_tool_calls']:.2f}")
    print(f"Avg latency        : {summary['avg_latency_s']:.3f}s")
    print(f"Total tokens       : {summary['total_tokens']:,}")
    print(f"Total cost (USD)   : ${summary['total_cost_usd']:.4f}")
    if summary["mode"] == "mock":
        print("\nNOTE: mock numbers measure the pipeline (routing, guardrails, execution, scoring)")
        print("with a hand-written router; they are NOT a measure of LLM accuracy. Unmatched")
        print("questions are skipped honestly. Run with ANTHROPIC_API_KEY set for real numbers.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-gold", action="store_true", help="only verify the gold SQL runs")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    args = parser.parse_args()
    if not args.db.exists():
        print(f"Lakehouse not found at {args.db}. Run `make data build` first.", file=sys.stderr)
        return 1

    questions = load_questions()[: args.limit]
    if args.check_gold:
        return 1 if check_gold(questions, args.db) else 0

    agent = LakehouseAgent()
    if agent.llm.is_mock:
        print(OFFLINE_BANNER)
    print(f"Evaluating {len(questions)} questions ...")
    report = evaluate(questions, agent, args.db)
    print_summary(report["summary"])
    out = DATA_DIR / f"eval_results_{report['summary']['mode']}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nPer-question results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
