"""Agent loop: mock routing end-to-end, plus self-correction and give-up paths with a scripted LLM."""

from __future__ import annotations

from lakehouse_agent.agent import LakehouseAgent
from lakehouse_agent.llm import LLMResponse, MockLLM, ToolCall, Usage, get_llm, is_offline
from lakehouse_agent.mock_router import OFFLINE_CANNOT_ANSWER, mock_responder


def scripted(*steps):
    """A fake LLM that returns pre-baked responses in order (to drive specific graph paths)."""
    queue = list(steps)

    def responder(messages, tools):
        return queue.pop(0)

    return MockLLM(responder)


def sql_call(sql: str, n: int) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"c{n}", name="run_sql", arguments={"sql": sql})],
                       usage=Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.001))


def test_offline_mode_selected_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4-5")
    assert is_offline()
    assert get_llm(mock_responder).is_mock


def test_mock_list_tables(tools):
    res = LakehouseAgent(tools).ask("What tables are in the silver layer?")
    assert res.mode == "mock" and res.ok
    assert "silver.orders" in res.answer and res.sql is None
    assert res.tool_calls == 1


def test_mock_describe_and_lineage(tools):
    agent = LakehouseAgent(tools)
    d = agent.ask("What is the primary key of gold.fact_orders?")
    assert "Primary key: order_id" in d.answer
    lin = agent.ask("Where does gold.agg_daily_sales come from?")
    assert "bronze.orders" in lin.answer and "get_lineage" in lin.explanation


def test_mock_analytic_question_runs_sql(tools):
    res = LakehouseAgent(tools).ask("What are the top 2 product categories by revenue?")
    assert res.ok and res.sql and "gold.fact_order_items" in res.sql
    assert res.rows == [["unknown", 200.0], ["health_beauty", 180.0]]  # canceled pc_gamer order excluded
    assert "Layer: gold" in res.explanation
    assert res.tool_calls == 2 and res.sql_attempts == 1


def test_mock_join_explanation(tools):
    res = LakehouseAgent(tools).ask("Which customer states have the most orders?")
    assert "foreign key -> primary key" in res.explanation
    assert {r[0] for r in res.rows} == {"SP", "RJ", "MG"}


def test_mock_unknown_question_is_honest(tools):
    res = LakehouseAgent(tools).ask("Write me a poem about logistics")
    assert res.answer == OFFLINE_CANNOT_ANSWER
    assert res.sql is None and res.tool_calls == 0


def test_self_correction_after_execution_error(tools):
    llm = scripted(
        sql_call("SELECT nope FROM gold.fact_orders", 1),
        sql_call("SELECT count(*) AS n FROM gold.fact_orders", 2),
        LLMResponse(content="There are 4 orders.", usage=Usage(50, 10, 0.0005)),
    )
    res = LakehouseAgent(tools, llm).ask("How many orders?")
    assert res.ok and res.rows == [[4]]
    assert res.sql_attempts == 2 and res.retries == 1
    assert [(t["step"], t["ok"]) for t in res.trace] == [
        ("validate", True), ("execute", False), ("validate", True), ("execute", True)]
    assert res.prompt_tokens == 250 and abs(res.cost_usd - 0.0025) < 1e-9


def test_guardrail_block_then_fix(tools):
    llm = scripted(
        sql_call("DROP TABLE gold.fact_orders", 1),
        sql_call("SELECT count(*) AS n FROM gold.fact_orders", 2),
        LLMResponse(content="4 orders."),
    )
    res = LakehouseAgent(tools, llm).ask("How many orders?")
    assert res.ok and res.rows == [[4]]
    assert res.trace[0]["step"] == "validate" and res.trace[0]["ok"] is False


def test_gives_up_after_two_retries(tools):
    bad = [sql_call("SELECT nope FROM gold.fact_orders", i) for i in range(3)]
    res = LakehouseAgent(tools, scripted(*bad)).ask("How many orders?")
    assert not res.ok and res.sql is None
    assert res.sql_attempts == 3
    assert "could not answer" in res.answer.lower()
