"""Exercise the real LiteLLM client path offline using LiteLLM's built-in mock responses.

No network and no key: litellm short-circuits when `mock_response` / `mock_tool_calls` is set.
This verifies tool-call parsing, message round-tripping and token/cost accounting through the
full LangGraph loop. It does NOT measure any real model's behaviour.
"""

from __future__ import annotations

import litellm

from lakehouse_agent.agent import LakehouseAgent
from lakehouse_agent.llm import LiteLLMClient


def test_real_client_path_with_litellm_mocks(tools, monkeypatch):
    original = litellm.completion
    script = [
        {"mock_tool_calls": [{"id": "call_1", "type": "function",
                              "function": {"name": "describe_table", "arguments": '{"name": "gold.fact_orders"}'}}]},
        {"mock_tool_calls": [{"id": "call_2", "type": "function",
                              "function": {"name": "run_sql",
                                           "arguments": '{"sql": "SELECT count(*) AS n FROM gold.fact_orders"}'}}]},
        {"mock_response": "There are 4 orders in gold.fact_orders."},
    ]
    seen_messages = []

    def fake_completion(**kwargs):
        seen_messages.append(kwargs["messages"])
        assert kwargs["tools"] and kwargs["tool_choice"] == "auto"
        return original(api_key="test-key", **kwargs, **script.pop(0))

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient(model="anthropic/claude-sonnet-4-5")
    res = LakehouseAgent(tools, client).ask("How many orders are there?")

    assert res.mode == "llm" and res.ok
    assert res.rows == [[4]] and res.tool_calls == 2
    assert res.answer.startswith("There are 4 orders")
    assert res.prompt_tokens > 0 and res.completion_tokens > 0
    # Tool results were fed back with matching tool_call_ids.
    last = seen_messages[-1]
    assert [m["tool_call_id"] for m in last if m["role"] == "tool"] == ["call_1", "call_2"]
