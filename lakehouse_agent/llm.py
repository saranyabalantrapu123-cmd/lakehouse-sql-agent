"""Shared LLM access layer (same pattern across the portfolio repos).

* Real mode: LiteLLM `completion()` with OpenAI-style tool calling. The model comes from
  `LLM_MODEL` (default `anthropic/claude-sonnet-4-5`), the key from `ANTHROPIC_API_KEY`
  (or `OPENAI_API_KEY` for `openai/...` models).
* Offline demo mode: when `LLM_MODEL=mock` or no key is present, a deterministic `MockLLM`
  is used instead. It delegates to a project-specific rule-based responder so the full
  pipeline runs with no key and no network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
OFFLINE_BANNER = "Running in offline demo mode (no LLM key) — outputs are rule-based"

Message = dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def as_message(self) -> Message:
        """Render as an OpenAI-format assistant message for the conversation history."""
        msg: Message = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    model: str
    is_mock: bool

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


def _model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _has_key_for(model: str) -> bool:
    if model.startswith("openai/") or model.startswith("gpt-"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if model.startswith("anthropic/") or model.startswith("claude"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    # Other LiteLLM providers: accept any of the common keys.
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))


def is_offline() -> bool:
    """True when we should use the deterministic mock instead of a real model."""
    model = _model_name()
    return model.lower() == "mock" or not _has_key_for(model)


class LiteLLMClient:
    """Thin wrapper over `litellm.completion` that normalises tool calls and usage."""

    is_mock = False

    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int = 1500):
        self.model = model or _model_name()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        import litellm  # imported lazily so offline mode never needs it at import time

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = litellm.completion(**kwargs)
        choice = response.choices[0].message

        calls: list[ToolCall] = []
        for tc in getattr(choice, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage()
        if getattr(response, "usage", None):
            usage.prompt_tokens = int(response.usage.prompt_tokens or 0)
            usage.completion_tokens = int(response.usage.completion_tokens or 0)
        try:
            usage.cost_usd = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # pricing unknown for this model: report tokens only
            usage.cost_usd = 0.0
        return LLMResponse(content=choice.content or "", tool_calls=calls, usage=usage)


MockResponder = Callable[[list[Message], list[dict[str, Any]] | None], LLMResponse]


class MockLLM:
    """Deterministic stand-in for a real model. Zero tokens, zero cost, no network."""

    is_mock = True
    model = "mock"

    def __init__(self, responder: MockResponder):
        self._responder = responder

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        return self._responder(messages, tools)


def get_llm(mock_responder: MockResponder) -> LLMClient:
    """Return a real LiteLLM client when a key is configured, else the MockLLM."""
    if is_offline():
        return MockLLM(mock_responder)
    return LiteLLMClient()
