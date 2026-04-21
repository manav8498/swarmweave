"""Test fixtures and a fake OpenAIClient for hermetic tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from swarmweave._openai_client import CallResult, ClientStats, ToolCall


@dataclass
class FakeResponse:
    """Test-side description of a single chat-completion turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def text_response(text: str) -> FakeResponse:
    return FakeResponse(text=text)


def tool_call_response(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> FakeResponse:
    return FakeResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def multi_tool_response(calls: list[tuple[str, dict[str, Any]]]) -> FakeResponse:
    return FakeResponse(
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls, 1)
        ]
    )


class FakeOpenAIClient:
    """Quacks like :class:`OpenAIClient` but returns canned responses.

    ``responder`` is called for every ``call`` invocation and must return a
    :class:`FakeResponse`. The ``finish_reason`` is inferred: if any tool
    calls are present it is set to ``tool_calls``, else ``stop``.
    """

    def __init__(
        self,
        responder: Callable[[dict[str, Any]], FakeResponse],
        model: str = "gpt-4o-mini",
    ) -> None:
        self.responder = responder
        self.model = model
        self.max_tokens = 1024
        self.stats = ClientStats()

    async def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        parallel_tool_calls: bool = True,
    ) -> CallResult:
        ctx = {
            "system": system,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "model": model,
        }
        response = self.responder(ctx)
        finish = "tool_calls" if response.tool_calls else "stop"
        raw_assistant: dict[str, Any] = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            import json as _json

            raw_assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
        self.stats.calls += 1
        self.stats.input_tokens += 10
        self.stats.output_tokens += 20
        return CallResult(
            text=response.text,
            tool_calls=list(response.tool_calls),
            finish_reason=finish,
            raw_assistant_message=raw_assistant,
            input_tokens=10,
            output_tokens=20,
        )


@pytest.fixture
def fake_client_factory() -> Callable[[Callable[[dict[str, Any]], FakeResponse]], FakeOpenAIClient]:
    def _factory(
        responder: Callable[[dict[str, Any]], FakeResponse],
    ) -> FakeOpenAIClient:
        return FakeOpenAIClient(responder=responder)

    return _factory
