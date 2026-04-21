"""Unit tests for the Worker class."""

from __future__ import annotations

from typing import Any

import pytest

from swarmweave import SharedContext, Worker
from swarmweave.backends import LocalBackend
from tests.conftest import FakeOpenAIClient, text_response, tool_call_response


def _ctx() -> SharedContext:
    return SharedContext(backend=LocalBackend(persist=False))


def test_worker_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        Worker(name="", role="x")
    with pytest.raises(ValueError):
        Worker(name="bad name with spaces", role="x")


async def test_worker_runs_without_tools() -> None:
    ctx = _ctx()
    client = FakeOpenAIClient(responder=lambda _: text_response("FINDING: done"))
    worker = Worker(name="w1", role="r")
    obs = await worker.arun("subtask alpha", ctx, client)  # type: ignore[arg-type]
    assert "FINDING" in obs.content
    trace = await ctx.aget_full_trace()
    assert len(trace) == 1
    assert trace[0].agent_id == "w1"


async def test_worker_invokes_tool_then_finishes() -> None:
    state = {"step": 0}

    def lookup(query: str) -> str:
        """Look up information."""
        return f"result for {query}"

    def responder(ctx: dict[str, Any]) -> Any:
        state["step"] += 1
        if state["step"] == 1:
            return tool_call_response("lookup", {"query": "foo"})
        return text_response("FINDING: synthesized result for foo")

    ctx = _ctx()
    client = FakeOpenAIClient(responder=responder)
    worker = Worker(name="w1", role="r", tools=[lookup])
    obs = await worker.arun("answer foo", ctx, client)  # type: ignore[arg-type]
    assert "synthesized result" in obs.content
    assert state["step"] == 2


async def test_worker_tool_loop_bounded() -> None:
    def loop_tool() -> str:
        """Loop forever."""
        return "again"

    def responder(_: dict[str, Any]) -> Any:
        return tool_call_response("loop_tool", {})

    ctx = _ctx()
    client = FakeOpenAIClient(responder=responder)
    worker = Worker(name="w1", role="r", tools=[loop_tool], max_tool_iterations=2)
    obs = await worker.arun("never finishes", ctx, client)  # type: ignore[arg-type]
    assert obs.agent_id == "w1"
    assert obs.content


async def test_worker_reads_shared_context_before_running() -> None:
    ctx = _ctx()
    from swarmweave import Observation

    await ctx.awrite(
        "other", Observation(agent_id="other", task="t", content="prior insight about X")
    )

    captured: dict[str, str] = {}

    def responder(call_ctx: dict[str, Any]) -> Any:
        captured["user"] = call_ctx["messages"][0]["content"]
        return text_response("FINDING: built on prior")

    client = FakeOpenAIClient(responder=responder)
    worker = Worker(name="w1", role="r")
    await worker.arun("synthesize about X", ctx, client)  # type: ignore[arg-type]

    assert "prior insight about X" in captured["user"]
