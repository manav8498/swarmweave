"""Unit tests for SharedContext + LocalBackend."""

from __future__ import annotations

import asyncio

import pytest

from swarmweave import Observation, Outcome, SharedContext
from swarmweave.backends import LocalBackend


def _ctx() -> SharedContext:
    return SharedContext(backend=LocalBackend(persist=False))


async def test_write_then_read_returns_observation() -> None:
    ctx = _ctx()
    obs = Observation(agent_id="a", task="find foo bar", content="foo bar baz")
    await ctx.awrite("a", obs)
    slice_ = await ctx.aread(agent_id="b", task="foo bar")
    assert any(o.id == obs.id for o in slice_.observations)


async def test_read_excludes_self() -> None:
    ctx = _ctx()
    obs = Observation(agent_id="a", task="t", content="hello world")
    await ctx.awrite("a", obs)
    slice_ = await ctx.aread(agent_id="a", task="hello")
    assert all(o.agent_id != "a" for o in slice_.observations)


async def test_read_relevance_ordering() -> None:
    ctx = _ctx()
    await ctx.awrite("a", Observation(agent_id="a", task="cats", content="cats are mammals"))
    await ctx.awrite("a", Observation(agent_id="a", task="cars", content="cars have wheels"))
    slice_ = await ctx.aread(agent_id="b", task="mammals biology cats", limit=2)
    assert slice_.observations
    assert "cats" in slice_.observations[0].content


async def test_finalize_records_outcome() -> None:
    backend = LocalBackend(persist=False)
    ctx = SharedContext(backend=backend)
    await ctx.afinalize(Outcome(success=True, score=0.9, notes="ok"))
    assert backend.last_outcome is not None
    assert backend.last_outcome.success is True
    assert backend.last_outcome.score == 0.9


async def test_get_full_trace_preserves_order() -> None:
    ctx = _ctx()
    for i in range(3):
        await ctx.awrite("a", Observation(agent_id="a", task="t", content=f"obs-{i}"))
    trace = await ctx.aget_full_trace()
    assert [o.content for o in trace] == ["obs-0", "obs-1", "obs-2"]


def test_sync_wrappers_work_without_running_loop() -> None:
    ctx = _ctx()
    ctx.write("a", Observation(agent_id="a", task="t", content="hello"))
    trace = ctx.get_full_trace()
    assert any(o.content == "hello" for o in trace)


async def test_sync_wrapper_inside_loop_raises() -> None:
    ctx = _ctx()
    with pytest.raises(RuntimeError):
        # `read` would create a coroutine; we expect the sync guard to fire first.
        _ = ctx.read("a", "t")


async def test_concurrent_writes_are_safe() -> None:
    ctx = _ctx()

    async def writer(i: int) -> None:
        await ctx.awrite("a", Observation(agent_id="a", task="t", content=f"n-{i}"))

    await asyncio.gather(*(writer(i) for i in range(20)))
    trace = await ctx.aget_full_trace()
    assert len(trace) == 20
