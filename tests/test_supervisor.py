"""Unit tests for Supervisor task decomposition + synthesis."""

from __future__ import annotations

from typing import Any

from swarmweave import SharedContext, Supervisor, Worker
from swarmweave.backends import LocalBackend
from tests.conftest import FakeOpenAIClient, text_response, tool_call_response


def _supervisor(client: FakeOpenAIClient, workers: list[Worker]) -> Supervisor:
    return Supervisor(
        workers=workers,
        shared_context=SharedContext(backend=LocalBackend(persist=False)),
        client=client,  # type: ignore[arg-type]
    )


async def test_decompose_returns_one_task_per_worker() -> None:
    workers = [Worker(name="a", role="A"), Worker(name="b", role="B")]
    client = FakeOpenAIClient(
        responder=lambda _: tool_call_response(
            "assign_subtasks",
            {
                "assignments": [
                    {"worker": "a", "subtask": "do A"},
                    {"worker": "b", "subtask": "do B"},
                ]
            },
        )
    )
    sup = _supervisor(client, workers)
    tasks = await sup.adecompose("the user task")
    assert {t.assigned_to for t in tasks} == {"a", "b"}


async def test_decompose_falls_back_when_tool_missing() -> None:
    workers = [Worker(name="a", role="role A"), Worker(name="b", role="role B")]
    client = FakeOpenAIClient(responder=lambda _: text_response("oops, plain text"))
    sup = _supervisor(client, workers)
    tasks = await sup.adecompose("user task")
    assert len(tasks) == 2
    assert all(t.assigned_to in {"a", "b"} for t in tasks)


async def test_decompose_fills_missing_workers() -> None:
    workers = [Worker(name="a", role="A"), Worker(name="b", role="B")]
    client = FakeOpenAIClient(
        responder=lambda _: tool_call_response(
            "assign_subtasks",
            {"assignments": [{"worker": "a", "subtask": "do A"}]},
        )
    )
    sup = _supervisor(client, workers)
    tasks = await sup.adecompose("user task")
    assert {t.assigned_to for t in tasks} == {"a", "b"}


async def test_decompose_drops_unknown_workers() -> None:
    workers = [Worker(name="a", role="A")]
    client = FakeOpenAIClient(
        responder=lambda _: tool_call_response(
            "assign_subtasks",
            {
                "assignments": [
                    {"worker": "ghost", "subtask": "nope"},
                    {"worker": "a", "subtask": "ok"},
                ]
            },
        )
    )
    sup = _supervisor(client, workers)
    tasks = await sup.adecompose("u")
    assert [t.assigned_to for t in tasks] == ["a"]


async def test_synthesize_concatenates_outputs() -> None:
    workers = [Worker(name="a", role="A")]
    captured: dict[str, Any] = {}

    def responder(ctx: dict[str, Any]) -> Any:
        captured["msgs"] = ctx["messages"]
        return text_response("final synthesis")

    client = FakeOpenAIClient(responder=responder)
    sup = _supervisor(client, workers)
    out = await sup.asynthesize("user task", {"a": "alpha output", "b": "beta output"})
    assert out == "final synthesis"
    assert "alpha output" in captured["msgs"][0]["content"]
    assert "beta output" in captured["msgs"][0]["content"]


def test_supervisor_requires_unique_worker_names() -> None:
    import pytest

    with pytest.raises(ValueError):
        Supervisor(
            workers=[Worker(name="a", role="A"), Worker(name="a", role="A2")],
            shared_context=SharedContext(backend=LocalBackend(persist=False)),
        )


def test_supervisor_requires_at_least_one_worker() -> None:
    import pytest

    with pytest.raises(ValueError):
        Supervisor(workers=[], shared_context=SharedContext(backend=LocalBackend(persist=False)))
