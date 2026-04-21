"""End-to-end integration: Supervisor.run against LocalBackend with a fake LLM."""

from __future__ import annotations

import re
from typing import Any

from swarmweave import SharedContext, Supervisor, Worker
from swarmweave.backends import LocalBackend
from tests.conftest import FakeOpenAIClient, text_response, tool_call_response


def _make_responder() -> Any:
    """Return a stateful responder that mimics a full swarm run."""

    state = {"synth_seen": False}

    def responder(ctx: dict[str, Any]) -> Any:
        system = ctx["system"]
        if "supervisor of a small expert swarm" in system and ctx.get("tools"):
            return tool_call_response(
                "assign_subtasks",
                {
                    "assignments": [
                        {"worker": "alpha", "subtask": "investigate angle A"},
                        {"worker": "beta", "subtask": "investigate angle B"},
                        {"worker": "gamma", "subtask": "synthesize"},
                    ]
                },
            )
        if "supervisor of a small expert swarm" in system:
            state["synth_seen"] = True
            return text_response("FINAL: integrated answer drawing on alpha, beta and gamma.")
        # Worker call — return a unique finding based on the subtask.
        msg = ctx["messages"][0]["content"]
        match = re.search(r"Your subtask:\n(.*?)\n", msg)
        subtask = match.group(1) if match else "unknown"
        return text_response(f"FINDING: did {subtask}")

    return responder, state


async def test_full_swarm_run() -> None:
    workers = [
        Worker(name="alpha", role="angle A specialist"),
        Worker(name="beta", role="angle B specialist"),
        Worker(name="gamma", role="synthesizer"),
    ]
    responder, state = _make_responder()
    client = FakeOpenAIClient(responder=responder)
    sup = Supervisor(
        workers=workers,
        shared_context=SharedContext(backend=LocalBackend(persist=False)),
        client=client,  # type: ignore[arg-type]
    )

    result = await sup.arun("Tell me about X")
    assert result.final_output.startswith("FINAL")
    assert set(result.per_worker_outputs.keys()) == {"alpha", "beta", "gamma"}
    assert state["synth_seen"] is True
    assert result.metrics.input_tokens > 0
    assert result.metrics.output_tokens > 0
    assert result.metrics.worker_calls >= 4
    assert len({o.agent_id for o in result.trace}) == 3


async def test_swarm_writes_findings_to_shared_context() -> None:
    workers = [
        Worker(name="alpha", role="A"),
        Worker(name="beta", role="B"),
        Worker(name="gamma", role="G"),
    ]
    responder, _ = _make_responder()
    client = FakeOpenAIClient(responder=responder)
    ctx = SharedContext(backend=LocalBackend(persist=False))
    sup = Supervisor(workers=workers, shared_context=ctx, client=client)  # type: ignore[arg-type]

    await sup.arun("user task")
    trace = await ctx.aget_full_trace()
    assert {o.agent_id for o in trace} == {"alpha", "beta", "gamma"}
    assert all("FINDING" in o.content for o in trace)


async def test_sequential_mode_runs_workers_in_declaration_order() -> None:
    """In sequential mode, each worker sees prior workers' writes via SharedContext."""

    call_order: list[str] = []
    contexts_seen_by_worker: dict[str, str] = {}

    def responder(ctx: dict[str, Any]) -> Any:
        system = ctx["system"]
        if "supervisor of a small expert swarm" in system and ctx.get("tools"):
            return tool_call_response(
                "assign_subtasks",
                {
                    "assignments": [
                        {"worker": "classifier", "subtask": "classify"},
                        {"worker": "resolver", "subtask": "draft"},
                        {"worker": "verifier", "subtask": "check"},
                    ]
                },
            )
        if "supervisor of a small expert swarm" in system:
            return text_response("FINAL: verified")

        msg = ctx["messages"][0]["content"]
        for name in ("classifier", "resolver", "verifier"):
            if f"You are {name}" in ctx["system"]:
                call_order.append(name)
                contexts_seen_by_worker[name] = msg
                return text_response(f"FINDING: {name} did its job")
        return text_response("FINDING: unknown")

    workers = [
        Worker(name="classifier", role="Classify."),
        Worker(name="resolver", role="Draft."),
        Worker(name="verifier", role="Verify."),
    ]
    client = FakeOpenAIClient(responder=responder)
    ctx = SharedContext(backend=LocalBackend(persist=False))
    sup = Supervisor(
        workers=workers,
        shared_context=ctx,
        client=client,  # type: ignore[arg-type]
        mode="sequential",
    )

    result = await sup.arun("triage this")

    # Workers executed in declaration order.
    assert call_order == ["classifier", "resolver", "verifier"]
    # classifier saw no prior context; resolver saw classifier's write; verifier saw both.
    assert "no prior shared context yet" in contexts_seen_by_worker["classifier"]
    assert "classifier did its job" in contexts_seen_by_worker["resolver"]
    assert "classifier did its job" in contexts_seen_by_worker["verifier"]
    assert "resolver did its job" in contexts_seen_by_worker["verifier"]
    assert result.final_output == "FINAL: verified"


def test_supervisor_rejects_bad_mode() -> None:
    import pytest

    with pytest.raises(ValueError):
        Supervisor(
            workers=[Worker(name="a", role="A")],
            shared_context=SharedContext(backend=LocalBackend(persist=False)),
            mode="nonsense",  # type: ignore[arg-type]
        )
