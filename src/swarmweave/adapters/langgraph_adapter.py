"""LangGraph integration for swarmweave.

Builds a compiled :class:`StateGraph` that mirrors the Supervisor's
decompose → fan-out → synthesize pipeline. Exposed publicly so users who
already run a LangGraph application can compose a swarm in as a sub-graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

if TYPE_CHECKING:
    from swarmweave.supervisor import Supervisor


def _merge_outputs(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    if left:
        merged.update(left)
    if right:
        merged.update(right)
    return merged


class SwarmState(TypedDict, total=False):
    """LangGraph state for one swarm run."""

    user_task: str
    subtasks_json: str
    worker_outputs: Annotated[dict[str, str], _merge_outputs]
    final_output: str


def build_swarm_graph(supervisor: Supervisor) -> Any:
    """Compile a LangGraph that runs ``supervisor`` end-to-end.

    Topology depends on ``supervisor.mode``:

    * ``"parallel"`` — ``decompose -> Send(...) -> [worker, ...] -> synthesize``
    * ``"sequential"`` — ``decompose -> worker_chain -> synthesize`` (each
      worker runs one at a time so its :meth:`SharedContext.aread` sees every
      prior worker's findings).

    The returned graph is compatible with LangGraph's standard ``ainvoke`` /
    ``astream`` calls.
    """

    async def decompose_node(state: SwarmState) -> dict[str, Any]:
        tasks = await supervisor.adecompose(state["user_task"])
        return {"subtasks_json": supervisor._serialize_tasks(tasks)}

    async def synthesize_node(state: SwarmState) -> dict[str, Any]:
        outputs = state.get("worker_outputs", {}) or {}
        final = await supervisor.asynthesize(state["user_task"], outputs)
        return {"final_output": final}

    graph = StateGraph(SwarmState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "decompose")
    graph.add_edge("synthesize", END)

    if supervisor.mode == "sequential":

        async def worker_chain_node(state: SwarmState) -> dict[str, Any]:
            tasks = supervisor._deserialize_tasks(state.get("subtasks_json", ""))
            order = {w.name: i for i, w in enumerate(supervisor.workers)}
            tasks.sort(key=lambda t: order.get(t.assigned_to, 1_000_000))
            outputs: dict[str, str] = {}
            for t in tasks:
                obs = await supervisor._run_one_worker(t.assigned_to, t.description)
                outputs[t.assigned_to] = obs.content
            return {"worker_outputs": outputs}

        graph.add_node("worker_chain", worker_chain_node)
        graph.add_edge("decompose", "worker_chain")
        graph.add_edge("worker_chain", "synthesize")
    else:

        def fan_out(state: SwarmState) -> list[Send]:
            tasks = supervisor._deserialize_tasks(state.get("subtasks_json", ""))
            return [
                Send("worker", {"worker_name": t.assigned_to, "subtask": t.description})
                for t in tasks
            ]

        async def worker_node(payload: dict[str, str]) -> dict[str, Any]:
            worker_name = payload["worker_name"]
            subtask = payload["subtask"]
            observation = await supervisor._run_one_worker(worker_name, subtask)
            return {"worker_outputs": {worker_name: observation.content}}

        graph.add_node("worker", worker_node)  # type: ignore[arg-type]
        graph.add_conditional_edges("decompose", fan_out, ["worker"])
        graph.add_edge("worker", "synthesize")

    return graph.compile()
