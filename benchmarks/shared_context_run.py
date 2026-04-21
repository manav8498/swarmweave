"""Run one task under the swarmweave shared-context condition."""

from __future__ import annotations

from typing import Any

from swarmweave import SharedContext, Supervisor, Worker
from swarmweave._openai_client import OpenAIClient
from swarmweave.backends import LocalBackend


def make_workers(lookup: Any) -> list[Worker]:
    """Same 3-worker pipeline as baseline_isolated — only the backend differs."""
    return [
        Worker(
            name="alpha",
            role=(
                "Call lookup() once to retrieve the evidence. Extract and summarize "
                "the facts in the evidence that are relevant to the question. "
                "End with 'FINDING: <concise facts>'."
            ),
            tools=[lookup],
        ),
        Worker(
            name="beta",
            role=(
                "Read alpha's facts from shared context (if available) or call "
                "lookup() yourself. Apply the facts to the question and produce a "
                "draft answer. End with 'FINDING: <draft answer>'."
            ),
            tools=[lookup],
        ),
        Worker(
            name="gamma",
            role=(
                "You are the verifier. ALWAYS call lookup() first to get the "
                "raw evidence, even if prior findings are already in shared "
                "context. Then compare beta's draft answer (if available) "
                "against the evidence. Your final answer MUST include every "
                "specific fact from the evidence needed to answer the question "
                "— exact years, dates, numbers, names. Do NOT use vague phrases "
                "like 'this year' or 'the conference'; use the specific value "
                "from the evidence. End with "
                "'FINDING: <final answer with specific facts>'."
            ),
            tools=[lookup],
        ),
    ]


async def run_shared(question: str, lookup: Any, client: OpenAIClient) -> str:
    """Run the swarm on ``question`` with swarmweave SharedContext.

    Uses sequential mode so later workers can actually read earlier
    workers' findings from shared context. Identical structure to the
    isolated baseline (same model, worker count, tools, execution mode) —
    only the backend differs.
    """
    ctx = SharedContext(backend=LocalBackend(persist=False))
    supervisor = Supervisor(
        workers=make_workers(lookup),
        shared_context=ctx,
        client=client,
        mode="sequential",
    )
    result = await supervisor.arun(question)
    return result.final_output
