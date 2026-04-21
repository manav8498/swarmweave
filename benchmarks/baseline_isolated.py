"""Run one task under the isolated-context baseline."""

from __future__ import annotations

from typing import Any

from benchmarks._isolated_backend import IsolatedBackend
from swarmweave import SharedContext, Supervisor, Worker
from swarmweave._openai_client import OpenAIClient


def make_workers(lookup: Any) -> list[Worker]:
    """3 workers with progressively-dependent roles.

    In sequential mode with shared context, beta and gamma can read alpha's
    findings and build on them. In the isolated baseline each worker starts
    fresh and must re-derive everything. This is the comparison the
    benchmark is designed to measure.
    """
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


async def run_isolated(question: str, lookup: Any, client: OpenAIClient) -> str:
    """Run the swarm on ``question`` with no shared context (isolated baseline).

    Uses sequential mode so the comparison vs. shared-context is meaningful:
    in parallel mode both conditions would produce identical results because
    workers that start simultaneously never see each other's writes either
    way. Sequential mode runs workers one after another; the IsolatedBackend
    returns empty on every read, modeling the standard "each subagent has
    only its own private context" pattern.
    """
    ctx = SharedContext(backend=IsolatedBackend())
    supervisor = Supervisor(
        workers=make_workers(lookup),
        shared_context=ctx,
        client=client,
        mode="sequential",
    )
    result = await supervisor.arun(question)
    return result.final_output
