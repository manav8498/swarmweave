"""Research swarm: three parallel researchers. Supervisor synthesizes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import (
    enable_progress_logging,
    print_banner,
    print_section,
    render_trace,
    require_api_key,
)

from swarmweave import SharedContext, Supervisor, Worker
from swarmweave.backends import LocalBackend

QUESTION = (
    "What are the most important shifts happening in AI memory systems "
    "for autonomous agents over the next 12 months?"
)
USER_TASK = QUESTION  # required by `swarmweave watch`


def build_swarm(events=None):  # type: ignore[no-untyped-def]
    """Used by `swarmweave watch`. Returns a fresh Supervisor."""
    return Supervisor(
        workers=[
            Worker(
                name="papers", role="Survey recent academic papers on agent memory architectures."
            ),
            Worker(
                name="industry", role="Survey industry reports, blog posts, and product launches."
            ),
            Worker(name="open_source", role="Survey active open-source projects in the area."),
        ],
        shared_context=SharedContext(backend=LocalBackend(persist=False), events=events),
        model="gpt-4o-mini",
        mode="parallel",
        events=events,
    )


def main() -> None:
    require_api_key()
    enable_progress_logging()
    print_banner("research swarm")
    print(f"Question: {QUESTION}\n")

    ctx = SharedContext(backend=LocalBackend(persist=False))

    workers = [
        Worker(
            name="papers",
            role="Survey recent academic papers on agent memory architectures.",
        ),
        Worker(
            name="industry",
            role="Survey industry reports, blog posts, and product launches.",
        ),
        Worker(
            name="open_source",
            role="Survey active open-source projects in the area.",
        ),
    ]

    # Three parallel researchers. The supervisor's built-in synthesis reads
    # each worker's output and produces the final cited report — no
    # dedicated "synthesizer" worker needed.
    supervisor = Supervisor(
        workers=workers,
        shared_context=ctx,
        model="gpt-4o-mini",
        mode="parallel",
    )

    print_section("running swarm")
    result = supervisor.run(QUESTION)

    print_section("final report")
    print(result.final_output)

    render_trace(result.trace)

    print_section("metrics")
    m = result.metrics
    print(
        f"  workers: {len(result.per_worker_outputs)}  "
        f"calls: {m.worker_calls}  "
        f"tokens in/out: {m.input_tokens}/{m.output_tokens}  "
        f"wall: {m.wall_clock_seconds}s"
    )


if __name__ == "__main__":
    main()
