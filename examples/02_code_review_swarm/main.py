"""Code-review swarm: four parallel reviewers. Supervisor prioritizes."""

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

HERE = Path(__file__).resolve().parent


def load_sample_code() -> str:
    return (HERE / "sample_code.py").read_text(encoding="utf-8")


def make_review_tools(source: str) -> list:
    def get_source() -> str:
        """Return the full source code under review."""
        return source

    def grep(pattern: str) -> str:
        """Find lines matching ``pattern`` (substring match) in the source."""
        return (
            "\n".join(
                f"L{i + 1}: {line}" for i, line in enumerate(source.splitlines()) if pattern in line
            )
            or f"(no lines matched {pattern!r})"
        )

    return [get_source, grep]


def main() -> None:
    require_api_key()
    enable_progress_logging()
    source = load_sample_code()

    print_banner("code review swarm")
    print(f"reviewing {len(source.splitlines())} lines from sample_code.py\n")

    ctx = SharedContext(backend=LocalBackend(persist=False))
    tools = make_review_tools(source)

    workers = [
        Worker(
            name="security",
            role="Find security issues: injection, secrets, unsafe deserialization.",
            tools=tools,
        ),
        Worker(
            name="performance",
            role="Find performance issues: algorithmic complexity, IO patterns.",
            tools=tools,
        ),
        Worker(
            name="style",
            role="Find style and idiomatic-Python issues.",
            tools=tools,
        ),
        Worker(
            name="coverage",
            role="Find test-coverage and testability concerns.",
            tools=tools,
        ),
    ]

    # Four specialists fan out in parallel. The supervisor's synthesis
    # consolidates them into a single prioritized (P0/P1/P2) review — no
    # dedicated "aggregator" worker needed.
    supervisor = Supervisor(
        workers=workers,
        shared_context=ctx,
        model="gpt-4o-mini",
        mode="parallel",
    )

    print_section("running swarm")
    result = supervisor.run(
        "Review sample_code.py and produce a prioritized review "
        "(P0: must-fix / P1: high / P2: nice-to-have)."
    )

    print_section("prioritized review")
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
