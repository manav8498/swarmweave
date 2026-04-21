"""Self-improving swarm: same task, run twice, watch it get better.

This example demonstrates the lessons system. Round 1 runs cold. The Mentor
extracts lessons from the synthesis and persists them under
~/.swarmweave/lessons/<book>.jsonl. Round 2 preloads those lessons into shared
context before any worker fires — the swarm starts from a smarter prior.

Run:

    python examples/04_self_improving_swarm/main.py

To watch it live in the TUI:

    swarmweave watch examples/04_self_improving_swarm/main.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import enable_progress_logging, print_banner, print_section, require_api_key

from swarmweave import (
    EventBus,
    Mentor,
    SharedContext,
    Supervisor,
    Worker,
)
from swarmweave.backends import LocalBackend

USER_TASK = (
    "A junior dev on our team asks: 'how should I structure a Python project "
    "for a new internal CLI tool we're building?' Give them a concrete, "
    "opinionated answer covering: directory layout, dependency management, "
    "testing, and the one mistake they're most likely to make."
)

BOOK = "python_cli_advice"


def _make_workers() -> list[Worker]:
    return [
        Worker(
            name="structurer",
            role="Recommend a concrete directory layout and dependency-management approach for the project.",
        ),
        Worker(
            name="tester",
            role="Recommend the minimum viable testing setup (framework, layout, what to cover first).",
        ),
        Worker(
            name="reality_check",
            role=(
                "Read the structurer's and tester's findings from shared "
                "context and call out the ONE mistake the junior dev is "
                "most likely to make in practice."
            ),
        ),
    ]


def build_swarm(events: EventBus | None = None) -> Supervisor:
    """Used by `swarmweave watch`. Returns a fresh Supervisor."""
    return Supervisor(
        workers=_make_workers(),
        shared_context=SharedContext(backend=LocalBackend(persist=False), events=events),
        model="gpt-4o-mini",
        mode="sequential",
        events=events,
    )


async def _one_round(round_label: str, mentor: Mentor) -> None:
    print_section(f"ROUND {round_label} — preloading lessons")
    sup = build_swarm()
    preloaded = await mentor.before_run(sup, USER_TASK)
    print(f"  preloaded {preloaded} lesson(s) from past runs")

    print_section(f"ROUND {round_label} — running swarm")
    result = await sup.arun(USER_TASK)

    print_section(f"ROUND {round_label} — final answer")
    print(result.final_output)

    print_section(f"ROUND {round_label} — extracting lessons")
    new_lessons = await mentor.after_run(sup, USER_TASK, result)
    if new_lessons:
        for lesson in new_lessons:
            print(f"  • {lesson}")
    else:
        print("  (no new lessons recorded)")

    m = result.metrics
    print(
        f"\n  metrics: {len(result.per_worker_outputs)} workers · "
        f"{m.worker_calls} calls · "
        f"{m.input_tokens}/{m.output_tokens} tokens · "
        f"{m.wall_clock_seconds}s"
    )


async def main_async() -> None:
    require_api_key()
    enable_progress_logging()

    print_banner("self-improving swarm")
    print(f"Task: {USER_TASK}\n")

    mentor = Mentor(book=BOOK)
    book_size_before = mentor.book.count()
    print(f"lesson book '{BOOK}' currently holds {book_size_before} lesson(s)")

    await _one_round(
        "1 (cold)" if book_size_before == 0 else f"N (book has {book_size_before})", mentor
    )
    await _one_round("N+1 (with lessons from this run preloaded)", mentor)

    print_section("Result")
    print(
        f"Lesson book grew from {book_size_before} to {mentor.book.count()} entries.\n"
        f"Run this script again — round 1 will start with a smarter prior than ever before."
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
