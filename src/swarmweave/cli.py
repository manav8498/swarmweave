"""``swarmweave`` command-line entry point.

Subcommands:

* ``swarmweave watch <example.py> [args...]`` — run a Python script that
  builds a swarm, attach the live TUI dashboard, and stream every event.

The watched script must define a top-level ``def build_swarm(events) -> Supervisor``
that returns a Supervisor wired with the given EventBus, and (optionally)
a top-level ``USER_TASK`` string. The CLI calls
``await supervisor.arun(USER_TASK)`` inside the dashboard.

Example:

    swarmweave watch examples/03_support_triage_swarm/main.py

For arbitrary swarms, just construct the EventBus + LiveDashboard yourself
in your own script — see ``LiveDashboard`` docstring.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from swarmweave.events import EventBus
from swarmweave.tui import LiveDashboard


def _load_module(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _run_watch(script: Path, task_override: str | None) -> None:
    if not script.exists():
        print(f"error: {script} does not exist", file=sys.stderr)
        sys.exit(1)
    module = _load_module(script)
    builder = getattr(module, "build_swarm", None)
    if not callable(builder):
        print(
            f"error: {script} does not define a top-level build_swarm(events) function.",
            file=sys.stderr,
        )
        print(
            "Add: `def build_swarm(events): return Supervisor(..., events=events)`",
            file=sys.stderr,
        )
        sys.exit(1)
    user_task = task_override or getattr(module, "USER_TASK", None)
    if not user_task:
        print(
            f"error: {script} does not define USER_TASK and no --task was provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    bus = EventBus()
    supervisor = builder(bus)

    async with LiveDashboard(bus):
        await supervisor.arun(user_task)


def main() -> None:
    parser = argparse.ArgumentParser(prog="swarmweave")
    sub = parser.add_subparsers(dest="cmd", required=True)

    watch = sub.add_parser("watch", help="run a swarm with the live TUI dashboard")
    watch.add_argument("script", type=Path, help="path to a script defining build_swarm(events)")
    watch.add_argument(
        "--task",
        default=None,
        help="override the script's USER_TASK with a custom prompt",
    )

    args = parser.parse_args()

    if args.cmd == "watch":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        asyncio.run(_run_watch(args.script, args.task))


if __name__ == "__main__":
    main()
