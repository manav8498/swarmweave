"""Shared helpers for example scripts (graceful API-key handling, banners)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


def enable_progress_logging() -> None:
    """Stream INFO-level logs from the library to stdout so progress is visible live."""
    logger = logging.getLogger("swarmweave")
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def require_api_key() -> str | None:
    """Return the OpenAI API key or print a friendly message and exit 0."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(
            "OPENAI_API_KEY is not set.\n"
            "Set it (e.g. `export OPENAI_API_KEY=sk-...`) and re-run "
            "this example. See .env.example at the repo root for the template."
        )
        sys.exit(0)
    return key


def print_banner(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def render_trace(trace: list[Any]) -> None:
    print_section("Shared-context trace")
    for obs in trace:
        snippet = obs.content.replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        print(f"  [{obs.agent_id}] {snippet}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
