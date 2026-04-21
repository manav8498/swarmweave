"""Real-world A/B test: review the swarmweave library with a 3-worker
sequential swarm, once under IsolatedBackend, once under LocalBackend.

This is NOT a synthetic benchmark. Workers use real filesystem tools
against the real repo, and their outputs are judgeable by reading them
side-by-side. The only variable between the two runs is the backend —
same model, same workers, same tools, same subtask decomposition,
same execution mode (sequential).

Usage:
    OPENAI_API_KEY=... python scripts/real_world_review.py

Writes:
    real_world_results/isolated_output.md
    real_world_results/shared_output.md
    real_world_results/summary.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _isolated_backend import IsolatedBackend  # noqa: E402

from swarmweave import SharedContext, Supervisor, Worker  # noqa: E402
from swarmweave._openai_client import OpenAIClient  # noqa: E402
from swarmweave.backends import LocalBackend  # noqa: E402

RESULTS_DIR = ROOT / "real_world_results"
RESULTS_DIR.mkdir(exist_ok=True)

USER_TASK = (
    "Review the swarmweave library source code (under swarmweave/ in "
    "this repo). Identify the ONE most important shippability concern for a "
    "v0.1 open-source release, investigate it concretely (with file/line "
    "citations from real code), and propose concrete fixes. This is for a "
    "public open-source launch — be decisive about what matters."
)


def _enable_logging() -> None:
    lib = logging.getLogger("swarmweave")
    if any(isinstance(h, logging.StreamHandler) for h in lib.handlers):
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    lib.addHandler(h)
    lib.setLevel(logging.INFO)


def build_tools(repo_root: Path) -> list:
    """Real filesystem tools scoped to the repo root."""

    def _safe(path: str) -> Path:
        full = (repo_root / path).resolve()
        if not str(full).startswith(str(repo_root)):
            raise ValueError(f"path {path!r} escapes repo root")
        return full

    def read_file(path: str) -> str:
        """Read a text file from the repo. Paths are relative to the repo root."""
        try:
            full = _safe(path)
            if not full.exists():
                return f"error: {path} does not exist"
            if not full.is_file():
                return f"error: {path} is not a file"
            text = full.read_text(encoding="utf-8", errors="replace")
            if len(text) > 12_000:
                return text[:12_000] + f"\n...(truncated; full size {len(text)} chars)"
            return text
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def list_dir(path: str) -> str:
        """List the contents of a directory in the repo. Use '.' for the repo root."""
        try:
            full = _safe(path if path else ".")
            if not full.is_dir():
                return f"error: {path} is not a directory"
            entries = sorted(full.iterdir())
            lines = []
            for e in entries:
                if e.name.startswith(".") or e.name == "__pycache__":
                    continue
                marker = "/" if e.is_dir() else ""
                lines.append(f"{e.name}{marker}")
            return "\n".join(lines) if lines else "(empty)"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def grep(pattern: str, path: str = "swarmweave") -> str:
        """Regex-search files under ``path`` for ``pattern``. Returns matching lines with file:line."""
        try:
            root = _safe(path)
            if not root.exists():
                return f"error: {path} does not exist"
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                return f"error: invalid regex: {exc}"
            files: list[Path] = []
            if root.is_file():
                files = [root]
            else:
                for p in root.rglob("*.py"):
                    if "__pycache__" in p.parts:
                        continue
                    files.append(p)
            hits: list[str] = []
            for f in files:
                try:
                    for i, line in enumerate(
                        f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                    ):
                        if rx.search(line):
                            rel = f.relative_to(repo_root)
                            hits.append(f"{rel}:{i}: {line.strip()}")
                            if len(hits) >= 60:
                                hits.append("...(truncated at 60 hits)")
                                return "\n".join(hits)
                except Exception:
                    continue
            return "\n".join(hits) if hits else f"(no matches for {pattern!r} in {path})"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    return [read_file, list_dir, grep]


def build_workers(tools: list) -> list[Worker]:
    """3-worker pipeline: scout identifies concerns, investigator deep-dives, fixer proposes patches."""
    return [
        Worker(
            name="scout",
            role=(
                "You SCAN the repo to build situational awareness. "
                "Call list_dir('swarmweave') first, then read a few of "
                "the most important files (__init__.py, supervisor.py, "
                "worker.py, shared_context.py, backends/local.py). "
                "Produce a numbered list of the top 3-5 concrete concerns "
                "you notice — each with a file:line reference and a one-"
                "sentence explanation of why it matters for a v0.1 "
                "open-source release."
            ),
            tools=tools,
            max_tool_iterations=8,
        ),
        Worker(
            name="investigator",
            role=(
                "You pick the SINGLE highest-priority concern from the "
                "scout's list (read the shared context to see it) and "
                "investigate it deeply. Use read_file and grep to gather "
                "concrete evidence — exact line numbers, related code in "
                "other files, whether tests cover it. Produce a detailed "
                "analysis: what's wrong, why it matters, and what code "
                "needs to change."
            ),
            tools=tools,
            max_tool_iterations=10,
        ),
        Worker(
            name="fixer",
            role=(
                "You read the investigator's analysis from shared context "
                "and propose 2-3 concrete, shippable fixes with exact "
                "file:line references and before/after code snippets. "
                "Each fix should be small enough to land as one PR. Say "
                "clearly which fix you'd land FIRST and why."
            ),
            tools=tools,
            max_tool_iterations=8,
        ),
    ]


async def run_condition(
    label: str,
    backend_factory: Any,
    client: OpenAIClient,
) -> tuple[str, dict[str, Any]]:
    print(f"\n{'=' * 60}\nRUNNING CONDITION: {label}\n{'=' * 60}")
    tools = build_tools(ROOT)
    workers = build_workers(tools)
    ctx = SharedContext(backend=backend_factory())
    sup = Supervisor(
        workers=workers,
        shared_context=ctx,
        client=client,
        mode="sequential",
    )
    start = time.perf_counter()
    result = await sup.arun(USER_TASK)
    elapsed = time.perf_counter() - start

    body = [
        f"# Real-world code-review run — {label}",
        "",
        f"Task: {USER_TASK}",
        "",
        f"Wall-clock: {elapsed:.1f}s",
        f"Calls: {result.metrics.worker_calls}",
        f"Tokens in/out: {result.metrics.input_tokens}/{result.metrics.output_tokens}",
        "",
        "## Final supervisor synthesis",
        "",
        result.final_output,
        "",
        "## Per-worker deliverables",
        "",
    ]
    for name in ("scout", "investigator", "fixer"):
        body.append(f"### {name}")
        body.append("")
        body.append(result.per_worker_outputs.get(name, "(missing)"))
        body.append("")
    body.append("## Shared-context trace")
    body.append("")
    for obs in result.trace:
        snippet = obs.content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        body.append(f"- [{obs.agent_id}] {snippet}")

    md = "\n".join(body)
    meta = {
        "label": label,
        "wall_clock_s": round(elapsed, 2),
        "worker_calls": result.metrics.worker_calls,
        "input_tokens": result.metrics.input_tokens,
        "output_tokens": result.metrics.output_tokens,
        "worker_names": list(result.per_worker_outputs.keys()),
        "trace_agents": [o.agent_id for o in result.trace],
        "final_output_len": len(result.final_output),
    }
    return md, meta


async def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Abort.")
        sys.exit(1)

    _enable_logging()

    # One client per condition so token stats don't pollute each other.
    iso_client = OpenAIClient(reasoning_effort="medium")
    sc_client = OpenAIClient(reasoning_effort="medium")

    iso_md, iso_meta = await run_condition("isolated", IsolatedBackend, iso_client)
    (RESULTS_DIR / "isolated_output.md").write_text(iso_md, encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'isolated_output.md'}")

    sc_md, sc_meta = await run_condition("shared", lambda: LocalBackend(persist=False), sc_client)
    (RESULTS_DIR / "shared_output.md").write_text(sc_md, encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'shared_output.md'}")

    summary = {"isolated": iso_meta, "shared": sc_meta, "task": USER_TASK}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS_DIR / 'summary.json'}")

    print(
        f"\nisolated: {iso_meta['wall_clock_s']}s, "
        f"{iso_meta['input_tokens']}/{iso_meta['output_tokens']} tokens, "
        f"final {iso_meta['final_output_len']} chars"
    )
    print(
        f"shared:   {sc_meta['wall_clock_s']}s, "
        f"{sc_meta['input_tokens']}/{sc_meta['output_tokens']} tokens, "
        f"final {sc_meta['final_output_len']} chars"
    )


if __name__ == "__main__":
    asyncio.run(main())
