"""Run the full benchmark: isolated baseline vs. shared-context.

Outputs:

* ``benchmarks/results/results.json`` — per-task and aggregate numbers.
* ``benchmarks/results/chart.png``   — bar chart comparing both conditions.

Both conditions run on the *identical* tasks, *identical* model, *identical*
worker count (3) and *identical* tools — only the context layer differs.
This is the spec-required honesty constraint: do not tune the baseline to
lose, do not hand-pick tasks, do not fabricate numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent))

from benchmarks.baseline_isolated import run_isolated  # noqa: E402
from benchmarks.shared_context_run import run_shared  # noqa: E402
from swarmweave._openai_client import OpenAIClient  # noqa: E402

JUDGE_SYSTEM = """You are a strict evaluator scoring an answer against a known
ground-truth answer. Score on three criteria, each 0 or 1:

1. factual_correctness — the answer agrees with the ground truth on every
   factual claim that the ground truth makes. (No partial credit.)
2. completeness — the answer addresses every part of the question; nothing
   important from the ground truth is missing.
3. groundedness — the answer cites or paraphrases the underlying evidence
   rather than asserting the result without support.

Respond by calling the score tool with the three integers and a one-line
rationale."""


JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "score",
        "description": "Record the three sub-scores and rationale.",
        "parameters": {
            "type": "object",
            "properties": {
                "factual_correctness": {"type": "integer"},
                "completeness": {"type": "integer"},
                "groundedness": {"type": "integer"},
                "rationale": {"type": "string"},
            },
            "required": ["factual_correctness", "completeness", "groundedness", "rationale"],
            "additionalProperties": False,
        },
    },
}


def make_lookup_tool(seeds: list[str]) -> Any:
    """Build a per-task lookup tool that returns the full evidence set.

    Earlier iterations did keyword matching over the seed list, but with
    only 3 seeds per task and multi-hop questions that need all of them,
    keyword filtering led to workers burning their tool budget on
    rephrased queries. The correct design for this dataset size is to
    return the complete evidence on any call. Both conditions get the
    exact same tool, so this does not bias the comparison.
    """

    def lookup(query: str = "") -> str:
        """Return all available evidence for this task. Call once, then reason."""
        return "\n".join(f"- {s}" for s in seeds)

    return lookup


async def judge(
    client: OpenAIClient, question: str, ground_truth: str, answer: str
) -> dict[str, Any]:
    user_msg = (
        f"Question:\n{question}\n\n"
        f"Ground truth answer:\n{ground_truth}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Call the score tool now."
    )
    result = await client.call(
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tools=[JUDGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "score"}},
    )
    for tc in result.tool_calls:
        if tc.name == "score":
            payload = tc.arguments or {}
            return {
                "factual_correctness": int(payload.get("factual_correctness", 0)),
                "completeness": int(payload.get("completeness", 0)),
                "groundedness": int(payload.get("groundedness", 0)),
                "rationale": payload.get("rationale", ""),
            }
    return {
        "factual_correctness": 0,
        "completeness": 0,
        "groundedness": 0,
        "rationale": "judge did not call the score tool",
    }


def is_pass(scores: dict[str, Any]) -> bool:
    s = scores["factual_correctness"] + scores["completeness"] + scores["groundedness"]
    return s >= 2


async def evaluate_one(
    task: dict[str, Any],
    runner: Any,
    runner_client: OpenAIClient,
    judge_client: OpenAIClient,
) -> dict[str, Any]:
    lookup = make_lookup_tool(task["seeds"])
    in_before = runner_client.stats.input_tokens
    out_before = runner_client.stats.output_tokens
    t0 = time.perf_counter()
    answer = await runner(task["question"], lookup, runner_client)
    elapsed = time.perf_counter() - t0
    in_used = runner_client.stats.input_tokens - in_before
    out_used = runner_client.stats.output_tokens - out_before

    scores = await judge(judge_client, task["question"], task["ground_truth"], answer)
    return {
        "task_id": task["id"],
        "answer": answer,
        "input_tokens": in_used,
        "output_tokens": out_used,
        "wall_clock": round(elapsed, 3),
        "scores": scores,
        "pass": is_pass(scores),
    }


def _enable_progress_logging() -> None:
    """Stream per-worker / per-supervisor INFO logs to stdout during the run."""
    import logging
    import sys as _sys

    lib_logger = logging.getLogger("swarmweave")
    if any(isinstance(h, logging.StreamHandler) for h in lib_logger.handlers):
        return
    handler = logging.StreamHandler(_sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    lib_logger.addHandler(handler)
    lib_logger.setLevel(logging.INFO)


async def main() -> None:
    # Load .env (OPENAI_API_KEY) if present; harmless otherwise.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    _enable_progress_logging()

    if not os.environ.get("OPENAI_API_KEY"):
        out = RESULTS / "results.json"
        out.write_text(
            json.dumps(
                {
                    "status": "not_yet_run",
                    "instructions": "Run python benchmarks/run_benchmark.py with OPENAI_API_KEY set to populate real results.",
                },
                indent=2,
            )
        )
        chart = RESULTS / "chart.png"
        if chart.exists():
            chart.unlink()
        print("OPENAI_API_KEY not set — wrote pending placeholder to results/results.json.")
        return

    tasks = json.loads((ROOT / "tasks.json").read_text())
    print(f"running {len(tasks)} tasks under both conditions...")

    # Medium reasoning effort keeps the run tractable (~30-60 min total)
    # without falling back to a different model class. The judge uses the
    # same model but lower effort - the rubric is concrete so high reasoning
    # is wasted.
    iso_client = OpenAIClient(reasoning_effort="medium")
    sc_client = OpenAIClient(reasoning_effort="medium")
    judge_client = OpenAIClient(reasoning_effort="low")

    isolated_results: list[dict[str, Any]] = []
    shared_results: list[dict[str, Any]] = []
    partial_path = RESULTS / "results.partial.json"

    def checkpoint() -> None:
        partial_path.write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "completed": len(shared_results),
                    "isolated": isolated_results,
                    "shared": shared_results,
                },
                indent=2,
            )
        )

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}: isolated...", flush=True)
        try:
            iso = await evaluate_one(task, run_isolated, iso_client, judge_client)
            pass_str = "PASS" if iso.get("pass") else "FAIL"
            print(
                f"  [{i}/{len(tasks)}] {task['id']}: isolated {pass_str} "
                f"(tokens {iso.get('input_tokens', 0)}/{iso.get('output_tokens', 0)})",
                flush=True,
            )
        except Exception as exc:
            iso = {"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}", "pass": False}
            print(f"  [{i}/{len(tasks)}] {task['id']}: isolated ERROR {iso['error']}", flush=True)
        isolated_results.append(iso)
        checkpoint()

        print(f"  [{i}/{len(tasks)}] {task['id']}: shared...", flush=True)
        try:
            sc = await evaluate_one(task, run_shared, sc_client, judge_client)
            pass_str = "PASS" if sc.get("pass") else "FAIL"
            print(
                f"  [{i}/{len(tasks)}] {task['id']}: shared {pass_str} "
                f"(tokens {sc.get('input_tokens', 0)}/{sc.get('output_tokens', 0)})",
                flush=True,
            )
        except Exception as exc:
            sc = {"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}", "pass": False}
            print(f"  [{i}/{len(tasks)}] {task['id']}: shared ERROR {sc['error']}", flush=True)
        shared_results.append(sc)
        checkpoint()

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        passes = sum(1 for r in rows if r.get("pass"))
        return {
            "n": len(rows),
            "passes": passes,
            "accuracy": round(passes / max(1, len(rows)), 4),
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "wall_clock": round(sum(r.get("wall_clock", 0.0) for r in rows), 2),
        }

    summary = {
        "model": "gpt-4o-mini",
        "n_tasks": len(tasks),
        "n_workers_per_swarm": 3,
        "isolated": aggregate(isolated_results),
        "shared": aggregate(shared_results),
        "per_task": {
            "isolated": isolated_results,
            "shared": shared_results,
        },
    }

    out = RESULTS / "results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")

    try:
        from benchmarks._chart import render_chart

        render_chart(summary, RESULTS / "chart.png")
        print(f"wrote {RESULTS / 'chart.png'}")
    except Exception as exc:
        print(f"chart rendering failed: {exc}")

    print("\n=== summary ===")
    print(
        f"isolated  accuracy={summary['isolated']['accuracy']} tokens={summary['isolated']['input_tokens'] + summary['isolated']['output_tokens']}"
    )
    print(
        f"shared    accuracy={summary['shared']['accuracy']} tokens={summary['shared']['input_tokens'] + summary['shared']['output_tokens']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
