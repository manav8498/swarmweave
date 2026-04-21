# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-20

First public release.

### Core library

- Public API: `SharedContext`, `Supervisor`, `Worker`, plus the supporting `Observation`, `ContextSlice`, `Outcome`, `Task`, `SwarmResult`, `SwarmMetrics`, `WorkerSpec` pydantic types.
- `Supervisor(mode="parallel" | "sequential")`. Sequential mode chains workers in declaration order so each worker's `SharedContext.aread` sees prior workers' findings — the correct topology for classify → draft → verify pipelines.
- `LocalBackend` with embedding-based retrieval (OpenAI `text-embedding-3-small` + cosine similarity + recency) and automatic Jaccard fallback when offline or no API key is set.
- Forced wrap-up call when a Worker exhausts `max_tool_iterations`. Reasoning models no longer silently return "(no final answer — tool loop exhausted)"; they get one more chance with tools removed to commit to a FINDING.
- OpenAI Chat Completions client wrapper with graceful kwarg fallback: transparently strips / clamps `reasoning_effort`, `parallel_tool_calls`, and `max_completion_tokens` when a specific model rejects them.
- Proactive backend close at end of run, to avoid spurious "no running event loop" warnings from async GC of the embedding HTTP pool.
- LangGraph adapter (`build_swarm_graph`) that mirrors the supervisor pipeline and composes cleanly into larger LangGraph applications.

### Live observability — the `swarmweave watch` TUI

- Opt-in `EventBus` emits events for run_start, decompose, worker_start/done, tool_call, context_read/write, synthesize, run_done. Zero overhead when not subscribed.
- Rich-based `LiveDashboard` renders a split-screen TUI with per-worker status panels, shared-context flow log, and a live cost/token/wall-clock meter.
- `swarmweave` CLI entry point: `swarmweave watch <script>` runs any swarm script that exposes a `build_swarm(events)` function and streams the dashboard.

### Self-improving — `Mentor` + `LessonBook`

- `Mentor.before_run()` retrieves relevant past lessons (via embeddings or Jaccard) and preloads them into shared context.
- `Mentor.after_run()` asks the model to distill 1-5 transferable lessons from the successful run and persists them as plain JSONL under `~/.swarmweave/lessons/<book>.jsonl`.
- Fully local, editable by hand, no telemetry.

### Examples

Four runnable examples under `examples/`:

- `01_research_swarm` — parallel research fan-out + supervisor synthesis
- `02_code_review_swarm` — four parallel specialists + prioritized output
- `03_support_triage_swarm` — sequential classify → resolve → verify pipeline
- `04_self_improving_swarm` — demonstrates Mentor end-to-end

### Benchmark

- 20 hand-written multi-hop questions under `benchmarks/tasks.json`, with ground truth and seed evidence per task.
- Identical setup across conditions (sequential mode, 3 workers, same tools, same model); only the backend differs.
- LLM-as-judge with a strict 3-criterion rubric (factual correctness / completeness / groundedness), committed to the repo.
- Both modes land in the 85–100% accuracy range across runs; differences are within `gpt-4o-mini` output variance. See `benchmarks/README.md` for methodology.

### Real-world harness

- `scripts/real_world_review.py` is a reusable A/B harness that runs a 3-worker sequential code-review swarm against this repo (or any repo) with real filesystem tools, once under `IsolatedBackend` and once under `LocalBackend`. Users run it to see the qualitative difference for themselves rather than shipping pre-computed outputs.

### Infrastructure

- Tests: 24 unit tests, hermetic (no network), covering shared-context reads/writes, Worker tool loop, Supervisor decomposition / synthesis / sequential mode, and full integration.
- Lint: `ruff format` + `ruff check` clean
- Types: `mypy --strict` clean across 17 source files
- CI: GitHub Actions matrix on Python 3.11 / 3.12 / 3.13 across Ubuntu + macOS

### Known limitations

- No MCP tool-URL support in `Worker` yet; plain Python callables only (deferred to v0.2).
- `SharedContext` synchronous methods raise if called inside an existing event loop. Use the async API (`aread`, `awrite`, `arun`) from async contexts.
- `LocalBackend` advertises multi-process sharing via `session_id` but uses only an intra-process `asyncio.Lock`; for true cross-process durability, wait for v0.2 or bring your own backend (see `docs/custom_backends.md`).
