# API reference

This is the public API surface of swarmweave v0.1. Everything here is importable from the top-level `swarmweave` package.

## `SharedContext`

```python
SharedContext(backend: Backend | None = None)
```

The user-facing wrapper around a `Backend`. Defaults to a fresh in-memory `LocalBackend`.

| Method                                                   | Description                                                                                  |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `aread(agent_id, task, limit=8) -> ContextSlice`         | Async. Returns the slice of context most relevant to `task`.                                 |
| `awrite(agent_id, observation) -> None`                  | Async. Appends an observation contributed by `agent_id`.                                     |
| `afinalize(outcome) -> None`                             | Async. Records the run's final outcome.                                                      |
| `aget_full_trace() -> list[Observation]`                 | Async. Returns every observation in the log.                                                 |
| `read(...)`, `write(...)`, `finalize(...)`, `get_full_trace(...)` | Synchronous wrappers around the corresponding async methods. They raise if called from inside a running event loop. |

## `Supervisor`

```python
Supervisor(
    workers: list[Worker],
    shared_context: SharedContext | None = None,
    model: str = "gpt-4o-mini",
    client: OpenAIClient | None = None,
    mode: Literal["parallel", "sequential"] = "parallel",
)
```

`mode`:
- `"parallel"` (default): all workers fan out simultaneously via LangGraph's `Send` API. Best when workers cover non-overlapping angles and don't depend on each other. The supervisor's final synthesis integrates every worker's output.
- `"sequential"`: workers run one at a time in the order they appear in `workers`. Each worker's `SharedContext.aread` returns every prior worker's findings — use this for pipelines like classify → draft → verify.

| Method                                  | Description                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| `arun(user_task) -> SwarmResult`        | Async. Runs the full decompose → fan-out → synthesize → finalize pipeline.        |
| `run(user_task) -> SwarmResult`         | Synchronous wrapper around `arun`.                                                |
| `adecompose(user_task) -> list[Task]`   | Async. Returns one `Task` per worker. Useful if you want to inspect the plan.     |
| `asynthesize(user_task, outputs) -> str`| Async. Produces the final answer from per-worker outputs.                         |

`Supervisor` validates that worker names are unique and that at least one worker is provided. Both checks raise `ValueError` at construction time.

## `Worker`

```python
Worker(
    name: str,
    role: str,
    model: str = "gpt-4o-mini",
    system_prompt: str | None = None,
    tools: list[Callable[..., Any]] | None = None,
    max_tool_iterations: int = 6,
)
```

| Method                                                                        | Description                                                       |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `arun(subtask, shared_context, client) -> Observation`                        | Async. Reads the slice, runs the tool loop, writes the finding.   |
| `run(subtask, shared_context, client) -> Observation`                         | Sync wrapper around `arun`.                                       |

The `tools` argument accepts plain Python callables. Each callable should have a docstring (used as the tool description) and typed parameters from the primitives `str | int | float | bool | list | dict`. Anything more exotic should be expressed as an OpenAI-format function tool dict (`{"type": "function", "function": {...}}`) and passed through. MCP server URLs are a v0.2 surface and currently raise `NotImplementedError`.

`name` must be a non-empty alphanumeric string (`-` and `_` allowed). It is used as the agent id in the shared log.

## Types

All re-exported from `swarmweave`:

- **`Observation`** — `id`, `agent_id`, `task`, `content`, `kind`, `timestamp`, `metadata`. The unit of memory.
- **`ContextSlice`** — `observations: list[Observation]`, `summary: str | None`, `tokens_estimate: int`. Has a `.render()` method that produces a flat string for an LLM prompt.
- **`Outcome`** — `success: bool`, `score: float | None`, `notes: str | None`, `metadata`. End-of-run signal.
- **`Task`** — `id`, `description`, `parent_id`, `assigned_to`. Internal-but-public; you'll see these in `Supervisor.adecompose()` results.
- **`WorkerSpec`** — Lightweight dataclass version of a worker config. Reserved for future use; current users should pass `Worker` instances directly.
- **`SwarmResult`** — `final_output`, `per_worker_outputs`, `trace`, `metrics`. The return value of `Supervisor.run()`.
- **`SwarmMetrics`** — `input_tokens`, `output_tokens`, `wall_clock_seconds`, `worker_calls`.

## Backends

```python
from swarmweave.backends import Backend, LocalBackend
```

`Backend` is the abstract base. `LocalBackend` is the default embedding-backed implementation (OpenAI `text-embedding-3-small` with Jaccard fallback). See [`custom_backends.md`](custom_backends.md) for a worked example of implementing your own (Postgres + pgvector, Redis, etc.).

## Adapter

```python
from swarmweave.adapters import build_swarm_graph
graph = build_swarm_graph(supervisor)
final_state = await graph.ainvoke({"user_task": "..."})
```

`build_swarm_graph` returns a compiled LangGraph `StateGraph` mirroring `Supervisor.arun()`. Use it when you want to embed a swarm into a larger LangGraph application.
