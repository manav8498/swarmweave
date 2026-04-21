# Architecture

This document is the thesis behind swarmweave. It explains why isolated-context multi-agent systems hit a coordination ceiling, what changes when you swap that for an outcome-optimized shared context layer, and how the Backend interface keeps the substrate pluggable across local development and any custom store you bring.

## The problem with isolated-context subagents

The standard multi-agent pattern is supervisor-and-subagents. A supervisor receives a user task, decomposes it, dispatches subtasks to subagent workers, and stitches the worker outputs back together. Every worker runs with its own private context window — the only thing it knows about the rest of the swarm is what the supervisor put into its prompt.

This pattern looks elegant on a whiteboard, and it scales linearly until it doesn't. The failure mode is coordination loss. Three concrete symptoms:

1. **Re-summarization tax.** Every handoff between supervisor and worker requires the supervisor to re-summarize the task and any relevant context for that worker. Re-summarization is lossy. The supervisor cannot anticipate every fact a worker will need, so it either over-includes (paying token cost) or under-includes (paying accuracy cost).
2. **Duplicate work.** Two workers given overlapping subtasks have no way to know that the other has already produced a result. The supervisor only learns this at synthesis time, after both have spent tokens.
3. **Brittle composition.** Adding a fourth worker means reworking the supervisor's decomposition logic, because the supervisor is the only place that "knows" the swarm's structure. Workers cannot be composed independently.

The deeper observation is that the supervisor is a coordination bottleneck. Every signal that needs to flow between workers has to pass through it. As the number of workers grows, the supervisor's context window — and its attention budget — becomes the limiting factor. Past a certain swarm size, accuracy degrades faster than any benefit you get from parallelism.

## What changes with shared, outcome-optimized context

swarmweave replaces the supervisor-as-bottleneck with a substrate. Workers still receive subtasks, still run in parallel, still report results — but they read and write a shared, outcome-optimized memory layer in between.

Two ideas matter here:

**Shared.** Every worker's findings are immediately visible to every other worker. A worker about to draft a section first asks the shared context what its peers have already established. A worker about to fact-check first asks what claims have already been made. There is no re-summarization tax, because the worker pulls exactly the slice it needs at the moment it needs it.

**Outcome-optimized.** The slice a worker receives is not a similarity-ranked dump of the log. It is a slice scored against the worker's current subtask and the swarm's end goal. In swarmweave's reference `LocalBackend`, this is implemented as embedding similarity (OpenAI `text-embedding-3-small`, cosine) plus a recency bonus, with a Jaccard fallback when offline. A production backend would add outcome-aware reranking — learning from past runs which observations actually contributed to the final outcome — but that is the job of the backend you bring, not of swarmweave itself.

The crucial thing is that the *interface* is the same. A worker calls `shared_context.read(agent_id, task)` and gets back a `ContextSlice` that is "what you should know now." Whether the slice was computed by Jaccard on a JSONL log, by embedding similarity on the local backend, or by a learned retrieval model in a service you wrote, is invisible to the worker — and to the user.

## Why this is a different shape from existing patterns

A few patterns sit nearby and are worth distinguishing:

* **Vector-store retrieval per agent.** Many agent frameworks already give each agent a private RAG store. That helps with grounding inside a single agent but does nothing for cross-agent coordination — agents still cannot see each other's reasoning or intermediate results.
* **Supervisor-as-router with shared scratchpad.** Some frameworks let the supervisor pass a scratchpad string between workers. This is a strict subset of what shared context offers: the scratchpad is unstructured, ordered by writes (not relevance), and grows monotonically. By the third worker, the supervisor is back to summarizing.
* **Multi-agent debate.** Debate-style architectures explicitly route agents at each other. They optimize for adversarial coverage, not for outcome efficiency. Shared context is complementary: a debate could absolutely sit on top of a swarmweave-style substrate, with each debater pulling from the same outcome-aware log.

swarmweave is the substrate underneath these patterns, not a replacement for them.

## The Backend interface

The whole architecture is held together by one abstract base class:

```python
class Backend(ABC):
    async def append(self, observation: Observation) -> None: ...
    async def query(self, agent_id: str, task: str, limit: int = 8) -> ContextSlice: ...
    async def trace(self) -> list[Observation]: ...
    async def finalize(self, outcome: Outcome) -> None: ...
```

Four methods. That's the whole substrate.

`LocalBackend` is the default implementation: an append-only JSONL log under `~/.swarmweave/<session>.jsonl` with similarity + recency scoring. It runs offline, has no external dependencies, and powers the examples and the benchmark.

Custom backends are first-class. To plug in your own store — Postgres + pgvector, Redis, an internal service — you subclass `Backend`, implement those four methods, and pass an instance to `SharedContext(backend=...)`. See [`custom_backends.md`](custom_backends.md) for a worked example.

## What this enables

Once the shared substrate is in place, several things become natural that were awkward before:

* **Swarms compose.** Two pre-built swarms can be run sequentially against the same `SharedContext` and downstream workers automatically benefit from upstream findings. No glue code.
* **Cross-framework portability.** Because the substrate is interface-defined and async, the same `SharedContext` can underlie a LangGraph swarm today and OpenAI Assistants / Agent SDK or Google ADK swarms tomorrow. Adapters live in `swarmweave/adapters/`.
* **Protocol-agnostic positioning.** swarmweave is not a competing protocol. It is the memory layer that agent-to-agent and team-orchestration protocols can sit on top of. A protocol-compliant agent that exposes a shared-context handle is just one more participant in the swarm.
* **Outcome-driven evaluation.** Because the substrate records the trace and the outcome, every run produces ground-truth signal that can train better retrieval models. The benchmark in `benchmarks/` is the simplest demonstration of this loop; any learned-retrieval backend you write would use the same trace shape to drive its policy.

## Limits we're honest about

* The reference scoring (Jaccard + recency) is intentionally weak. It is *good enough* to demonstrate the architectural delta on the benchmark, but it is not what production deployments should ship. Bring your own backend.
* swarmweave v0.1 supports plain Python callables as worker tools. MCP server URLs are accepted in the constructor but raise `NotImplementedError` until v0.2.
* The default sync API will refuse to run inside an existing event loop. Use `await supervisor.arun(...)` from async contexts.

These are spec choices, not bugs. Each is documented at its call site so users get an actionable error, not a surprise.
