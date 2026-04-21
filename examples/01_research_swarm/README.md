# 01 — research swarm

Three researchers cover different angles of the same question in parallel.
The supervisor's built-in synthesis reads every researcher's output and
produces a single cited report.

## Roster

| Worker         | Role                                                 |
| -------------- | ---------------------------------------------------- |
| `papers`       | Surveys the academic literature.                     |
| `industry`     | Surveys industry reports, blog posts, product news.  |
| `open_source`  | Surveys the active open-source projects in the area. |

All three fan out in parallel via LangGraph's `Send` API
(`Supervisor(mode="parallel")` is the default). The supervisor itself
performs the final synthesis — there is no "synthesizer" worker, because
putting one in the worker roster would make it race the researchers and
read empty shared context. This is the correct pattern for any task that
decomposes into independent sub-investigations + one aggregate answer.

## What shared context is doing here

During this run, shared context does not feed cross-worker signals (the
researchers cover independent angles). It records every finding into the
append-only log, which the supervisor's `asynthesize()` draws on along
with the per-worker outputs. The end-of-run trace is available via
`result.trace` for audit / benchmark analysis.

## Run

```bash
python examples/01_research_swarm/main.py
```

Expected output (truncated):

```
=== research swarm ===
> decomposing task across 3 workers
> running workers in parallel
  - papers      done in 4.2s
  - industry    done in 3.9s
  - open_source done in 4.4s
> final report:
...
```
