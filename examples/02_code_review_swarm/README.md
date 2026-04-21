# 02 — code review swarm

Four reviewers with non-overlapping expertise read the same Python file in
parallel. The supervisor consolidates their findings into a single
prioritized review labeled P0 / P1 / P2.

## Roster

| Worker        | Role                                                         |
| ------------- | ------------------------------------------------------------ |
| `security`    | Security issues — injection, unsafe deserialization, secrets.|
| `performance` | Performance issues — algorithmic, allocations, IO patterns.  |
| `style`       | Style and idiomatic-Python concerns.                         |
| `coverage`    | Test-coverage and testability concerns.                      |

Four specialists run concurrently (`Supervisor(mode="parallel")`). The
supervisor's `asynthesize()` call ranks findings by severity — that is the
right place for cross-cutting consolidation, because it sees every worker's
full output and does not race them.

The file under review is `sample_code.py`, an intentionally buggy script
with at least one finding for each reviewer to surface.

## What shared context is doing here

Every finding is appended to the shared log as it's produced. The trace
is available as `result.trace` after the run — useful for debugging and
for feeding back into a benchmark. Cross-worker reading is not the value
here (the four specialists cover independent domains); the supervisor's
synthesis is.

## Run

```bash
python examples/02_code_review_swarm/main.py
```
