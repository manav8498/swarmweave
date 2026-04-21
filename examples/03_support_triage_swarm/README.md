# 03 — support triage swarm

A customer support ticket flows through three workers: a classifier, a
resolver, and a verifier. This example uses `Supervisor(mode="sequential")`,
so each worker runs one at a time and its `SharedContext.aread()` returns
every prior worker's findings. This is the right topology for true
pipelines where each step builds on the previous step.

## Roster

| Worker       | Role                                                                  |
| ------------ | --------------------------------------------------------------------- |
| `classifier` | Categorize the ticket (billing / technical / abuse / feature request).|
| `resolver`   | Draft a customer-facing response, informed by the classifier's tag.   |
| `verifier`   | Read the draft and the company `policies.txt`; flag any policy violations. |

## What shared context is doing here

This is the canonical *sequential-with-shared-context* pattern:

* `classifier` writes a category into shared context.
* `resolver` runs next, reads the classifier's category via
  `SharedContext.aread`, and tailors its draft.
* `verifier` runs last, reads the draft from shared context, and checks
  it against `policies.txt`.

None of the workers takes another worker's output as a function argument.
They all read and write the same shared log, which means adding a fourth
worker (e.g. a `tone_coach`) requires zero changes to any existing worker.

## Run

```bash
python examples/03_support_triage_swarm/main.py
```
