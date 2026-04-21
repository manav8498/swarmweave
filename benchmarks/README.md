# Benchmarks

This directory contains the public, reproducible benchmark for swarmweave.

## What it measures

20 multi-hop questions, run under two conditions:

| Condition | Description                                                                 |
| --------- | --------------------------------------------------------------------------- |
| isolated  | Standard subagent pattern: each worker runs with its own private context.   |
| shared    | swarmweave pattern: workers read and write a shared context layer.       |

Both conditions use the **same model** (`gpt-4o-mini` at
`reasoning_effort="medium"`), the **same number of workers** (3),
the **same lookup tool**, and the **same execution mode** (`sequential`).
Only the context layer differs:

- **isolated**: `IsolatedBackend` — every `read()` returns an empty slice;
  each worker sees only its own private context.
- **shared**: `LocalBackend` — each worker reads prior workers' findings
  from the shared log via similarity + recency scoring.

Sequential execution is essential for a meaningful comparison: in parallel
mode, workers that start simultaneously never see each other's writes in
either condition, so the two would be indistinguishable. Sequential mode
is also the topology where shared context actually matters in practice.

## Metrics

* **Accuracy** — pass / fail vs. the ground-truth answer, judged by an
  LLM-as-judge (also `gpt-4o-mini`) using the rubric below.
* **Total tokens** — sum of input and output tokens across the entire run.
* **Wall-clock time** — total seconds spent in the runner.

## LLM-as-judge rubric

The judge calls a `score` tool with three integer sub-scores (each 0 or 1):

1. **factual_correctness** — every factual claim made by the ground truth
   appears, accurately, in the candidate answer.
2. **completeness** — every required piece of the question is addressed.
3. **groundedness** — the candidate cites or paraphrases evidence rather
   than asserting the result.

A task counts as a **pass** if the sum of sub-scores is **≥ 2 of 3**.

## Reproducing

```bash
export OPENAI_API_KEY=sk-...
python benchmarks/run_benchmark.py
```

Results land in `results/results.json` and `results/chart.png`. With
`OPENAI_API_KEY` unset the script writes a pending-status placeholder
into `results/results.json` and exits cleanly.

## Honest reporting

We commit the numbers we get. We do not tune the baseline to lose. The
baseline is a faithful "private context per subagent" implementation
sharing every other piece of plumbing with the shared-context condition,
which is the most direct comparison we can run without changing the model
or the toolset.

If you re-run the benchmark and get different numbers, open an issue with
the seed log and we'll investigate.
