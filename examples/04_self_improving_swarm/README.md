# 04 — self-improving swarm

The same swarm, on the same task, gets better the second time it runs.

This is what `Mentor` and the lessons system are for.

## What happens

1. **Round 1** — the Mentor checks `~/.swarmweave/lessons/python_cli_advice.jsonl`. If empty, no lessons are preloaded. The 3-worker swarm runs cold, the synthesizer produces an answer, and the Mentor asks the model to extract 1-5 transferable lessons from the run. Those lessons are persisted to disk.
2. **Round 2** — the Mentor retrieves the lessons most relevant to the user task (via embeddings if available, Jaccard otherwise) and writes them into the shared context as observations *before any worker fires*. The 3-worker swarm now reads from a context that already contains hard-won patterns from round 1.

Run it once and watch the second-round answer get noticeably more specific. Run it 5 times and the lesson book accumulates a real knowledge base for this kind of task — every run is starting on the shoulders of every previous run.

## Run it

```bash
# Default: prints to stdout
python examples/04_self_improving_swarm/main.py

# Or watch it live in the TUI dashboard:
swarmweave watch examples/04_self_improving_swarm/main.py
```

## Where the lessons live

`~/.swarmweave/lessons/python_cli_advice.jsonl` — plain JSONL, fully readable, fully editable, fully deletable. Edit by hand to teach the swarm domain knowledge directly. Delete to start over.

No telemetry leaves your machine.
