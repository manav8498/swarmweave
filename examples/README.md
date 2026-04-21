# Examples

Three runnable swarms that illustrate different coordination patterns.

| Example                  | Pattern                            | What shared context does                                                                |
| ------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `01_research_swarm`      | Fan-out + synthesis                | Lets the synthesizer read every researcher's findings without supervisor middleman.     |
| `02_code_review_swarm`   | Domain specialists + aggregator    | Lets each reviewer see what other reviewers have already flagged and avoid duplicates.  |
| `03_support_triage_swarm`| Sequential pipeline with sharing   | Lets the verifier read both the classification and the draft response before checking. |

## Running

```bash
export OPENAI_API_KEY=sk-...
python examples/01_research_swarm/main.py
```

If `OPENAI_API_KEY` is not set, each example prints a clear message and
exits cleanly.
