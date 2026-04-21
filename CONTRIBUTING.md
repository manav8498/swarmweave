# Contributing

Thanks for your interest in swarmweave. The bar for contribution is the same as the bar for the original codebase: small, focused changes that pass the lint, type, and test gates, and that respect the existing API surface.

## Development setup

```bash
git clone https://github.com/manav8498/swarmweave
cd swarmweave
./init.sh                          # creates .venv, installs the package + dev extras
source .venv/bin/activate
```

Required: Python 3.11+ on `PATH`. The init script will pick up `python3.13` / `python3.12` / `python3.11` automatically.

## The gates

Every PR must pass these locally before review:

```bash
ruff format .
ruff check .
mypy --strict src/swarmweave
pytest -q
```

The suite is hermetic — it does not call the OpenAI API. The included `FakeOpenAIClient` (in `tests/conftest.py`) is the convention for new tests that need to stand in for the model.

## Codebase conventions

- Public types live in `swarmweave/types.py` as pydantic v2 models.
- Public surface is exported from `swarmweave/__init__.py`. Keep it small.
- Async-first inside the package; user-facing classes provide sync wrappers around `arun`/`aread`/etc.
- No `print()` calls in library code — use the module-level logger named `"swarmweave"`.
- File I/O uses `pathlib.Path`, not raw strings.
- Docstrings are required on every public function, class, and method. Private helpers may skip them when the name is obvious.

## Backend contributions

Backends are an explicit extension point. If you want to ship a backend implementation against another store, the path is:

1. Implement `Backend` in your own package (or in a draft PR here).
2. Add a brief section to `docs/custom_backends.md` linking the implementation.
3. The example in `docs/custom_backends.md` is the canonical template; mirror its structure where it fits.

Backends do not need to live inside swarmweave to be first-class.

## Commit style

Conventional commits, lower-case prefixes:

- `feat:` new behavior or surface area
- `fix:` bug fix
- `docs:` docs-only changes
- `chore:` infra / formatting / housekeeping
- `test:` test-only changes

Atomic commits within a PR are encouraged. Avoid god-commits that rewrite multiple subsystems in one go.

## What's out of scope right now

- Adding new model providers beyond OpenAI. v0.1 is opinionated about GPT models as the substrate. Adapter layers may land in later releases.
- Supporting MCP server URLs as worker tools. This is a v0.2 surface tracked in the roadmap.
- Adding telemetry that calls out to third-party services by default.

If you're not sure whether something is in scope, open an issue first and we'll work it out before you spend implementation time.
