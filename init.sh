#!/usr/bin/env bash
# One-shot setup for swarmweave.
#
# Creates a Python 3.11+ virtualenv in ./.venv, installs the package in
# editable mode with dev extras, and copies .env.example to .env if missing.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
  echo "error: Python 3.11+ is required but was not found on PATH." >&2
  echo "Install Python 3.11 or newer and re-run this script." >&2
  exit 1
fi

echo "Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip >/dev/null
pip install -e ".[dev]"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in OPENAI_API_KEY before running examples."
fi

echo
echo "Setup complete. Activate the environment with:  source .venv/bin/activate"
echo "Then try:  python examples/01_research_swarm/main.py"
