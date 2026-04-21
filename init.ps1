# One-shot setup for swarmweave on Windows PowerShell.
#
# Creates a Python 3.11+ virtualenv in .\.venv, installs the package in
# editable mode with dev extras, and copies .env.example to .env if missing.

$ErrorActionPreference = "Stop"

function Test-PythonVersion {
    param([string]$Exe)
    try {
        & $Exe -c "import sys; assert sys.version_info >= (3, 11)" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$PythonBin = $env:PYTHON_BIN
if (-not $PythonBin -or -not (Test-PythonVersion $PythonBin)) {
    foreach ($candidate in @("py -3.13", "py -3.12", "py -3.11", "python3.13", "python3.12", "python3.11", "python")) {
        if (Test-PythonVersion $candidate) {
            $PythonBin = $candidate
            break
        }
    }
}

if (-not $PythonBin -or -not (Test-PythonVersion $PythonBin)) {
    Write-Error "Python 3.11+ is required but was not found on PATH. Install from https://www.python.org/ and retry."
    exit 1
}

Write-Host "Using $(& $PythonBin --version) at $(Get-Command $PythonBin.Split()[0] | Select-Object -ExpandProperty Source)"

if (-not (Test-Path ".venv")) {
    & $PythonBin -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip | Out-Null
pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example — fill in OPENAI_API_KEY before running examples."
}

Write-Host ""
Write-Host "Setup complete. Activate the environment with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then try:  python examples\01_research_swarm\main.py"
