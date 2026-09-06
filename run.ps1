# One-command local start for Windows PowerShell.
#   .\run.ps1            # launch the Streamlit app
#   .\run.ps1 -Ingest    # (re)build the demo corpus and ingest it first
param([switch]$Ingest)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtualenv (Python 3.12)..." -ForegroundColor Cyan
    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]" -q
}
if (-not (Test-Path ".env")) { Copy-Item .env.example .env }

# Ollama reachable?
try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null }
catch { Write-Warning "Ollama does not seem to be running. Start it, then: ollama pull qwen2.5:7b nomic-embed-text" }

if ($Ingest) {
    .\.venv\Scripts\python.exe scripts\make_samples.py
    .\.venv\Scripts\python.exe -m grounded.cli ingest "data/samples/*"
}

Write-Host "`nGrounded is starting -> http://localhost:8501  (Ctrl+C to stop)`n" -ForegroundColor Green
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
