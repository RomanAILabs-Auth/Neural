# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
<#
.SYNOPSIS
  GGUF-runner readiness gate: nrlpy import + stub golden + optional real golden.

.DESCRIPTION
  Runs the stub-mode golden harness unconditionally (no model required) and
  then, if NRL_GGUF_GOLDEN_MODEL or -Model points at a real .gguf, runs the
  real-mode harness too. Artifacts land in build/gguf_golden/.

  Exit codes:
    0 — all selected modes passed
    1 — a regression (golden assertion failed)
    2 — configuration error (e.g. -Mode real with no model)

  See docs/nrl_gguf_runner_architecture.md §7 for the honesty contract this
  harness enforces, and scripts/live_readiness.ps1 for the umbrella gate.
#>
[CmdletBinding()]
param(
    [ValidateSet("stub", "p2active-sim", "p2active-prefill", "mm-replay", "real", "auto")]
    [string]$Mode = "auto",
    [string]$Model,
    [string]$Prompt = "Reply with one short sentence about the number two.",
    [int]$MaxTokens = 8,
    [int]$Seed = 42,
    [string]$ChatFormat = "phi3",
    [double]$MinWps = 0,
    [ValidateSet("executed", "virtual", "effective")]
    [string]$WpsMetric = "effective"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "nrlpy\src"

Write-Host "[live-gguf] nrlpy import probe" -ForegroundColor Cyan
$probe = & python -c "from nrlpy import gguf, gguf_chat; print(gguf.__name__, gguf_chat.__name__)"
if ($LASTEXITCODE -ne 0) { throw "nrlpy import failed (PYTHONPATH=$env:PYTHONPATH)" }
Write-Host "  $probe"

$harness = Join-Path $root "benchmarks\gguf_golden.py"
if (-not (Test-Path $harness)) { throw "missing $harness" }

$argv = @("--mode", $Mode, "--prompt", $Prompt, "--max-tokens", $MaxTokens, "--seed", $Seed, "--chat-format", $ChatFormat)
if ($Model) { $argv += @("--model", $Model) }
if ($MinWps -gt 0) { $argv += @("--min-wps", $MinWps, "--wps-metric", $WpsMetric) }

Write-Host "[live-gguf] golden harness $Mode" -ForegroundColor Cyan
& python $harness @argv
$rc = $LASTEXITCODE
if ($rc -eq 0) {
    Write-Host "[live-gguf] OK" -ForegroundColor Green
} elseif ($rc -eq 2) {
    Write-Host "[live-gguf] CONFIG ERROR (rc=$rc): pass -Model or set NRL_GGUF_GOLDEN_MODEL" -ForegroundColor Yellow
} else {
    Write-Host "[live-gguf] FAIL (rc=$rc) — see build\gguf_golden\gguf_golden.md" -ForegroundColor Red
}
exit $rc
