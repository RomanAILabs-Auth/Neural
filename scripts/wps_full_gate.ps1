# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
<#
.SYNOPSIS
  One-shot WPS + GGUF automation (pytest slice, golden harness, autopilot report).

.PARAMETER Ci
  When set, enforces mm-replay floor >= 1000 effective WPS (same as CI).

.EXAMPLE
  .\scripts\wps_full_gate.ps1
  .\scripts\wps_full_gate.ps1 -Ci
#>
[CmdletBinding()]
param(
    [switch]$Ci
)

$ErrorActionPreference = "Stop"
# Resolve repo: script lives in <repo>/scripts/ ; or use NRL_REPO / nrlpy on PATH.
$ScriptRoot = if ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { $PSScriptRoot }
$FromScript = Join-Path $ScriptRoot ".."
if (Test-Path (Join-Path $FromScript "benchmarks\gguf_golden.py")) {
    $Root = (Resolve-Path $FromScript).Path
    Set-Location $Root
} elseif ($env:NRL_REPO -and (Test-Path (Join-Path $env:NRL_REPO "benchmarks\gguf_golden.py"))) {
    $Root = $env:NRL_REPO.TrimEnd('\')
    Set-Location $Root
} else {
    Write-Host "Could not find NRL repo (benchmarks\gguf_golden.py)." -ForegroundColor Yellow
    Write-Host "  Option A — from anywhere (after install sets NRL_REPO):" -ForegroundColor Cyan
    Write-Host "    nrlpy wps-gate" -ForegroundColor White
    Write-Host "  Option B — set clone path then:" -ForegroundColor Cyan
    Write-Host '    $env:NRL_REPO = "C:\Users\Asus\Desktop\Documents\NRL"' -ForegroundColor White
    Write-Host "    nrlpy wps-gate" -ForegroundColor White
    Write-Host "  Option C — cd to your clone then: .\scripts\wps_full_gate.ps1" -ForegroundColor Cyan
    if (Get-Command nrlpy -ErrorAction SilentlyContinue) {
        if ($Ci) {
            & nrlpy wps-gate --ci
        } else {
            & nrlpy wps-gate
        }
        exit $LASTEXITCODE
    }
    exit 2
}

$env:PYTHONPATH = Join-Path $Root "nrlpy\src"

Write-Host "=== [wps-full-gate] specs ===" -ForegroundColor Cyan
Write-Host "  repo:       $Root"
Write-Host "  PYTHONPATH: $env:PYTHONPATH"
Write-Host "  python:     $(python --version 2>&1)"
Write-Host "  Ci switch:  $Ci"
Write-Host ""

Write-Host "=== [wps-full-gate] pytest (GGUF + chat + autopilot) ===" -ForegroundColor Cyan
python -m pytest `
  nrlpy/tests/test_wps_autopilot.py `
  nrlpy/tests/test_gguf.py `
  nrlpy/tests/test_gguf_chat.py `
  nrlpy/tests/test_cli_chat.py `
  -q --tb=line
if ($LASTEXITCODE -ne 0) { throw "pytest failed (rc=$LASTEXITCODE)" }

Write-Host ""
Write-Host "=== [wps-full-gate] gguf_golden --mode auto ===" -ForegroundColor Cyan
$gargv = @("benchmarks/gguf_golden.py", "--mode", "auto")
python @gargv
if ($LASTEXITCODE -ne 0) { throw "gguf_golden failed (rc=$LASTEXITCODE)" }

Write-Host ""
Write-Host "=== [wps-full-gate] wps_autopilot ===" -ForegroundColor Cyan
$aargv = @(
    "benchmarks/wps_autopilot.py",
    "--markdown", "build/wps_autopilot.md"
)
if ($Ci) {
    $aargv += @("--min-mm-effective-wps", "1000")
    python @aargv
    if ($LASTEXITCODE -ne 0) { throw "wps_autopilot CI floor failed (rc=$LASTEXITCODE)" }
    python benchmarks/gguf_golden.py --mode mm-replay --mm-min-wps 1000
    if ($LASTEXITCODE -ne 0) { throw "gguf_golden mm-replay CI floor failed (rc=$LASTEXITCODE)" }
} else {
    python @aargv
    if ($LASTEXITCODE -ne 0) { throw "wps_autopilot failed (rc=$LASTEXITCODE)" }
}

Write-Host ""
Write-Host "=== [wps-full-gate] artifacts ===" -ForegroundColor Green
Write-Host "  $(Join-Path $Root 'build\gguf_golden\gguf_golden.json')"
Write-Host "  $(Join-Path $Root 'build\wps_autopilot.json')"
Write-Host "  $(Join-Path $Root 'build\wps_autopilot.md')"
Write-Host "[wps-full-gate] OK" -ForegroundColor Green
