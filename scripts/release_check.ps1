# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "[release] build + tests" -ForegroundColor Cyan
& (Join-Path $root "build.ps1") -Tests
if ($LASTEXITCODE -ne 0) { throw "build/tests failed" }

Write-Host "[release] locked nrl-vs-cpp artifact" -ForegroundColor Cyan
& python (Join-Path $root "benchmarks\nrl_vs_cpp.py") `
    --neurons 1048576 --iterations 4096 --reps 4 --threshold 8
if ($LASTEXITCODE -ne 0) { throw "nrl_vs_cpp harness failed" }

Write-Host "[release] verify workload_identity in bench JSON" -ForegroundColor Cyan
$artifact = Join-Path $root "build\bench\nrl_vs_cpp.json"
& python -c "import json; d=json.load(open(r'$artifact', encoding='utf-8')); assert 'workload_identity' in d and 'structural_hash' in d['workload_identity']"
if ($LASTEXITCODE -ne 0) { throw "workload_identity missing from nrl_vs_cpp.json" }

$pyd = Get-ChildItem -Path (Join-Path $root "nrlpy\src\nrlpy") -Filter "_core*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pyd) {
    Write-Host "[release] nrlpy learn + chat one-shot" -ForegroundColor Cyan
    $env:PYTHONPATH = Join-Path $root "nrlpy\src"
    & python -m nrlpy.cli learn status | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "nrlpy learn status failed" }
    & python -m nrlpy.cli chat --one "status" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "nrlpy chat --one failed" }
    & python -m nrlpy.cli control status | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "nrlpy control status failed" }
} else {
    Write-Warning "nrlpy._core pyd not found; skipping nrlpy learn/chat in release_check"
}

Write-Host "[release] smoke status/chat" -ForegroundColor Cyan
$st = & (Join-Path $root "build\bin\nrl.exe") status 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "nrl status failed" }
if ($st -notmatch "control_preferences_path") { throw "nrl status missing control_preferences_path" }
$st | Out-Host
& (Join-Path $root "build\bin\nrl.exe") inquire speed
if ($LASTEXITCODE -ne 0) { throw "nrl inquire failed" }

Write-Host "[release] smoke nrl control (sandbox)" -ForegroundColor Cyan
$nrl = Join-Path $root "build\bin\nrl.exe"
$blocked = & $nrl control "buy stocks" 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "nrl control blocked path failed" }
if ($blocked -notmatch "BLOCKED") { throw "expected BLOCKED in control output" }
$defer = & $nrl control "maximum power for one hour" 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "nrl control defer path failed" }
if ($defer -notmatch "DEFER") { throw "expected DEFER in control output" }

Write-Host "[release] OK" -ForegroundColor Green
