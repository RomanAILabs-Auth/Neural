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

Write-Host "[release] smoke status/chat" -ForegroundColor Cyan
& (Join-Path $root "build\bin\nrl.exe") status
if ($LASTEXITCODE -ne 0) { throw "nrl status failed" }
& (Join-Path $root "build\bin\nrl.exe") inquire speed
if ($LASTEXITCODE -ne 0) { throw "nrl inquire failed" }

Write-Host "[release] OK" -ForegroundColor Green
