# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
<#
.SYNOPSIS
  Full live-readiness gate: release_check + optional nrlpy pytest (if extension built).

.DESCRIPTION
  Run from repo root after .\build.ps1 -Config Release -Tests (or at least a successful
  engine + pyd build). Uses scripts\release_check.ps1 then nrlpy tests when _core exists.
#>
[CmdletBinding()]
param(
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "[live] release_check.ps1" -ForegroundColor Cyan
& (Join-Path $root "scripts\release_check.ps1")
if ($LASTEXITCODE -ne 0) { throw "release_check failed" }

if (-not $SkipPytest) {
    $pyd = Get-ChildItem -Path (Join-Path $root "nrlpy\src\nrlpy") -Filter "_core*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pyd) {
        Write-Host "[live] nrlpy pytest" -ForegroundColor Cyan
        $env:PYTHONPATH = Join-Path $root "nrlpy\src"
        $hasPytest = & python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
        if ($LASTEXITCODE -eq 0) {
            & python -m pytest (Join-Path $root "nrlpy\tests") -q
            if ($LASTEXITCODE -ne 0) { throw "nrlpy pytest failed" }
        } else {
            Write-Warning "pytest not installed; skipping nrlpy tests"
        }
    } else {
        Write-Warning "nrlpy._core pyd not found; skipping pytest (build.ps1 -Tests builds it)"
    }
}

Write-Host "[live] OK — ready for live testing on this host" -ForegroundColor Green
