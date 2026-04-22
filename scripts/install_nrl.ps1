[CmdletBinding()]
param(
    [switch]$OptInLMAI,
    [switch]$SkipBuild,
    [switch]$NoPathUpdate
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$installRoot = Join-Path $env:LOCALAPPDATA "Programs\NRL"
$binDir = Join-Path $installRoot "bin"
$consentDir = Join-Path $env:USERPROFILE ".nrl"
$consentPath = Join-Path $consentDir "consent.json"
$nrlExe = Join-Path $root "build\bin\nrl.exe"

if (-not $SkipBuild) {
    Write-Host "[install] building NRL release artifacts ..." -ForegroundColor Cyan
    & (Join-Path $root "build.ps1")
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

if (-not (Test-Path $nrlExe)) {
    throw "missing nrl.exe at $nrlExe"
}

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
# Replace fails if another shell still has ``nrl.exe`` loaded (Windows file lock).
Get-Process -Name "nrl" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 400
Copy-Item -Force $nrlExe (Join-Path $binDir "nrl.exe")

# Legacy path: older demos looked for ``%NRL_ROOT%\build\bin\nrl.exe`` (repo layout).
$legacyBin = Join-Path $installRoot "build\bin"
New-Item -ItemType Directory -Force -Path $legacyBin | Out-Null
Copy-Item -Force $nrlExe (Join-Path $legacyBin "nrl.exe")

# Launcher so ``nrlpy`` is on PATH (same idea as pip Scripts; no separate pip step required).
$nrlpyCmd = Join-Path $binDir "nrlpy.cmd"
@'
@echo off
setlocal
if not defined NRL_ROOT (
  echo nrlpy: NRL_ROOT is not set. Re-run install_nrl.ps1 or set NRL_ROOT to your NRL install root 1>&2
  exit /b 1
)
if exist "%NRL_ROOT%\bin\nrl.exe" set "NRL_BIN=%NRL_ROOT%\bin\nrl.exe"
set "PYTHONPATH=%NRL_ROOT%\py;%PYTHONPATH%"
python -m nrlpy.cli %*
exit /b %ERRORLEVEL%
'@ | Set-Content -Path $nrlpyCmd -Encoding ascii

# Ship demo + nrlpy package for ``nrl demo`` (PYTHONPATH = %NRL_ROOT%\py)
$examplesDest = Join-Path $installRoot "examples"
$pyDest = Join-Path $installRoot "py"
New-Item -ItemType Directory -Force -Path $examplesDest | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $pyDest "nrlpy") | Out-Null
Copy-Item -Path (Join-Path $root "examples\*") -Destination $examplesDest -Recurse -Force -ErrorAction SilentlyContinue
$nrlpySrc = Join-Path $root "nrlpy\src\nrlpy"
$nrlpyDst = Join-Path $pyDest "nrlpy"
# ``_core*.pyd`` is locked if Python imported nrlpy from this tree; stop only those PIDs (narrow match).
$installMarker = "Programs\NRL"
foreach ($wp in Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue) {
    $cl = $wp.CommandLine
    if ($null -ne $cl -and ($cl -like "*${installMarker}*" -or $cl -like "*$($installRoot.Replace('\', '/'))*")) {
        Stop-Process -Id $wp.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Milliseconds 600
# Retries help if AV briefly scans the new exe/pyd.
$null = & robocopy.exe $nrlpySrc $nrlpyDst /E /IS /IT /R:12 /W:2 /NFL /NDL /NJH /NJS
if ($LASTEXITCODE -ge 8) {
    throw "robocopy nrlpy failed (exit $LASTEXITCODE). Close Python/nrlpy sessions using $nrlpyDst then re-run install_nrl.ps1 -SkipBuild."
}
[Environment]::SetEnvironmentVariable("NRL_ROOT", $installRoot, "User")
$env:NRL_ROOT = $installRoot

if (-not $NoPathUpdate) {
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $currentPath) { $currentPath = "" }
    $entries = $currentPath -split ";" | Where-Object { $_ -ne "" }
    if ($entries -notcontains $binDir) {
        $newPath = ($entries + $binDir) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:Path = "$env:Path;$binDir"
    }
}

New-Item -ItemType Directory -Force -Path $consentDir | Out-Null
$enableLMAI = $false
if ($OptInLMAI) {
    $enableLMAI = $true
} else {
    $reply = Read-Host "Opt in to LM/AI features? (y/N)"
    if ($reply -match "^(y|Y|yes|YES)$") {
        $enableLMAI = $true
    }
}

$consent = [ordered]@{
    lm_ai_opt_in = $enableLMAI
    source = "install_nrl.ps1"
    updated_utc = [DateTime]::UtcNow.ToString("o")
}
$consent | ConvertTo-Json | Set-Content -Path $consentPath -Encoding UTF8

if ($enableLMAI) {
    [Environment]::SetEnvironmentVariable("NRL_LM_AI_OPT_IN", "1", "User")
    $env:NRL_LM_AI_OPT_IN = "1"
} else {
    [Environment]::SetEnvironmentVariable("NRL_LM_AI_OPT_IN", "0", "User")
    $env:NRL_LM_AI_OPT_IN = "0"
}

Write-Host "[install] complete" -ForegroundColor Green
Write-Host "  binary:  $(Join-Path $binDir 'nrl.exe')" -ForegroundColor DarkGray
Write-Host "  consent: $consentPath" -ForegroundColor DarkGray
Write-Host "  lm_ai_opt_in: $enableLMAI" -ForegroundColor DarkGray
Write-Host "  NRL_ROOT: $installRoot (examples + py/nrlpy for ``nrl demo``)" -ForegroundColor DarkGray
Write-Host "  nrlpy:     $(Join-Path $binDir 'nrlpy.cmd') (open a new terminal, then ``nrlpy script.py``)" -ForegroundColor DarkGray
