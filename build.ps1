# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
[CmdletBinding()]
param(
    [ValidateSet('Release','Debug')]
    [string]$Config = 'Release',
    [switch]$Tests,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$zigCandidates = @(
    (Join-Path $root 'tools\zig\zig.exe'),
    (Join-Path $root '..\trinary\tools\zig\zig.exe'),
    'zig'
)

$zig = $null
foreach ($candidate in $zigCandidates) {
    try {
        if ($candidate -eq 'zig') {
            & $candidate version *> $null
            if ($LASTEXITCODE -eq 0) { $zig = $candidate; break }
        } elseif (Test-Path $candidate) {
            $zig = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $zig) {
    throw "zig not found. Install zig or place it in tools\zig\zig.exe"
}

$build = Join-Path $root 'build'
$obj = Join-Path $build 'obj'
$lib = Join-Path $build 'lib'
$bin = Join-Path $build 'bin'

if ($Clean -and (Test-Path $build)) {
    Remove-Item $build -Recurse -Force
}
foreach ($d in @($obj, $lib, $bin)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

$target = 'x86_64-windows-gnu'
$include = "-I$root\engine\include"
$warn = @('-Wall','-Wextra','-Wpedantic')
$arch = @('-mavx2','-mxsave','-mxsaveopt','-mpopcnt','-mbmi2','-mfma')

if ($Config -eq 'Release') {
    $opt = @('-O3','-DNDEBUG') + $arch
} else {
    $opt = @('-O0','-g','-DDEBUG') + $arch
}

$std = '-std=c11'
$sources = @(
    'engine\src\cpuid.c',
    'engine\src\dispatch.c',
    'engine\src\runtime_status.c',
    'engine\src\capi.c',
    'engine\src\braincore_int4_scalar.c',
    'engine\src\braincore_int4_avx2.c',
    'engine\src\zpm_int4_static.c',
    'engine\src\zpm_omega_router.c',
    'engine\src\version.c',
    'engine\src\llama_bridge.c',
    'engine\src\ladder_native.c',
    'engine\src\ladder_full.c'
)

function Invoke-Cc {
    param([string[]]$CcArgs)
    & $script:zig cc -target $script:target @CcArgs
    if ($LASTEXITCODE -ne 0) {
        throw "zig cc failed: $($CcArgs -join ' ')"
    }
}

$objs = @()
Write-Host "[build] compiling engine sources ..." -ForegroundColor Cyan
foreach ($s in $sources) {
    $src = Join-Path $root $s
    $dst = Join-Path $obj ([IO.Path]::ChangeExtension([IO.Path]::GetFileName($s),'.obj'))
    Invoke-Cc -CcArgs (@('-c', $src, '-o', $dst, $std, $include) + $opt + $warn)
    $objs += $dst
}

$libPath = Join-Path $lib 'libnrl.a'
if (Test-Path $libPath) { Remove-Item $libPath -Force }
& $zig ar rcs $libPath @objs
if ($LASTEXITCODE -ne 0) { throw "zig ar failed" }

$exePath = Join-Path $bin 'nrl.exe'
$mainSrc = Join-Path $root 'engine\src\main.c'
Write-Host "[build] linking nrl.exe ..." -ForegroundColor Cyan
Invoke-Cc -CcArgs (@($mainSrc, '-o', $exePath, $libPath, $std, $include) + $opt + $warn + @('-lpsapi'))

$pyVer = ''
$pydOut = ''
Write-Host "[build] nrlpy._core extension ..." -ForegroundColor Cyan
try {
    $pyInc = & python -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
    $pyLibDir = & python -c "import sysconfig, os; print(os.path.join(sysconfig.get_config_var('installed_base'), 'libs'))"
    $pyVer = & python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
    $coreSrc = Join-Path $root 'nrlpy\src\_core\module.c'
    $pydOut = Join-Path $root "nrlpy\src\nrlpy\_core.cp$pyVer-win_amd64.pyd"
    Invoke-Cc -CcArgs (@('-shared','-o', $pydOut, $coreSrc, $libPath, "-I$pyInc", $include,
                 "-L$pyLibDir", "-lpython$pyVer") + $opt + $warn)
} catch {
    Write-Warning "Skipping nrlpy extension build (python toolchain unavailable): $($_.Exception.Message)"
}

Write-Host "[build] OK" -ForegroundColor Green
Write-Host "  exe: $exePath" -ForegroundColor DarkGray
Write-Host "  lib: $libPath" -ForegroundColor DarkGray
if ($pydOut) {
    Write-Host "  pyd: $pydOut" -ForegroundColor DarkGray
}

if ($Tests) {
    $testSrc = Join-Path $root 'engine\tests\test_runtime.c'
    $testExe = Join-Path $bin 'nrl-tests.exe'
    Write-Host "[test] compiling nrl-tests.exe ..." -ForegroundColor Cyan
    Invoke-Cc -CcArgs (@($testSrc, '-o', $testExe, $libPath, $std, $include) + $opt + $warn)
    & $testExe
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    if ($pydOut) {
        Write-Host "[test] running nrlpy tests ..." -ForegroundColor Cyan
        $hasPytest = & python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
        if ($LASTEXITCODE -eq 0) {
            $env:PYTHONPATH = Join-Path $root 'nrlpy\src'
            & python -m pytest (Join-Path $root 'nrlpy\tests') -q
            if ($LASTEXITCODE -ne 0) { throw "nrlpy tests failed" }
        } else {
            Write-Warning "pytest not installed; skipping nrlpy tests"
        }
    }
}
