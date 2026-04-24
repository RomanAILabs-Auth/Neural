#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
set -euo pipefail

CONFIG="${1:-Release}"
RUN_TESTS="${2:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"
OBJ="$BUILD/obj"
LIB="$BUILD/lib"
BIN="$BUILD/bin"

CC="${CC:-cc}"
AR="${AR:-ar}"

mkdir -p "$OBJ" "$LIB" "$BIN"

INCLUDE="-I$ROOT/engine/include"
STD="-std=c11"
WARN="-Wall -Wextra -Wpedantic"
ARCH="-mavx2 -mxsave -mxsaveopt -mpopcnt -mbmi2 -mfma"

if [ "$CONFIG" = "Debug" ]; then
  OPT="-O0 -g -DDEBUG $ARCH"
else
  OPT="-O3 -DNDEBUG $ARCH"
fi

SOURCES=(
  engine/src/cpuid.c
  engine/src/dispatch.c
  engine/src/runtime_status.c
  engine/src/capi.c
  engine/src/braincore_int4_scalar.c
  engine/src/braincore_int4_avx2.c
  engine/src/zpm_int4_static.c
  engine/src/zpm_omega_router.c
  engine/src/version.c
  engine/src/llama_bridge.c
  engine/src/ladder_native.c
  engine/src/ladder_full.c
)

OBJS=()
echo "[build] compiling engine sources ..."
for s in "${SOURCES[@]}"; do
  out="$OBJ/$(basename "$s" .c).o"
  $CC -c "$ROOT/$s" -o "$out" $STD $INCLUDE $OPT $WARN
  OBJS+=("$out")
done

LIB_PATH="$LIB/libnrl.a"
rm -f "$LIB_PATH"
$AR rcs "$LIB_PATH" "${OBJS[@]}"

EXE_PATH="$BIN/nrl"
echo "[build] linking nrl ..."
$CC "$ROOT/engine/src/main.c" -o "$EXE_PATH" "$LIB_PATH" $STD $INCLUDE $OPT $WARN

echo "[build] OK"
echo "  exe: $EXE_PATH"
echo "  lib: $LIB_PATH"

if [ "$RUN_TESTS" = "1" ]; then
  TEST_OUT="$BIN/nrl-tests"
  echo "[test] compiling nrl-tests ..."
  $CC "$ROOT/engine/tests/test_runtime.c" -o "$TEST_OUT" "$LIB_PATH" $STD $INCLUDE $OPT $WARN
  "$TEST_OUT"
fi
