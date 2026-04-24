#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
# One-shot WPS + GGUF automation (pytest slice, golden harness, autopilot).
# Usage:
#   ./scripts/wps_full_gate.sh       # measure-only autopilot
#   ./scripts/wps_full_gate.sh ci    # enforce mm-replay >= 1000 effective WPS
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../benchmarks/gguf_golden.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -n "${NRL_REPO:-}" && -f "$NRL_REPO/benchmarks/gguf_golden.py" ]]; then
  ROOT="$(cd "$NRL_REPO" && pwd)"
else
  echo "Could not find NRL repo. Try: nrlpy wps-gate   (set NRL_REPO to your clone)" >&2
  if [[ "${1:-}" == "ci" ]]; then
    exec nrlpy wps-gate --ci
  else
    exec nrlpy wps-gate
  fi
fi
cd "$ROOT"
export PYTHONPATH="$ROOT/nrlpy/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== [wps-full-gate] specs ==="
echo "  repo:       $ROOT"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  python:     $(python --version 2>&1)"
echo "  ci:         ${1:-}"
echo ""

echo "=== [wps-full-gate] pytest (GGUF + chat + autopilot) ==="
python -m pytest \
  nrlpy/tests/test_wps_autopilot.py \
  nrlpy/tests/test_gguf.py \
  nrlpy/tests/test_gguf_chat.py \
  nrlpy/tests/test_cli_chat.py \
  -q --tb=line

echo ""
echo "=== [wps-full-gate] gguf_golden --mode auto ==="
python benchmarks/gguf_golden.py --mode auto

echo ""
echo "=== [wps-full-gate] wps_autopilot ==="
if [[ "${1:-}" == "ci" ]]; then
  python benchmarks/wps_autopilot.py --markdown build/wps_autopilot.md --min-mm-effective-wps 1000
  python benchmarks/gguf_golden.py --mode mm-replay --mm-min-wps 1000
else
  python benchmarks/wps_autopilot.py --markdown build/wps_autopilot.md
fi

echo ""
echo "=== [wps-full-gate] artifacts ==="
echo "  $ROOT/build/gguf_golden/gguf_golden.json"
echo "  $ROOT/build/wps_autopilot.json"
echo "  $ROOT/build/wps_autopilot.md"
echo "[wps-full-gate] OK"
