#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
# GGUF-runner readiness gate: nrlpy import + stub golden + optional real golden.
#
# Usage:
#   ./scripts/live_readiness_gguf.sh                    # auto: stub + real if $NRL_GGUF_GOLDEN_MODEL
#   ./scripts/live_readiness_gguf.sh stub               # stub only
#   ./scripts/live_readiness_gguf.sh real /path/model.gguf   # real only (hard-gated)
#
# Env overrides: NRL_GGUF_GOLDEN_MODEL, NRL_INFERENCE, NRL_LLAMA_CLI,
#                NRL_GGUF_MIN_WPS, NRL_GGUF_WPS_METRIC.
#
# Exit codes:
#   0 - all selected modes passed
#   1 - regression (golden assertion failed)
#   2 - config error (e.g. mode=real with no model)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/nrlpy/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-auto}"
MODEL_ARG="${2:-}"

if [[ "$MODE" != "stub" && "$MODE" != "p2active-sim" && "$MODE" != "p2active-prefill" && "$MODE" != "mm-replay" && "$MODE" != "real" && "$MODE" != "auto" ]]; then
  echo "usage: $0 [stub|p2active-sim|p2active-prefill|mm-replay|real|auto] [model.gguf]" >&2
  exit 2
fi

echo "[live-gguf] nrlpy import probe"
python -c "from nrlpy import gguf, gguf_chat; print(gguf.__name__, gguf_chat.__name__)"

HARNESS="$ROOT/benchmarks/gguf_golden.py"
if [[ ! -f "$HARNESS" ]]; then
  echo "missing $HARNESS" >&2
  exit 2
fi

ARGS=(--mode "$MODE" --prompt "Reply with one short sentence about the number two." --max-tokens 8 --seed 42 --chat-format phi3)
if [[ -n "$MODEL_ARG" ]]; then
  ARGS+=(--model "$MODEL_ARG")
fi
if [[ -n "${NRL_GGUF_MIN_WPS:-}" ]]; then
  ARGS+=(--min-wps "$NRL_GGUF_MIN_WPS")
  ARGS+=(--wps-metric "${NRL_GGUF_WPS_METRIC:-effective}")
fi

echo "[live-gguf] golden harness $MODE"
set +e
python "$HARNESS" "${ARGS[@]}"
rc=$?
set -e
case "$rc" in
  0) echo "[live-gguf] OK" ;;
  2) echo "[live-gguf] CONFIG ERROR (rc=$rc): pass a model path or set NRL_GGUF_GOLDEN_MODEL" ;;
  *) echo "[live-gguf] FAIL (rc=$rc) - see build/gguf_golden/gguf_golden.md" ;;
esac
exit $rc
