#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
# Live readiness: release_check.sh + nrlpy pytest when _core is present.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_PYTEST="${SKIP_PYTEST:-0}"

echo "[live] release_check.sh"
"$ROOT/scripts/release_check.sh"

if [[ "$SKIP_PYTEST" != "1" ]]; then
  if compgen -G "$ROOT/nrlpy/src/nrlpy/_core*.so" > /dev/null || compgen -G "$ROOT/nrlpy/src/nrlpy/_core*.pyd" > /dev/null; then
    echo "[live] nrlpy pytest"
    export PYTHONPATH="$ROOT/nrlpy/src"
    if python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"; then
      python -m pytest "$ROOT/nrlpy/tests" -q
    else
      echo "[live] pytest not installed; skipping" >&2
    fi
  else
    echo "[live] nrlpy._core not built; skipping pytest" >&2
  fi
fi

echo "[live] OK — ready for live testing on this host"
