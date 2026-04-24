#!/usr/bin/env bash
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[release] build + tests"
if [ -x "$ROOT/build.sh" ]; then
  "$ROOT/build.sh" Release 1
else
  echo "build.sh missing or not executable" >&2
  exit 1
fi

echo "[release] locked nrl-vs-cpp artifact"
python "$ROOT/benchmarks/nrl_vs_cpp.py" \
  --neurons 1048576 --iterations 4096 --reps 4 --threshold 8

echo "[release] verify workload_identity in bench JSON"
python -c "import json, pathlib; p=pathlib.Path('$ROOT/build/bench/nrl_vs_cpp.json'); d=json.loads(p.read_text(encoding='utf-8')); assert 'workload_identity' in d and 'structural_hash' in d['workload_identity']"

if compgen -G "$ROOT/nrlpy/src/nrlpy/_core"*.so >/dev/null 2>&1 || compgen -G "$ROOT/nrlpy/src/nrlpy/_core"*.pyd >/dev/null 2>&1; then
  echo "[release] nrlpy learn + chat one-shot"
  export PYTHONPATH="$ROOT/nrlpy/src"
  python -m nrlpy.cli learn status
  python -m nrlpy.cli chat --one "status"
  python -m nrlpy.cli control status
else
  echo "[release] skip nrlpy learn/chat (no _core extension)" >&2
fi

echo "[release] smoke status/chat"
st="$("$ROOT/build/bin/nrl" status 2>&1)" || exit 1
case "$st" in *control_preferences_path*) ;; *) echo "nrl status missing control_preferences_path" >&2; exit 1;; esac
printf '%s\n' "$st"
"$ROOT/build/bin/nrl" inquire speed

echo "[release] smoke nrl control (sandbox)"
NRL="$ROOT/build/bin/nrl"
out="$("$NRL" control "buy stocks" 2>&1)" || exit 1
case "$out" in *BLOCKED*) ;; *) echo "expected BLOCKED in control output" >&2; exit 1;; esac
out="$("$NRL" control "maximum power for one hour" 2>&1)" || exit 1
case "$out" in *DEFER*) ;; *) echo "expected DEFER in control output" >&2; exit 1;; esac

echo "[release] OK"
