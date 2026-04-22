#!/usr/bin/env bash
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

echo "[release] smoke status/chat"
"$ROOT/build/bin/nrl" status
"$ROOT/build/bin/nrl" inquire speed

echo "[release] OK"
