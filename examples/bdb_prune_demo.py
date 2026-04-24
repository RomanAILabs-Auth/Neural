# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bio-Digital Brain example: preview or apply quota pruning."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "nrlpy" / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview/apply LMO ZPM pruning.")
    parser.add_argument("target", nargs="?", help="LMO dir, GGUF path, or SHA prefix")
    parser.add_argument("--apply", action="store_true", help="Apply prune; default is dry-run")
    parser.add_argument("--aggressive", action="store_true", help="Use stronger 80 percent quota target")
    args = parser.parse_args()
    if not args.target:
        parser.print_help()
        return 0
    cmd = [sys.executable, "-m", "nrlpy", "lmo", "prune", args.target]
    if not args.apply:
        cmd.append("--dry-run")
    if args.aggressive:
        cmd.append("--aggressive")
    return subprocess.call(cmd, env=_subprocess_env())


if __name__ == "__main__":
    raise SystemExit(main())
