# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bio-Digital Brain example: print coverage and disk quota summaries."""

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
    parser = argparse.ArgumentParser(description="Check Drift Conqueror coverage and LMO footprint.")
    parser.add_argument("target", nargs="?", help="LMO dir, GGUF path, or SHA prefix")
    parser.add_argument("--json", action="store_true", help="Use JSON output")
    args = parser.parse_args()
    if not args.target:
        parser.print_help()
        return 0
    extra = ["--json"] if args.json else []
    env = _subprocess_env()
    code = subprocess.call([sys.executable, "-m", "nrlpy", "lmo", "coverage", args.target, *extra], env=env)
    if code != 0:
        return code
    return subprocess.call([sys.executable, "-m", "nrlpy", "lmo", "info", args.target, *extra], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
