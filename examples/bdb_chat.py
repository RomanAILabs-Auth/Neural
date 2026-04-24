# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bio-Digital Brain example: absorb (optional) and start rewired chat."""

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
    parser = argparse.ArgumentParser(description="Start NRL rewired chat for a local GGUF.")
    parser.add_argument("model", nargs="?", help="Path to a local .gguf model")
    parser.add_argument("--absorb-first", action="store_true", help="Run nrlpy absorb before chat")
    args = parser.parse_args()
    if not args.model:
        parser.print_help()
        return 0
    env = _subprocess_env()
    if args.absorb_first:
        subprocess.check_call([sys.executable, "-m", "nrlpy", "absorb", args.model], env=env)
    return subprocess.call([sys.executable, "-m", "nrlpy", "chat", args.model, "--rewired"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
