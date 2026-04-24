# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bio-Digital Brain example: configure Learn Mode for overnight idle mapping."""

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
    parser = argparse.ArgumentParser(description="Enable Learn Mode and show health/coverage commands.")
    parser.add_argument("target", nargs="?", help="LMO dir, GGUF path, or SHA prefix for coverage check")
    parser.add_argument("--idle-sec", default="300", help="Seconds idle before Drift Conqueror starts")
    parser.add_argument("--growth-pct", default="5", help="Max ZPM growth percent per 24h window")
    args = parser.parse_args()

    os.environ["NRL_LEARN_MODE"] = "1"
    os.environ["NRL_LEARN_CONQUEST_IDLE_SEC"] = str(args.idle_sec)
    os.environ["NRL_LEARN_MAX_GROWTH_PCT"] = str(args.growth_pct)

    print("Learn Mode configured for this process:")
    print(f"  NRL_LEARN_MODE={os.environ['NRL_LEARN_MODE']}")
    print(f"  NRL_LEARN_CONQUEST_IDLE_SEC={os.environ['NRL_LEARN_CONQUEST_IDLE_SEC']}")
    print(f"  NRL_LEARN_MAX_GROWTH_PCT={os.environ['NRL_LEARN_MAX_GROWTH_PCT']}")
    env = _subprocess_env()
    env["NRL_LEARN_MODE"] = os.environ["NRL_LEARN_MODE"]
    env["NRL_LEARN_CONQUEST_IDLE_SEC"] = os.environ["NRL_LEARN_CONQUEST_IDLE_SEC"]
    env["NRL_LEARN_MAX_GROWTH_PCT"] = os.environ["NRL_LEARN_MAX_GROWTH_PCT"]
    subprocess.call([sys.executable, "-m", "nrlpy", "doctor"], env=env)
    if args.target:
        return subprocess.call([sys.executable, "-m", "nrlpy", "lmo", "coverage", args.target], env=env)
    print("When a model is absorbed, run: nrlpy lmo coverage <sha-prefix>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
