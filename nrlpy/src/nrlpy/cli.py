"""NRLPy CLI.

Contract (aligned with TriPy-style ergonomics):
  nrlpy <file.py>     Run a Python file with NRL assimilation globals injected (same as ``nrlpy run``).
  nrlpy <file.nrl>    Run a control file via native ``nrl file``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import runtime
from .shell import run_path

USAGE = """\
nrlpy - Python front-end for the NRL machine-code engine

Usage:
  nrlpy <file.py>              Run Python with ``nrl`` / ``NRL`` pre-injected (TriPy-style; raw speed via _core + nrl bench).
  nrlpy run <script.py> [-- extra args...]   Same as above; explicit form.
  nrlpy --version
  nrlpy --features
  nrlpy variant <kernel>
  nrlpy <file.nrl>
  nrlpy braincore4 [N] [ITER] [THRESHOLD]
  nrlpy bench [N] [ITER] [REPS] [THRESHOLD] [PROFILE]
  nrlpy assimilate [N] [ITER] [THRESHOLD]
  nrlpy demo
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    if args[0] == "--version":
        print(runtime.version())
        return 0
    if args[0] == "--features":
        print(json.dumps(runtime.features(), indent=2))
        return 0
    if args[0] == "variant" and len(args) >= 2:
        print(runtime.active_variant(args[1]))
        return 0
    if args[0] == "braincore4":
        neurons = int(args[1]) if len(args) >= 2 else 8_000_000
        iters = int(args[2]) if len(args) >= 3 else 1000
        threshold = int(args[3]) if len(args) >= 4 else 12
        print(
            json.dumps(
                runtime.braincore_int4(neurons=neurons, iterations=iters, threshold=threshold),
                indent=2,
            )
        )
        return 0
    if args[0] == "demo":
        repo = Path(__file__).resolve().parents[3]
        demo = repo / "examples" / "ultimate_power_demo.py"
        if not demo.is_file():
            print(f"error: demo script not found at {demo}\n\n" + USAGE, file=sys.stderr)
            return 2
        run_path(str(demo), [])
        return 0
    if args[0] == "run":
        if len(args) < 2:
            print("error: nrlpy run requires a script path\n\n" + USAGE, file=sys.stderr)
            return 2
        script = args[1]
        extra: list[str] = []
        if "--" in args:
            i = args.index("--")
            extra = args[i + 1 :]
        else:
            extra = args[2:]
        run_path(script, extra)
        return 0
    if args[0] == "assimilate":
        neurons = int(args[1]) if len(args) >= 2 else 4096
        iters = int(args[2]) if len(args) >= 3 else 256
        threshold = int(args[3]) if len(args) >= 4 else 10
        print(json.dumps(runtime.assimilate_cli(neurons, iters, threshold), indent=2))
        return 0
    if args[0] == "bench":
        neurons = int(args[1]) if len(args) >= 2 else 1_048_576
        iters = int(args[2]) if len(args) >= 3 else 4096
        reps = int(args[3]) if len(args) >= 4 else 12
        threshold = int(args[4]) if len(args) >= 5 else 8
        profile = args[5] if len(args) >= 6 else "sovereign"
        print(
            json.dumps(
                runtime.bench_cli(
                    neurons=neurons,
                    iterations=iters,
                    reps=reps,
                    threshold=threshold,
                    profile=profile,
                ),
                indent=2,
            )
        )
        return 0
    if args[0].endswith(".nrl"):
        print(runtime.run_nrl_file(args[0]), end="")
        return 0

    if args[0].endswith(".py"):
        script = Path(args[0])
        if not script.is_file():
            print(f"error: Python file not found: {args[0]}\n\n{USAGE}", file=sys.stderr)
            return 2
        extra: list[str] = []
        if "--" in args:
            i = args.index("--")
            extra = args[i + 1 :]
        else:
            extra = args[1:]
        run_path(str(script.resolve()), extra)
        return 0

    print(f"error: unknown args: {' '.join(args)}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
