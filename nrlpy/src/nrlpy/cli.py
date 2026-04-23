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
  nrlpy <file.py>              Run Python with ``nrl`` / ``NRL`` + seamless builtins
                               (``next_prime``, ``is_prime``, ``fabric_pulse``) pre-injected.
  nrlpy run <script.py> [-- extra args...]   Same as above; explicit form.
  nrlpy --version
  nrlpy --features
  nrlpy variant <kernel>
  nrlpy <file.nrl>
  nrlpy braincore4 [N] [ITER] [THRESHOLD]
  nrlpy bench [N] [ITER] [REPS] [THRESHOLD] [PROFILE] [--respect-control-hints]
  nrlpy control status                     JSON: prefs path, parsed file, hints_active
  nrlpy control audit tail [N]            Last N lines of control_audit.jsonl
  nrlpy assimilate [N] [ITER] [THRESHOLD]
  nrlpy demo
  nrlpy chat | talk            English-friendly status loop (telemetry + evidence tail)
  nrlpy chat --one "…"         Single non-interactive reply
  nrlpy learn status           Bounded vocab store: disk use vs cap (default 4 GiB)
  nrlpy learn cap BYTES        Set learn-store byte cap (min 1 MiB)
  nrlpy evidence tail [N]     Last N lines of immune JSONL (NRL_EVIDENCE_LOG or build/immune/events.jsonl)
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
    if args[0] == "control":
        if len(args) < 2:
            print("error: nrlpy control status | nrlpy control audit tail [N]\n\n" + USAGE, file=sys.stderr)
            return 2
        if args[1] == "status":
            prefs = runtime.load_control_preferences()
            payload: dict[str, object] = {
                "control_preferences_path": str(runtime.control_preferences_path()),
                "control_audit_log_path": str(runtime.control_audit_log_path()),
                "preferences": prefs,
                "hints_active": runtime.control_hints_active(prefs),
            }
            print(json.dumps(payload, indent=2))
            return 0
        if args[1] == "audit":
            from .evidence import read_jsonl_tail  # noqa: PLC0415

            if len(args) < 3 or args[2] != "tail":
                print("error: nrlpy control audit tail [N]\n\n" + USAGE, file=sys.stderr)
                return 2
            n = int(args[3]) if len(args) >= 4 else 20
            p = runtime.control_audit_log_path()
            if not p.is_file():
                print(
                    f"error: no control audit log at {p} (run `nrl control` first)\n\n" + USAGE,
                    file=sys.stderr,
                )
                return 2
            for line in read_jsonl_tail(p, n):
                print(line)
            return 0
        print(f"error: unknown control subcommand: {args[1]!r}\n\n" + USAGE, file=sys.stderr)
        return 2
    if args[0] == "evidence":
        from .evidence import read_jsonl_tail
        from .paths import first_existing_evidence_log

        if len(args) < 2 or args[1] != "tail":
            print("error: nrlpy evidence tail [N]\n\n" + USAGE, file=sys.stderr)
            return 2
        n = int(args[2]) if len(args) >= 3 else 20
        p = first_existing_evidence_log()
        if p is None:
            print("error: no evidence log found (set NRL_EVIDENCE_LOG or create build/immune/events.jsonl)", file=sys.stderr)
            return 2
        for line in read_jsonl_tail(p, n):
            print(line)
        return 0
    if args[0] == "learn":
        from .learn_store import LearnStore  # noqa: PLC0415

        if len(args) < 2:
            print("error: nrlpy learn status | nrlpy learn cap BYTES\n\n" + USAGE, file=sys.stderr)
            return 2
        store = LearnStore()
        if args[1] == "status":
            print(store.stats().summary())
            return 0
        if args[1] == "cap":
            if len(args) < 3:
                print("error: nrlpy learn cap BYTES\n\n" + USAGE, file=sys.stderr)
                return 2
            store.set_max_bytes(int(args[2]))
            print(f"max_bytes set to {store.max_bytes:,}")
            return 0
        print(f"error: unknown learn subcommand: {args[1]!r}\n\n" + USAGE, file=sys.stderr)
        return 2
    if args[0] in ("chat", "talk"):
        from .chat import main_chat  # noqa: PLC0415

        return main_chat(args[1:])
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
        bargs = [a for a in args if a != "--respect-control-hints"]
        respect = len(bargs) != len(args)
        neurons = int(bargs[1]) if len(bargs) >= 2 else 1_048_576
        iters = int(bargs[2]) if len(bargs) >= 3 else 4096
        reps = int(bargs[3]) if len(bargs) >= 4 else 12
        threshold = int(bargs[4]) if len(bargs) >= 5 else 8
        profile = bargs[5] if len(bargs) >= 6 else "sovereign"
        print(
            json.dumps(
                runtime.bench_cli(
                    neurons=neurons,
                    iterations=iters,
                    reps=reps,
                    threshold=threshold,
                    profile=profile,
                    respect_control_hints=respect,
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
