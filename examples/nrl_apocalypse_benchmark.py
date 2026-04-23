# nrl_apocalypse_benchmark.py
# RomanAILabs — stress harness (determinism + lane comparison)
# Copyright Daniel Harding - RomanAILabs
#
# IMPORTANT (read before interpreting numbers):
# - ``nrl bench`` argument order is:  neurons  iterations  reps  threshold  profile
# - Varying **threshold** changes reset dynamics (valid 1..15). Varying **reps** changes
#   measurement noise / warmup averaging — not "entropy injected into the lattice."
# - Omega on the **default synthetic zero start + fixed drive** is highly skip-prone by
#   construction; flat skip curves here mean **structural routing on this harness**, not
#   "entropy blindness" or "learning." See ``docs/nrl_alive_language_evolution_architecture.md``
#   for where real bounded adaptation is specified (control plane + shadow promotion, not
#   silent hot-loop ML).
#
# Run from repo (or anywhere ``nrl`` is on PATH / ``NRL_BIN`` is set):
#   python examples/nrl_apocalypse_benchmark.py
#   python examples/nrl_apocalypse_benchmark.py --quick   # smaller size list

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

SIZES_FULL = [131072, 262144, 524288, 1048576, 2097152, 4194304]
SIZES_QUICK = [131072, 262144, 524288]
ITER = 256
BENCH_REPS = 2
THRESHOLD_BASELINE = 8
ENTROPY_THRESHOLDS = [2, 4, 6, 8, 10, 12, 14]  # sweep threshold (real knob), not "random lattice"


def nrl_exe() -> str:
    return os.environ.get("NRL_BIN") or shutil.which("nrl") or "nrl"


def run_cmd(args: list[str], timeout: float = 300.0) -> str:
    proc = subprocess.run(
        [nrl_exe(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"nrl {' '.join(args)} failed ({proc.returncode}):\n{out}")
    return out


def parse(pattern: str, text: str, cast: type[Any] = float) -> Any | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return cast(m.group(1))


def bench(n: int, mode: str, *, reps: int = BENCH_REPS, threshold: int = THRESHOLD_BASELINE) -> dict[str, float | None]:
    # Correct order: neurons iterations reps threshold profile
    out = run_cmd(
        ["bench", str(n), str(ITER), str(reps), str(threshold), mode],
    )
    return {
        "skip": parse(r"skip_ratio:\s*([\d.]+)", out),
        "exec": parse(r"executed_gops:\s*([\d.]+)", out),
        "virt": parse(r"virtual_gops:\s*([\d.]+)", out),
    }


def assimilate_checksum(n: int, *, iterations: int = ITER, threshold: int = THRESHOLD_BASELINE) -> int | None:
    out = run_cmd(["assimilate", str(n), str(iterations), str(threshold)])
    return parse(r"checksum_fnv1a64:\s*(\d+)", out, int)


def _f(x: float | None, fmt: str) -> str:
    if x is None:
        return "n/a"
    return format(x, fmt)


def main() -> None:
    p = argparse.ArgumentParser(description="NRL stress sweep with correct bench CLI semantics.")
    p.add_argument("--quick", action="store_true", help="Use smaller neuron list")
    args = p.parse_args()
    sizes = SIZES_QUICK if args.quick else SIZES_FULL

    print("\n==============================")
    print(" NRL APOCALYPSE BENCHMARK (CLI-correct)")
    print("==============================")
    print(f"  nrl: {nrl_exe()}")
    print(f"  bench: neurons iter={ITER} reps={BENCH_REPS} threshold=<sweep> profile=<mode>")
    print()

    for n in sizes:
        print(f"\n--- SIZE {n} ---")

        c1 = assimilate_checksum(n)
        c2 = assimilate_checksum(n)
        ok = c1 is not None and c2 is not None and c1 == c2
        print(f"[Determinism assimilate] {ok}  checksum_fnv1a64={c1}")

        sov = bench(n, "sovereign")
        omg = bench(n, "omega")
        print("[Baseline]")
        print(
            f"  SOV exec={_f(sov['exec'], '.3f')} virt={_f(sov['virt'], '.3f')} skip={_f(sov['skip'], '.6f')}"
        )
        print(
            f"  OMG exec={_f(omg['exec'], '.4f')} virt={_f(omg['virt'], '.3f')} skip={_f(omg['skip'], '.6f')}"
        )

        print("\n[Threshold sweep on omega (same reps, real threshold axis)]")
        for t in ENTROPY_THRESHOLDS:
            r = bench(n, "omega", threshold=t)
            print(
                f"  threshold={t:2d}  skip={_f(r['skip'], '.6f')}  "
                f"exec={_f(r['exec'], '.4f')}  virt={_f(r['virt'], '.3f')}"
            )

        # Adversarial: random threshold (still valid 1..15), not random "entropy field"
        t_rand = random.randint(1, 15)
        shock = bench(n, "omega", threshold=t_rand)
        print(f"\n[Adversarial threshold shock t={t_rand}]")
        print(f"  exec={_f(shock['exec'], '.4f')} skip={_f(shock['skip'], '.6f')}")

        if n < sizes[-1]:
            t0 = time.perf_counter()
            bench(n * 2, "omega")
            dt = time.perf_counter() - t0
            print(f"\n[Scaling x2 omega] wall={dt:.5f}s")

    print("\n==============================")
    print(" HOW TO READ THIS (honest)")
    print("==============================")
    print("  - High omega skip here measures structural sparse routing on a synthetic harness,")
    print("    not 'emergent intelligence' — interpret with nrl-architecture §0.5 accounting.")
    print("  - Bounded AI/ML adaptation = control plane: workload_id + shadow + evidence (alive doc).")
    print("  - nrlpy plasticity today: stub only (see nrlpy/plasticity.py).")
    print("\nEND\n")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
