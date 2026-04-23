# real_prime_flex.py
# Copyright RomanAILabs - Daniel Harding
#
# Stress/compare ``nrl assimilate`` + ``nrl bench`` with a **computed** lattice spec
# (no hard-coded giant literals). Verification is the **checksum_fnv1a64** from the
# native harness: two independent assimilate runs must agree on the same digest.
#
# Usage (``nrl`` on PATH, or set NRL_BIN):
#   python examples/real_prime_flex.py
#   python examples/real_prime_flex.py 262144 128 8
#   python examples/real_prime_flex.py 262144 128 8 4    # last arg = bench reps
# Env overrides (used when argv omits that field): NRL_FLEX_NEURONS, NRL_FLEX_ITERATIONS,
#   NRL_FLEX_THRESHOLD, NRL_FLEX_BENCH_REPS

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any


def nrl_bin() -> str:
    return os.environ.get("NRL_BIN") or shutil.which("nrl") or "nrl"


def _env_u64(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip(), 10)


def lattice_spec(argv: list[str]) -> tuple[int, int, int, int]:
    """Return neurons, iterations, threshold, bench_reps from argv (positional) then env then defaults."""
    # Defaults from geometry (not pasted decimals): 2**20 neurons, 2**8 iters, threshold 8.
    neurons = _env_u64("NRL_FLEX_NEURONS", 1 << 20)
    iterations = _env_u64("NRL_FLEX_ITERATIONS", 1 << 8)
    threshold = _env_u64("NRL_FLEX_THRESHOLD", 8)
    bench_reps = _env_u64("NRL_FLEX_BENCH_REPS", 2)
    if len(argv) >= 1:
        neurons = int(argv[0])
    if len(argv) >= 2:
        iterations = int(argv[1])
    if len(argv) >= 3:
        threshold = int(argv[2])
    if len(argv) >= 4:
        bench_reps = int(argv[3])
    if neurons <= 0 or (neurons & 1) != 0:
        raise SystemExit(f"neurons must be a positive even integer, got {neurons}")
    if iterations <= 0:
        raise SystemExit(f"iterations must be positive, got {iterations}")
    if not 1 <= threshold <= 15:
        raise SystemExit(f"threshold must be 1..15, got {threshold}")
    if bench_reps <= 0:
        raise SystemExit(f"bench reps must be positive, got {bench_reps}")
    return neurons, iterations, threshold, bench_reps


def parse_nrl_lines(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().replace(" ", "_")
        val = val.strip()
        if key in {"variant", "lane", "profile", "mode"}:
            out[key] = val
            continue
        if key == "checksum_fnv1a64":
            out[key] = int(val.split()[0], 10)
            continue
        try:
            if "." in val.split()[0]:
                out[key] = float(val.split()[0])
            else:
                out[key] = int(val.split()[0], 10)
        except (ValueError, IndexError):
            out[key] = val
    return out


def run_nrl(args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    exe = nrl_bin()
    return subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_assimilate(neurons: int, iterations: int, threshold: int) -> dict[str, Any]:
    proc = run_nrl(
        ["assimilate", str(neurons), str(iterations), str(threshold)],
        timeout=180.0,
    )
    merged = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"nrl assimilate failed ({proc.returncode}):\n{merged}")
    return parse_nrl_lines(merged)


def run_bench(neurons: int, iterations: int, reps: int, threshold: int, profile: str) -> dict[str, Any]:
    proc = run_nrl(
        ["bench", str(neurons), str(iterations), str(reps), str(threshold), profile],
        timeout=300.0,
    )
    merged = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"nrl bench {profile} failed ({proc.returncode}):\n{merged}")
    return parse_nrl_lines(merged)


def main() -> None:
    neurons, iterations, threshold, bench_reps = lattice_spec(sys.argv[1:])

    # “Flex” number: entirely determined by the chosen spec (recomputable by hand / script).
    baseline_equiv_updates = neurons * iterations

    print("=" * 72)
    print(" NRL real_prime_flex - computed spec + digest verification")
    print("=" * 72)
    print(f"  nrl binary: {nrl_bin()}")
    print(f"  neurons:   {neurons}  (hex {neurons:#x})")
    print(f"  iterations:{iterations}")
    print(f"  threshold: {threshold}")
    print(f"  baseline_equiv_updates (neurons*iterations): {baseline_equiv_updates}")
    print()

    print("[1] First assimilate (sovereign INT4 digest)")
    t0 = time.perf_counter()
    a1 = run_assimilate(neurons, iterations, threshold)
    t1 = time.perf_counter()
    print(f"  elapsed_s: {t1 - t0:.6f}")
    for k in ("packed_bytes", "neurons", "iterations", "threshold", "checksum_fnv1a64"):
        if k in a1:
            print(f"  {k}: {a1[k]}")

    print("\n[2] Second assimilate (independent process - digest must match)")
    a2 = run_assimilate(neurons, iterations, threshold)
    c1 = int(a1["checksum_fnv1a64"])
    c2 = int(a2["checksum_fnv1a64"])
    if c1 != c2:
        raise SystemExit(f"VERIFICATION FAILED: checksum {c1} != {c2}")
    print(f"  checksum_fnv1a64: {c2}")
    print()
    print("VERIFIED: both assimilate runs agree on checksum_fnv1a64")
    print(f"  >>> checksum_fnv1a64 = {c1}  <<<  (copy this value to compare across machines / builds)")
    print()

    print(f"[3] Bench sovereign vs omega (reps={bench_reps}, same N/I/T)")
    t0 = time.perf_counter()
    b_sov = run_bench(neurons, iterations, bench_reps, threshold, "sovereign")
    cold = time.perf_counter() - t0
    print("  sovereign:")
    for k in ("executed_updates", "baseline_equiv_updates", "skip_ratio", "executed_gops", "virtual_gops"):
        if k in b_sov:
            print(f"    {k}: {b_sov[k]}")

    t0 = time.perf_counter()
    b_om = run_bench(neurons, iterations, bench_reps, threshold, "omega")
    warm = time.perf_counter() - t0
    print("  omega:")
    for k in ("executed_updates", "baseline_equiv_updates", "skip_ratio", "executed_gops", "virtual_gops"):
        if k in b_om:
            print(f"    {k}: {b_om[k]}")

    speedup = cold / warm if warm > 0 else float("inf")
    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"  Digest witness (assimilate): checksum_fnv1a64 = {c1}")
    print(f"  Sovereign wall time: {cold:.4f}s")
    print(f"  Omega wall time:     {warm:.4f}s")
    print(f"  Wall speedup (bench): {speedup:.2f}x")
    print()
    print("Re-run with other sizes, e.g.:")
    print("  python examples/real_prime_flex.py 524288 256 8 4")


if __name__ == "__main__":
    main()
