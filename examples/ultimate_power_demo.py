# -*- coding: utf-8 -*-
"""
ULTIMATE RAW POWER DEMO — NRL + NRLPy
=====================================
Run (from repository root, with nrlpy on PYTHONPATH or installed editable):

    python -m nrlpy.cli run examples/ultimate_power_demo.py

Or use the native launcher (sets ``PYTHONPATH`` for you):

    nrl demo

What this script actually proves (no fairy tales):
  • Packed INT4 tensors are mutated in **machine code** via ``braincore_int4_inplace``.
  • ``nrl assimilate`` checksum parity is **bit-exact** against the same packed layout.
  • ``nrl bench`` **sovereign** vs **zpm** vs **omega** profiles show the real split between
    executed updates and baseline-equivalent accounting — the heart of ZPM / Omega.
  • A second **zpm** bench on identical args is often slightly faster (OS/CPU cache);
    the *structural* “muscle memory” is the ZPM static map **inside each bench process**,
    not magic persistence between Python subprocesses (see comments in Phase Z).

Console drama uses ANSI colors when stdout is a TTY; numbers come only from live APIs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Imports: work under ``python -m nrlpy.cli run`` (injected ``nrl``) or plain
# ``python`` with PYTHONPATH=nrlpy/src.
# ---------------------------------------------------------------------------
try:
    nrl  # type: ignore[name-defined]
except NameError:
    from nrlpy.compat import nrl

from nrlpy import native, runtime


# =============================================================================
# ANSI theatre (disabled when piped / non-TTY)
# =============================================================================
def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GRN = "\033[92m"
    YEL = "\033[93m"
    BLU = "\033[94m"
    MAG = "\033[95m"
    CYA = "\033[96m"
    WHT = "\033[97m"


def paint(s: str, *codes: str) -> str:
    if not _use_color():
        return s
    return "".join(codes) + s + C.RST


def hr(char: str = "=", width: int = 76) -> None:
    print(paint(char * width, C.DIM, C.BOLD))


def title(msg: str) -> None:
    hr()
    print(paint("  " + msg, C.BOLD, C.CYA))
    hr()


def sub(msg: str) -> None:
    print(paint(" >> " + msg, C.YEL, C.BOLD))


def bar(ratio: float, width: int = 44) -> str:
    ratio = max(0.0, min(1.0, ratio))
    n = int(round(ratio * width))
    body = "#" * n + "." * (width - n)
    return "[" + body + "]"


def sci(x: float) -> str:
    """Human-readable scientific notation for large accounting rates."""
    if x == 0.0:
        return "0"
    ax = abs(x)
    exp = 0
    while ax >= 10.0:
        ax /= 10.0
        exp += 1
    while ax < 1.0 and ax > 0.0:
        ax *= 10.0
        exp -= 1
    sign = "-" if x < 0 else ""
    return f"{sign}{ax:.3f}e{exp:+d}"


# =============================================================================
# Locate native ``nrl`` (installer: NRL_ROOT/bin; dev: NRL_ROOT/build/bin)
# =============================================================================
def nrl_executable() -> Path:
    return runtime.nrl_binary_path()


def require_nrl(exe: Path) -> None:
    if not exe.is_file():
        print(
            paint(f"FATAL: native binary not found at {exe}", C.RED, C.BOLD),
            file=sys.stderr,
        )
        print(
            paint(
                "Set NRL_BIN to nrl.exe, or ensure NRL_ROOT/bin (install) or "
                "./build/bin (dev checkout) exists. Build: .\\build.ps1 or ./build.sh Release",
                C.DIM,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)


# =============================================================================
# Full ``nrl bench`` parse — includes omega-only fields absent from BenchCliResult
# =============================================================================
def parse_nrl_bench_stdout(stdout: str) -> dict[str, Any]:
    """
    Parse every ``key: value`` line from ``nrl bench`` output.
    Integer-looking keys use int(); floats use float().
    """
    out: dict[str, Any] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or ":" not in line or line.startswith("NRL bench"):
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        key = key.replace(" ", "_")
        if key in {"profile", "mode", "variant"}:
            out[key] = val
            continue
        if key in {
            "neurons",
            "iterations",
            "reps",
            "threshold",
            "executed_updates",
            "baseline_equiv_updates",
        }:
            out[key] = int(val.replace(",", "").split()[0], 10)
            continue
        if key == "checksum_fnv1a64":
            out[key] = int(val.split()[0], 10)
            continue
        try:
            out[key] = float(val.split()[0])
        except ValueError:
            out[key] = val
    return out


def bench_raw(
    exe: Path,
    neurons: int,
    iterations: int,
    reps: int,
    threshold: int,
    profile: str,
) -> dict[str, Any]:
    proc = subprocess.run(
        [
            str(exe),
            "bench",
            str(neurons),
            str(iterations),
            str(reps),
            str(threshold),
            profile,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "nrl bench failed")
    return parse_nrl_bench_stdout(proc.stdout)


def print_bench_frame(tag: str, d: dict[str, Any]) -> None:
    """Pretty-print one parsed bench snapshot."""
    skip = float(d.get("skip_ratio", 0.0))
    vg = float(d.get("virtual_gops", 0.0))
    eg = float(d.get("executed_gops", 0.0))
    ex = int(d.get("executed_updates", 0))
    base = int(d.get("baseline_equiv_updates", 0))
    bps = float(d.get("baseline_equiv_updates_per_sec", vg * 1e9))

    sub(tag)
    print(
        f"    skip_ratio        {paint(f'{skip:.8f}', C.GRN)}  {bar(skip)}  "
        f"(1.0 = all baseline-equiv elided from *executed* path)"
    )
    print(f"    executed_gops     {eg:.6f}    (real silicon work / s)")
    print(f"    virtual_gops      {paint(f'{vg:.3f}', C.MAG, C.BOLD)}    "
          f"(baseline-equivalent accounting / s - architecture definition)")
    print(f"    baseline-equiv op/s  {sci(bps)}  ( = virtual_gops x 1e9 )")
    print(f"    executed_updates       {ex:,}")
    print(f"    baseline_equiv_updates {base:,}")
    aa = d.get("avg_active_sublattices")
    ap = d.get("avg_pruned_sublattices")
    at = d.get("avg_total_sublattices")
    if aa is not None and ap is not None:
        print(
            f"    avg_active_sublattices  {aa:.3f}    "
            f"avg_pruned_sublattices {ap:.3f}    "
            f"avg_total_sublattices {at if at is not None else float('nan'):.3f}"
        )
    print(f"    elapsed_s         {float(d.get('elapsed_s', 0.0)):.6f}")


# =============================================================================
# PHASE A — Binary assimilation: abstract bitwise “interference law” on lattice
# =============================================================================
def phase_assimilated_wavefront(exe: Path) -> int:
    title("PHASE A - ABYSSAL WAVEPACKET (PACKED INT4 ASSIMILATION)")
    print(
        paint(
            "Abstract model: each packed byte is a two-neuron cell; we apply repeated\n"
            "braincore_int4 updates - a discrete threshold-and-saturate law - while\n"
            "XOR-rotating the drive field. This is *binary-level physics theatre* on\n"
            "real machine code, not a separate PDE solver.",
            C.DIM,
        )
    )

    neurons = 262_144  # 128 Ki packed cells - even, heavy enough to feel, still demo-fast
    pot, inp = native.assimilation_tensors(neurons)
    waves = 12
    inner_iters = 48

    print(f"\n  Packed bytes: {len(pot):,}  (neurons={neurons:,})\n")

    for w in range(1, waves + 1):
        # Bitwise law on the *input* field — cheap, deterministic, visibly mutates drive.
        for i in range(len(inp)):
            inp[i] = ((inp[i] ^ ((w * 17 + i * 3) & 0xFF)) & 0x77) & 0xFF

        t0 = time.perf_counter()
        out = runtime.braincore_int4_inplace(
            pot, inp, neurons=neurons, iterations=inner_iters, threshold=9
        )
        dt = time.perf_counter() - t0
        ck = runtime.fnv1a64_packed(memoryview(pot)[: runtime.braincore_packed_bytes(neurons)])

        skip_vis = min(1.0, w / float(waves))
        print(
            paint(f"  WAVE {w:02d}/{waves}", C.BOLD, C.BLU),
            bar(skip_vis),
            f"  checksum_fnv1a64={ck}  variant={out['variant']}  wall={dt*1000:.2f}ms",
        )

    # Bit-exact parity: one contiguous assimilate run must match ``nrl assimilate``
    # with the same (neurons, iterations, threshold) on the **same** initial tensors.
    total_iters = inner_iters * waves
    pot2, inp2 = native.assimilation_tensors(neurons)
    runtime.braincore_int4_inplace(
        pot2, inp2, neurons=neurons, iterations=total_iters, threshold=9
    )
    ck_py = runtime.fnv1a64_packed(
        memoryview(pot2)[: runtime.braincore_packed_bytes(neurons)]
    )
    cli = runtime.assimilate_cli(
        neurons=neurons,
        iterations=total_iters,
        threshold=9,
        nrl_bin=str(exe),
    )
    cli_ck = int(cli["checksum_fnv1a64"])
    match = ck_py == cli_ck
    print()
    print(
        paint(
            "  FNV1A64 PARITY (Python packed state vs ``nrl assimilate`` same N/I/T): "
            + ("PASS" if match else "MISMATCH - investigate build mismatch"),
            C.GRN if match else C.RED,
            C.BOLD,
        )
    )
    print(f"    inplace tensor checksum . . . . {ck_py}")
    print(f"    nrl assimilate checksum . . . . {cli_ck}")
    return ck_py


# =============================================================================
# PHASE B — Sovereign “full cognition” (no skips)
# =============================================================================
def phase_sovereign_grind(exe: Path) -> dict[str, Any]:
    title("PHASE B - SOVEREIGN GRIND (SYSTEM 2 / FULL DELIBERATION)")
    print(
        paint(
            "Every baseline-equivalent update is executed on the hot path.\n"
            "skip_ratio -> 0. This is the honest cost floor for this workload.",
            C.DIM,
        )
    )
    d = bench_raw(exe, 1_048_576, 256, 2, 8, "sovereign")
    print_bench_frame("SOVEREIGN SNAPSHOT", d)
    return d


# =============================================================================
# PHASE C — ZPM muscle memory (static transition collapse, exact lane)
# =============================================================================
def phase_zpm_twins(exe: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    title("PHASE C - ZPM MUSCLE MEMORY (SYSTEM 1 / STATIC LAW INSIDE ``nrl bench``)")
    print(
        paint(
            "ZPM rebuilds its transition machinery **inside this process** each bench.\n"
            "First vs second **identical** CLI invocation measures wall-time stability;\n"
            "the *physics* is: executed_updates << baseline_equiv_updates at high skip_ratio.",
            C.DIM,
        )
    )
    args = (1_048_576, 4096, 3, 8, "zpm")
    sub("ZPM FIRST STRIKE (muscle memory formation / map build + collapse)")
    z1 = bench_raw(exe, *args)
    print_bench_frame("ZPM RUN #1", z1)

    sub("ZPM IMMEDIATE REPEAT (same argv - cache + warm silicon, same structural skip)")
    z2 = bench_raw(exe, *args)
    print_bench_frame("ZPM RUN #2", z2)

    t1 = float(z1["elapsed_s"])
    t2 = float(z2["elapsed_s"])
    if t2 > 0 and t1 > 0:
        speedup = t1 / t2
        msg = f"Wall-time ratio run#1/run#2 = {speedup:.3f}x (!= structural skip_ratio; honest cache story)"
        print(paint("\n  " + msg, C.CYA))
    return z1, z2


# =============================================================================
# PHASE D — Omega collapse frames (each frame = one full ``nrl bench`` with reps=1)
# =============================================================================
def phase_omega_collapse_frames(exe: Path) -> None:
    title("PHASE D - OMEGA COLLAPSE FRAMES (FRACTAL ROUTING / VIRTUAL ENVELOPE)")
    print(
        paint(
            "Each *frame* below is an independent ``nrl bench`` with reps=1.\n"
            "virtual_gops is **baseline-equivalent updates per second** / 1e9 - an accounting\n"
            "measure from NRL's contract, not a claim of physical exa-scale FLOPS.",
            C.DIM,
        )
    )
    neurons = 1_048_576
    iters = 16_384
    threshold = 8
    frames = 8

    peak_virtual = 0.0
    peak_skip = 0.0

    for f in range(1, frames + 1):
        print(paint(f"\n  === OMEGA COLLAPSE FRAME {f:02d}/{frames} INITIATED ===", C.MAG, C.BOLD))
        d = bench_raw(exe, neurons, iters, 1, threshold, "omega")
        print_bench_frame("OMEGA FRAME STATS", d)
        vg = float(d.get("virtual_gops", 0.0))
        sk = float(d.get("skip_ratio", 0.0))
        peak_virtual = max(peak_virtual, vg)
        peak_skip = max(peak_skip, sk)
        # micro sleep so operator eyes can track stdout
        time.sleep(0.04)

    print()
    print(
        paint(
            f"  PEAK virtual_gops (max over frames): {peak_virtual:.3f}\n"
            f"  PEAK skip_ratio (max over frames):    {peak_skip:.8f}",
            C.GRN,
            C.BOLD,
        )
    )
    if peak_skip >= 0.99999:
        print(paint("  * skip_ratio >= 0.99999 - EXTREME PRUNING ACHIEVED (real silicon)", C.YEL, C.BOLD))
    else:
        print(
            paint(
                f"  (Peak skip {peak_skip:.8f} - raise neurons/iters on AVX2 host to chase 0.99999x)",
                C.DIM,
            )
        )


# =============================================================================
# PHASE E — Hybrid profile: executed throughput preservation + partial skip
# =============================================================================
def phase_hybrid_echo(exe: Path) -> None:
    title("PHASE E — Hybrid profile (executed GOPS + partial skip balance)")
    d = bench_raw(exe, 1_048_576, 4096, 2, 8, "omega-hybrid")
    print_bench_frame("HYBRID SNAPSHOT", d)


# =============================================================================
# INTERACTIVE — proxy to ``nrl chat`` (no persistent daemon; one question per line)
# =============================================================================
def phase_interactive_oracle(exe: Path) -> None:
    title("PHASE F - INTERACTIVE ORACLE (``nrl chat`` BRIDGE)")
    print(
        paint(
            "You are now talking to the **rule-based** CLI chat surface - lightweight,\n"
            "deterministic answers about speed / safety / modes (not a hidden LLM).\n"
            "Empty line exits.",
            C.DIM,
        )
    )
    print()
    while True:
        try:
            prompt = paint("nrl chat> ", C.BOLD, C.GRN)
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        try:
            parts = shlex.split(line, posix=os.name != "nt")
        except ValueError as e:
            print(paint(f"  parse error: {e}", C.RED))
            continue
        proc = subprocess.run(
            [str(exe), "chat"] + parts,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            print(paint(f"  (exit {proc.returncode})", C.DIM))


# =============================================================================
# main
# =============================================================================
def main() -> None:
    exe = nrl_executable()
    require_nrl(exe)

    print()
    print(paint("  +----------------------------------------------------------------+", C.CYA, C.BOLD))
    print(paint("  |     N R L   +   N R L P Y   -   U L T I M A T E   P O W E R      |", C.CYA, C.BOLD))
    print(paint("  |     RomanAILabs - neural lattice / machine-code fabric         |", C.CYA, C.BOLD))
    print(paint("  +----------------------------------------------------------------+", C.CYA, C.BOLD))
    print(
        paint(
            f"  native binary . . . . . {exe}\n"
            "  control plane . . . . . nrlpy.runtime + compat ``nrl`` (injected under ``nrlpy run``)",
            C.WHT,
        )
    )
    hr("-")

    phase_assimilated_wavefront(exe)
    phase_sovereign_grind(exe)
    phase_zpm_twins(exe)
    phase_omega_collapse_frames(exe)
    phase_hybrid_echo(exe)

    title("MUSCLE MEMORY SUMMARY (WHAT ACTUALLY HAPPENED)")
    print(
        paint(
            "  * Assimilation waves ran on **your** packed buffers in AVX2/scalar code.\n"
            "  * Sovereign showed the **full** executed path (skip ~ 0).\n"
            "  * ZPM showed **massive** skip_ratio - static-input muscle inside one bench.\n"
            "  * Omega frames maximized **virtual_gops** accounting - fractal prune story.\n"
            "  * Hybrid showed you can keep **executed_gops** high while still skipping.\n",
            C.DIM,
        )
    )

    phase_interactive_oracle(exe)

    print(paint("\n  DEMO COMPLETE - STAY SOVEREIGN. STAY REPRODUCIBLE.\n", C.BOLD, C.GRN))


if __name__ == "__main__":
    main()
