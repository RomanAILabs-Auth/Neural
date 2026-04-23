"""High-level Python API over machine-code NRL kernels.

Use this module when you want explicit, typed calls. For LLM-style scripts that
expect globals like ``nrl`` / ``NRL`` without imports, see ``nrlpy.compat`` and
``nrlpy run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from . import runtime
from .runtime import BraincoreInt4InplaceResult


def packed_lattice_bytes(neurons: int) -> int:
    """Packed tensor size for ``neurons`` INT4 cells (even count); 0 if invalid."""
    return runtime.braincore_packed_bytes(neurons)


def assimilate_packed(
    potentials: bytearray,
    inputs: bytes | bytearray,
    neurons: int,
    iterations: int,
    threshold: int,
) -> BraincoreInt4InplaceResult:
    """In-place neural lattice update on caller-owned packed bytes (binary assimilation)."""
    return runtime.braincore_int4_inplace(
        potentials, inputs, neurons, iterations, threshold
    )


def assimilation_tensors(neurons: int) -> tuple[bytearray, bytearray]:
    """Allocate ``potentials`` (zero) and ``inputs`` (deterministic drive) like ``nrl assimilate``."""
    if neurons <= 0 or (neurons & 1) != 0:
        raise ValueError("neurons must be a positive even integer")
    nb = runtime.braincore_packed_bytes(neurons)
    if nb == 0:
        raise ValueError("invalid neuron count for packed layout")
    pot = bytearray(nb)
    inp = bytearray(nb)
    for i in range(nb):
        inp[i] = ((i * 17) & 0x77) & 0xFF
    return pot, inp


@dataclass(frozen=True)
class LatticeSpec:
    """INT4 packed lattice dynamics (braincore_int4)."""

    neurons: int = 8_000_000
    iterations: int = 1000
    threshold: int = 12


def run_lattice(spec: LatticeSpec | None = None, **overrides: Any) -> runtime.BraincoreInt4Result:
    """Execute native INT4 lattice updates; returns timing and throughput metrics."""
    s = spec or LatticeSpec()
    if overrides:
        s = LatticeSpec(
            neurons=int(overrides.get("neurons", s.neurons)),
            iterations=int(overrides.get("iterations", s.iterations)),
            threshold=int(overrides.get("threshold", s.threshold)),
        )
    return runtime.braincore_int4(
        neurons=s.neurons,
        iterations=s.iterations,
        threshold=s.threshold,
    )


def run_bench(
    neurons: int = 1_048_576,
    iterations: int = 4096,
    reps: int = 12,
    threshold: int = 8,
    profile: str = "sovereign",
    nrl_bin: str | None = None,
    *,
    respect_control_hints: bool = False,
) -> runtime.BenchCliResult:
    """Drive the native ``nrl bench`` harness (profiles: sovereign, omega, …)."""
    return runtime.bench_cli(
        neurons=neurons,
        iterations=iterations,
        reps=reps,
        threshold=threshold,
        profile=profile,
        nrl_bin=nrl_bin,
        respect_control_hints=respect_control_hints,
    )


def run_nrl(path: str, nrl_bin: str | None = None) -> str:
    """Execute a ``.nrl`` control file via the native ``nrl`` CLI."""
    return runtime.run_nrl_file(path, nrl_bin=nrl_bin)


def engine_info() -> dict[str, Any]:
    """Version + CPU features; safe to print from generated scripts."""
    return cast(
        dict[str, Any],
        {
            "version": runtime.version(),
            "features": runtime.features(),
            "braincore_int4_variant": runtime.active_variant("braincore_int4"),
        },
    )
