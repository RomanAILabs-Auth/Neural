"""LLM- and script-friendly facade: same neural/binary entry points without NumPy.

Use ``from nrlpy.compat import nrl`` or ``nrlpy run script.py`` so generated code
calls into machine-code assimilation instead of failing on missing scientific
stacks.
"""

from __future__ import annotations

from typing import Any

from . import native, runtime


class NRLRuntime:
    """Object-shaped API (``nrl.lattice()``, ``nrl.assimilate(...)``)."""

    __slots__ = ()

    def lattice(
        self, neurons: int = 8_000_000, iterations: int = 1000, threshold: int = 12
    ) -> dict[str, Any]:
        return runtime.braincore_int4(neurons=neurons, iterations=iterations, threshold=threshold)

    def assimilate(
        self,
        potentials: bytearray | memoryview,
        inputs: bytes | bytearray | memoryview,
        neurons: int,
        iterations: int,
        threshold: int,
    ) -> dict[str, Any]:
        return runtime.braincore_int4_inplace(
            potentials, inputs, neurons, iterations, threshold
        )

    def packed_bytes(self, neurons: int) -> int:
        return runtime.braincore_packed_bytes(neurons)

    def bench(
        self,
        neurons: int = 1_048_576,
        iterations: int = 4096,
        reps: int = 12,
        threshold: int = 8,
        profile: str = "sovereign",
        nrl_bin: str | None = None,
    ) -> dict[str, Any]:
        return runtime.bench_cli(
            neurons, iterations, reps, threshold, profile, nrl_bin=nrl_bin
        )

    def assimilate_cli(
        self,
        neurons: int = 4096,
        iterations: int = 256,
        threshold: int = 10,
        nrl_bin: str | None = None,
    ) -> dict[str, Any]:
        return runtime.assimilate_cli(neurons, iterations, threshold, nrl_bin=nrl_bin)

    def run_program(self, path: str, nrl_bin: str | None = None) -> str:
        return runtime.run_nrl_file(path, nrl_bin=nrl_bin)

    def engine(self) -> dict[str, Any]:
        return native.engine_info()


def llm_globals() -> dict[str, Any]:
    """Globals merged by ``nrlpy run`` so scripts can use ``nrl`` / ``NRL`` without setup."""
    r = NRLRuntime()
    return {
        "NRL": r,
        "nrl": r,
        "nrl_lattice": r.lattice,
        "nrl_assimilate": r.assimilate,
        "nrl_packed_bytes": r.packed_bytes,
        "nrl_bench": r.bench,
        "nrl_run_program": r.run_program,
        "nrl_assimilate_cli": r.assimilate_cli,
        "nrl_engine": r.engine,
    }
