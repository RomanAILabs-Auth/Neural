"""Typed runtime wrappers for the nrlpy C extension."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict, cast

from . import _core


class BraincoreInt4Result(TypedDict):
    kernel: str
    variant: str
    neurons: int
    iterations: int
    threshold: int
    seconds: float
    giga_neurons_per_sec: float


class BraincoreInt4InplaceResult(TypedDict):
    kernel: str
    variant: str
    neurons: int
    iterations: int
    threshold: int
    seconds: float
    checksum_fnv1a64: int


class BenchCliResult(TypedDict):
    profile: str
    mode: str
    variant: str
    neurons: int
    iterations: int
    reps: int
    threshold: int
    elapsed_s: float
    executed_updates: int
    baseline_equiv_updates: int
    skip_ratio: float
    executed_gops: float
    virtual_gops: float


def version() -> str:
    return _core.version()


def features() -> dict[str, bool]:
    return _core.features()


def active_variant(kernel: str) -> str:
    return _core.active_variant(kernel)


def fnv1a64_packed(data: bytes | bytearray | memoryview) -> int:
    """FNV-1a 64-bit over bytes; matches ``checksum_u64`` in ``engine/src/main.c``."""
    x = 1469598103934665603
    prime = 1099511628211
    for b in data:
        x ^= int(b) & 0xFF
        x = (x * prime) & 0xFFFFFFFFFFFFFFFF
    return x


def braincore_packed_bytes(neurons: int) -> int:
    """Byte length of packed INT4 lattice for ``neurons`` (even, positive); 0 if invalid."""
    return int(_core.braincore_packed_bytes(neurons))


def braincore_int4_inplace(
    potentials: bytearray | memoryview,
    inputs: bytes | bytearray | memoryview,
    neurons: int,
    iterations: int,
    threshold: int,
) -> BraincoreInt4InplaceResult:
    """Mutate ``potentials`` in-place via machine-code kernel (binary assimilation path)."""
    raw = cast(
        dict[str, object],
        _core.braincore_int4_inplace(
            potentials,
            inputs,
            neurons,
            iterations,
            threshold,
        ),
    )
    need = braincore_packed_bytes(neurons)
    view = memoryview(potentials)
    prefix = view[:need] if len(view) >= need else view
    merged = dict(raw)
    merged["checksum_fnv1a64"] = fnv1a64_packed(prefix)
    return cast(BraincoreInt4InplaceResult, merged)


def assimilate_cli(
    neurons: int = 4096,
    iterations: int = 256,
    threshold: int = 10,
    nrl_bin: str | None = None,
) -> dict[str, int | float | str]:
    """Run ``nrl assimilate`` and parse key metrics (parity with native CLI)."""
    nrl_path = Path(nrl_bin) if nrl_bin else _default_nrl_bin()
    proc = subprocess.run(
        [
            str(nrl_path),
            "assimilate",
            str(neurons),
            str(iterations),
            str(threshold),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nrl assimilate failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    parsed: dict[str, object] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("NRL assimilate"):
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        key = key.replace(" ", "_")
        if key in {"variant", "lane"}:
            parsed[key] = value
            continue
        if key == "checksum_fnv1a64":
            parsed[key] = int(value.split()[0], 10)
            continue
        if key in {"packed_bytes", "neurons", "iterations", "threshold"}:
            parsed[key] = int(float(value))
            continue
        parsed[key] = float(value)
    required = {
        "lane",
        "variant",
        "packed_bytes",
        "neurons",
        "iterations",
        "threshold",
        "elapsed_s",
        "checksum_fnv1a64",
    }
    missing = sorted(required - parsed.keys())
    if missing:
        raise RuntimeError(
            f"nrl assimilate output missing keys: {json.dumps(missing)}; raw={proc.stdout!r}"
        )
    return cast(dict[str, int | float | str], parsed)


def braincore_int4(
    neurons: int = 8_000_000, iterations: int = 1000, threshold: int = 12
) -> BraincoreInt4Result:
    return cast(BraincoreInt4Result, _core.braincore_int4(neurons, iterations, threshold))


def nrl_binary_candidates() -> list[Path]:
    """Ordered locations for the native ``nrl`` executable (installer vs dev tree)."""
    exe = "nrl.exe" if os.name == "nt" else "nrl"
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    if env := os.environ.get("NRL_BIN"):
        add(Path(env))
    nrl_root = os.environ.get("NRL_ROOT")
    if nrl_root:
        root = Path(nrl_root)
        add(root / "bin" / exe)
        add(root / "build" / "bin" / exe)
    here = Path(__file__).resolve().parent
    for anc in [here, *here.parents]:
        add(anc / "bin" / exe)
        add(anc / "build" / "bin" / exe)
        if anc.parent == anc:
            break
    return out


def nrl_binary_path() -> Path:
    """First existing ``nrl`` on the search path, else the first candidate (for error messages)."""
    c = nrl_binary_candidates()
    for p in c:
        if p.is_file():
            return p
    if c:
        return c[0]
    return Path("nrl.exe" if os.name == "nt" else "nrl")


def _default_nrl_bin() -> Path:
    return nrl_binary_path()


def bench_cli(
    neurons: int,
    iterations: int,
    reps: int,
    threshold: int,
    profile: str,
    nrl_bin: str | None = None,
) -> BenchCliResult:
    nrl_path = Path(nrl_bin) if nrl_bin else _default_nrl_bin()
    proc = subprocess.run(
        [
            str(nrl_path),
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
        raise RuntimeError(f"nrl bench failed: {proc.stderr.strip() or proc.stdout.strip()}")

    parsed: dict[str, object] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("NRL bench"):
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        key = key.replace(" ", "_")
        if key in {"profile", "mode", "variant"}:
            parsed[key] = value
            continue
        if key in {"neurons", "iterations", "reps", "threshold", "executed_updates", "baseline_equiv_updates"}:
            parsed[key] = int(float(value))
            continue
        parsed[key] = float(value)

    required = {
        "profile",
        "mode",
        "variant",
        "neurons",
        "iterations",
        "reps",
        "threshold",
        "elapsed_s",
        "executed_updates",
        "baseline_equiv_updates",
        "skip_ratio",
        "executed_gops",
        "virtual_gops",
    }
    missing = sorted(required - parsed.keys())
    if missing:
        raise RuntimeError(
            f"nrl bench output missing keys: {json.dumps(missing)}; raw={proc.stdout!r}"
        )
    return cast(BenchCliResult, parsed)


def run_nrl_file(path: str, nrl_bin: str | None = None) -> str:
    nrl_path = Path(nrl_bin) if nrl_bin else _default_nrl_bin()
    proc = subprocess.run(
        [str(nrl_path), "file", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nrl file failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout
