# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Typed runtime wrappers for the nrlpy C extension."""

from __future__ import annotations

import json
import os
import subprocess
import time
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


class ControlPreferences(TypedDict, total=False):
    """Subset of ``build/control/preferences.json`` written by ``nrl control``."""

    schema_id: str
    updated_unix: int
    recommended_profile: str
    power_until_unix: int
    throttle_hint: str


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


# Phase 11 — prefer the native C FNV when ``_core`` exports it. The
# pure-Python loop below is O(n) Python-level overhead per byte and is a
# pathological hot path for LMO absorption on multi-GB GGUFs (it was the
# cause of the hour-plus absorption stall on Phi-3 mini). The native
# export in ``engine/src/capi.c`` matches ``checksum_u64`` bit-for-bit,
# so the on-disk attestation format is unchanged.
_native_fnv1a64 = getattr(_core, "fnv1a64_bytes", None)


def fnv1a64_packed(data: bytes | bytearray | memoryview) -> int:
    """FNV-1a 64-bit over bytes; matches ``checksum_u64`` in ``engine/src/main.c``."""
    if _native_fnv1a64 is not None:
        # ``_core.fnv1a64_bytes`` accepts anything implementing the buffer
        # protocol; ``bytes(...)`` is a cheap guard for exotic types
        # (e.g. memoryview of non-contiguous slices) that the C binding
        # would otherwise reject.
        try:
            return int(_native_fnv1a64(data))
        except TypeError:
            return int(_native_fnv1a64(bytes(data)))
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


def control_preferences_path() -> Path:
    """Path to CLI ``preferences.json`` (same layout as native ``nrl control``)."""
    nrl_root = os.environ.get("NRL_ROOT")
    if nrl_root:
        return Path(nrl_root) / "build" / "control" / "preferences.json"
    return Path.cwd() / "build" / "control" / "preferences.json"


def control_audit_log_path() -> Path:
    """Append-only audit log from ``nrl control`` / ``nrl chat`` (JSON lines)."""
    return control_preferences_path().parent / "control_audit.jsonl"


def _control_hint_bias_applies(prefs: ControlPreferences) -> bool:
    """True when ``resolve_bench_profile_with_control_hints`` would replace the requested profile."""
    now = int(time.time())
    power_until = int(prefs.get("power_until_unix") or 0)
    if power_until > now:
        return True
    th = prefs.get("throttle_hint") or "none"
    return th in {"conservative", "gated"}


def control_hints_active(prefs: ControlPreferences | None = None) -> bool:
    """Whether sandboxed control preferences currently bias nrlpy bench resolution."""
    p = prefs if prefs is not None else load_control_preferences()
    if p is None:
        return False
    return _control_hint_bias_applies(p)


def load_control_preferences() -> ControlPreferences | None:
    """Read control preferences if present and schema matches; otherwise ``None``."""
    path = control_preferences_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_id") != "nrl.control_preferences.v1":
        return None
    out = cast(ControlPreferences, {})
    if isinstance(raw.get("recommended_profile"), str):
        out["recommended_profile"] = raw["recommended_profile"]
    for key in ("updated_unix", "power_until_unix"):
        v = raw.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[key] = v
        elif isinstance(v, float):
            out[key] = int(v)
        elif isinstance(v, str) and v.isdigit():
            out[key] = int(v)
    if isinstance(raw.get("throttle_hint"), str):
        out["throttle_hint"] = raw["throttle_hint"]
    out["schema_id"] = "nrl.control_preferences.v1"
    return out


def resolve_bench_profile_with_control_hints(requested_profile: str, prefs: ControlPreferences | None) -> str:
    """Pick bench profile from ``nrl control`` hints without touching hot-path kernels.

    - Active **power** window (``power_until_unix`` in the future): use ``recommended_profile``.
    - **throttle_hint** ``conservative`` or ``gated``: use ``recommended_profile`` when set.
    - Otherwise return ``requested_profile``.
    """
    if prefs is None:
        return requested_profile
    rec = prefs.get("recommended_profile")
    if not isinstance(rec, str) or not rec:
        return requested_profile
    if not _control_hint_bias_applies(prefs):
        return requested_profile
    return rec


def bench_cli(
    neurons: int,
    iterations: int,
    reps: int,
    threshold: int,
    profile: str,
    nrl_bin: str | None = None,
    *,
    respect_control_hints: bool = False,
) -> BenchCliResult:
    nrl_path = Path(nrl_bin) if nrl_bin else _default_nrl_bin()
    eff_profile = (
        resolve_bench_profile_with_control_hints(profile, load_control_preferences())
        if respect_control_hints
        else profile
    )
    proc = subprocess.run(
        [
            str(nrl_path),
            "bench",
            str(neurons),
            str(iterations),
            str(reps),
            str(threshold),
            eff_profile,
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
