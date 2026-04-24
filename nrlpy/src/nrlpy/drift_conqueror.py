# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Bio-Digital Blueprint P5 — Drift Conqueror (weak-bucket targeting + growth cap).
"""Lattice coverage heuristics, weak-bucket detection, and conquest prompt plans.

All persistence is under ``<lmo_dir>/learn_probe/`` (same contract as
:class:`nrlpy.learn_daemon.LearnDaemon`). Does **not** modify P4 WAL paths
under ``cache/zpm/`` directly — growth is *observed* via ``index.bin`` size
and decode calls are gated by the learn daemon when the budget is exhausted.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

ANCHOR_BUCKETS = 256

__all__ = [
    "ANCHOR_BUCKETS",
    "anchor_bucket",
    "coverage_percent",
    "weak_buckets",
    "conquest_prompts",
    "load_coverage_state",
    "save_coverage_state_atomic",
    "growth_window_state_path",
    "load_growth_window",
    "save_growth_window_atomic",
    "growth_budget_exhausted",
    "maybe_roll_growth_window",
    "summarize_for_cli",
    "max_growth_fraction",
    "max_growth_pct",
]


def anchor_bucket(material: str) -> int:
    """Stable 0..255 bucket for synthetic anchor material (prompt head, etc.)."""
    h = hashlib.sha256(material.encode("utf-8", errors="replace")).digest()
    return h[0] % ANCHOR_BUCKETS


def coverage_percent(bucket_counts: dict[str, int]) -> float:
    """Share of 256 anchor buckets that have at least one recorded probe."""
    if not bucket_counts:
        return 0.0
    hit = sum(1 for _b, c in bucket_counts.items() if int(c) > 0)
    return round(100.0 * hit / float(ANCHOR_BUCKETS), 4)


def weak_buckets(
    bucket_counts: dict[str, int],
    *,
    floor: int = 1,
    ratio: float = 0.35,
    decode_fail_by_bucket: dict[str, int] | None = None,
) -> list[int]:
    """Buckets with count strictly below ``ratio * max(count,1)`` (or zero).

    Low-hit regions are candidates for targeted micro-queries. Buckets with
    decode-hook failures (R5 / consolidator stress proxy) are merged in.
    """
    if not bucket_counts:
        base = list(range(ANCHOR_BUCKETS))
    else:
        vals = [int(bucket_counts.get(str(i), 0) or 0) for i in range(ANCHOR_BUCKETS)]
        mx = max(max(vals), 1)
        thr = max(floor, int(mx * ratio))
        out = [i for i, v in enumerate(vals) if v < thr]
        base = out if out else [i for i, v in enumerate(vals) if v == min(vals)][:32]
    if decode_fail_by_bucket:
        extra = [
            i
            for i in range(ANCHOR_BUCKETS)
            if int(decode_fail_by_bucket.get(str(i), 0) or 0) >= 1
        ]
        return sorted(set(base) | set(extra))
    return sorted(set(base))


def conquest_prompts(weak: list[int], *, n_min: int = 10, n_max: int = 30) -> list[str]:
    """Focused synthetic prompts aimed at weak ZPM anchor neighborhoods."""
    if not weak:
        weak = list(range(0, ANCHOR_BUCKETS, 8))  # coarse sweep if no signal yet
    weak = weak[: ANCHOR_BUCKETS]
    rng = random.Random(0xC0DA1)  # deterministic variety per call site in tests
    rng.shuffle(weak)
    n = min(n_max, max(n_min, min(len(weak), n_max)))
    pick = weak[:n]
    prompts: list[str] = []
    for b in pick:
        prompts.append(
            "drift_conquest: stress ZPM anchor neighborhood "
            f"bucket={b} omega_verify=1 nullspace_tight=1 budget_ms=8"
        )
    while len(prompts) < n_min:
        prompts.append(
            f"drift_conquest: synthetic stability sweep bucket={rng.randint(0, 255)}"
        )
    return prompts[:n_max]


def load_coverage_state(probe_dir: Path) -> dict[str, Any]:
    p = probe_dir / "coverage_state.json"
    if not p.is_file():
        return {
            "schema_id": "nrl.drift_coverage.v1",
            "bucket_counts": {},
            "decode_fail_by_bucket": {},
            "last_conquest_unix": 0.0,
            "conquest_cycles": 0,
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_id": "nrl.drift_coverage.v1",
            "bucket_counts": {},
            "decode_fail_by_bucket": {},
            "last_conquest_unix": 0.0,
            "conquest_cycles": 0,
        }


def save_coverage_state_atomic(probe_dir: Path, state: dict[str, Any]) -> None:
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / "coverage_state.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def growth_window_state_path(probe_dir: Path) -> Path:
    return probe_dir / "conquest_growth_window.json"


def load_growth_window(probe_dir: Path) -> dict[str, Any]:
    p = growth_window_state_path(probe_dir)
    if not p.is_file():
        return {
            "schema_id": "nrl.drift_growth.v1",
            "window_start_unix": 0.0,
            "baseline_index_bytes": 0,
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_id": "nrl.drift_growth.v1",
            "window_start_unix": 0.0,
            "baseline_index_bytes": 0,
        }


def save_growth_window_atomic(probe_dir: Path, state: dict[str, Any]) -> None:
    probe_dir.mkdir(parents=True, exist_ok=True)
    p = growth_window_state_path(probe_dir)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def maybe_roll_growth_window(
    probe_dir: Path,
    *,
    now: float,
    zpm_index_bytes: int,
    window_sec: float = 86_400.0,
) -> dict[str, Any]:
    """Start or roll the 24h growth accounting window; returns current window dict."""
    gw = load_growth_window(probe_dir)
    start = float(gw.get("window_start_unix", 0.0) or 0.0)
    if start <= 0.0 or (now - start) >= window_sec:
        gw = {
            "schema_id": "nrl.drift_growth.v1",
            "window_start_unix": now,
            "baseline_index_bytes": int(max(0, zpm_index_bytes)),
        }
        save_growth_window_atomic(probe_dir, gw)
    return gw


def growth_budget_exhausted(
    zpm_index_bytes: int,
    baseline_bytes: int,
    max_growth_fraction: float,
) -> bool:
    """True when index grew more than ``max_growth_fraction`` over baseline."""
    base = max(int(baseline_bytes), 1)
    cur = max(int(zpm_index_bytes), 0)
    growth = (cur - base) / float(base)
    return growth > max(0.0, max_growth_fraction)


def summarize_for_cli(
    *,
    probe_dir: Path,
    zpm_index_path: Path,
) -> dict[str, Any]:
    """Aggregate for ``nrlpy lmo coverage`` (read-only)."""
    cov = load_coverage_state(probe_dir)
    bc = {str(k): int(v) for k, v in (cov.get("bucket_counts") or {}).items()}
    fails = {str(k): int(v) for k, v in (cov.get("decode_fail_by_bucket") or {}).items()}
    weak = weak_buckets(bc, decode_fail_by_bucket=fails if fails else None)
    zsz = zpm_index_path.stat().st_size if zpm_index_path.is_file() else 0
    now = time.time()
    maybe_roll_growth_window(probe_dir, now=now, zpm_index_bytes=zsz)
    gw = load_growth_window(probe_dir)
    base = int(gw.get("baseline_index_bytes", 0) or 0)
    base = max(base, 1)
    used_pct = 100.0 * (zsz - base) / float(base) if zsz >= base else 0.0
    max_pct = _env_max_growth_pct()
    headroom_pct = max(0.0, max_pct - used_pct)
    return {
        "coverage_percent": coverage_percent(bc),
        "weak_bucket_count": len(weak),
        "weak_buckets_preview": weak[:24],
        "zpm_index_bytes": zsz,
        "growth_baseline_bytes": base,
        "growth_used_pct": round(used_pct, 4),
        "growth_headroom_pct": round(headroom_pct, 4),
        "last_conquest_unix": float(cov.get("last_conquest_unix", 0.0) or 0.0),
        "conquest_cycles": int(cov.get("conquest_cycles", 0) or 0),
        "max_growth_pct": _env_max_growth_pct(),
    }


def _env_max_growth_pct() -> float:
    return _env_float("NRL_LEARN_MAX_GROWTH_PCT", 5.0)


def _env_max_growth_frac() -> float:
    return max(0.0, min(1.0, _env_max_growth_pct() / 100.0))


def max_growth_fraction() -> float:
    """``NRL_LEARN_MAX_GROWTH_PCT`` as a fraction in ``[0, 1]`` (default 0.05)."""
    return _env_max_growth_frac()


def max_growth_pct() -> float:
    """Raw percent cap from ``NRL_LEARN_MAX_GROWTH_PCT`` (default ``5``)."""
    return _env_max_growth_pct()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
