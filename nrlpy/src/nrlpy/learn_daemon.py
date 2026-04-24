# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Bio-Digital Blueprint P2 — Learn Daemon MVP (subconscious / curiosity loop).
"""Background learn daemon: idle-gated synthetic exploration (no LMO tensor writes).

Environment
-----------

* ``NRL_LEARN_MODE`` — ``0``/``off`` (default): supervisor thread idles; **no**
  curiosity cycles or decode hooks run.
* ``NRL_LEARN_MODE`` — ``1``/``on``: cycles run only after **idle** window.
* ``NRL_LEARN_IDLE_SEC`` — seconds since last user activity (or last cycle
  anchor) before a cycle runs (default ``60``).
* ``NRL_LEARN_CPU_CAP`` — target max duty cycle for learn work in ``(0,1]``
  (default ``0.25``). Implemented as sleep padding after each cycle.
* ``NRL_LEARN_CONQUEST_IDLE_SEC`` — user-idle seconds before a cycle may use
  Drift Conquest prompts (default ``300``; P5).
* ``NRL_LEARN_MAX_GROWTH_PCT`` — max ZPM ``index.bin`` growth vs 24h baseline
  (default ``5``; P5). When exceeded, decode hooks are skipped (no unbounded
  disk growth).

Writes under ``<lmo_dir>/learn_probe/`` only (atomic JSON snapshots + optional
append-only journal). Never modifies GGUF bytes or ``retained.blob`` /
``packed.blob`` / ``tiles``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from . import drift_conqueror as _drift
from .lmo import LmoHandle

__all__ = ["LearnDaemon", "learn_mode_enabled"]


def learn_mode_enabled() -> bool:
    """True when subconscious learn cycles are allowed (supervisor gate)."""
    if os.environ.get("NRL_SAFE_MODE", "0").strip().lower() in ("1", "true", "yes", "on"):
        return False
    v = os.environ.get("NRL_LEARN_MODE", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class LearnDaemon:
    """Subconscious thread: idle → synthetic prompts → optional R5 hook (MVP)."""

    def __init__(
        self,
        *,
        idle_sec: float | None = None,
        cpu_cap: float | None = None,
        conquest_idle_sec: float | None = None,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._idle_sec = float(idle_sec if idle_sec is not None else _env_float("NRL_LEARN_IDLE_SEC", 60.0))
        self._conquest_idle_sec = float(
            conquest_idle_sec
            if conquest_idle_sec is not None
            else _env_float("NRL_LEARN_CONQUEST_IDLE_SEC", 300.0)
        )
        cap = float(cpu_cap if cpu_cap is not None else _env_float("NRL_LEARN_CPU_CAP", 0.25))
        self._cpu_cap = min(1.0, max(0.01, cap))
        self._mono: Callable[[], float] = monotonic_fn or time.monotonic
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

        self._lmo: LmoHandle | None = None
        self._decode_runner: Callable[[str], None] | None = None

        self._history: deque[str] = deque(maxlen=10)
        self._last_user_activity = 0.0
        self._last_cycle_anchor = 0.0
        self._last_sample_prompts: list[str] = []

        self._stats: dict[str, Any] = {
            "curiosity_cycles": 0,
            "synthetic_prompts_generated": 0,
            "decode_calls": 0,
            "decode_errors": 0,
            "coverage_percent": 0.0,
            "weak_bucket_count": 0,
            "weak_buckets_preview": [],
            "last_conquest_unix": 0.0,
            "conquest_cycles": 0,
            "growth_used_pct": 0.0,
            "growth_headroom_pct": 5.0,
            "growth_cap_exhausted": False,
        }

    def start(
        self,
        lmo_handle: LmoHandle,
        *,
        decode_runner: Callable[[str], None] | None = None,
    ) -> None:
        """Start background thread. ``decode_runner`` is optional R5 hook (prompt str in)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._lmo = lmo_handle
            self._decode_runner = decode_runner
            self._stop.clear()
            now = self._mono()
            self._last_user_activity = now
            self._last_cycle_anchor = 0.0
            self._thread = threading.Thread(target=self._supervisor_loop, name="nrl-learn-daemon", daemon=True)
            self._thread.start()

    def stop(self, *, join_timeout_s: float = 5.0) -> None:
        """Signal stop and wait for thread (tests / shutdown)."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout_s)
        self._thread = None
        self._paused.clear()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def notify_user_interaction(self) -> None:
        """Tier-1 chat should call this on each user send (resets idle clock)."""
        with self._lock:
            self._last_user_activity = self._mono()

    def feed_history_snippet(self, text: str) -> None:
        """Append a bounded history line for curiosity templates (max 10)."""
        s = (text or "").strip()
        if not s:
            return
        with self._lock:
            self._history.append(s[:512])

    def get_status(self) -> dict[str, Any]:
        """JSON-safe status dict (MVP placeholders for coverage / index size)."""
        with self._lock:
            lmo_dir = str(self._lmo.lmo_dir) if self._lmo is not None else ""
            zpm_guess = 0
            if self._lmo is not None:
                zpm_p = self._lmo.lmo_dir.parent.parent / "zpm" / self._lmo.model_sha256 / "index.bin"
                if zpm_p.is_file():
                    try:
                        zpm_guess = zpm_p.stat().st_size
                    except OSError:
                        zpm_guess = 0
            wprev = self._stats.get("weak_buckets_preview") or []
            if not isinstance(wprev, list):
                wprev = []
            return {
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "paused": bool(self._paused.is_set()),
                "learn_mode_enabled": bool(learn_mode_enabled()),
                "idle_sec": float(self._idle_sec),
                "conquest_idle_sec": float(self._conquest_idle_sec),
                "cpu_cap": float(self._cpu_cap),
                "lmo_dir": lmo_dir,
                "model_sha256": self._lmo.model_sha256 if self._lmo else "",
                "curiosity_cycles": int(self._stats["curiosity_cycles"]),
                "synthetic_prompts_generated": int(self._stats["synthetic_prompts_generated"]),
                "decode_calls": int(self._stats["decode_calls"]),
                "decode_errors": int(self._stats["decode_errors"]),
                "coverage_percent": float(self._stats["coverage_percent"]),
                "weak_bucket_count": int(self._stats.get("weak_bucket_count", 0)),
                "weak_buckets_preview": [int(x) for x in wprev[:32]],
                "last_conquest_unix": float(self._stats.get("last_conquest_unix", 0.0)),
                "conquest_cycles": int(self._stats.get("conquest_cycles", 0)),
                "growth_used_pct": float(self._stats.get("growth_used_pct", 0.0)),
                "growth_headroom_pct": float(self._stats.get("growth_headroom_pct", 0.0)),
                "growth_cap_exhausted": bool(self._stats.get("growth_cap_exhausted", False)),
                "max_growth_pct": float(_drift.max_growth_pct()),
                "zpm_index_bytes_estimate": int(zpm_guess),
                "history_snippets": len(self._history),
                "sample_prompts": list(self._last_sample_prompts),
            }

    def _idle_anchor(self) -> float:
        with self._lock:
            return max(self._last_user_activity, self._last_cycle_anchor)

    def _supervisor_loop(self) -> None:
        probe_dir: Path | None = None
        while not self._stop.is_set():
            if not learn_mode_enabled():
                self._sleep(1.0)
                continue
            if self._paused.is_set():
                self._sleep(0.2)
                continue
            if self._lmo is None:
                self._sleep(1.0)
                continue
            probe_dir = self._lmo.lmo_dir / "learn_probe"
            anchor = self._idle_anchor()
            now = self._mono()
            if now - anchor < self._idle_sec:
                wait = min(1.0, max(0.05, self._idle_sec - (now - anchor)))
                self._sleep(wait)
                continue

            t0 = self._mono()
            self._run_curiosity_cycle(probe_dir)
            dt = max(1e-6, self._mono() - t0)
            # Target duty cycle ~= cpu_cap: sleep (1/cap - 1) * dt
            pad = dt * (1.0 / self._cpu_cap - 1.0)
            if pad > 0:
                self._sleep(pad)

            with self._lock:
                self._last_cycle_anchor = self._mono()

    def _run_curiosity_cycle(self, probe_dir: Path) -> None:
        assert self._lmo is not None
        probe_dir.mkdir(parents=True, exist_ok=True)
        lmo = self._lmo
        zpm_p = lmo.lmo_dir.parent.parent / "zpm" / lmo.model_sha256 / "index.bin"
        zpm_bytes = zpm_p.stat().st_size if zpm_p.is_file() else 0
        now_wall = float(time.time())
        gw = _drift.maybe_roll_growth_window(probe_dir, now=now_wall, zpm_index_bytes=zpm_bytes)
        baseline = int(gw.get("baseline_index_bytes", 0) or 0)
        frac = _drift.max_growth_fraction()
        exhausted = _drift.growth_budget_exhausted(zpm_bytes, baseline, frac)
        base = max(baseline, 1)
        used_pct = 100.0 * (zpm_bytes - base) / float(base) if zpm_bytes >= base else 0.0
        headroom_pct = max(0.0, _drift.max_growth_pct() - used_pct)

        cov = _drift.load_coverage_state(probe_dir)
        bc: dict[str, int] = {str(k): int(v) for k, v in (cov.get("bucket_counts") or {}).items()}
        fails: dict[str, int] = {
            str(k): int(v) for k, v in (cov.get("decode_fail_by_bucket") or {}).items()
        }

        with self._lock:
            user_idle = self._mono() - self._last_user_activity
        weak = _drift.weak_buckets(bc, decode_fail_by_bucket=fails if fails else None)
        conquest_mode = user_idle >= self._conquest_idle_sec
        if conquest_mode:
            prompts = _drift.conquest_prompts(weak)
        else:
            prompts = self._synthetic_prompts()

        journal_path = probe_dir / "curiosity_journal.jsonl"
        for p in prompts:
            if self._stop.is_set():
                break
            anchor = _drift.anchor_bucket(p[:128])
            key = str(anchor)
            bc[key] = int(bc.get(key, 0)) + 1
            kind = "conquest" if conquest_mode else "curiosity"
            line = json.dumps(
                {
                    "ts_unix": int(time.time()),
                    "kind": kind,
                    "anchor_bucket": anchor,
                    "prompt": p[:2048],
                },
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            self._append_jsonl_line(journal_path, line)

        runner = None if exhausted else self._decode_runner
        decode_n = min(10, len(prompts)) if conquest_mode else min(3, len(prompts))
        if runner is not None:
            for p in prompts[:decode_n]:
                if self._stop.is_set():
                    break
                bkt = str(_drift.anchor_bucket(p[:128]))
                with self._lock:
                    self._stats["decode_calls"] = int(self._stats["decode_calls"]) + 1
                try:
                    runner(p)
                except Exception:
                    with self._lock:
                        self._stats["decode_errors"] = int(self._stats["decode_errors"]) + 1
                    fails[bkt] = int(fails.get(bkt, 0)) + 1

        cov["bucket_counts"] = bc
        cov["decode_fail_by_bucket"] = fails
        if conquest_mode:
            cov["last_conquest_unix"] = float(time.time())
            cov["conquest_cycles"] = int(cov.get("conquest_cycles", 0) or 0) + 1
        _drift.save_coverage_state_atomic(probe_dir, cov)

        cov_pct = _drift.coverage_percent(bc)
        weak_after = _drift.weak_buckets(bc, decode_fail_by_bucket=fails if fails else None)
        sample = prompts[:8]
        with self._lock:
            self._stats["curiosity_cycles"] = int(self._stats["curiosity_cycles"]) + 1
            self._stats["synthetic_prompts_generated"] = int(
                self._stats["synthetic_prompts_generated"]
            ) + len(prompts)
            self._stats["coverage_percent"] = float(cov_pct)
            self._stats["weak_bucket_count"] = int(len(weak_after))
            self._stats["weak_buckets_preview"] = [int(x) for x in weak_after[:24]]
            self._stats["last_conquest_unix"] = float(cov.get("last_conquest_unix", 0.0) or 0.0)
            self._stats["conquest_cycles"] = int(cov.get("conquest_cycles", 0) or 0)
            self._stats["growth_used_pct"] = float(round(used_pct, 4))
            self._stats["growth_headroom_pct"] = float(round(headroom_pct, 4))
            self._stats["growth_cap_exhausted"] = bool(exhausted)
            self._last_sample_prompts = list(sample)
            zpm_guess = zpm_bytes
            snap = {
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "paused": bool(self._paused.is_set()),
                "learn_mode_enabled": bool(learn_mode_enabled()),
                "idle_sec": float(self._idle_sec),
                "conquest_idle_sec": float(self._conquest_idle_sec),
                "cpu_cap": float(self._cpu_cap),
                "lmo_dir": str(self._lmo.lmo_dir) if self._lmo else "",
                "model_sha256": self._lmo.model_sha256 if self._lmo else "",
                "curiosity_cycles": int(self._stats["curiosity_cycles"]),
                "synthetic_prompts_generated": int(self._stats["synthetic_prompts_generated"]),
                "decode_calls": int(self._stats["decode_calls"]),
                "decode_errors": int(self._stats["decode_errors"]),
                "coverage_percent": float(self._stats["coverage_percent"]),
                "weak_bucket_count": int(self._stats["weak_bucket_count"]),
                "weak_buckets_preview": list(self._stats["weak_buckets_preview"]),
                "last_conquest_unix": float(self._stats["last_conquest_unix"]),
                "conquest_cycles": int(self._stats["conquest_cycles"]),
                "growth_used_pct": float(self._stats["growth_used_pct"]),
                "growth_headroom_pct": float(self._stats["growth_headroom_pct"]),
                "growth_cap_exhausted": bool(self._stats["growth_cap_exhausted"]),
                "zpm_index_bytes_estimate": int(zpm_guess),
                "history_snippets": len(self._history),
                "sample_prompts": list(sample),
            }
        _atomic_write_json(probe_dir / "daemon_snapshot.json", snap)

    def _append_jsonl_line(self, path: Path, line: bytes) -> None:
        """Append one JSONL record via tmp+replace (single-writer MVP)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".append.tmp")
        with self._lock:
            prev = path.read_bytes() if path.is_file() else b""
            tmp.write_bytes(prev + line)
            os.replace(tmp, path)

    def _synthetic_prompts(self) -> list[str]:
        with self._lock:
            hist = list(self._history)
        if not hist:
            return [
                "what if the lattice explored an unused router edge under budget?",
                "what if token 0 and token 1 were swapped in a shadow probe?",
                "what if coherence_lane toggled for one micro-batch only?",
            ]
        out: list[str] = []
        for i, s in enumerate(hist[-8:]):
            out.append(f"what_if: rephrase {i!r} as a stability probe: {s[:120]!r}")
        while len(out) < 5:
            out.append(f"synthetic_explore_{len(out)}")
        return out[:20]
