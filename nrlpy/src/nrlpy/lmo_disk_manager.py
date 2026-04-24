# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Bio-Digital Blueprint P6 — HD quota + ZPM LRU prune (bounded footprint).
"""LMO / ZPM / muscle-memory footprint accounting and safe pruning."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import zpm
from . import zpm_persist as zp

__all__ = [
    "LmoDiskManager",
    "bump_access_stat",
    "state_fingerprint",
    "footprint_bytes",
    "max_quota_bytes",
    "access_stats_path",
]

_MIN_ENTRIES_AFTER_PRUNE = 8


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def max_quota_bytes() -> int:
    """``NRL_LMO_MAX_GB`` (default 100) — total GiB cap per model footprint."""
    gib = _env_float("NRL_LMO_MAX_GB", 100.0)
    # Allow sub-megabyte caps for CI / harness (fractional GiB).
    return int(max(1e-12, gib) * (1024**3))


def target_footprint_bytes(*, aggressive: bool) -> int:
    """Max allowed bytes after prune (10% or 20% headroom below quota)."""
    q = max_quota_bytes()
    frac = 0.8 if aggressive else 0.9
    return int(q * frac)


def state_fingerprint(st: zpm.State) -> str:
    """Stable key for :class:`zpm.ZpmEntry` access stats (hex words)."""
    return ":".join(f"{int(w) & 0xFFFFFFFFFFFFFFFF:016x}" for w in st)


def access_stats_path(model_sha256: str) -> Path:
    return zp.zpm_dir(model_sha256) / "access_stats.json"


def _read_access_stats(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = obj.get("hits") if isinstance(obj, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items() if int(v) > 0}


def _write_access_stats(path: Path, hits: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "nrl.zpm_access_stats.v1",
        "hits": dict(sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))),
        "updated_unix": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def bump_access_stat(model_sha256: str, state: zpm.State) -> None:
    """Increment LRU score for a served ZPM state (hot path; must not raise)."""
    sha = model_sha256 or "unknown"
    lk = zp._lock_for(sha)
    try:
        with lk:
            p = access_stats_path(sha)
            hits = _read_access_stats(p)
            fp = state_fingerprint(state)
            hits[fp] = int(hits.get(fp, 0)) + 1
            _write_access_stats(p, hits)
    except OSError:
        pass


def dir_tree_bytes(root: Path | None) -> int:
    if root is None or not root.exists():
        return 0
    total = 0
    try:
        if root.is_file():
            return root.stat().st_size
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total


def footprint_bytes(
    *,
    model_sha256: str,
    lmo_dir: Path | None,
    mm_root: Path,
) -> int:
    """On-disk bytes for LMO dir + ZPM dir + muscle-memory shard."""
    sha = model_sha256 or "unknown"
    zd = zp.zpm_dir(sha)
    mm = mm_root / sha
    lm = 0
    if lmo_dir is not None and lmo_dir.is_dir():
        lm = dir_tree_bytes(lmo_dir)
    return lm + dir_tree_bytes(zd) + dir_tree_bytes(mm)


def _trial_index_bytes(idx: zpm.ZpmIndex, path: Path) -> int:
    tmp = path.with_suffix(path.suffix + ".sizeprobe.tmp")
    try:
        idx.save(tmp)
        return tmp.stat().st_size if tmp.is_file() else 0
    except OSError:
        return 0
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _prune_old_snapshots(zd: Path, *, keep_last: int = 2, dry_run: bool) -> tuple[int, int]:
    """Delete oldest ``zpm_index_*.bin`` under ``snapshots/``; returns (files, bytes)."""
    snap = zd / "snapshots"
    if not snap.is_dir():
        return 0, 0
    files = sorted(snap.glob("zpm_index_*.bin"), key=lambda p: p.stat().st_mtime)
    victims = files[:-keep_last] if len(files) > keep_last else []
    freed = 0
    n = 0
    for p in victims:
        try:
            sz = p.stat().st_size
            if not dry_run:
                p.unlink(missing_ok=True)
            freed += sz
            n += 1
        except OSError:
            pass
    return n, freed


def _compact_wal(sha: str) -> None:
    wpath = zp.wal_path(sha)
    try:
        wpath.write_bytes(b"")
    except OSError:
        return
    st = zp._read_learn_state(sha)
    st["wal_applied_bytes"] = 0
    st.setdefault("schema_id", "nrl.zpm_persistence.v1")
    zp._atomic_write_json(zp.learn_state_path(sha), st)


class LmoDiskManager:
    """Quota-aware pruning for one absorbed model (ZPM LRU + WAL compact)."""

    def __init__(
        self,
        *,
        model_sha256: str,
        lmo_dir: Path | None,
        index_path: Path,
        mm_root: Path,
    ) -> None:
        self.sha = model_sha256 or "unknown"
        self.lmo_dir = lmo_dir
        self.index_path = Path(index_path)
        self.mm_root = Path(mm_root)

    def footprint(self) -> int:
        return footprint_bytes(
            model_sha256=self.sha,
            lmo_dir=self.lmo_dir,
            mm_root=self.mm_root,
        )

    def prune_if_needed(
        self,
        *,
        aggressive: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """If footprint exceeds quota, evict cold ZPM rows until under target.

        Uses ``access_stats.json`` hit counts (updated on ZPM cache hits) with
        ``wall_s_at_write`` as tie-breaker. Atomic: tmp index + ``os.replace``,
        then WAL truncate (same contract as post-snapshot compaction).
        """
        quota = max_quota_bytes()
        target = target_footprint_bytes(aggressive=aggressive)
        out: dict[str, Any] = {
            "model_sha256": self.sha,
            "dry_run": dry_run,
            "aggressive": aggressive,
            "force": force,
            "quota_bytes": quota,
            "target_bytes": target,
            "pruned": False,
            "evicted_entries": 0,
            "freed_bytes_estimate": 0,
            "snapshots_removed": 0,
            "snapshot_freed_bytes": 0,
        }
        # Recover outside our lock: recover_zpm_for_model uses the same per-model lock.
        zp.recover_zpm_for_model(self.sha, self.index_path)
        lk = zp._lock_for(self.sha)
        with lk:
            cur_fp = self.footprint()
            out["footprint_bytes_before"] = cur_fp
            if cur_fp <= quota and not force:
                out["footprint_bytes_after"] = cur_fp
                return out

            idx = zpm.ZpmIndex.load(self.index_path) if self.index_path.is_file() else zpm.ZpmIndex()
            stats_path = access_stats_path(self.sha)
            stats = _read_access_stats(stats_path)

            zd = zp.zpm_dir(self.sha)
            sn_n, sn_b = _prune_old_snapshots(zd, keep_last=2, dry_run=dry_run)
            out["snapshots_removed"] = sn_n
            out["snapshot_freed_bytes"] = sn_b

            cur_fp = self.footprint()
            out["footprint_bytes_mid"] = cur_fp
            if cur_fp <= target:
                out["footprint_bytes_after"] = cur_fp
                out["pruned"] = sn_n > 0
                if not dry_run and sn_n:
                    self._record_prune_meta(0, sn_b)
                return out

            entries = list(idx)
            ranked = sorted(
                range(len(entries)),
                key=lambda i: (
                    int(stats.get(state_fingerprint(entries[i].state), 0)),
                    float(entries[i].wall_s_at_write),
                ),
            )
            bytes_needed = max(0, cur_fp - target)
            if force and bytes_needed == 0 and len(entries) > _MIN_ENTRIES_AFTER_PRUNE + 5:
                bytes_needed = max(1_048_576, cur_fp // 50)
            old_index_sz = self.index_path.stat().st_size if self.index_path.is_file() else 0
            victims: set[int] = set()
            for i in ranked:
                if len(entries) - len(victims) <= _MIN_ENTRIES_AFTER_PRUNE:
                    break
                victims.add(i)
                trial = zpm.ZpmIndex([e for j, e in enumerate(entries) if j not in victims])
                new_index_sz = _trial_index_bytes(trial, self.index_path)
                index_delta = max(0, old_index_sz - new_index_sz)
                thr = int(bytes_needed * 0.85) if bytes_needed > 0 else 0
                if bytes_needed > 0 and index_delta + sn_b >= thr:
                    break
                if force and bytes_needed > 0 and len(victims) >= max(1, len(entries) // 20):
                    break

            if not victims:
                out["footprint_bytes_after"] = cur_fp
                return out

            trial_final = zpm.ZpmIndex([e for j, e in enumerate(entries) if j not in victims])
            new_index_sz = _trial_index_bytes(trial_final, self.index_path)
            index_delta = max(0, old_index_sz - new_index_sz)
            est_freed = index_delta + sn_b

            if dry_run:
                out["evicted_entries"] = len(victims)
                out["freed_bytes_estimate"] = int(est_freed)
                out["footprint_bytes_after"] = max(0, cur_fp - est_freed)
                out["pruned"] = True
                return out

            idx.remove_entry_indices(victims)
            idx.save(self.index_path)
            _compact_wal(self.sha)
            zpm.invalidate_prefetch(self.sha)
            after_fp = self.footprint()
            out["evicted_entries"] = len(victims)
            out["freed_bytes_estimate"] = max(0, cur_fp - after_fp)
            out["footprint_bytes_after"] = after_fp
            out["pruned"] = True
            self._record_prune_meta(len(victims), int(out["freed_bytes_estimate"]))
        return out

    def _record_prune_meta(self, evicted: int, freed: int) -> None:
        st = zp._read_learn_state(self.sha)
        st.setdefault("schema_id", "nrl.zpm_persistence.v1")
        now = time.time()
        st["last_prune_unix"] = now
        st["last_prune_evicted_entries"] = int(evicted)
        st["last_prune_freed_bytes"] = int(freed)
        fp = self.footprint()
        st["last_footprint_bytes"] = fp
        st["last_footprint_unix"] = now
        zp._atomic_write_json(zp.learn_state_path(self.sha), st)
