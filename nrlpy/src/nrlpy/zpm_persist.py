# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Bio-Digital Blueprint P4 — ZPM WAL + snapshots + recovery (<2s target).
"""Crash-safe ZPM persistence: append-only WAL, atomic index snapshots, recovery.

Layout under ``$NRL_ROOT/cache/zpm/<model_sha256>/``:

* ``index.bin`` — full :class:`nrlpy.zpm.ZpmIndex` (atomic tmp+replace on write).
* ``zpm_wal.log`` — append-only JSONL; each line is one :class:`ZpmEntry`
  payload; **fsync** after every append.
* ``learn_state.json`` — JSON metadata (snapshot times, WAL offset, counts).
* ``snapshots/zpm_index_<unix>.bin`` — periodic full copies of ``index.bin``.

Recovery runs before loading an index (see :func:`recover_zpm_for_model`):
load ``index.bin``, replay any WAL records past the committed offset, then
rewrite ``index.bin`` if the merged index grew.

Muscle-memory files remain per-key atomic writes in ``cache/mm/``; callers
should fsync the temp file before ``os.replace`` (handled in :mod:`nrlpy.gguf`).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from . import zpm

__all__ = [
    "recover_zpm_for_model",
    "persist_zpm_entry",
    "maybe_auto_prune_lmo",
    "zpm_dir",
    "wal_path",
    "learn_state_path",
    "count_wal_pending_lines",
    "gather_lmo_persistence_info",
]

_LOCKS: dict[str, threading.Lock] = {}
_INIT_LOCK = threading.Lock()


def _safe_mode_enabled() -> bool:
    return os.environ.get("NRL_SAFE_MODE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _lock_for(sha: str) -> threading.Lock:
    with _INIT_LOCK:
        if sha not in _LOCKS:
            _LOCKS[sha] = threading.Lock()
    return _LOCKS[sha]


def zpm_dir(model_sha256: str) -> Path:
    env = os.environ.get("NRL_ROOT")
    base = Path(env) if env else Path.cwd()
    return base / "cache" / "zpm" / (model_sha256 or "unknown")


def wal_path(model_sha256: str) -> Path:
    return zpm_dir(model_sha256) / "zpm_wal.log"


def learn_state_path(model_sha256: str) -> Path:
    return zpm_dir(model_sha256) / "learn_state.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_learn_state(model_sha256: str) -> dict[str, Any]:
    p = learn_state_path(model_sha256)
    if not p.is_file():
        return {
            "schema_id": "nrl.zpm_persistence.v1",
            "wal_applied_bytes": 0,
            "since_snapshot_appends": 0,
            "last_snapshot_unix": 0.0,
            "total_persist_calls": 0,
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_id": "nrl.zpm_persistence.v1",
            "wal_applied_bytes": 0,
            "since_snapshot_appends": 0,
            "last_snapshot_unix": 0.0,
            "total_persist_calls": 0,
        }


def _entry_to_json(entry: zpm.ZpmEntry) -> dict[str, Any]:
    return {
        "state": [int(entry.state[i]) for i in range(4)],
        "reply": entry.reply_text,
        "tokens": int(entry.tokens),
        "wall": float(entry.wall_s_at_write),
        "meta": dict(entry.metadata),
    }


def _json_to_entry(obj: dict[str, Any]) -> zpm.ZpmEntry | None:
    try:
        s = obj["state"]
        if len(s) != 4:
            return None
        st = (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
        return zpm.ZpmEntry(
            state=st,
            reply_text=str(obj.get("reply", "")),
            tokens=int(obj.get("tokens", 0)),
            wall_s_at_write=float(obj.get("wall", 0.0)),
            metadata={str(k): str(v) for k, v in dict(obj.get("meta") or {}).items()},
        )
    except (KeyError, TypeError, ValueError):
        return None


def _index_has_state(idx: zpm.ZpmIndex, st: zpm.State) -> bool:
    for e in idx:
        if e.state == st:
            return True
    return False


def recover_zpm_for_model(model_sha256: str, index_path: Path) -> bool:
    """Merge un-applied WAL lines into ``index.bin``. Returns True if index changed."""
    if _safe_mode_enabled():
        return False
    if os.environ.get("NRL_ZPM_WAL", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    sha = model_sha256 or "unknown"
    lk = _lock_for(sha)
    with lk:
        wpath = wal_path(sha)
        state = _read_learn_state(sha)
        applied = int(state.get("wal_applied_bytes", 0) or 0)
        idx = zpm.ZpmIndex.load(index_path) if index_path.is_file() else zpm.ZpmIndex()
        dirty = False
        if wpath.is_file():
            blob = wpath.read_bytes()
            tail = blob[applied:]
            if tail:
                for line in tail.split(b"\n"):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    ent = _json_to_entry(obj)
                    if ent is None or ent.tokens <= 0:
                        continue
                    if not _index_has_state(idx, ent.state):
                        idx.add(ent)
                        dirty = True
                applied = len(blob)
        if dirty:
            idx.save(index_path)
        new_applied = wpath.stat().st_size if wpath.is_file() else 0
        if dirty or new_applied != int(state.get("wal_applied_bytes", 0) or 0):
            state["wal_applied_bytes"] = new_applied
            state.setdefault("schema_id", "nrl.zpm_persistence.v1")
            _atomic_write_json(learn_state_path(sha), state)
        return dirty


def _wal_append_line(model_sha256: str, line: bytes) -> None:
    wpath = wal_path(model_sha256)
    wpath.parent.mkdir(parents=True, exist_ok=True)
    with open(wpath, "ab", buffering=0) as wf:
        wf.write(line)
        if not line.endswith(b"\n"):
            wf.write(b"\n")
        wf.flush()
        os.fsync(wf.fileno())


def _maybe_snapshot(
    model_sha256: str,
    index_path: Path,
    idx: zpm.ZpmIndex,
    state: dict[str, Any],
) -> None:
    now = time.time()
    last = float(state.get("last_snapshot_unix", 0.0) or 0.0)
    since = int(state.get("since_snapshot_appends", 0) or 0)
    first = float(state.get("first_persist_unix", 0.0) or 0.0)
    if first <= 0:
        first = now
        state["first_persist_unix"] = first
    time_due = (now - first) >= 300.0 or (last > 0.0 and (now - last) >= 300.0)
    if since < 1000 and not time_due:
        return
    zd = zpm_dir(model_sha256)
    snap_dir = zd / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = int(now)
    dst = snap_dir / f"zpm_index_{ts}.bin"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if index_path.is_file():
        shutil.copy2(index_path, tmp)
        os.replace(tmp, dst)
    prev = _read_learn_state(model_sha256)
    learn = {
        "schema_id": "nrl.zpm_persistence.v1",
        "model_sha256": model_sha256,
        "last_snapshot_unix": now,
        "zpm_entries": len(idx),
        "index_bytes": index_path.stat().st_size if index_path.is_file() else 0,
        "snapshot_path": str(dst),
        "wal_applied_bytes": 0,
        "since_snapshot_appends": 0,
        "total_persist_calls": int(state.get("total_persist_calls", 0) or 0),
        "first_persist_unix": now,
    }
    for _k in (
        "last_prune_unix",
        "last_prune_evicted_entries",
        "last_prune_freed_bytes",
        "last_footprint_bytes",
        "last_footprint_unix",
    ):
        if _k in prev:
            learn[_k] = prev[_k]
    _atomic_write_json(learn_state_path(model_sha256), learn)
    # P6: WAL compaction — truncate after a durable snapshot (bounded disk).
    wpath = wal_path(model_sha256)
    try:
        wpath.write_bytes(b"")
    except OSError:
        pass
    state.clear()
    state.update(learn)


def maybe_auto_prune_lmo(model_sha256: str, index_path: Path) -> None:
    """If ``NRL_LMO_AUTO_PRUNE`` is on, shrink footprint after a successful persist (no-op by default)."""
    if _safe_mode_enabled():
        return
    if os.environ.get("NRL_LMO_AUTO_PRUNE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    sha = model_sha256 or "unknown"
    try:
        from . import lmo_disk_manager as _ldm

        base = Path(os.environ.get("NRL_ROOT") or Path.cwd())
        lmo_dir = base / "cache" / "lmo" / sha
        if not (lmo_dir / "lmo.header").is_file():
            lmo_dir = None
        mm_root = base / "cache" / "mm"
        _ldm.LmoDiskManager(
            model_sha256=sha,
            lmo_dir=lmo_dir,
            index_path=index_path,
            mm_root=mm_root,
        ).prune_if_needed(aggressive=False, dry_run=False, force=False)
    except Exception:
        pass


def persist_zpm_entry(model_sha256: str, index_path: Path, idx: zpm.ZpmIndex, entry: zpm.ZpmEntry) -> None:
    """WAL-first durable write: append+fsync, atomic full index save, optional snapshot."""
    if _safe_mode_enabled():
        idx.save(index_path)
        return
    if os.environ.get("NRL_ZPM_WAL", "1").strip().lower() in ("0", "false", "off", "no"):
        idx.save(index_path)
        maybe_auto_prune_lmo(model_sha256 or "unknown", index_path)
        return
    sha = model_sha256 or "unknown"
    lk = _lock_for(sha)
    with lk:
        line = json.dumps(_entry_to_json(entry), sort_keys=True).encode("utf-8") + b"\n"
        _wal_append_line(sha, line)
        idx.save(index_path)
        state = _read_learn_state(sha)
        state.setdefault("schema_id", "nrl.zpm_persistence.v1")
        if not float(state.get("first_persist_unix", 0.0) or 0.0):
            state["first_persist_unix"] = time.time()
        state["wal_applied_bytes"] = wal_path(sha).stat().st_size if wal_path(sha).is_file() else 0
        state["since_snapshot_appends"] = int(state.get("since_snapshot_appends", 0) or 0) + 1
        state["total_persist_calls"] = int(state.get("total_persist_calls", 0) or 0) + 1
        state["last_snapshot_unix"] = float(state.get("last_snapshot_unix", 0.0) or 0.0)
        state["zpm_entries"] = len(idx)
        state["index_bytes"] = index_path.stat().st_size if index_path.is_file() else 0
        snap_now = state["since_snapshot_appends"] >= 1000
        now = time.time()
        first = float(state.get("first_persist_unix", now) or now)
        if not snap_now and (now - first) >= 300.0:
            snap_now = True
        last = float(state.get("last_snapshot_unix", 0.0) or 0.0)
        if not snap_now and last > 0.0 and (now - last) >= 300.0:
            snap_now = True
        if snap_now:
            _maybe_snapshot(sha, index_path, idx, state)
        else:
            _atomic_write_json(learn_state_path(sha), state)

    maybe_auto_prune_lmo(sha, index_path)


def count_wal_pending_lines(model_sha256: str, index_path: Path) -> tuple[int, int]:
    """Return ``(wal_file_bytes, estimated_unapplied_lines)`` for UI."""
    wpath = wal_path(model_sha256)
    if not wpath.is_file():
        return 0, 0
    total = wpath.stat().st_size
    applied = int(_read_learn_state(model_sha256).get("wal_applied_bytes", 0) or 0)
    tail = max(0, total - applied)
    # Rough line count in tail
    if tail == 0:
        return total, 0
    try:
        chunk = wpath.read_bytes()[applied:]
        lines = [x for x in chunk.split(b"\n") if x.strip()]
        return total, len(lines)
    except OSError:
        return total, 0


def gather_lmo_persistence_info(
    *,
    model_sha256: str,
    index_path: Path,
    mm_root: Path,
    lmo_dir: Path | None = None,
) -> dict[str, Any]:
    """JSON-safe stats for ``nrlpy lmo info``."""
    sha = model_sha256 or "unknown"
    zd = zpm_dir(sha)
    st = _read_learn_state(sha)
    idx_size = index_path.stat().st_size if index_path.is_file() else 0
    wal_sz, pending_lines = count_wal_pending_lines(sha, index_path)
    snap_dir = zd / "snapshots"
    snaps = sorted(snap_dir.glob("zpm_index_*.bin")) if snap_dir.is_dir() else []
    last_snap_age_s: float | None = None
    last_snap = float(st.get("last_snapshot_unix", 0.0) or 0.0)
    if last_snap > 0:
        last_snap_age_s = max(0.0, time.time() - last_snap)
    mm_dir = mm_root / sha
    mm_count = 0
    mm_bytes = 0
    if mm_dir.is_dir():
        for p in mm_dir.glob("*.mm"):
            mm_count += 1
            try:
                mm_bytes += p.stat().st_size
            except OSError:
                pass
    try:
        n_entries = len(zpm.ZpmIndex.load(index_path)) if index_path.is_file() else 0
    except Exception:
        n_entries = 0
    from . import lmo_disk_manager as _ldm

    fp = _ldm.footprint_bytes(
        model_sha256=sha,
        lmo_dir=lmo_dir,
        mm_root=mm_root,
    )
    qb = _ldm.max_quota_bytes()
    last_fp = int(st.get("last_footprint_bytes", 0) or 0)
    last_ft = float(st.get("last_footprint_unix", 0.0) or 0.0)
    now = time.time()
    growth_bps: float | None = None
    if last_fp > 0 and last_ft > 0.0 and now > last_ft + 0.5:
        growth_bps = round((fp - last_fp) / (now - last_ft), 6)
    info = {
        "model_sha256": sha,
        "zpm_index_path": str(index_path),
        "zpm_index_bytes": idx_size,
        "zpm_entry_count": n_entries,
        "wal_bytes": wal_sz,
        "wal_pending_lines_estimate": pending_lines,
        "wal_applied_bytes": int(st.get("wal_applied_bytes", 0) or 0),
        "last_snapshot_unix": last_snap,
        "last_snapshot_age_s": last_snap_age_s,
        "snapshot_count": len(snaps),
        "muscle_memory_files": mm_count,
        "muscle_memory_bytes": mm_bytes,
        "learn_state_path": str(learn_state_path(sha)),
        "lmo_footprint_bytes": fp,
        "lmo_quota_bytes": qb,
        "lmo_quota_used_ratio": round(fp / float(qb), 6) if qb > 0 else 0.0,
        "footprint_growth_bytes_per_sec": growth_bps,
        "last_prune_unix": float(st.get("last_prune_unix", 0.0) or 0.0),
        "last_prune_evicted_entries": int(st.get("last_prune_evicted_entries", 0) or 0),
        "last_prune_freed_bytes": int(st.get("last_prune_freed_bytes", 0) or 0),
    }
    return info
