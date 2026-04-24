# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.lmo_disk_manager` — Bio-Digital P6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ("test_lmo",)


def test_max_quota_bytes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRL_LMO_MAX_GB", "2.5")
    from nrlpy.lmo_disk_manager import max_quota_bytes

    assert max_quota_bytes() == int(2.5 * (1024**3))


def test_bump_access_stat_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    from nrlpy import zpm
    from nrlpy.lmo_disk_manager import access_stats_path, bump_access_stat, state_fingerprint

    sha = "ab" * 32
    st: zpm.State = (1, 2, 3, 4)
    bump_access_stat(sha, st)
    bump_access_stat(sha, st)
    p = access_stats_path(sha)
    assert p.is_file()
    body = json.loads(p.read_text(encoding="utf-8"))
    fp = state_fingerprint(st)
    assert int(body["hits"][fp]) == 2


def test_prune_dry_run_then_apply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_gguf: Path,
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    from nrlpy import gguf
    from nrlpy import zpm as Z
    from nrlpy import zpm_persist as zp
    from nrlpy.lmo import absorb_gguf
    from nrlpy.lmo_disk_manager import LmoDiskManager, bump_access_stat

    h = absorb_gguf(
        fixture_gguf,
        out_root=tmp_path / "cache" / "lmo",
        attempt_libllama=False,
    )
    sha = h.model_sha256
    zpath = gguf._zpm_index_path(sha)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    idx = Z.ZpmIndex()
    for i in range(48):
        idx.add(
            Z.ZpmEntry(
                state=(i, i, i, i),
                reply_text="Z" * 6000,
                tokens=1,
                wall_s_at_write=float(i),
            )
        )
    idx.save(zpath)
    bump_access_stat(sha, (0, 0, 0, 0))
    mm_root = gguf._muscle_memory_root()
    mgr = LmoDiskManager(
        model_sha256=sha,
        lmo_dir=h.lmo_dir,
        index_path=zpath,
        mm_root=mm_root,
    )
    before_fp = mgr.footprint()
    qgib = max(0.00015, (before_fp / (1024**3)) * 0.35)
    monkeypatch.setenv("NRL_LMO_MAX_GB", str(qgib))

    dry = mgr.prune_if_needed(dry_run=True)
    assert dry["dry_run"] is True
    assert dry["footprint_bytes_before"] > int(dry["quota_bytes"] * 0.99)

    wet = mgr.prune_if_needed(dry_run=False)
    assert wet["dry_run"] is False
    assert int(wet.get("evicted_entries", 0) or 0) >= 1
    assert int(wet["footprint_bytes_after"]) < int(wet["footprint_bytes_before"])


def test_zpm_remove_indices() -> None:
    from nrlpy import zpm as Z

    idx = Z.ZpmIndex(
        [
            Z.ZpmEntry((i, 0, 0, 0), str(i), 1, 0.0) for i in range(10)
        ]
    )
    n = idx.remove_entry_indices({0, 2, 4})
    assert n == 3
    assert len(idx) == 7

