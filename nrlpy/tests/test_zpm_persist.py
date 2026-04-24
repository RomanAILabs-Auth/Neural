# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.zpm_persist` — P4 WAL + recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrlpy import zpm
from nrlpy.gguf import _zpm_index_path
from nrlpy.zpm_persist import (
    learn_state_path,
    persist_zpm_entry,
    recover_zpm_for_model,
    wal_path,
)


@pytest.fixture()
def isolated_zpm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    return "a" * 64


def test_wal_append_then_recover_rehydrates_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_zpm: str
) -> None:
    monkeypatch.setenv("NRL_ZPM_WAL", "1")
    sha = isolated_zpm
    zpath = _zpm_index_path(sha)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    idx = zpm.ZpmIndex()
    ent = zpm.ZpmEntry(
        state=(1, 2, 3, 4),
        reply_text="hello",
        tokens=1,
        wall_s_at_write=0.01,
        metadata={"t": "x"},
    )
    idx.add(ent)
    persist_zpm_entry(sha, zpath, idx, ent)
    assert wal_path(sha).is_file()
    assert zpath.is_file()
    # Simulate crash: index lost but WAL retained; reset offset so replay runs.
    learn_state_path(sha).unlink(missing_ok=True)
    zpath.unlink()
    assert not zpath.is_file()
    recover_zpm_for_model(sha, zpath)
    assert zpath.is_file()
    idx2 = zpm.ZpmIndex.load(zpath)
    assert len(idx2) == 1
    hit, e2 = idx2.lookup((1, 2, 3, 4), threshold_bits=0)
    assert hit.exact and e2 is not None
    assert e2.reply_text == "hello"


def test_learn_state_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_zpm: str) -> None:
    monkeypatch.setenv("NRL_ZPM_WAL", "1")
    sha = isolated_zpm
    zpath = _zpm_index_path(sha)
    idx = zpm.ZpmIndex()
    ent = zpm.ZpmEntry(state=(9, 8, 7, 6), reply_text="z", tokens=1, metadata={})
    idx.add(ent)
    persist_zpm_entry(sha, zpath, idx, ent)
    st = learn_state_path(sha)
    assert st.is_file()
    data = json.loads(st.read_text(encoding="utf-8"))
    assert data.get("schema_id") == "nrl.zpm_persistence.v1"
    assert int(data.get("wal_applied_bytes", 0)) >= 1


def test_wal_disabled_skips_append(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_zpm: str) -> None:
    monkeypatch.setenv("NRL_ZPM_WAL", "0")
    sha = isolated_zpm
    zpath = _zpm_index_path(sha)
    idx = zpm.ZpmIndex()
    ent = zpm.ZpmEntry(state=(5, 5, 5, 5), reply_text="n", tokens=1, metadata={})
    idx.add(ent)
    persist_zpm_entry(sha, zpath, idx, ent)
    assert zpath.is_file()
    assert not wal_path(sha).is_file()
