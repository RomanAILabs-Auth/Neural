# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.learn_daemon` — Bio-Digital P2 MVP."""

from __future__ import annotations

import json
import time

import pytest

from nrlpy.learn_daemon import LearnDaemon, learn_mode_enabled
from nrlpy.lmo import absorb_gguf

from test_lmo import fixture_gguf


@pytest.fixture()
def lmo_handle(fixture_gguf: Path, tmp_path: Path):
    return absorb_gguf(
        fixture_gguf,
        out_root=tmp_path / "cache" / "lmo",
        attempt_libllama=False,
    )


def test_learn_mode_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "0")
    assert learn_mode_enabled() is False
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    assert learn_mode_enabled() is True


def test_learn_mode_off_generates_no_curiosity(
    monkeypatch: pytest.MonkeyPatch, lmo_handle
) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "0")
    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=1.0, cpu_cap=0.25)
    d.start(lmo_handle)
    for _ in range(40):
        sleep(0.25)
    d.stop()
    st = d.get_status()
    assert st["curiosity_cycles"] == 0
    assert st["synthetic_prompts_generated"] == 0
    assert st["decode_calls"] == 0


def test_idle_triggers_curiosity_and_decode_hook(
    monkeypatch: pytest.MonkeyPatch, lmo_handle
) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    clock = [0.0]
    decodes: list[str] = []

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=5.0, cpu_cap=0.25)
    d.start(lmo_handle, decode_runner=lambda p: decodes.append(p))
    for _ in range(400):
        sleep(0.1)
        st = d.get_status()
        if st["curiosity_cycles"] >= 1 and st["decode_calls"] >= 3:
            break
    sleep(0.3)
    d.stop()
    st = d.get_status()
    assert st["curiosity_cycles"] >= 1
    assert st["synthetic_prompts_generated"] >= 3
    assert len(decodes) == 3
    snap = lmo_handle.lmo_dir / "learn_probe" / "daemon_snapshot.json"
    assert snap.is_file()
    body = json.loads(snap.read_text(encoding="utf-8"))
    assert body["curiosity_cycles"] >= 1
    assert body["learn_mode_enabled"] is True


def test_pause_blocks_cycles(monkeypatch: pytest.MonkeyPatch, lmo_handle) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=3.0)
    d.pause()
    d.start(lmo_handle)
    for _ in range(80):
        sleep(0.1)
    d.stop()
    assert d.get_status()["curiosity_cycles"] == 0


def test_resume_after_pause_runs_cycle(monkeypatch: pytest.MonkeyPatch, lmo_handle) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=4.0)
    d.start(lmo_handle)
    d.pause()
    for _ in range(20):
        sleep(0.2)
    assert d.get_status()["curiosity_cycles"] == 0
    d.resume()
    for _ in range(200):
        sleep(0.1)
        if d.get_status()["curiosity_cycles"] >= 1:
            break
    d.stop()
    assert d.get_status()["curiosity_cycles"] >= 1


def test_notify_user_interaction_resets_idle(
    monkeypatch: pytest.MonkeyPatch, lmo_handle
) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=6.0)
    d.start(lmo_handle)
    for _ in range(45):
        sleep(0.1)
    assert d.get_status()["curiosity_cycles"] == 0
    d.notify_user_interaction()
    for _ in range(30):
        sleep(0.1)
    assert d.get_status()["curiosity_cycles"] == 0
    for _ in range(120):
        sleep(0.1)
        if d.get_status()["curiosity_cycles"] >= 1:
            break
    d.stop()
    assert d.get_status()["curiosity_cycles"] >= 1


def test_feed_history_shapes_prompts(monkeypatch: pytest.MonkeyPatch, lmo_handle) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(monotonic_fn=mono, sleep_fn=sleep, idle_sec=2.0)
    d.feed_history_snippet("hello lattice")
    d.start(lmo_handle)
    snap = lmo_handle.lmo_dir / "learn_probe" / "daemon_snapshot.json"
    ok = False
    for _ in range(400):
        sleep(0.1)
        time.sleep(0.001)  # fake clock does not yield; let supervisor thread run
        if not snap.is_file():
            continue
        body = json.loads(snap.read_text(encoding="utf-8"))
        if int(body.get("curiosity_cycles", 0)) < 1:
            continue
        blob = " ".join(body.get("sample_prompts") or [])
        if "hello lattice" in blob:
            ok = True
            break
    assert ok, "timed out waiting for snapshot with history-shaped sample_prompts"
    sleep(0.3)
    d.stop()
    journal = lmo_handle.lmo_dir / "learn_probe" / "curiosity_journal.jsonl"
    if journal.is_file():
        lines = journal.read_text(encoding="utf-8").strip().splitlines()
        assert any("hello lattice" in ln for ln in lines)


def test_long_user_idle_triggers_conquest_prompts(
    monkeypatch: pytest.MonkeyPatch, lmo_handle
) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    # Wide cap so prior curiosity decode growth does not suppress conquest decode.
    monkeypatch.setenv("NRL_LEARN_MAX_GROWTH_PCT", "100")
    clock = [0.0]
    decodes: list[str] = []

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    d = LearnDaemon(
        monotonic_fn=mono,
        sleep_fn=sleep,
        idle_sec=2.0,
        conquest_idle_sec=9.0,
        cpu_cap=0.25,
    )
    d.start(lmo_handle, decode_runner=lambda p: decodes.append(p))
    for _ in range(800):
        sleep(0.05)
        time.sleep(0.001)
        st = d.get_status()
        if int(st.get("conquest_cycles", 0) or 0) >= 1:
            break
    sleep(0.2)
    d.stop()
    assert d.get_status().get("conquest_cycles", 0) >= 1
    assert any("drift_conquest" in p for p in decodes)
    assert sum(1 for p in decodes if "drift_conquest" in p) >= 10
    journal = lmo_handle.lmo_dir / "learn_probe" / "curiosity_journal.jsonl"
    assert journal.is_file()
    blob = journal.read_text(encoding="utf-8")
    assert "drift_conquest" in blob
    cov = lmo_handle.lmo_dir / "learn_probe" / "coverage_state.json"
    assert cov.is_file()
    body = json.loads(cov.read_text(encoding="utf-8"))
    assert float(body.get("coverage_percent", 0) or 0) >= 0.0


def test_growth_cap_skips_decode_runner(
    monkeypatch: pytest.MonkeyPatch, lmo_handle
) -> None:
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    monkeypatch.setenv("NRL_LEARN_MAX_GROWTH_PCT", "1")
    probe = lmo_handle.lmo_dir / "learn_probe"
    probe.mkdir(parents=True)
    zpm_p = lmo_handle.lmo_dir.parent.parent / "zpm" / lmo_handle.model_sha256 / "index.bin"
    zpm_p.parent.mkdir(parents=True, exist_ok=True)
    zpm_p.write_bytes(b"z" * 2000)
    gw = {
        "schema_id": "nrl.drift_growth.v1",
        "window_start_unix": time.time() - 3600.0,
        "baseline_index_bytes": 1000,
    }
    (probe / "conquest_growth_window.json").write_text(
        json.dumps(gw, indent=2), encoding="utf-8"
    )

    clock = [0.0]

    def mono() -> float:
        return clock[0]

    def sleep(dt: float) -> None:
        clock[0] += dt

    decodes: list[str] = []

    d = LearnDaemon(
        monotonic_fn=mono,
        sleep_fn=sleep,
        idle_sec=2.0,
        conquest_idle_sec=5.0,
        cpu_cap=0.25,
    )
    d.start(lmo_handle, decode_runner=lambda p: decodes.append(p))
    for _ in range(400):
        sleep(0.05)
        if d.get_status().get("curiosity_cycles", 0) >= 3:
            break
    sleep(0.15)
    d.stop()
    assert decodes == []
    assert d.get_status().get("growth_cap_exhausted") is True
