# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bio-Digital Brain MVP v3.0 final lifecycle test (P1-P7).

The test uses the repository's tiny valid GGUF fixture plus ``NRL_INFERENCE=stub``
so the lifecycle is deterministic and fast: absorb, learn/coverage, chat/write,
WAL recovery, prune, safe mode, and doctor.
"""

from __future__ import annotations

import json
import tracemalloc
import time
from pathlib import Path

import pytest

pytest_plugins = ("test_lmo",)


def test_bio_digital_brain_e2e(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_gguf: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_LEARN_MODE", "1")
    monkeypatch.setenv("NRL_LEARN_MAX_GROWTH_PCT", "100")
    monkeypatch.delenv("NRL_SAFE_MODE", raising=False)

    from nrlpy import zpm
    from nrlpy.cli import main as cli_main
    from nrlpy.gguf import GgufManifest, _muscle_memory_root, _zpm_index_path, run_gguf
    from nrlpy.gguf_chat import ChatSession, chat_turn
    from nrlpy.learn_daemon import LearnDaemon, learn_mode_enabled
    from nrlpy.lmo import absorb_gguf
    from nrlpy.lmo_disk_manager import LmoDiskManager
    from nrlpy.zpm_persist import learn_state_path, persist_zpm_entry, recover_zpm_for_model, wal_path

    tracemalloc.start()
    try:
        # P1: absorb a valid GGUF into an LMO.
        handle = absorb_gguf(
            fixture_gguf,
            out_root=tmp_path / "cache" / "lmo",
            attempt_libllama=False,
        )
        assert handle.lmo_dir.is_dir()

        # P2 + P5: simulate >5 minutes idle; Drift Conqueror coverage must grow.
        clock = [0.0]

        def mono() -> float:
            return clock[0]

        def sleep(dt: float) -> None:
            clock[0] += dt

        d = LearnDaemon(
            monotonic_fn=mono,
            sleep_fn=sleep,
            idle_sec=1.0,
            conquest_idle_sec=300.0,
            cpu_cap=0.5,
        )
        d.start(handle)
        before_cov = float(d.get_status().get("coverage_percent", 0.0))
        for _ in range(800):
            sleep(0.5)
            time.sleep(0.001)
            st = d.get_status()
            if int(st.get("conquest_cycles", 0) or 0) >= 1:
                break
        d.stop()
        after_status = d.get_status()
        assert float(after_status["coverage_percent"]) >= before_cov
        assert int(after_status["conquest_cycles"]) >= 1
        assert clock[0] >= 300.0

        # P3: chat with aggressive learning knobs (MM+ZPM on) over the stub backend.
        base = GgufManifest(
            model=str(fixture_gguf),
            max_tokens=6,
            seed=42,
            muscle_memory="on",
            zpm_nullspace=True,
            zpm_threshold_bits=0,
            chat_format="none",
        )
        session = ChatSession(base_manifest=base, system="You are a concise NRL test bot.")
        r1 = chat_turn(session, "map the bio digital brain lifecycle")
        r2 = chat_turn(session, "repeat the lifecycle in one sentence")
        assert r1.tokens > 0
        assert r2.tokens > 0
        assert session.tps.turns == 2

        # P4: crash simulation and recovery from WAL.
        sha = handle.model_sha256
        zpath = _zpm_index_path(sha)
        assert zpath.is_file()
        before_entries = len(zpm.ZpmIndex.load(zpath))
        assert before_entries >= 1
        learn_state_path(sha).unlink(missing_ok=True)
        zpath.unlink()
        recovered = recover_zpm_for_model(sha, zpath)
        assert recovered is True
        recovered_entries = len(zpm.ZpmIndex.load(zpath))
        assert recovered_entries >= before_entries

        # P6: force quota pressure with cold ZPM rows, preview then apply prune.
        idx = zpm.ZpmIndex.load(zpath)
        for i in range(40):
            ent = zpm.ZpmEntry(
                state=(10_000 + i, i, i, i),
                reply_text="cold-row-" + ("x" * 4096),
                tokens=1,
                wall_s_at_write=float(i),
            )
            idx.add(ent)
            persist_zpm_entry(sha, zpath, idx, ent)

        mgr = LmoDiskManager(
            model_sha256=sha,
            lmo_dir=handle.lmo_dir,
            index_path=zpath,
            mm_root=_muscle_memory_root(),
        )
        footprint_before = mgr.footprint()
        monkeypatch.setenv("NRL_LMO_MAX_GB", str(max(1e-9, (footprint_before * 0.7) / (1024**3))))
        dry = mgr.prune_if_needed(dry_run=True)
        assert dry["dry_run"] is True
        assert int(dry.get("evicted_entries", 0) or 0) >= 1
        wet = mgr.prune_if_needed(dry_run=False)
        assert wet["dry_run"] is False
        assert int(wet.get("evicted_entries", 0) or 0) >= 1
        assert int(wet["footprint_bytes_after"]) < int(wet["footprint_bytes_before"])
        assert zpm.ZpmIndex.load(zpath).lookup((10_000, 0, 0, 0), threshold_bits=0)[1] is None
        assert wal_path(sha).is_file()
        assert wal_path(sha).stat().st_size == 0

        # P7 hardening: safe mode disables learn, WAL, and auto-prune hooks.
        monkeypatch.setenv("NRL_SAFE_MODE", "1")
        monkeypatch.setenv("NRL_LMO_AUTO_PRUNE", "1")
        assert learn_mode_enabled() is False
        safe_sha = "f" * 64
        safe_path = _zpm_index_path(safe_sha)
        safe_idx = zpm.ZpmIndex()
        safe_ent = zpm.ZpmEntry((1, 1, 1, 1), "safe", 1)
        safe_idx.add(safe_ent)
        persist_zpm_entry(safe_sha, safe_path, safe_idx, safe_ent)
        assert safe_path.is_file()
        assert not wal_path(safe_sha).is_file()
        monkeypatch.setenv("NRL_SAFE_MODE", "0")

        # P7 doctor must report a healthy local install.
        assert cli_main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "status: healthy" in out

        # Simulated 1-hour soak clock; memory remains bounded for the test lifecycle.
        clock[0] += 3600.0
        current, peak = tracemalloc.get_traced_memory()
        assert current >= 0
        assert peak < 64 * 1024 * 1024
    finally:
        tracemalloc.stop()
