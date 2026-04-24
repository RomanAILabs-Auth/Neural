# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.drift_conqueror` — Bio-Digital P5."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest_plugins = ("test_lmo",)

from nrlpy import drift_conqueror as dc


def test_coverage_percent_monotonic_with_new_buckets() -> None:
    bc: dict[str, int] = {"0": 5}
    c0 = dc.coverage_percent(bc)
    for i in range(1, 20):
        bc[str(i)] = 1
    c1 = dc.coverage_percent(bc)
    assert c1 >= c0
    assert c1 > c0 or c0 == 0


def test_weak_buckets_includes_decode_failures() -> None:
    bc = {str(i): 100 for i in range(256)}
    fails = {"42": 3}
    w = dc.weak_buckets(bc, decode_fail_by_bucket=fails)
    assert 42 in w


def test_growth_budget_exhausted_respects_pct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRL_LEARN_MAX_GROWTH_PCT", "1")
    assert dc.max_growth_fraction() == 0.01
    assert not dc.growth_budget_exhausted(1000, 1000, dc.max_growth_fraction())
    assert not dc.growth_budget_exhausted(1010, 1000, dc.max_growth_fraction())
    assert dc.growth_budget_exhausted(1011, 1000, dc.max_growth_fraction())


def test_maybe_roll_resets_window(tmp_path: Path) -> None:
    probe = tmp_path / "learn_probe"
    probe.mkdir(parents=True)
    now = 1_000_000.0
    gw0 = dc.maybe_roll_growth_window(probe, now=now, zpm_index_bytes=5000)
    assert int(gw0["baseline_index_bytes"]) == 5000
    gw1 = dc.maybe_roll_growth_window(probe, now=now + 100.0, zpm_index_bytes=9000)
    assert int(gw1["baseline_index_bytes"]) == 5000
    gw2 = dc.maybe_roll_growth_window(probe, now=now + 86_500.0, zpm_index_bytes=9000)
    assert int(gw2["baseline_index_bytes"]) == 9000


def test_summarize_for_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRL_LEARN_MAX_GROWTH_PCT", "5")
    probe = tmp_path / "learn_probe"
    probe.mkdir(parents=True)
    zpath = tmp_path / "index.bin"
    zpath.write_bytes(b"x" * 2000)
    dc.save_coverage_state_atomic(
        probe,
        {
            "schema_id": "nrl.drift_coverage.v1",
            "bucket_counts": {str(i): 1 for i in range(50)},
            "decode_fail_by_bucket": {},
            "last_conquest_unix": time.time(),
            "conquest_cycles": 2,
        },
    )
    summ = dc.summarize_for_cli(probe_dir=probe, zpm_index_path=zpath)
    assert summ["coverage_percent"] > 0
    assert summ["weak_bucket_count"] >= 1
    assert summ["zpm_index_bytes"] == 2000
    assert "growth_headroom_pct" in summ


def test_conquest_prompts_length() -> None:
    weak = list(range(40))
    ps = dc.conquest_prompts(weak, n_min=10, n_max=30)
    assert 10 <= len(ps) <= 30
    assert all("drift_conquest" in p for p in ps)


def test_lmo_coverage_cli_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fixture_gguf: Path,
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    from nrlpy.cli import main as cli_main
    from nrlpy.lmo import absorb_gguf

    h = absorb_gguf(fixture_gguf, out_root=tmp_path / "cache" / "lmo", attempt_libllama=False)
    assert cli_main(["lmo", "coverage", str(h.lmo_dir), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "coverage_percent" in out
    assert "weak_bucket_count" in out
    assert "growth_headroom_pct" in out
