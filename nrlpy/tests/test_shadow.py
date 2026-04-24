# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Control-plane shadow harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrlpy.shadow import control_plane_probe_identity, shadow_bench_pair


def test_control_plane_probe_identity_stable_version() -> None:
    b = control_plane_probe_identity()
    assert "workload_id" in b and "structural_hash" in b
    assert b["descriptor"]["harness_id"] == "plasticity_shadow_stub"
    assert b["descriptor"]["neurons"] == 2048


def test_shadow_bench_pair_with_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = {
        "executed_updates": 100,
        "baseline_equiv_updates": 100,
        "skip_ratio": 0.0,
        "executed_gops": 1.0,
        "virtual_gops": 1.0,
        "profile": "x",
        "mode": "m",
        "variant": "v",
        "neurons": 16,
        "iterations": 2,
        "reps": 1,
        "threshold": 8,
        "elapsed_s": 0.01,
    }

    def fake_bench(
        _n: int,
        _it: int,
        _r: int,
        _th: int,
        profile: str,
        nrl_bin: str | None = None,
        *,
        respect_control_hints: bool = False,
    ) -> dict[str, object]:
        del nrl_bin, respect_control_hints
        m = dict(metrics)
        m["profile"] = profile
        return m

    monkeypatch.setattr("nrlpy.shadow.runtime.bench_cli", fake_bench)
    log = tmp_path / "ev.jsonl"
    out = shadow_bench_pair(
        harness_id="unit_shadow",
        neurons=16,
        iterations=2,
        reps=1,
        threshold=8,
        profile_reference="sovereign",
        profile_candidate="omega",
        evidence_path=log,
    )
    assert out["executed_updates_match"] is True
    assert log.is_file()
    line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    ev = json.loads(line)
    assert ev["signal_id"] == "SHADOW_BENCH_PAIR"
    assert ev.get("workload_id", "").startswith("unit_shadow|")
