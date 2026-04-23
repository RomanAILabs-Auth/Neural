"""``nrlpy control`` and bench control-hints CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrlpy.cli import main as cli_main


def test_cli_control_status_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = tmp_path / "build" / "control"
    ctrl.mkdir(parents=True)
    prefs = {
        "schema_id": "nrl.control_preferences.v1",
        "updated_unix": 1,
        "recommended_profile": "sovereign",
        "power_until_unix": 0,
        "throttle_hint": "gated",
    }
    (ctrl / "preferences.json").write_text(json.dumps(prefs), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["control", "status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hints_active"] is True
    assert out["preferences"]["recommended_profile"] == "sovereign"
    assert str(tmp_path) in out["control_preferences_path"]


def test_cli_control_audit_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = tmp_path / "build" / "control"
    ctrl.mkdir(parents=True)
    (ctrl / "control_audit.jsonl").write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["control", "audit", "tail", "1"]) == 0
    assert capsys.readouterr().out.strip() == '{"b":2}'


def test_cli_control_audit_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli_main(["control", "audit", "tail"]) == 2


def test_cli_bench_respect_control_hints_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    def fake_bench(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "profile": "x",
            "mode": "m",
            "variant": "v",
            "neurons": 1,
            "iterations": 1,
            "reps": 1,
            "threshold": 1,
            "elapsed_s": 0.0,
            "executed_updates": 0,
            "baseline_equiv_updates": 0,
            "skip_ratio": 0.0,
            "executed_gops": 0.0,
            "virtual_gops": 0.0,
        }

    monkeypatch.setattr("nrlpy.cli.runtime.bench_cli", fake_bench)
    assert cli_main(["bench", "2048", "32", "2", "8", "omega", "--respect-control-hints"]) == 0
    assert captured.get("respect_control_hints") is True
    assert json.loads(capsys.readouterr().out)["profile"] == "x"
