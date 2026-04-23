"""Control-plane preferences (``nrl control``) consumed by nrlpy runtime helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrlpy import runtime


def test_control_hints_active_matches_resolve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    now = 1_700_000_000
    monkeypatch.setattr(runtime.time, "time", lambda: float(now))
    prefs: runtime.ControlPreferences = {
        "schema_id": "nrl.control_preferences.v1",
        "updated_unix": now,
        "recommended_profile": "sovereign",
        "power_until_unix": 0,
        "throttle_hint": "gated",
    }
    assert runtime.control_hints_active(prefs) is True
    assert runtime.resolve_bench_profile_with_control_hints("omega", prefs) == "sovereign"
    monkeypatch.chdir(tmp_path)
    assert runtime.control_hints_active(None) is False


def test_resolve_bench_profile_power_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctrl = tmp_path / "build" / "control"
    ctrl.mkdir(parents=True)
    now = 1_700_000_000
    monkeypatch.setattr(runtime.time, "time", lambda: float(now))
    prefs_path = ctrl / "preferences.json"
    prefs_path.write_text(
        json.dumps(
            {
                "schema_id": "nrl.control_preferences.v1",
                "updated_unix": now,
                "recommended_profile": "omega-hybrid",
                "power_until_unix": now + 60,
                "throttle_hint": "none",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = runtime.load_control_preferences()
    assert loaded is not None
    assert runtime.resolve_bench_profile_with_control_hints("sovereign", loaded) == "omega-hybrid"
    assert runtime.resolve_bench_profile_with_control_hints("sovereign", None) == "sovereign"


def test_resolve_bench_profile_gated_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctrl = tmp_path / "build" / "control"
    ctrl.mkdir(parents=True)
    now = 1_700_000_000
    monkeypatch.setattr(runtime.time, "time", lambda: float(now))
    (ctrl / "preferences.json").write_text(
        json.dumps(
            {
                "schema_id": "nrl.control_preferences.v1",
                "updated_unix": now,
                "recommended_profile": "sovereign",
                "power_until_unix": 0,
                "throttle_hint": "gated",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = runtime.load_control_preferences()
    assert loaded is not None
    assert runtime.resolve_bench_profile_with_control_hints("omega", loaded) == "sovereign"


def test_control_preferences_path_uses_nrl_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    assert runtime.control_preferences_path() == tmp_path / "build" / "control" / "preferences.json"


def test_load_control_preferences_bad_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctrl = tmp_path / "build" / "control"
    ctrl.mkdir(parents=True)
    (ctrl / "preferences.json").write_text('{"schema_id":"other"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert runtime.load_control_preferences() is None
