import json
from pathlib import Path

import pytest

from nrlpy.plasticity import plasticity_snapshot


def test_sovereign_default() -> None:
    s = plasticity_snapshot(None)
    assert s["writes_enabled"] is False
    assert s["mode"] == "sovereign"


def test_adaptive_stub() -> None:
    s = plasticity_snapshot("adaptive")
    assert s["writes_enabled"] is False
    assert "stub" in s["detail"].lower()


def test_adaptive_shadow_log_emits_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    monkeypatch.chdir(root)
    monkeypatch.setenv("NRL_PLASTICITY_SHADOW_LOG", "1")
    monkeypatch.delenv("NRL_EVIDENCE_LOG", raising=False)
    plasticity_snapshot("adaptive")
    log = root / "build" / "immune" / "events.jsonl"
    assert log.is_file()
    line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    ev = json.loads(line)
    assert ev["signal_id"] == "PLASTICITY_SHADOW_STUB"
    assert ev["action"] == "log_only"
