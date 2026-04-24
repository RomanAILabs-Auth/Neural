# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""LM/AI consent toggle (``nrlpy -ai``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrlpy.cli import main as cli_main
from nrlpy.consent_ai import lm_ai_cli_toggle


def test_lm_ai_cli_toggle_writes_consent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nrlpy.consent_ai.Path.home", lambda: tmp_path)
    assert lm_ai_cli_toggle("on") == 0
    p = tmp_path / ".nrl" / "consent.json"
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["lm_ai_opt_in"] is True
    assert data["source"] == "nrlpy -ai"


def test_cli_ai_invocation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nrlpy.consent_ai.Path.home", lambda: tmp_path)
    assert cli_main(["--ai", "off"]) == 0
    data = json.loads((tmp_path / ".nrl" / "consent.json").read_text(encoding="utf-8"))
    assert data["lm_ai_opt_in"] is False


def test_cli_ai_missing_arg() -> None:
    assert cli_main(["-ai"]) == 2
