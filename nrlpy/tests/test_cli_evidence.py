# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""``nrlpy evidence tail`` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from nrlpy.cli import main as cli_main


def test_evidence_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text('{"x":1}\n{"y":2}\n{"z":3}\n', encoding="utf-8")
    monkeypatch.setenv("NRL_EVIDENCE_LOG", str(p))
    assert cli_main(["evidence", "tail", "2"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
