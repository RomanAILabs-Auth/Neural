from __future__ import annotations

from pathlib import Path

import pytest

from nrlpy.paths import first_existing_evidence_log, immune_evidence_log_paths


def test_first_existing_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "build" / "immune" / "events.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    p = first_existing_evidence_log()
    assert p is not None
    assert p.name == "events.jsonl"


def test_immune_paths_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NRL_EVIDENCE_LOG", raising=False)
    paths = immune_evidence_log_paths()
    assert len(paths) >= 1
