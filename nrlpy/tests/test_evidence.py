# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Evidence log (JSONL) tests."""

from __future__ import annotations

import json
from pathlib import Path

from nrlpy.evidence import append_immune_event, append_jsonl


def test_append_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"b": 2})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_append_immune_event_schema(tmp_path: Path) -> None:
    p = tmp_path / "immune.jsonl"
    out = append_immune_event(
        p,
        {"level": 1, "signal_id": "SIG_TEST", "action": "warn", "message": "ok"},
    )
    assert out["schema_id"] == "nrl.immune_event.v1"
    assert "ts_utc" in out
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["level"] == 1
