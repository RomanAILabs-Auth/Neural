"""Shared path resolution for control-plane tools (evidence, chat, CLI)."""

from __future__ import annotations

import os
from pathlib import Path


def immune_evidence_log_paths() -> list[Path]:
    """Ordered candidate paths for immune JSONL (first existing wins for tail/display)."""
    out: list[Path] = []
    env = os.environ.get("NRL_EVIDENCE_LOG")
    if env:
        out.append(Path(env))
    here = Path.cwd()
    for anc in [here, *here.parents]:
        out.append(anc / "build" / "immune" / "events.jsonl")
        if anc.parent == anc:
            break
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def first_existing_evidence_log() -> Path | None:
    for p in immune_evidence_log_paths():
        if p.is_file():
            return p
    return None
