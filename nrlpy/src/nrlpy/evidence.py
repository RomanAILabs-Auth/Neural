"""Append-only evidence log helpers (immune and specialization events).

Schema: ``docs/schemas/immune_event_v1.schema.json``. Events are JSON lines (JSONL).
Control-plane only; not used on neuron hot paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


IMMUNE_SCHEMA_ID = "nrl.immune_event.v1"


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object per line to ``path`` (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_immune_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and append an immune-style event; returns the emitted record."""
    ev: dict[str, Any] = {
        "schema_id": IMMUNE_SCHEMA_ID,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **dict(event),
    }
    append_jsonl(path, ev)
    return ev


def read_jsonl_tail(path: Path, max_lines: int) -> list[str]:
    """Return up to ``max_lines`` non-empty lines from the end of ``path``."""
    if not path.is_file() or max_lines <= 0:
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [ln for ln in raw if ln.strip()]
    return nonempty[-max_lines:]
