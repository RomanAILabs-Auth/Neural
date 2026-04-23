"""Bounded plasticity controller stub (P4).

Sovereign mode: all mutation APIs are logically disabled per immune spec §4.3.
Adaptive mode (future): logging-only stub — no data-plane persistence yet.
Set ``NRL_PLASTICITY_SHADOW_LOG=1`` to append a stub ``immune_event`` line when
``plasticity_snapshot("adaptive")`` runs (first candidate path from
``immune_evidence_log_paths()``).

**Roadmap (where "real ML" is allowed to land in this repo):** control-plane only —
``workload_id`` / ``structural_hash``, shadow executor vs reference, append-only evidence,
specialization store promotion — see ``docs/nrl_alive_language_evolution_architecture.md``
(contracts NAL-001 / NAL-002: no adaptation inside INT4 inner loops unless a future
signed coprocessor contract says otherwise).
"""

from __future__ import annotations

import os
from typing import Any, Literal

Mode = Literal["sovereign", "adaptive"]


def _maybe_emit_adaptive_shadow_stub() -> None:
    """Append one immune JSONL line when ``NRL_PLASTICITY_SHADOW_LOG=1`` (control plane only)."""
    if os.environ.get("NRL_PLASTICITY_SHADOW_LOG") != "1":
        return
    from .evidence import append_immune_event
    from .paths import immune_evidence_log_paths

    paths = immune_evidence_log_paths()
    if not paths:
        return
    path = paths[0]
    append_immune_event(
        path,
        {
            "level": 2,
            "signal_id": "PLASTICITY_SHADOW_STUB",
            "action": "log_only",
            "message": "Adaptive plasticity snapshot: evidence hook (no lattice mutation).",
        },
    )


def plasticity_snapshot(mode: str | None) -> dict[str, Any]:
    """Non-mutating status for harnesses and CLI; does not touch lattice buffers."""
    m = (mode or "sovereign").strip().lower()
    if m not in ("sovereign", "adaptive"):
        m = "sovereign"
    if m == "sovereign":
        return {
            "mode": m,
            "writes_enabled": False,
            "persistence": "none",
            "detail": "Sovereign: plasticity APIs disabled (immune spec §4.3).",
        }
    _maybe_emit_adaptive_shadow_stub()
    return {
        "mode": m,
        "writes_enabled": False,
        "persistence": "none",
        "detail": "Adaptive stub: logging-only path; bounded updates not yet implemented.",
    }
