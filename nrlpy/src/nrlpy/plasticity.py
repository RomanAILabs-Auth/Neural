"""Bounded plasticity controller stub (P4).

Sovereign mode: all mutation APIs are logically disabled per immune spec §4.3.
Adaptive mode (future): logging-only stub — no data-plane persistence yet.

**Roadmap (where "real ML" is allowed to land in this repo):** control-plane only —
``workload_id`` / ``structural_hash``, shadow executor vs reference, append-only evidence,
specialization store promotion — see ``docs/nrl_alive_language_evolution_architecture.md``
(contracts NAL-001 / NAL-002: no adaptation inside INT4 inner loops unless a future
signed coprocessor contract says otherwise).
"""

from __future__ import annotations

from typing import Any, Literal

Mode = Literal["sovereign", "adaptive"]


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
    return {
        "mode": m,
        "writes_enabled": False,
        "persistence": "none",
        "detail": "Adaptive stub: logging-only path; bounded updates not yet implemented.",
    }
