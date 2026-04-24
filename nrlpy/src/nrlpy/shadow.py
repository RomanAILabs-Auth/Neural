# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Control-plane shadow compare (reference vs candidate benches).

No lattice mutation; uses ``bench_cli`` observables only. See alive §4.4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import runtime
from .evidence import append_immune_event
from .workload import build_workload_descriptor, workload_identity_block


def control_plane_probe_identity() -> dict[str, Any]:
    """Stable tiny descriptor for plasticity shadow logging (identity only)."""
    desc = build_workload_descriptor(
        harness_id="plasticity_shadow_stub",
        neurons=2048,
        iterations=32,
        reps=2,
        threshold=8,
        profiles=["sovereign"],
        nrl_version=runtime.version(),
    )
    return workload_identity_block(desc)


def shadow_bench_pair(
    *,
    harness_id: str,
    neurons: int,
    iterations: int,
    reps: int,
    threshold: int,
    profile_reference: str,
    profile_candidate: str,
    nrl_bin: str | None = None,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Run two ``nrl bench`` calls; return workload ids, metrics, and simple diff flags.

    If ``evidence_path`` is set, appends one ``immune_event_v1`` line (``log_only``).
    """
    desc_ref = build_workload_descriptor(
        harness_id=harness_id,
        neurons=neurons,
        iterations=iterations,
        reps=reps,
        threshold=threshold,
        profiles=[profile_reference],
        nrl_version=runtime.version(),
    )
    desc_cand = build_workload_descriptor(
        harness_id=harness_id,
        neurons=neurons,
        iterations=iterations,
        reps=reps,
        threshold=threshold,
        profiles=[profile_candidate],
        nrl_version=runtime.version(),
    )
    id_ref = workload_identity_block(desc_ref)
    id_cand = workload_identity_block(desc_cand)

    ref_m = runtime.bench_cli(
        neurons, iterations, reps, threshold, profile_reference, nrl_bin=nrl_bin
    )
    cand_m = runtime.bench_cli(
        neurons, iterations, reps, threshold, profile_candidate, nrl_bin=nrl_bin
    )

    def _pick(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: d[k] for k in keys if k in d}

    keys = ("executed_updates", "baseline_equiv_updates", "skip_ratio", "executed_gops", "virtual_gops")
    ref_s = _pick(ref_m, keys)
    cand_s = _pick(cand_m, keys)
    executed_match = ref_s.get("executed_updates") == cand_s.get("executed_updates")
    skip_match = ref_s.get("skip_ratio") == cand_s.get("skip_ratio")

    out: dict[str, Any] = {
        "reference_identity": id_ref,
        "candidate_identity": id_cand,
        "reference_metrics": ref_s,
        "candidate_metrics": cand_s,
        "executed_updates_match": executed_match,
        "skip_ratio_match": skip_match,
    }

    if evidence_path is not None:
        summary = {
            "harness_id": harness_id,
            "executed_updates_match": executed_match,
            "skip_ratio_match": skip_match,
            "ref_profile": profile_reference,
            "cand_profile": profile_candidate,
        }
        append_immune_event(
            evidence_path,
            {
                "level": 3,
                "signal_id": "SHADOW_BENCH_PAIR",
                "action": "log_only",
                "message": json.dumps(summary, separators=(",", ":")),
                "workload_id": id_cand["workload_id"],
                "structural_hash": id_cand["structural_hash"],
                "profile": profile_candidate,
                "nrl_version": runtime.version(),
            },
        )

    return out
