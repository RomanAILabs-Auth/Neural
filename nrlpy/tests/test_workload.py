"""Tests for workload identity (P1 harness contract)."""

from __future__ import annotations

from nrlpy.workload import (
    build_workload_descriptor,
    canonical_json_bytes,
    structural_hash,
    workload_id,
    workload_identity_block,
)


def test_structural_hash_stable() -> None:
    a = build_workload_descriptor(
        harness_id="nrl_vs_cpp",
        neurons=1024,
        iterations=128,
        reps=4,
        threshold=8,
        profiles=["omega", "sovereign"],
        nrl_version="nrl test",
    )
    b = build_workload_descriptor(
        harness_id="nrl_vs_cpp",
        neurons=1024,
        iterations=128,
        reps=4,
        threshold=8,
        profiles=["sovereign", "omega"],
        nrl_version="nrl test",
    )
    assert structural_hash(a) == structural_hash(b)
    assert canonical_json_bytes(a) == canonical_json_bytes(b)


def test_profiles_order_matters_for_hash() -> None:
    a = build_workload_descriptor(
        harness_id="h",
        neurons=4,
        iterations=2,
        reps=1,
        threshold=8,
        profiles=["a"],
    )
    b = build_workload_descriptor(
        harness_id="h",
        neurons=4,
        iterations=2,
        reps=1,
        threshold=8,
        profiles=["a", "b"],
    )
    assert structural_hash(a) != structural_hash(b)


def test_workload_identity_block() -> None:
    d = build_workload_descriptor(
        harness_id="nrl_vs_cpp",
        neurons=8,
        iterations=2,
        reps=1,
        threshold=8,
        profiles=["sovereign"],
    )
    blk = workload_identity_block(d)
    assert blk["workload_id"] == workload_id("nrl_vs_cpp", blk["structural_hash"])
    assert blk["descriptor"]["schema_id"] == "nrl.workload_descriptor.v1"
