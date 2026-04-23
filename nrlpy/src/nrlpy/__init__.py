"""NRLPy public API."""

from __future__ import annotations

from . import native
from .compat import NRLRuntime, llm_globals
from .learn_store import LearnStats, LearnStore, default_store
from .plasticity import plasticity_snapshot
from .shadow import control_plane_probe_identity, shadow_bench_pair
from .seamless import fabric_pulse, injected_globals, is_prime, next_prime
from .workload import (
    build_workload_descriptor,
    canonical_json_bytes,
    structural_hash,
    workload_id,
    workload_identity_block,
)
from .runtime import (
    BenchCliResult,
    BraincoreInt4InplaceResult,
    BraincoreInt4Result,
    ControlPreferences,
    active_variant,
    assimilate_cli,
    bench_cli,
    braincore_int4,
    braincore_int4_inplace,
    braincore_packed_bytes,
    control_audit_log_path,
    control_hints_active,
    control_preferences_path,
    features,
    fnv1a64_packed,
    load_control_preferences,
    nrl_binary_path,
    resolve_bench_profile_with_control_hints,
    run_nrl_file,
    version,
)

__all__ = [
    "BenchCliResult",
    "BraincoreInt4InplaceResult",
    "BraincoreInt4Result",
    "ControlPreferences",
    "NRLRuntime",
    "build_workload_descriptor",
    "canonical_json_bytes",
    "active_variant",
    "assimilate_cli",
    "bench_cli",
    "braincore_int4",
    "control_audit_log_path",
    "control_hints_active",
    "control_preferences_path",
    "braincore_int4_inplace",
    "braincore_packed_bytes",
    "fabric_pulse",
    "features",
    "fnv1a64_packed",
    "load_control_preferences",
    "injected_globals",
    "is_prime",
    "LearnStats",
    "LearnStore",
    "default_store",
    "llm_globals",
    "native",
    "next_prime",
    "nrl_binary_path",
    "plasticity_snapshot",
    "control_plane_probe_identity",
    "shadow_bench_pair",
    "resolve_bench_profile_with_control_hints",
    "run_nrl_file",
    "structural_hash",
    "version",
    "workload_id",
    "workload_identity_block",
]
