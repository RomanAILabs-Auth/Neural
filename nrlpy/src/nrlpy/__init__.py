"""NRLPy public API."""

from __future__ import annotations

from . import native
from .compat import NRLRuntime, llm_globals
from .runtime import (
    BenchCliResult,
    BraincoreInt4InplaceResult,
    BraincoreInt4Result,
    active_variant,
    assimilate_cli,
    bench_cli,
    braincore_int4,
    braincore_int4_inplace,
    braincore_packed_bytes,
    features,
    fnv1a64_packed,
    nrl_binary_path,
    run_nrl_file,
    version,
)

__all__ = [
    "BenchCliResult",
    "BraincoreInt4InplaceResult",
    "BraincoreInt4Result",
    "NRLRuntime",
    "active_variant",
    "assimilate_cli",
    "bench_cli",
    "braincore_int4",
    "braincore_int4_inplace",
    "braincore_packed_bytes",
    "features",
    "fnv1a64_packed",
    "llm_globals",
    "native",
    "nrl_binary_path",
    "run_nrl_file",
    "version",
]
