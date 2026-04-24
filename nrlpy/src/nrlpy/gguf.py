# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""GGUF runner — NRL as execution supervisor, libllama as numerics library.

Implements the P1 contract from ``docs/nrl_gguf_runner_architecture.md``:

* Parses ``.nrl`` manifest v1 files (schema-gated extension of v0).
* Loads the GGUF model through ``llama-cpp-python`` (lazy import).
* Streams tokens with zero subprocess calls in the decode hot loop.
* Emits the four-metric TPS block (``executed_tps``, ``virtual_tps``,
  ``cache_tps``, ``effective_tps``) with honest §15 accounting.
* Runs one ``nrl assimilate`` attestation at init (not per token).
* On-disk FNV-1a64 muscle-memory cache under ``$NRL_ROOT/cache/mm/``.
* Appends ``nrl.gguf_run.v1`` events to the immune evidence log.

L1 layer/expert/KV gating (``gate_*`` manifest keys) parses today but has no
effect until P2 lands. When P2 lands, this module is the only place that
consumes the gate output; the manifest format does not change.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import runtime
from . import ladder as _ladder
from . import lmo as _lmo
from .evidence import append_jsonl
from .gates import GateReport, PrefillGate

MANIFEST_SCHEMA_V1 = "nrl.manifest.v1"
EVIDENCE_SCHEMA = "nrl.gguf_run.v1"
MUSCLE_MEMORY_MAGIC = b"NRLMM1\x00\x00"  # 8-byte header magic
DEFAULT_MM_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB, aligned with LearnStore default

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}
_VALID_PROFILES = {
    "sovereign",
    "adaptive",
    "war-drive",
    "zpm",
    "automatic",
    "omega",
    "omega-hybrid",
}
_VALID_GATE_POLICIES = {"none", "omega", "omega-hybrid", "zpm"}
_VALID_MUSCLE_MODES = {"on", "off", "replay-only"}
_VALID_PREFILL_CACHE = {"off", "session"}
_VALID_CHAT_FORMATS = {"none", "chatml", "phi3", "llama2"}
_VALID_BENCH_CLASSES = {"A", "B"}
_VALID_TELEMETRY = {"summary", "per_token"}
_VALID_COHERENCE_LANES = {"fast-stable", "fast-balanced", "max-throughput"}
_VALID_RUNNER_BACKENDS = {
    "python",
    "native",
    "native_strict",
    "native_full",
    "native_full_strict",
}
_VALID_KV_CACHE_DTYPES = {
    "",  # default (f16)
    "f32", "f16", "bf16",
    "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class GgufManifest:
    """Typed view of a parsed ``.nrl`` manifest v1 (``mode = gguf_run``)."""

    schema: str = MANIFEST_SCHEMA_V1
    mode: str = "gguf_run"
    profile: str = "sovereign"

    model: str = ""
    model_sha256: str = ""
    prompt: str = ""
    prompt_file: str = ""
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 0
    n_ctx: int = 2048
    n_threads: int = 0
    n_batch: int = 512
    chat_format: str = "none"

    gate_layer_policy: str = "none"
    gate_expert_policy: str = "none"
    gate_kv_policy: str = "none"
    gate_min_active: int = 4
    gate_wake_rate: float = 0.25

    respect_control_hints: bool = True
    muscle_memory: str = "on"
    muscle_memory_key_fields: str = "model_sha256,prompt,sampler,seed,n_ctx"
    # Phase 11 — cache-scoping for multi-turn chat. When non-empty, the
    # muscle-memory and ZPM keys are derived from ``chat_intent`` (the
    # latest user message) instead of the full rendered ``prompt`` that
    # carries accumulated conversation history. The decoder still sees
    # the full ``prompt`` for coherent generation, but cache lookups
    # match on intent so rephrased follow-ups and repeated questions
    # actually hit R0/R1. Empty string = use ``prompt`` (legacy single-
    # shot behaviour), which is also what ``nrlpy run`` does for one-
    # shot inference.
    chat_intent: str = ""
    # Phase 13 — KV-cache prefix reuse across chat turns. When ``True``
    # (set by the chat REPL, never by one-shot callers), libllama is
    # asked to *not* reset its KV cache between calls. The internal
    # ``Llama._input_ids`` prefix matcher then re-evaluates only the
    # token tail that differs from the previous turn. This is the
    # honest ``2 × 64`` computation win: the shared transcript prefix
    # was already multiplied through every transformer layer on the
    # prior turn, so we don't redo it. Default ``False`` keeps
    # ``nrlpy run``, the bench-wps release gates, and every golden
    # fixture byte-identical to previous phases.
    chat_kv_reuse: bool = False
    evidence_log: str = ""
    telemetry_granularity: str = "summary"
    benchmark_class: str = "B"

    kv_cache_dtype: str = ""  # "" = engine default (f16); or one of _VALID_KV_CACHE_DTYPES
    no_repack: bool = False  # CPU_REPACK bypass for tight-RAM hosts

    # P2-Active simulation hinge (see docs/nrl_gguf_runner_architecture.md §1.0).
    # This is an *explicit* numeric override of TpsReport.gate_skip_ratio BEFORE
    # the real libllama layer gate ships in P3. It exists so the downstream math,
    # banner, evidence-log, and golden-harness invariants can be exercised in CI
    # on a known flipped-hinge input. It is *not* a performance claim — banner
    # and evidence log explicitly label it as simulation. Ignored when <= 0.
    gate_skip_ratio_override: float = 0.0

    # P2-Active structural gate. ``off`` = never attach a gate. ``session`` =
    # the caller (REPL / golden harness) may attach a PrefillGate to run_gguf
    # whose shared-prefix report drives gate_skip_ratio. A live PrefillGate
    # always wins over gate_skip_ratio_override (structural > numeric fixture).
    prefill_cache: str = "off"

    # ZPM (Zero-Point Mapping) identity resolver — Plane A.5.
    # When ``zpm_nullspace`` is True, run_gguf consults an on-disk topological
    # index before muscle-memory and libllama: if the turn's 256-bit anchor
    # state is within ``zpm_threshold_bits`` Hamming distance of a stored
    # unity state, serve the stored reply (gate_source="zpm_nullspace").
    # Threshold 0 = exact-only (near-miss behaves like a full miss).
    zpm_nullspace: bool = False
    zpm_threshold_bits: int = 0

    # Phase 5-EG — Rung R2 (Omega Native Resolve), SHADOW mode only.
    # ``coherence_lane`` maps to the §4.5 whitelists:
    #   * fast-stable     → R0, R1, R3, R5        (Class-A legal, default)
    #   * fast-balanced   → R0, R1, R3, R4, R5    (layer gate engaged)
    #   * max-throughput  → R0, R1, R2, R3, R4, R5 (R2 shadow today; active in P6-EG)
    # R2 shadow runs in a background thread and *never* emits tokens; its
    # result is recorded in the evidence log under
    # ``gate_source = "omega_resolve_shadow"``.
    coherence_lane: str = "fast-stable"
    r2_shadow_enabled: bool = True
    omega_budget_ms: float = 2.0
    omega_candidates: int = 4
    omega_iterations: int = 3
    omega_shadow_join_timeout_s: float = 0.5

    # Phase 7-EG — Native C runner. ``"python"`` (default) keeps the
    # current pure-Python Resolution Ladder. ``"native"`` routes the §4.2
    # rung-selection decision through the C dispatcher in
    # ``engine/src/ladder_native.c`` and the libllama bridge in
    # ``engine/src/llama_bridge.c``. The native path is byte-parity with
    # the Python path during the parity-gate window: deterministic
    # R0/R1/R2 candidate computation stays in Python, only dispatch +
    # bridge call move to C. If ``nrlpy._core`` is not built with the
    # Phase 7-EG bindings the runner silently falls back to the Python
    # ladder; pass ``runner_backend = "native_strict"`` to fail loudly
    # instead.
    runner_backend: str = "python"

    manifest_path: str = ""

    def sampler_fingerprint(self) -> str:
        """Stable fingerprint of sampler-visible params, for muscle-memory keys."""
        return (
            f"t={self.temperature:.6f};p={self.top_p:.6f};k={self.top_k};"
            f"r={self.repeat_penalty:.6f};n={self.n_ctx};b={self.n_batch};"
            f"cf={self.chat_format};prof={self.profile}"
        )


def _parse_bool(value: str, key: str, line_no: int) -> bool:
    low = value.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    raise ManifestError(f"line {line_no}: {key!r} expected bool, got {value!r}")


def _parse_int(value: str, key: str, line_no: int) -> int:
    try:
        return int(value.strip(), 10)
    except ValueError as e:
        raise ManifestError(f"line {line_no}: {key!r} expected int, got {value!r}") from e


def _parse_float(value: str, key: str, line_no: int) -> float:
    try:
        return float(value.strip())
    except ValueError as e:
        raise ManifestError(f"line {line_no}: {key!r} expected float, got {value!r}") from e


def _parse_string(value: str) -> str:
    s = value.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


class ManifestError(ValueError):
    """Raised for any manifest parse / validation failure."""


def load_manifest(path: str | Path) -> GgufManifest:
    """Parse a ``.nrl`` v1 manifest from disk. Strict: unknown keys fail the parse."""
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest not found: {p}")
    text = p.read_text(encoding="utf-8")
    manifest = parse_manifest_text(text, manifest_dir=p.parent)
    manifest.manifest_path = str(p.resolve())
    return manifest


def parse_manifest_text(text: str, manifest_dir: Path | None = None) -> GgufManifest:
    """Parse manifest text. ``manifest_dir`` scopes relative ``model`` / ``prompt_file`` paths."""
    m = GgufManifest()
    has_schema = False
    seen_prompt = False
    seen_prompt_file = False
    _ = manifest_dir  # currently only used by the caller for path resolution

    for raw_line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ManifestError(f"line {raw_line_no}: missing '=' in {raw!r}")
        key_raw, _, value_raw = line.partition("=")
        key = key_raw.strip()
        value = value_raw

        if key == "schema":
            sv = _parse_string(value)
            if sv != MANIFEST_SCHEMA_V1:
                raise ManifestError(
                    f"line {raw_line_no}: unknown schema {sv!r} "
                    f"(expected {MANIFEST_SCHEMA_V1})"
                )
            m.schema = sv
            has_schema = True
            continue

        # All keys below are v1-only.
        if key == "mode":
            sv = _parse_string(value)
            if sv not in {"run", "bench", "gguf_run"}:
                raise ManifestError(f"line {raw_line_no}: invalid mode {sv!r}")
            m.mode = sv
        elif key == "profile":
            sv = _parse_string(value)
            if sv not in _VALID_PROFILES:
                raise ManifestError(f"line {raw_line_no}: invalid profile {sv!r}")
            m.profile = sv
        elif key == "model":
            m.model = _parse_string(value)
        elif key == "model_sha256":
            sv = _parse_string(value).lower()
            if sv and not re.fullmatch(r"[0-9a-f]{64}", sv):
                raise ManifestError(f"line {raw_line_no}: model_sha256 must be 64 hex chars")
            m.model_sha256 = sv
        elif key == "prompt":
            m.prompt = _parse_string(value)
            seen_prompt = True
        elif key == "prompt_file":
            m.prompt_file = _parse_string(value)
            seen_prompt_file = bool(m.prompt_file)
        elif key == "max_tokens":
            m.max_tokens = _parse_int(value, key, raw_line_no)
        elif key == "temperature":
            m.temperature = _parse_float(value, key, raw_line_no)
        elif key == "top_p":
            m.top_p = _parse_float(value, key, raw_line_no)
        elif key == "top_k":
            m.top_k = _parse_int(value, key, raw_line_no)
        elif key == "repeat_penalty":
            m.repeat_penalty = _parse_float(value, key, raw_line_no)
        elif key == "seed":
            m.seed = _parse_int(value, key, raw_line_no)
        elif key == "n_ctx":
            m.n_ctx = _parse_int(value, key, raw_line_no)
        elif key == "n_threads":
            m.n_threads = _parse_int(value, key, raw_line_no)
        elif key == "n_batch":
            m.n_batch = _parse_int(value, key, raw_line_no)
        elif key == "chat_format":
            sv = _parse_string(value)
            if sv not in _VALID_CHAT_FORMATS:
                raise ManifestError(
                    f"line {raw_line_no}: chat_format must be one of {sorted(_VALID_CHAT_FORMATS)}"
                )
            m.chat_format = sv
        elif key == "gate_layer_policy":
            sv = _parse_string(value)
            if sv not in _VALID_GATE_POLICIES:
                raise ManifestError(f"line {raw_line_no}: invalid gate_layer_policy {sv!r}")
            m.gate_layer_policy = sv
        elif key == "gate_expert_policy":
            sv = _parse_string(value)
            if sv not in _VALID_GATE_POLICIES:
                raise ManifestError(f"line {raw_line_no}: invalid gate_expert_policy {sv!r}")
            m.gate_expert_policy = sv
        elif key == "gate_kv_policy":
            sv = _parse_string(value)
            if sv not in _VALID_GATE_POLICIES:
                raise ManifestError(f"line {raw_line_no}: invalid gate_kv_policy {sv!r}")
            m.gate_kv_policy = sv
        elif key == "gate_min_active":
            m.gate_min_active = _parse_int(value, key, raw_line_no)
        elif key == "gate_wake_rate":
            m.gate_wake_rate = _parse_float(value, key, raw_line_no)
        elif key == "respect_control_hints":
            m.respect_control_hints = _parse_bool(value, key, raw_line_no)
        elif key == "muscle_memory":
            sv = _parse_string(value)
            if sv not in _VALID_MUSCLE_MODES:
                valid = sorted(_VALID_MUSCLE_MODES)
                raise ManifestError(
                    f"line {raw_line_no}: muscle_memory must be one of {valid}"
                )
            m.muscle_memory = sv
        elif key == "muscle_memory_key_fields":
            m.muscle_memory_key_fields = _parse_string(value)
        elif key == "evidence_log":
            m.evidence_log = _parse_string(value)
        elif key == "telemetry_granularity":
            sv = _parse_string(value)
            if sv not in _VALID_TELEMETRY:
                raise ManifestError(
                    f"line {raw_line_no}: telemetry_granularity must be in "
                    f"{sorted(_VALID_TELEMETRY)}"
                )
            m.telemetry_granularity = sv
        elif key == "benchmark_class":
            sv = _parse_string(value).upper()
            if sv not in _VALID_BENCH_CLASSES:
                raise ManifestError(f"line {raw_line_no}: benchmark_class must be A or B")
            m.benchmark_class = sv
        elif key == "kv_cache_dtype":
            sv = _parse_string(value).lower()
            if sv not in _VALID_KV_CACHE_DTYPES:
                valid = sorted(x for x in _VALID_KV_CACHE_DTYPES if x)
                raise ManifestError(
                    f"line {raw_line_no}: kv_cache_dtype must be one of {valid} or empty"
                )
            m.kv_cache_dtype = sv
        elif key == "no_repack":
            m.no_repack = _parse_bool(value, key, raw_line_no)
        elif key == "gate_skip_ratio_override":
            fv = _parse_float(value, key, raw_line_no)
            if not (0.0 <= fv < 1.0):
                raise ManifestError(
                    f"line {raw_line_no}: gate_skip_ratio_override must be in [0.0, 1.0); "
                    f"got {fv!r}"
                )
            m.gate_skip_ratio_override = fv
        elif key == "prefill_cache":
            sv = _parse_string(value).lower()
            if sv not in _VALID_PREFILL_CACHE:
                raise ManifestError(
                    f"line {raw_line_no}: prefill_cache must be one of "
                    f"{sorted(_VALID_PREFILL_CACHE)}"
                )
            m.prefill_cache = sv
        elif key == "coherence_lane":
            sv = _parse_string(value).lower()
            if sv not in _VALID_COHERENCE_LANES:
                raise ManifestError(
                    f"line {raw_line_no}: coherence_lane must be one of "
                    f"{sorted(_VALID_COHERENCE_LANES)}; got {sv!r}"
                )
            m.coherence_lane = sv
        elif key == "r2_shadow_enabled":
            m.r2_shadow_enabled = _parse_bool(value, key, raw_line_no)
        elif key == "omega_budget_ms":
            fv = _parse_float(value, key, raw_line_no)
            if fv < 0.0:
                raise ManifestError(
                    f"line {raw_line_no}: omega_budget_ms must be >= 0.0"
                )
            m.omega_budget_ms = fv
        elif key == "omega_candidates":
            iv = _parse_int(value, key, raw_line_no)
            if iv < 1:
                raise ManifestError(
                    f"line {raw_line_no}: omega_candidates must be >= 1"
                )
            m.omega_candidates = iv
        elif key == "omega_iterations":
            iv = _parse_int(value, key, raw_line_no)
            if iv < 1:
                raise ManifestError(
                    f"line {raw_line_no}: omega_iterations must be >= 1"
                )
            m.omega_iterations = iv
        elif key == "runner_backend":
            sv = _parse_string(value).lower()
            if sv not in _VALID_RUNNER_BACKENDS:
                raise ManifestError(
                    f"line {raw_line_no}: runner_backend must be one of "
                    f"{sorted(_VALID_RUNNER_BACKENDS)}; got {sv!r}"
                )
            m.runner_backend = sv
        else:
            raise ManifestError(f"line {raw_line_no}: unknown key {key!r}")

    if not has_schema:
        raise ManifestError(
            f"missing 'schema = {MANIFEST_SCHEMA_V1}' directive "
            "(v0 files are parsed by the native CLI, not this loader)"
        )
    if m.mode == "gguf_run" and not m.model:
        raise ManifestError("mode=gguf_run requires 'model = <path>.gguf'")
    if seen_prompt and seen_prompt_file:
        raise ManifestError("'prompt' and 'prompt_file' are mutually exclusive")
    if m.benchmark_class == "A" and m.seed == 0:
        raise ManifestError("benchmark_class=A requires a non-zero 'seed' for replay lock")
    return m


# --------------------------------------------------------------------------- #
# Model-file hashing
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 hex of a file (read in 1 MiB chunks)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Muscle memory (on-disk FNV-1a64 keyed cache)
# --------------------------------------------------------------------------- #


def _muscle_memory_root() -> Path:
    env = os.environ.get("NRL_ROOT")
    base = Path(env) if env else Path.cwd()
    return base / "cache" / "mm"


def _zpm_root() -> Path:
    """Root directory for ZPM topological indexes. Parallels ``cache/mm``.

    ``$NRL_ROOT/cache/zpm/<model_sha>/index.bin`` holds the near-match table
    used by the Plane-A.5 identity resolver (see :mod:`nrlpy.zpm`).
    """
    env = os.environ.get("NRL_ROOT")
    base = Path(env) if env else Path.cwd()
    return base / "cache" / "zpm"


def _zpm_index_path(model_sha256: str) -> Path:
    return _zpm_root() / (model_sha256 or "unknown") / "index.bin"


def _zpm_index_load(model_sha256: str) -> Any:
    """Load the on-disk ZPM index, honoring a Phase-15 one-shot prefetch."""
    from . import zpm as _zpm  # noqa: PLC0415
    from . import zpm_persist as _zp  # noqa: PLC0415

    pf = _zpm.take_prefetched_zpm_index(model_sha256)
    if pf is not None:
        return pf
    p = _zpm_index_path(model_sha256)
    try:
        _zp.recover_zpm_for_model(model_sha256 or "unknown", p)
    except Exception:
        pass
    return _zpm.ZpmIndex.load(p) if p.is_file() else _zpm.ZpmIndex()


def _phase15_gate_enabled() -> bool:
    v = os.environ.get("NRL_PHASE15", "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def _phase15_pre_turn(
    preloaded_llm: Any,
    manifest: "GgufManifest",
    prompt: str,
    model_sha: str,
    intent_anchor_bytes: bytes,
) -> int:
    """Phase 15-EG warm-restart when consecutive R5 drift exceeds 3 tokens.

    Returns ``1`` if :func:`zpm.prime` ran this turn, else ``0``. Does not
    touch :class:`TpsReport` fields — evidence-only side channel.
    """
    if preloaded_llm is None or not _phase15_gate_enabled():
        return 0
    streak = int(getattr(preloaded_llm, "_nrl_drift_r5_tokens", 0))
    if streak <= 3:
        return 0
    from . import zpm as _zpm  # noqa: PLC0415

    ids = list(getattr(preloaded_llm, "_input_ids", ()) or ())
    tail = [int(x) for x in ids[-4:]]
    try:
        _zpm.prime(
            model_sha,
            bytes(intent_anchor_bytes),
            tail,
            _zpm_index_path(model_sha),
        )
    except Exception:
        pass
    preloaded_llm._nrl_drift_r5_tokens = 0
    return 1


def _phase15_post_turn_native(preloaded_llm: Any, served_rung: int, tokens: int) -> None:
    if preloaded_llm is None or not _phase15_gate_enabled():
        return
    from . import native_ladder as _nl  # noqa: PLC0415

    if served_rung in (
        _nl.RUNG_R0_MUSCLE_MEMORY,
        _nl.RUNG_R1_ZPM,
        _nl.RUNG_R2_OMEGA_ACTIVE,
    ):
        preloaded_llm._nrl_drift_r5_tokens = 0
    elif served_rung == _nl.RUNG_R5_LIBLLAMA:
        cur = int(getattr(preloaded_llm, "_nrl_drift_r5_tokens", 0))
        preloaded_llm._nrl_drift_r5_tokens = cur + int(max(0, tokens))


def _phase15_post_turn_python(
    preloaded_llm: Any, *, lattice: bool, tokens: int
) -> None:
    if preloaded_llm is None or not _phase15_gate_enabled():
        return
    if lattice:
        preloaded_llm._nrl_drift_r5_tokens = 0
    else:
        cur = int(getattr(preloaded_llm, "_nrl_drift_r5_tokens", 0))
        preloaded_llm._nrl_drift_r5_tokens = cur + int(max(0, tokens))


def _cache_scope_text(manifest: "GgufManifest", prompt: str) -> str:
    """Phase 11 — the text that seeds MM/ZPM keys.

    Chat flows set ``manifest.chat_intent`` to the latest user turn so
    repeated / rephrased questions match despite accumulating history
    in ``prompt``. Single-shot flows leave ``chat_intent`` empty and
    fall back to the full rendered prompt (legacy behavior, preserves
    every existing evidence log and golden fixture).
    """
    intent = (manifest.chat_intent or "").strip()
    return intent if intent else prompt


def _zpm_anchor_bytes(manifest: "GgufManifest", prompt: str) -> bytes:
    """Closed set of bytes the ZPM 256-bit anchor is computed over.

    Keep this list small + stable: changing it invalidates every stored
    unity state on disk. Fields match the muscle-memory key (so an MM hit
    is always ≤ the corresponding ZPM exact hit) *plus* the fully-rendered
    prompt (so the ZPM state is a strict refinement, never looser than MM).
    """
    parts = [
        (manifest.model_sha256 or "unknown").encode("utf-8"),
        _cache_scope_text(manifest, prompt).encode("utf-8"),
        manifest.sampler_fingerprint().encode("utf-8"),
        str(manifest.seed).encode("ascii"),
        str(manifest.max_tokens).encode("ascii"),
    ]
    return b"\x1e".join(parts)


def _muscle_memory_key(manifest: GgufManifest) -> int:
    """FNV-1a64 over the closed set of determinism-bearing fields."""
    parts = [
        manifest.model_sha256,
        _cache_scope_text(manifest, manifest.prompt),
        manifest.sampler_fingerprint(),
        str(manifest.seed),
        str(manifest.max_tokens),
    ]
    blob = "\u001f".join(parts).encode("utf-8")
    return runtime.fnv1a64_packed(blob)


def _muscle_memory_path(manifest: GgufManifest, key_fnv: int) -> Path:
    model_tag = manifest.model_sha256 or "unknown"
    return _muscle_memory_root() / model_tag / f"{key_fnv:016x}.mm"


@dataclass
class MuscleMemoryHit:
    """Result of a muscle-memory cache read."""

    text: str
    tokens: int
    cache_read_s: float
    key_fnv1a64: int


def muscle_memory_lookup(manifest: GgufManifest) -> MuscleMemoryHit | None:
    """Return a cache hit for this manifest or ``None``. Honors ``muscle_memory`` mode."""
    if manifest.muscle_memory == "off":
        return None
    key = _muscle_memory_key(manifest)
    path = _muscle_memory_path(manifest, key)
    if not path.is_file():
        return None
    t0 = time.perf_counter()
    try:
        with path.open("rb") as f:
            head = f.read(16)
            if len(head) < 16 or head[:8] != MUSCLE_MEMORY_MAGIC:
                return None
            token_count, text_bytes = struct.unpack("<II", head[8:16])
            if token_count == 0 or text_bytes == 0:
                return None
            body = f.read(text_bytes)
    except OSError:
        return None
    elapsed = time.perf_counter() - t0
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return MuscleMemoryHit(
        text=text, tokens=int(token_count), cache_read_s=elapsed, key_fnv1a64=key
    )


def muscle_memory_store(manifest: GgufManifest, text: str, tokens: int) -> Path | None:
    """Write a cache entry. Returns the path, or ``None`` when writes are disabled."""
    if manifest.muscle_memory != "on":
        return None
    key = _muscle_memory_key(manifest)
    path = _muscle_memory_path(manifest, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    header = MUSCLE_MEMORY_MAGIC + struct.pack("<II", tokens, len(body))
    with tmp.open("wb") as f:
        f.write(header)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def _promote_rescue_to_r0(
    manifest: "GgufManifest",
    *,
    prompt: str,
    text: str,
    tokens: int,
    model_sha: str,
    wall_s: float,
) -> None:
    """Phase 11 — R2-rescue → R0 Muscle Memory promotion.

    When an R2 turn is served via the n-gram rescue path
    (``omega_shadow.ngram_rescued=True``) the served reply came from a
    stored ZPM entry keyed on the *original* prompt's state. The
    rephrased prompt itself was never in R0 or R1 before this turn —
    the rescue bridge is what admitted it. Without promotion the same
    rephrasing would re-incur the full Omega probe (12 ms budget) on
    every subsequent ask.

    This helper writes the rescue-served reply into R0 under the
    **current** prompt's muscle-memory key, and into R1 under the
    current prompt's ZPM state. After promotion, the second and
    subsequent asks of the same rephrased prompt hit R0 at replay
    speed (microseconds) instead of going through R2 again.

    Closed-write discipline: the helper silently no-ops when
    ``muscle_memory`` is off, ``tokens`` is 0, or ``text`` is empty —
    the same gate the R5 writeback path applies. ZPM promotion also
    requires ``zpm_nullspace = True``.

    Never raises: storage failures are logged to stderr so the hot
    path keeps serving.
    """
    if manifest.muscle_memory != "on" or tokens <= 0 or not text:
        return
    try:
        muscle_memory_store(manifest, text, tokens)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[nrl.gguf] R2 rescue → R0 promotion (MM) skipped "
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return
    if not manifest.zpm_nullspace:
        return
    try:
        from . import zpm as _zpm  # noqa: PLC0415

        z_state = _zpm.anchor(_zpm_anchor_bytes(manifest, prompt))
        z_path = _zpm_index_path(model_sha)
        idx = _zpm_index_load(model_sha)
        existing_hit, existing_entry = idx.lookup(z_state, threshold_bits=0)
        if existing_entry is None or not existing_hit.exact:
            from . import zpm_persist as _zp  # noqa: PLC0415

            ent = _zpm.ZpmEntry(
                state=z_state,
                reply_text=text,
                tokens=tokens,
                wall_s_at_write=float(wall_s),
                metadata={
                    "model": Path(manifest.model).name,
                    "chat_format": manifest.chat_format,
                    "seed": str(manifest.seed),
                    "prompt_head": prompt[:256],
                    "promoted_from": "r2_ngram_rescue",
                },
            )
            idx.add(ent)
            _zp.persist_zpm_entry(model_sha, z_path, idx, ent)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[nrl.gguf] R2 rescue → R1 promotion (ZPM) skipped "
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# TPS report
# --------------------------------------------------------------------------- #


@dataclass
class TpsReport:
    """Four-metric TPS accounting (see ``docs/nrl_gguf_runner_architecture.md`` §1).

    The two ``gate_*`` fields are **applied** skip ratios — the fraction of layer
    or KV work the active gate caused libllama to skip. They are 0.0 in P1 and
    P2-Shadow; they diverge from 0 only when a gate policy is active and actually
    elides numerics (P2-Active+). This is the honesty hinge: observational
    NRL-lattice skip_ratio lives on :class:`NrlLatticeObservation`, not here.
    """

    executed_tokens: int = 0
    executed_wall_s: float = 0.0
    cache_tokens: int = 0
    cache_wall_s: float = 0.0

    # Applied skip ratios (libllama work that was elided). 0 in P1 + P2-Shadow.
    gate_skip_ratio: float = 0.0
    gate_kv_skip_ratio: float = 0.0

    # Filled after computation so consumers see the exact numbers that were printed.
    executed_tps: float = 0.0
    virtual_tps: float = 0.0
    cache_tps: float = 0.0
    effective_tps: float = 0.0

    # Non-native conditions honest-accounting requires the banner to surface.
    stream_chunk_ms: float = 0.0  # >0 = paced stream, not a native throughput claim

    def finalize(self) -> None:
        self.executed_tps = (
            self.executed_tokens / self.executed_wall_s if self.executed_wall_s > 0 else 0.0
        )
        denom = 1.0 - self.gate_skip_ratio
        self.virtual_tps = self.executed_tps / denom if denom > 1e-12 else self.executed_tps
        self.cache_tps = (
            self.cache_tokens / self.cache_wall_s if self.cache_wall_s > 0 else 0.0
        )
        total_tokens = self.executed_tokens + self.cache_tokens
        total_wall = self.executed_wall_s + self.cache_wall_s
        self.effective_tps = total_tokens / total_wall if total_wall > 0 else 0.0

    def headline(self) -> tuple[str, float]:
        """Pick the banner metric and return ``(name, value)`` — always labeled."""
        candidates = (
            ("virtual_tps", self.virtual_tps),
            ("effective_tps", self.effective_tps),
            ("cache_tps", self.cache_tps),
            ("executed_tps", self.executed_tps),
        )
        name, value = max(candidates, key=lambda kv: kv[1])
        return name, value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WordRateReport:
    """Words/sec views derived from measured TPS and emitted text."""

    word_count: int = 0
    words_per_token: float = 0.0
    executed_wps: float = 0.0
    virtual_wps: float = 0.0
    cache_wps: float = 0.0
    effective_wps: float = 0.0


# --------------------------------------------------------------------------- #
# NRL attestation (one subprocess call at init)
# --------------------------------------------------------------------------- #


@dataclass
class NrlAttestation:
    """Proof that NRL's INT4 kernels ran on this host during this session."""

    variant: str = ""
    checksum_fnv1a64: int = 0
    elapsed_s: float = 0.0
    version: str = ""
    executed_gops: float = 0.0
    virtual_gops: float = 0.0
    skip_ratio: float = 0.0
    profile: str = ""
    available: bool = False


def nrl_attest(
    profile: str = "omega",
    *,
    neurons: int = 65_536,
    iterations: int = 64,
) -> NrlAttestation:
    """Run a short ``nrl bench`` probe and parse results. One-shot, not per-token."""
    att = NrlAttestation(profile=profile)
    try:
        att.version = runtime.version()
    except Exception:
        att.version = ""
    try:
        res = runtime.bench_cli(
            neurons=neurons,
            iterations=iterations,
            reps=1,
            threshold=8,
            profile=profile,
        )
        att.variant = res["variant"]
        att.executed_gops = float(res["executed_gops"])
        att.virtual_gops = float(res["virtual_gops"])
        att.skip_ratio = float(res["skip_ratio"])
        att.elapsed_s = float(res["elapsed_s"])
        att.available = True
    except Exception:
        att.available = False
    return att


# --------------------------------------------------------------------------- #
# NRL lattice observation (P2-Shadow: advisory, does NOT inflate TPS)
# --------------------------------------------------------------------------- #


@dataclass
class NrlLatticeObservation:
    """What the NRL lattice *would* skip under a balanced gate, if one were active.

    This is **advisory information**, captured by running ``omega-hybrid`` on the
    NRL lattice in a background thread during decode. It is never used to inflate
    :class:`TpsReport` values. Its purpose is to give the operator a concrete
    preview of the gate-skip signal that P2-Active wiring would consume.

    Honesty contract:

    * ``skip_ratio`` here is a property of the lattice under its own workload,
      not a property of libllama's decode path.
    * Until P2-Active wires this into libllama's forward pass,
      :attr:`TpsReport.gate_skip_ratio` stays ``0.0`` and
      ``virtual_tps == executed_tps`` on fresh-decode runs.
    """

    profile: str = "omega-hybrid"
    available: bool = False
    skip_ratio: float = 0.0  # lattice-level, advisory only
    executed_gops: float = 0.0
    virtual_gops: float = 0.0
    elapsed_s: float = 0.0
    note: str = (
        "advisory; not applied to token flow in P1 / P2-Shadow"
    )


def _collect_lattice_observation(
    profile: str,
    *,
    neurons: int,
    iterations: int,
) -> NrlLatticeObservation:
    obs = NrlLatticeObservation(profile=profile)
    try:
        res = runtime.bench_cli(
            neurons=neurons,
            iterations=iterations,
            reps=1,
            threshold=8,
            profile=profile,
        )
        obs.skip_ratio = float(res["skip_ratio"])
        obs.executed_gops = float(res["executed_gops"])
        obs.virtual_gops = float(res["virtual_gops"])
        obs.elapsed_s = float(res["elapsed_s"])
        obs.available = True
    except Exception:
        obs.available = False
    return obs


class _LatticeObservationThread(threading.Thread):
    """Background ``omega-hybrid`` probe. Never blocks the decode hot loop.

    If the probe is slower than decode, we time out on ``join()`` and report an
    unavailable observation rather than stalling the user-visible stream.
    """

    def __init__(self, profile: str, neurons: int, iterations: int) -> None:
        super().__init__(daemon=True, name="nrl-lattice-observation")
        self._profile = profile
        self._neurons = neurons
        self._iterations = iterations
        self._result: NrlLatticeObservation | None = None
        self._lock = threading.Lock()

    def run(self) -> None:
        obs = _collect_lattice_observation(
            self._profile,
            neurons=self._neurons,
            iterations=self._iterations,
        )
        with self._lock:
            self._result = obs

    def result(self, timeout_s: float) -> NrlLatticeObservation:
        self.join(timeout=timeout_s)
        with self._lock:
            if self._result is None:
                return NrlLatticeObservation(
                    profile=self._profile,
                    available=False,
                    note=f"observation probe did not finish within {timeout_s:.2f}s",
                )
            return self._result


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class GgufRunResult:
    """Payload returned by :func:`run_gguf`. Stable for tests and external tools."""

    text: str
    tokens: int
    tps: TpsReport
    cache_hit: bool
    nrl_attestation: NrlAttestation
    manifest: GgufManifest
    model_sha256: str
    evidence_path: str = ""
    lattice_observation: NrlLatticeObservation = field(
        default_factory=NrlLatticeObservation
    )
    artifact: dict[str, Any] = field(default_factory=dict)
    # Provenance for TpsReport.gate_skip_ratio. ``None`` when no gate
    # contributed (P1 / P2-Shadow). Otherwise one of the labels from
    # :mod:`nrlpy.gates` (``prefill_cache`` / ``override`` / ``layer_skip``).
    gate_source: str | None = None
    # Full gate report when the gate fired. Carried for the banner and the
    # evidence log; ``None`` when no gate contributed.
    gate_report: dict[str, Any] = field(default_factory=dict)
    word_rates: WordRateReport = field(default_factory=WordRateReport)
    # Phase 5-EG R2 shadow. Advisory-only, *never* affects token output.
    # Always present in the result (default: ``skipped`` with reason
    # ``"r2_shadow_disabled"``) so downstream tools don't need Optional
    # handling. Evidence-log payload: ``omega_shadow_hits``,
    # ``omega_shadow_demotion_reasons``, ``omega_shadow_wall_ms``.
    omega_shadow: _lmo.OmegaShadowReport = field(
        default_factory=lambda: _lmo.OmegaShadowReport.skipped(
            "fast-stable", "r2_shadow_not_run"
        )
    )
    # Phase 15-EG — hinge-collapse warm-restart (``zpm.prime`` invocations
    # this turn). Does not participate in TPS / WPS math; evidence only.
    drift_reprime_count: int = 0


def _resolve_manifest_paths(manifest: GgufManifest) -> None:
    base = Path(manifest.manifest_path).parent if manifest.manifest_path else Path.cwd()
    if manifest.model and not Path(manifest.model).is_absolute():
        candidate = (base / manifest.model).resolve()
        if candidate.is_file():
            manifest.model = str(candidate)
    if manifest.prompt_file and not Path(manifest.prompt_file).is_absolute():
        cand = (base / manifest.prompt_file).resolve()
        if cand.is_file():
            manifest.prompt_file = str(cand)


def _resolve_prompt(manifest: GgufManifest) -> str:
    if manifest.prompt_file:
        return Path(manifest.prompt_file).read_text(encoding="utf-8")
    return manifest.prompt


def _auto_threads(requested: int) -> int:
    """Pick a thread count for libllama.

    ``requested > 0`` wins. Otherwise prefer **physical** cores — on most
    CPU-only hosts hyperthreads contend for the same FMA/vector units and
    actually hurt decode tokens/sec. ``NRL_N_THREADS`` overrides the auto-pick
    without needing to thread the argument through every callsite.
    """
    if requested > 0:
        return requested
    env = os.environ.get("NRL_N_THREADS", "").strip()
    if env:
        try:
            v = int(env, 10)
            if v > 0:
                return v
        except ValueError:
            pass
    try:
        import psutil  # noqa: PLC0415

        phys = psutil.cpu_count(logical=False)
        if isinstance(phys, int) and phys > 0:
            return phys
    except Exception:
        pass
    cpu = os.cpu_count() or 1
    return max(1, cpu)


def _n_gpu_layers_from_env() -> int:
    """``NRL_GPU_LAYERS``: unset/empty = 0 (CPU). ``-1`` / ``auto`` / ``all`` = all layers on GPU."""
    raw = os.environ.get("NRL_GPU_LAYERS", "").strip()
    if not raw:
        return 0
    low = raw.lower()
    if low in {"-1", "auto", "all", "max"}:
        return -1
    try:
        return int(raw, 10)
    except ValueError:
        return 0


def _stream_pacing_s() -> float:
    """``NRL_STREAM_CHUNK_MS`` per-token sleep; ``0`` (default) disables pacing.

    Intended for video demos and UI stream smoothing. Never used by benchmarks —
    when set, the TPS banner adds a ``paced=Xms`` tag so results are never quoted
    as native throughput.
    """
    raw = os.environ.get("NRL_STREAM_CHUNK_MS", "").strip()
    if not raw:
        return 0.0
    try:
        ms = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, ms) / 1000.0


def _apply_env_overrides(manifest: GgufManifest) -> None:
    """Let env vars override manifest defaults for host-specific RAM tuning."""
    kv = os.environ.get("NRL_KV_CACHE", "").strip().lower()
    if kv and kv in _VALID_KV_CACHE_DTYPES and not manifest.kv_cache_dtype:
        manifest.kv_cache_dtype = kv
    no_repack_env = os.environ.get("NRL_NO_REPACK", "").strip().lower()
    if no_repack_env in _BOOL_TRUE and not manifest.no_repack:
        manifest.no_repack = True
    ctx_env = os.environ.get("NRL_CTX", "").strip()
    if ctx_env:
        try:
            ctx_val = int(ctx_env, 10)
        except ValueError:
            ctx_val = 0
        if ctx_val > 0:
            manifest.n_ctx = ctx_val
    # P2-Active simulation override. Only honoured when the manifest didn't
    # already set one; env loses to explicit manifest intent. Out-of-range
    # values are ignored (not a crash) since this is a dev/CI knob.
    sim_env = os.environ.get("NRL_GATE_SKIP_RATIO_OVERRIDE", "").strip()
    if sim_env and manifest.gate_skip_ratio_override == 0.0:
        try:
            sim_val = float(sim_env)
        except ValueError:
            sim_val = -1.0
        if 0.0 <= sim_val < 1.0:
            manifest.gate_skip_ratio_override = sim_val
    # ZPM Plane-A.5 identity resolver.
    zpm_env = os.environ.get("NRL_ZPM", "").strip().lower()
    if zpm_env in _BOOL_TRUE and not manifest.zpm_nullspace:
        manifest.zpm_nullspace = True
    elif zpm_env in {"off", "0", "false", "no"}:
        manifest.zpm_nullspace = False
    zpm_thresh_env = os.environ.get("NRL_ZPM_THRESHOLD", "").strip()
    if zpm_thresh_env:
        try:
            thresh = int(zpm_thresh_env, 10)
        except ValueError:
            thresh = -1
        if 0 <= thresh <= 256:
            manifest.zpm_threshold_bits = thresh
    # Phase 5-EG R2 shadow overrides (host-local tuning only; never change
    # observable token flow — shadow never emits).
    lane_env = os.environ.get("NRL_COHERENCE_LANE", "").strip().lower()
    if lane_env and lane_env in _VALID_COHERENCE_LANES:
        manifest.coherence_lane = lane_env
    r2_env = os.environ.get("NRL_R2_SHADOW", "").strip().lower()
    if r2_env in _BOOL_TRUE:
        manifest.r2_shadow_enabled = True
    elif r2_env in _BOOL_FALSE:
        manifest.r2_shadow_enabled = False
    budget_env = os.environ.get("NRL_OMEGA_BUDGET_MS", "").strip()
    if budget_env:
        try:
            bv = float(budget_env)
        except ValueError:
            bv = -1.0
        if bv >= 0.0:
            manifest.omega_budget_ms = bv


def _diagnose_bad_model(gguf: Path) -> None:
    """Print actionable help when ``model = X.gguf`` doesn't resolve.

    Pattern borrowed from RomanAILabs Ghost_Compressor ``runner.py`` — cheap,
    saves real time for users who typo a path or pick the wrong drive.
    """
    print(f"[nrl.gguf] not a file (or no access): {gguf}", file=sys.stderr)
    if not gguf.exists():
        print(
            "[nrl.gguf] that path does not exist — check drive letter and spelling.",
            file=sys.stderr,
        )
    elif gguf.is_dir():
        print(
            "[nrl.gguf] that path is a folder, not a .gguf file.",
            file=sys.stderr,
        )
    parent = gguf.parent
    if parent.is_dir():
        candidates = sorted(parent.glob("*.gguf"))[:12]
        if candidates:
            print(f"[nrl.gguf] .gguf files in folder: {parent}", file=sys.stderr)
            for c in candidates:
                print(f"   {c}", file=sys.stderr)
        else:
            print(
                f"[nrl.gguf] no .gguf files in {parent} — point 'model' at a real file.",
                file=sys.stderr,
            )


def _apply_control_hints(manifest: GgufManifest) -> None:
    """Downshift gate policy when ``nrl control`` says so. Never touches libllama numerics."""
    if not manifest.respect_control_hints:
        return
    prefs = runtime.load_control_preferences()
    if prefs is None or not runtime.control_hints_active(prefs):
        return
    rec = prefs.get("recommended_profile")
    if isinstance(rec, str) and rec in _VALID_PROFILES:
        manifest.profile = rec
    th = prefs.get("throttle_hint")
    if th in {"conservative", "gated"}:
        manifest.gate_layer_policy = "none"


def _evidence_log_path(manifest: GgufManifest) -> Path:
    if manifest.evidence_log:
        return Path(manifest.evidence_log)
    root = os.environ.get("NRL_EVIDENCE_LOG")
    if root:
        return Path(root)
    nrl_root = os.environ.get("NRL_ROOT")
    base = Path(nrl_root) if nrl_root else Path.cwd()
    return base / "build" / "immune" / "events.jsonl"


def _build_chat_prompt(manifest: GgufManifest, prompt: str) -> str:
    """Very small set of templates. ``none`` returns the prompt verbatim."""
    fmt = manifest.chat_format
    if fmt == "none":
        return prompt
    if fmt == "chatml":
        return (
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    if fmt == "phi3":
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    if fmt == "llama2":
        return f"[INST] {prompt} [/INST]"
    return prompt


def _estimate_word_count(text: str) -> int:
    """Portable, tokenizer-agnostic word count estimate."""
    return len(re.findall(r"\S+", text))


def _compute_word_rates(text: str, tokens: int, tps: TpsReport) -> WordRateReport:
    words = _estimate_word_count(text)
    if tokens <= 0:
        return WordRateReport(word_count=words, words_per_token=0.0)
    wpt = words / float(tokens)
    return WordRateReport(
        word_count=words,
        words_per_token=wpt,
        executed_wps=tps.executed_tps * wpt,
        virtual_wps=tps.virtual_tps * wpt,
        cache_wps=tps.cache_tps * wpt,
        effective_wps=tps.effective_tps * wpt,
    )


class _StubLlm:
    """Deterministic fake backend. Selected by ``NRL_INFERENCE=stub`` (CI / tests)."""

    def __init__(self, manifest: GgufManifest) -> None:
        self._manifest = manifest

    def create_completion(
        self, *_args: Any, **_kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        seed = self._manifest.seed or 1
        pieces = [" token_" + str((seed + i) % 997) for i in range(self._manifest.max_tokens)]
        for p in pieces:
            yield {"choices": [{"text": p}]}


class _CliLlm:
    """``llama-cli.exe`` subprocess backend. Selected by ``NRL_INFERENCE=cli``.

    One subprocess per run (not per token). The decode hot loop reads stdout chunks
    from the child process; wall time is measured by the parent, so TPS is
    approximate (stream-chunk boundaries, not tokenizer-exact) and the banner
    labels it as such.

    Inspired by RomanAILabs Ghost_Compressor ``runner.py`` — kept honest: we call
    what ``llama-cli`` is, not "manifold projection".
    """

    def __init__(self, manifest: GgufManifest) -> None:
        self._manifest = manifest

    def _resolve_llama_cli(self) -> str:
        env = os.environ.get("NRL_LLAMA_CLI", "").strip()
        if not env:
            env = os.environ.get("LLAMA_CLI", "").strip()
        if env and Path(env).is_file():
            return str(Path(env).resolve())
        for name in ("llama-cli.exe", "llama-cli"):
            hit = shutil.which(name)
            if hit:
                return hit
        raise RuntimeError(
            "NRL_INFERENCE=cli requires llama-cli on PATH or NRL_LLAMA_CLI=<full-path>"
        )

    def _build_argv(self, prompt: str) -> list[str]:
        m = self._manifest
        argv = [
            self._resolve_llama_cli(),
            "-m", m.model,
            "-p", prompt,
            "-n", str(m.max_tokens),
            "-c", str(m.n_ctx),
            "-t", str(_auto_threads(m.n_threads)),
            "--temp", f"{m.temperature:.6f}",
            "--top-p", f"{m.top_p:.6f}",
            "--top-k", str(m.top_k),
            "--repeat-penalty", f"{m.repeat_penalty:.6f}",
            "-no-cnv",              # one-shot, no interactive REPL
            "--simple-io",          # unbuffered stdout
            "-no-display-prompt",   # don't echo the prompt back
            "--log-disable",        # keep stderr quiet
        ]
        if m.seed:
            argv.extend(["-s", str(m.seed)])
        if m.kv_cache_dtype:
            argv.extend(["-ctk", m.kv_cache_dtype, "-ctv", m.kv_cache_dtype])
        if m.no_repack:
            argv.append("--no-repack")
        return argv

    def create_completion(
        self, *_args: Any, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        prompt = str(kwargs.get("prompt", ""))
        argv = self._build_argv(prompt)
        # Text=True, UTF-8 so Windows doesn't cp1252 the model's tokens.
        # argv[0] is resolved from PATH or NRL_LLAMA_CLI above, not user-typed.
        proc = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                yield {"choices": [{"text": line}]}
        finally:
            try:
                proc.terminate()
            except OSError:
                pass
            proc.wait(timeout=5)


def _load_llm(manifest: GgufManifest) -> Any:
    """Select backend by ``NRL_INFERENCE`` env.

    * ``native`` (default) → ``llama-cpp-python`` (in-process).
    * ``cli`` → spawn ``llama-cli.exe`` once and stream stdout.
    * ``stub`` → deterministic fake, for CI / tests.
    """
    backend = os.environ.get("NRL_INFERENCE", "native").lower().strip()
    if backend == "stub":
        return _StubLlm(manifest)
    if backend == "cli":
        return _CliLlm(manifest)
    if backend not in {"native", ""}:
        raise RuntimeError(
            f"unknown NRL_INFERENCE backend {backend!r} "
            "(expected 'native', 'cli', or 'stub')"
        )
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python is required for `nrl run <model>.gguf` "
            "(`pip install llama-cpp-python`), set NRL_INFERENCE=cli to use "
            "llama-cli, or NRL_INFERENCE=stub for tests"
        ) from e
    n_gpu = _n_gpu_layers_from_env()
    kw: dict[str, Any] = dict(
        model_path=manifest.model,
        n_ctx=manifest.n_ctx,
        n_threads=_auto_threads(manifest.n_threads),
        n_batch=manifest.n_batch,
        n_gpu_layers=n_gpu,
        verbose=False,
        seed=manifest.seed if manifest.seed else -1,
        logits_all=False,
        use_mmap=True,
    )
    if manifest.kv_cache_dtype:
        kw["type_k"] = manifest.kv_cache_dtype
        kw["type_v"] = manifest.kv_cache_dtype
    # ``no_repack`` is a llama.cpp CLI flag; not all llama-cpp-python builds
    # surface it on Llama(). If the user set it, try the `additional_args` path,
    # otherwise silently ignore so the run still starts.
    if manifest.no_repack:
        kw["additional_args"] = ("--no-repack",)

    def _slim_llama_kwargs(k: dict[str, Any]) -> None:
        k.pop("additional_args", None)
        k.pop("type_k", None)
        k.pop("type_v", None)
        k.pop("use_mmap", None)

    def _try_build(k: dict[str, Any]) -> Any:
        return Llama(**k)

    # Progressive fallback chain. llama-cpp-python >= 0.3 dropped the
    # ``seed`` kwarg, and newer builds reject ``n_batch > n_ctx`` with
    # ``Failed to create llama_context``. We peel off those knobs one
    # by one so a broken chat doesn't fall through the cracks and waste
    # the user's time.
    attempts: list[tuple[str, dict[str, Any]]] = [("initial", dict(kw))]

    kw_no_gpu = dict(kw)
    if n_gpu != 0:
        kw_no_gpu["n_gpu_layers"] = 0
        attempts.append(("cpu-only", kw_no_gpu))

    kw_slim = dict(kw_no_gpu)
    _slim_llama_kwargs(kw_slim)
    attempts.append(("slim-kwargs", kw_slim))

    kw_no_seed = dict(kw_slim)
    kw_no_seed.pop("seed", None)
    attempts.append(("no-seed", kw_no_seed))

    kw_small_batch = dict(kw_no_seed)
    kw_small_batch["n_batch"] = min(int(kw_small_batch.get("n_batch", 512) or 512), 128)
    attempts.append(("small-batch", kw_small_batch))

    kw_safe_ctx = dict(kw_small_batch)
    kw_safe_ctx["n_ctx"] = min(int(kw_safe_ctx.get("n_ctx", 2048) or 2048), 2048)
    kw_safe_ctx["n_batch"] = 32
    attempts.append(("safe-ctx-batch", kw_safe_ctx))

    kw_minimal = {
        "model_path": kw["model_path"],
        "n_ctx": 2048,
        "n_threads": kw.get("n_threads"),
        "n_batch": 32,
        "n_gpu_layers": 0,
        "verbose": False,
        "use_mmap": True,
    }
    attempts.append(("minimal", kw_minimal))

    last_exc: Exception | None = None
    for label, k in attempts:
        try:
            if label != "initial":
                print(
                    f"[nrl.gguf] retrying Llama() with profile={label} "
                    f"(prev: {type(last_exc).__name__ if last_exc else 'n/a'})",
                    file=sys.stderr,
                )
            return _try_build(k)
        except TypeError as exc:
            last_exc = exc
            continue
        except ValueError as exc:
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    assert last_exc is not None
    raise RuntimeError(
        "libllama failed to initialize after every fallback profile. "
        "Common causes: (1) llama-cpp-python ABI mismatch "
        "(try: pip install --upgrade llama-cpp-python), "
        "(2) insufficient RAM for this model at n_ctx=2048, "
        "(3) corrupted .gguf file. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _longest_common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Return the length of the longest common prefix of two integer sequences.

    Used by Phase 13 KV-reuse telemetry. Pure Python; called at most once
    per chat turn; the sequences are token-id lists, ~hundreds of ints.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _measure_kv_reuse(llm: Any, prompt_text: str) -> tuple[int, int]:
    """Best-effort KV-reuse measurement (reused_prefix_tokens, total_tokens).

    Tokenizes ``prompt_text`` with the loaded ``llm`` and compares against
    whatever libllama currently holds in ``_input_ids`` (the prior turn's
    full eval history). Returns ``(0, 0)`` if llama-cpp-python's internals
    are not accessible in this version — telemetry is informational and
    must never raise into the decode path.
    """
    try:
        tokenize = getattr(llm, "tokenize", None)
        if tokenize is None:
            return (0, 0)
        tokens = list(tokenize(prompt_text.encode("utf-8"), add_bos=False))
        prior = list(getattr(llm, "_input_ids", ()) or ())
        lcp = _longest_common_prefix_len(prior, tokens)
        return (int(lcp), int(len(tokens)))
    except Exception:  # noqa: BLE001 — telemetry must not crash decode
        return (0, 0)


def _stream_tokens(
    llm: Any,
    manifest: GgufManifest,
    prompt_text: str,
) -> Iterator[str]:
    """Stream completion chunks from libllama with the low-level ``create_completion`` API.

    Phase 13: when ``manifest.chat_kv_reuse`` is set, we measure the LCP
    of the new prompt against libllama's current ``_input_ids`` (stashed
    on the Llama object as ``_nrl_kv_reuse_info`` for the chat footer)
    and pass ``reset=False`` so libllama skips re-evaluating the shared
    prefix. One-shot callers never opt in, so their behaviour is
    byte-identical to previous phases.
    """
    stop_seqs: list[str] = []
    if manifest.chat_format == "chatml":
        stop_seqs = ["<|im_end|>", "<|im_start|>"]
    elif manifest.chat_format == "phi3":
        stop_seqs = ["<|end|>", "<|user|>"]
    elif manifest.chat_format == "llama2":
        stop_seqs = ["[INST]", "</s>"]

    reset_kv = True
    if getattr(manifest, "chat_kv_reuse", False):
        reused, total = _measure_kv_reuse(llm, prompt_text)
        try:
            llm._nrl_kv_reuse_info = {"reused": reused, "total": total}
        except Exception:  # noqa: BLE001
            pass
        reset_kv = False

    kwargs: dict[str, Any] = dict(
        prompt=prompt_text,
        max_tokens=manifest.max_tokens,
        temperature=manifest.temperature,
        top_p=manifest.top_p,
        top_k=manifest.top_k,
        repeat_penalty=manifest.repeat_penalty,
        stream=True,
        stop=stop_seqs or None,
    )
    # Not every llama-cpp-python release accepts ``reset`` on
    # ``create_completion``. Try it, fall back cleanly if the kwarg
    # is unknown — KV reuse is then controlled implicitly by the shared
    # Llama instance's ``_input_ids`` prefix matcher.
    try:
        iterator: Iterator[Mapping[str, Any]] = llm.create_completion(
            reset=reset_kv, **kwargs
        )
    except TypeError:
        iterator = llm.create_completion(**kwargs)
    for chunk in iterator:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        piece = choices[0].get("text") or ""
        if piece:
            yield str(piece)


# --------------------------------------------------------------------------- #
# Phase 7-EG native runner integration
# --------------------------------------------------------------------------- #

class _NativeLadderUnavailable(RuntimeError):
    """Internal marker for the native fallback path."""


def _run_gguf_native(
    manifest: GgufManifest,
    *,
    stream_to: Any,
    attest_profile: str,
    observation_profile: str,
    observation_timeout_s: float,
    prefill_gate: PrefillGate | None,
    preloaded_llm: Any,
    trust_model_sha: bool,
) -> GgufRunResult:
    """Phase 7-EG native dispatch path. See :func:`run_gguf` docstring.

    Mirrors :func:`run_gguf`'s control flow but routes the rung-selection
    decision through the C dispatcher in ``engine/src/ladder_native.c``.
    All deterministic candidate computation (R0/R1/R2 anchors, Stage-VI
    verify, ZPM index walk) stays in Python so byte-identical evidence
    is structurally guaranteed during the parity-gate window.

    For R5 the native dispatcher invokes the libllama bridge; we install
    a Python callback (:func:`_native_libllama_callback`) that drives the
    existing :func:`_load_llm` / :func:`_stream_tokens` path, so token
    streams match the Python runner exactly.
    """
    try:
        from . import native_ladder as _nl
    except ImportError as exc:  # pragma: no cover — defensive
        raise _NativeLadderUnavailable(str(exc)) from exc
    if not _nl.is_available():
        raise _NativeLadderUnavailable(
            "nrlpy._core does not expose Phase 7-EG ladder bindings"
        )

    _resolve_manifest_paths(manifest)
    _apply_env_overrides(manifest)
    _apply_control_hints(manifest)

    model_path = Path(manifest.model)
    if not model_path.is_file():
        _diagnose_bad_model(model_path)
        raise FileNotFoundError(f"model not found: {manifest.model}")
    declared_sha = manifest.model_sha256
    if trust_model_sha and declared_sha:
        model_sha = declared_sha
    else:
        actual_sha = sha256_file(model_path)
        if declared_sha and declared_sha != actual_sha:
            raise RuntimeError(
                f"model_sha256 mismatch: manifest={declared_sha} actual={actual_sha}"
            )
        manifest.model_sha256 = actual_sha
        model_sha = actual_sha

    prompt = _resolve_prompt(manifest)
    manifest.prompt = prompt

    tps = TpsReport()
    attestation = (
        nrl_attest(profile=attest_profile)
        if attest_profile
        else NrlAttestation(profile="", available=False)
    )
    observation_thread: _LatticeObservationThread | None = None
    if observation_profile:
        observation_thread = _LatticeObservationThread(
            profile=observation_profile,
            neurons=65_536,
            iterations=64,
        )
        observation_thread.start()

    intent_anchor_b = _zpm_anchor_bytes(manifest, prompt)
    drift_reprime = _phase15_pre_turn(
        preloaded_llm, manifest, prompt, model_sha, bytes(intent_anchor_b)
    )
    shadow_ctx = _ladder.LadderShadowContext(
        model_sha256=model_sha,
        intent_anchor_bytes=intent_anchor_b,
        coherence_lane=_ladder.resolve_coherence_lane(manifest.coherence_lane),
        omega_budget_ms=manifest.omega_budget_ms,
        omega_candidates=manifest.omega_candidates,
        omega_iterations=manifest.omega_iterations,
        zpm_threshold_bits=manifest.zpm_threshold_bits,
        r2_shadow_enabled=manifest.r2_shadow_enabled,
        prompt_text=prompt,
    )
    shadow_report: _lmo.OmegaShadowReport = _lmo.OmegaShadowReport.skipped(
        shadow_ctx.coherence_lane, "r2_not_evaluated_yet"
    )

    # Pre-compute R0 (muscle memory) candidate.
    r0_candidate = _nl.LadderCandidate()
    hit = muscle_memory_lookup(manifest)
    if hit is not None:
        r0_candidate = _nl.LadderCandidate(
            available=True,
            text=hit.text,
            tokens=int(hit.tokens),
            wall_s=max(hit.cache_read_s, 1e-9),
        )

    # Pre-compute R1 (ZPM nullspace) candidate.
    r1_candidate = _nl.LadderCandidate()
    zpm_hit_meta: dict[str, Any] = {}
    if not r0_candidate.available and manifest.zpm_nullspace:
        try:
            from . import zpm as _zpm  # noqa: PLC0415

            z_state = _zpm.anchor(_zpm_anchor_bytes(manifest, prompt))
            idx = _zpm_index_load(model_sha)
            z_hit, z_entry = idx.lookup(z_state, threshold_bits=manifest.zpm_threshold_bits)
            if z_entry is not None:
                t_read = time.perf_counter()
                tokens_out = int(z_entry.tokens) or max(
                    1, len(z_entry.reply_text.split())
                )
                elapsed = max(time.perf_counter() - t_read, 1e-9)
                r1_candidate = _nl.LadderCandidate(
                    available=True,
                    text=z_entry.reply_text,
                    tokens=tokens_out,
                    wall_s=elapsed,
                )
                zpm_hit_meta = {
                    "source": "zpm_nullspace",
                    "distance_bits": z_hit.distance_bits,
                    "threshold_bits": z_hit.threshold_bits,
                    "exact": z_hit.exact,
                    "stored_index": z_hit.entry_index,
                }
                try:
                    from .lmo_disk_manager import bump_access_stat

                    bump_access_stat(model_sha, z_state)
                except Exception:
                    pass
        except Exception as _zpm_exc:  # noqa: BLE001
            print(
                f"[nrl.gguf] ZPM lookup skipped ({type(_zpm_exc).__name__}): {_zpm_exc}",
                file=sys.stderr,
            )

    # Pre-compute R2 active candidate (if lane allows).
    r2_candidate = _nl.LadderCandidate()
    r2_rung_result: _lmo.RungResult | None = None
    if (
        not r0_candidate.available
        and not r1_candidate.available
        and _ladder.lane_allows_r2_active_in_ctx(shadow_ctx)
    ):
        r2_rung_result, r2_report = _ladder.execute_r2_active(shadow_ctx)
        shadow_report = r2_report
        if (not r2_rung_result.coherence_demoted) and r2_report.served:
            r2_candidate = _nl.LadderCandidate(
                available=True,
                text=r2_report.served_text,
                tokens=int(r2_report.served_tokens),
                wall_s=max(r2_rung_result.wall_s, 1e-9),
            )
    elif not r0_candidate.available and not r1_candidate.available:
        shadow_report = _lmo.OmegaShadowReport.skipped(
            shadow_ctx.coherence_lane,
            "r2_shadow_disabled"
            if not shadow_ctx.r2_shadow_enabled
            else "coherence_lane_disallows_r2_active",
        )
    elif r0_candidate.available or r1_candidate.available:
        shadow_report = _lmo.OmegaShadowReport.skipped(
            shadow_ctx.coherence_lane, "cache_hit_preempted_r2"
        )

    # Build the R5 request (only consulted if all earlier rungs miss).
    r5_request = {
        "model": manifest.model,
        "prompt": prompt,
        "max_tokens": int(manifest.max_tokens),
        "seed": int(manifest.seed),
        "n_ctx": int(manifest.n_ctx),
        "n_threads": int(manifest.n_threads),
        "n_batch": int(manifest.n_batch),
        "temperature": float(manifest.temperature),
        "top_p": float(manifest.top_p),
        "top_k": int(manifest.top_k),
        "repeat_penalty": float(manifest.repeat_penalty),
    }

    # Wire the libllama bridge backend. The callback drives the same
    # _stream_tokens path the Python runner uses, with optional live
    # streaming to ``stream_to`` so users see tokens as they arrive.
    _native_register_callback(manifest, stream_to, preloaded_llm)
    _nl.set_backend(_nl.BACKEND_CALLBACK)

    # Capacity bound: enough room for max_tokens worth of UTF-8 bytes
    # plus a generous margin. 64 bytes/token is conservative.
    text_buf_capacity = max(16384, int(manifest.max_tokens) * 64 + 4096)
    native_out = _nl.resolve_turn(
        coherence_lane=shadow_ctx.coherence_lane,
        r2_shadow_enabled=manifest.r2_shadow_enabled,
        r0=r0_candidate,
        r1=r1_candidate,
        r2_active=r2_candidate,
        r5_request=r5_request,
        text_buf_capacity=text_buf_capacity,
    )

    # Build the same GgufRunResult shape the Python path produces.
    served_rung = native_out.served_rung
    text = native_out.text
    tokens = int(native_out.tokens)
    cache_hit = served_rung in (
        _nl.RUNG_R0_MUSCLE_MEMORY,
        _nl.RUNG_R1_ZPM,
    )

    if served_rung == _nl.RUNG_R0_MUSCLE_MEMORY:
        tps.cache_tokens = tokens
        tps.cache_wall_s = max(native_out.wall_seconds, 1e-9)
        gate_source: str | None = None
        gate_report_dict: dict[str, Any] = {}
    elif served_rung == _nl.RUNG_R1_ZPM:
        tps.cache_tokens = tokens
        tps.cache_wall_s = max(native_out.wall_seconds, 1e-9)
        gate_source = "zpm_nullspace"
        gate_report_dict = zpm_hit_meta
    elif served_rung == _nl.RUNG_R2_OMEGA_ACTIVE:
        tps.executed_tokens = tokens
        tps.executed_wall_s = max(native_out.wall_seconds, 1e-9)
        gate_source = _lmo.OMEGA_ACTIVE_GATE_SOURCE
        gate_report_dict = {
            "source": _lmo.OMEGA_ACTIVE_GATE_SOURCE,
            "coherence_lane": shadow_ctx.coherence_lane,
            "stored_entry_index": shadow_report.stored_entry_index,
            "zpm_distance_bits": shadow_report.zpm_distance_bits,
            "zpm_threshold_bits": shadow_report.zpm_threshold_bits,
            "wall_ms": shadow_report.wall_ms,
            "omega_iterations": shadow_report.omega_iterations,
            "sub_lattices_visited": shadow_report.sub_lattices_visited,
            "candidate_continuation_fnv": shadow_report.candidate_continuation_fnv,
        }
        # Phase 11 — promote n-gram rescues to R0 so follow-up asks of
        # the same rephrased prompt bypass the Omega probe entirely.
        if shadow_report.ngram_rescued:
            _promote_rescue_to_r0(
                manifest,
                prompt=prompt,
                text=text,
                tokens=tokens,
                model_sha=model_sha,
                wall_s=native_out.wall_seconds,
            )
    else:
        # R5 (libllama) — accounting + post-decode bookkeeping mirror
        # the Python path exactly so the banner numbers match.
        tps.executed_tokens = tokens
        tps.executed_wall_s = max(native_out.wall_seconds, 1e-9)
        gate_source = None
        gate_report_dict = {}

        if manifest.muscle_memory == "on" and tokens > 0:
            muscle_memory_store(manifest, text, tokens)
        if (
            manifest.zpm_nullspace
            and manifest.muscle_memory == "on"
            and tokens > 0
        ):
            try:
                from . import zpm as _zpm  # noqa: PLC0415

                z_state = _zpm.anchor(_zpm_anchor_bytes(manifest, prompt))
                z_path = _zpm_index_path(model_sha)
                idx = _zpm_index_load(model_sha)
                existing_hit, existing_entry = idx.lookup(
                    z_state, threshold_bits=0
                )
                if existing_entry is None or not existing_hit.exact:
                    from . import zpm_persist as _zp  # noqa: PLC0415

                    ent = _zpm.ZpmEntry(
                        state=z_state,
                        reply_text=text,
                        tokens=tokens,
                        wall_s_at_write=float(tps.executed_wall_s),
                        metadata={
                            "model": Path(manifest.model).name,
                            "chat_format": manifest.chat_format,
                            "seed": str(manifest.seed),
                            "prompt_head": prompt[:256],
                        },
                    )
                    idx.add(ent)
                    _zp.persist_zpm_entry(model_sha, z_path, idx, ent)
            except Exception as _zpm_exc:  # noqa: BLE001
                print(
                    f"[nrl.gguf] ZPM store skipped ({type(_zpm_exc).__name__}): {_zpm_exc}",
                    file=sys.stderr,
                )

        # P2-Active gate resolution (same logic as the Python path).
        if prefill_gate is not None and manifest.prefill_cache == "session":
            shaped = _build_chat_prompt(manifest, prompt)
            report = prefill_gate.compute_for(shaped)
            if report.skip_ratio > 0.0:
                tps.gate_skip_ratio = report.skip_ratio
                gate_source = report.source
                gate_report_dict = {
                    "source": report.source,
                    "skip_ratio": report.skip_ratio,
                    "shared_prefix_len": report.shared_prefix_len,
                    "prompt_token_count": report.prompt_token_count,
                }
            prefill_gate.observe(shaped)
        if gate_source is None and manifest.gate_skip_ratio_override > 0.0:
            tps.gate_skip_ratio = manifest.gate_skip_ratio_override
            gate_source = "override"
            gate_report_dict = {
                "source": "override",
                "skip_ratio": manifest.gate_skip_ratio_override,
            }

    tps.finalize()
    observation = (
        observation_thread.result(observation_timeout_s)
        if observation_thread is not None
        else NrlLatticeObservation(available=False, note="observation disabled")
    )

    # Stamp the runner_backend in the gate_report so evidence consumers
    # can tell at a glance which dispatcher served this turn. Never
    # overwrites an existing source key.
    if not isinstance(gate_report_dict, dict):
        gate_report_dict = {}
    gate_report_dict.setdefault("runner_backend", "native")

    _phase15_post_turn_native(preloaded_llm, served_rung, tokens)

    result = GgufRunResult(
        text=text,
        tokens=tokens,
        tps=tps,
        cache_hit=cache_hit,
        nrl_attestation=attestation,
        manifest=manifest,
        model_sha256=model_sha,
        lattice_observation=observation,
        gate_source=gate_source,
        gate_report=gate_report_dict,
        word_rates=_compute_word_rates(text, tokens, tps),
        omega_shadow=shadow_report,
        drift_reprime_count=int(drift_reprime),
    )
    _record_evidence(result)
    return result


def _native_register_callback(
    manifest: GgufManifest,
    stream_to: Any,
    preloaded_llm: Any,
) -> None:
    """Install the libllama bridge callback for native R5 dispatch.

    The callback lives in this module (not in :mod:`nrlpy.native_ladder`)
    because it has direct access to :func:`_load_llm` and the manifest
    state. Streams to ``stream_to`` while collecting the full reply so
    live token output behaves the same as the Python runner.
    """
    from . import native_ladder as _nl

    def _callback(req: dict[str, Any]) -> dict[str, Any]:
        try:
            llm = preloaded_llm if preloaded_llm is not None else _load_llm(manifest)
            shaped = _build_chat_prompt(manifest, str(req.get("prompt", "")))
            pieces: list[str] = []
            tokens = 0
            chunk_pacing_s = _stream_pacing_s()
            for piece in _stream_tokens(llm, manifest, shaped):
                pieces.append(piece)
                tokens += 1
                if stream_to is not None:
                    stream_to.write(piece)
                    stream_to.flush()
                if chunk_pacing_s > 0:
                    time.sleep(chunk_pacing_s)
            return {"text": "".join(pieces), "tokens": int(tokens)}
        except Exception as exc:  # noqa: BLE001 — bridge must never raise into C
            print(
                f"[nrl.gguf] native R5 callback failed ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            return {"text": "", "tokens": 0}

    _nl.register_libllama_callback(_callback)


# --------------------------------------------------------------------------- #
# Phase 8-EG full-native runner integration
# --------------------------------------------------------------------------- #


def _muscle_memory_root_for(manifest: GgufManifest) -> Path:  # pragma: no cover
    """Thin shim around :func:`_muscle_memory_root` parameterised by manifest.

    Kept as a helper so the full-native path and the Python path always
    resolve to the same directory (Python: ``_muscle_memory_root``,
    C: we pass this path in through the request).
    """
    return _muscle_memory_root()


def _native_full_register_r2_callback(manifest: GgufManifest) -> None:
    """Install the R2 probe bridge for the full-native hot path.

    R2 still executes the existing :func:`nrlpy.ladder.execute_r2_active`
    code in Python (byte-parity with Phase 6-EG / Phase 7-EG). The native
    ladder invokes this callback once per eligible turn; on a Stage-VI
    hit the callback hands back the served reply and the C ladder emits
    it as the Phase 6-EG ``omega_resolve_active`` gate source.
    """
    from . import native_ladder as _nl

    # C-side lane id -> Python lane name. Mirrors ``NRL_COHERENCE_LANE_*``
    # in engine/include/nrl/ladder_native.h.
    _LANE_ID_TO_NAME = {
        0: "fast-stable",
        1: "fast-balanced",
        2: "max-throughput",
    }

    def _callback(req: dict[str, Any]) -> dict[str, Any]:
        try:
            lane = _LANE_ID_TO_NAME.get(
                int(req.get("coherence_lane", 0)), "fast-stable"
            )
            shadow_ctx = _ladder.LadderShadowContext(
                model_sha256=str(req.get("model_sha256", "") or manifest.model_sha256),
                intent_anchor_bytes=bytes(req.get("intent_anchor_bytes", b"")),
                coherence_lane=lane,
                omega_budget_ms=float(
                    req.get("omega_budget_ms", manifest.omega_budget_ms)
                ),
                omega_candidates=int(
                    req.get("omega_candidates", manifest.omega_candidates)
                ),
                omega_iterations=int(
                    req.get("omega_iterations", manifest.omega_iterations)
                ),
                zpm_threshold_bits=int(
                    req.get("zpm_threshold_bits", manifest.zpm_threshold_bits)
                ),
                r2_shadow_enabled=bool(req.get("r2_shadow_enabled", 1)),
                prompt_text=str(req.get("prompt_text", "") or manifest.prompt),
            )
            if not _ladder.lane_allows_r2_active_in_ctx(shadow_ctx):
                return {
                    "available": 0,
                    "tokens": 0,
                    "text": "",
                    "stored_entry_index": -1,
                    "distance_bits": 256,
                    "wall_seconds": 0.0,
                }
            t0 = time.perf_counter()
            rung_result, report = _ladder.execute_r2_active(shadow_ctx)
            wall = max(time.perf_counter() - t0, 0.0)
            if rung_result.coherence_demoted or not report.served:
                return {
                    "available": 0,
                    "tokens": 0,
                    "text": "",
                    "stored_entry_index": int(report.stored_entry_index),
                    "distance_bits": int(report.zpm_distance_bits),
                    "wall_seconds": wall,
                }
            return {
                "available": 1,
                "tokens": int(report.served_tokens),
                "text": str(report.served_text),
                "stored_entry_index": int(report.stored_entry_index),
                "distance_bits": int(report.zpm_distance_bits),
                "wall_seconds": wall,
            }
        except Exception as exc:  # noqa: BLE001 — bridge must never raise into C
            print(
                f"[nrl.gguf] native R2 callback failed ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            return {
                "available": 0,
                "tokens": 0,
                "text": "",
                "stored_entry_index": -1,
                "distance_bits": 256,
                "wall_seconds": 0.0,
            }

    _nl.register_r2_callback(_callback)


def _run_gguf_native_full(
    manifest: GgufManifest,
    *,
    stream_to: Any,
    attest_profile: str,
    observation_profile: str,
    observation_timeout_s: float,
    prefill_gate: PrefillGate | None,
    preloaded_llm: Any,
    trust_model_sha: bool,
) -> GgufRunResult:
    """Phase 8-EG full-native dispatch path.

    The critical difference from :func:`_run_gguf_native` is that the
    *candidate-generation* stages for R0 (muscle memory) and R1 (ZPM
    nullspace) run entirely inside the C runtime. Python is only
    re-entered for the R2 and R5 bridge callbacks and for the
    post-turn bookkeeping (evidence logging, cache writes), so the
    critical decode path contains zero Python frames for the common
    cache-hit case.

    Byte-parity contract: the native C implementation is a line-by-line
    port of :func:`muscle_memory_lookup` and :mod:`nrlpy.zpm`; test
    suite ``nrlpy/tests/test_native_full_path.py`` pins the two
    implementations against each other on every release.
    """
    try:
        from . import native_ladder as _nl
    except ImportError as exc:  # pragma: no cover — defensive
        raise _NativeLadderUnavailable(str(exc)) from exc
    if not _nl.is_full_native_available():
        raise _NativeLadderUnavailable(
            "nrlpy._core does not expose Phase 8-EG full-native bindings"
        )

    _resolve_manifest_paths(manifest)
    _apply_env_overrides(manifest)
    _apply_control_hints(manifest)

    model_path = Path(manifest.model)
    if not model_path.is_file():
        _diagnose_bad_model(model_path)
        raise FileNotFoundError(f"model not found: {manifest.model}")
    declared_sha = manifest.model_sha256
    if trust_model_sha and declared_sha:
        model_sha = declared_sha
    else:
        actual_sha = sha256_file(model_path)
        if declared_sha and declared_sha != actual_sha:
            raise RuntimeError(
                f"model_sha256 mismatch: manifest={declared_sha} actual={actual_sha}"
            )
        manifest.model_sha256 = actual_sha
        model_sha = actual_sha

    prompt = _resolve_prompt(manifest)
    manifest.prompt = prompt

    tps = TpsReport()
    attestation = (
        nrl_attest(profile=attest_profile)
        if attest_profile
        else NrlAttestation(profile="", available=False)
    )
    observation_thread: _LatticeObservationThread | None = None
    if observation_profile:
        observation_thread = _LatticeObservationThread(
            profile=observation_profile,
            neurons=65_536,
            iterations=64,
        )
        observation_thread.start()

    intent_bytes = _zpm_anchor_bytes(manifest, prompt)
    drift_reprime = _phase15_pre_turn(
        preloaded_llm, manifest, prompt, model_sha, bytes(intent_bytes)
    )
    coherence_lane = _ladder.resolve_coherence_lane(manifest.coherence_lane)

    # Wire the R5 and R2 bridges before calling into the native runtime.
    # Both are same-process Python callables (matching the Phase 7-EG
    # design); the native ladder owns all rung-selection and wall-clock
    # bookkeeping.
    _native_register_callback(manifest, stream_to, preloaded_llm)
    _nl.set_backend(_nl.BACKEND_CALLBACK)
    _native_full_register_r2_callback(manifest)

    # Build the full-turn request. Text-buffer capacity is bounded above
    # so a runaway R5 reply cannot blow stack on long answers.
    text_buf_capacity = max(16384, int(manifest.max_tokens) * 64 + 4096)
    mm_root = str(_muscle_memory_root())
    zpm_index_path = str(_zpm_index_path(model_sha))
    # Phase 11 — the key string fed into the native FNV must match the
    # Python ``_cache_scope_text``: the user's current intent when set
    # by the chat layer, otherwise the full prompt. Without this the
    # native hot path would hash the accumulated chat transcript and
    # never hit R0 on a repeated question.
    cache_key_text = _cache_scope_text(manifest, prompt)
    turn_req = _nl.FullTurnRequest(
        mm_root_dir=mm_root,
        model_sha256=model_sha,
        prompt=cache_key_text,
        sampler_fingerprint=manifest.sampler_fingerprint(),
        seed=int(manifest.seed),
        max_tokens=int(manifest.max_tokens),
        muscle_memory_on=(manifest.muscle_memory != "off"),
        zpm_index_path=zpm_index_path,
        zpm_enabled=bool(manifest.zpm_nullspace),
        zpm_threshold_bits=int(manifest.zpm_threshold_bits),
        coherence_lane=coherence_lane,
        r2_shadow_enabled=bool(manifest.r2_shadow_enabled),
        omega_iterations=int(manifest.omega_iterations),
        omega_candidates=int(manifest.omega_candidates),
        omega_budget_ms=float(manifest.omega_budget_ms),
        intent_anchor_bytes=bytes(intent_bytes),
        r5_request={
            "model": manifest.model,
            "prompt": prompt,
            "max_tokens": int(manifest.max_tokens),
            "seed": int(manifest.seed),
            "n_ctx": int(manifest.n_ctx),
            "n_threads": int(manifest.n_threads),
            "n_batch": int(manifest.n_batch),
            "temperature": float(manifest.temperature),
            "top_p": float(manifest.top_p),
            "top_k": int(manifest.top_k),
            "repeat_penalty": float(manifest.repeat_penalty),
        },
    )
    turn_out = _nl.run_turn_full(turn_req, text_buf_capacity=text_buf_capacity)

    # Reconstruct OmegaShadowReport to match the Python path's evidence
    # shape so downstream tools don't need to distinguish backends.
    r2_report_dict = turn_out.r2_report
    zpm_report_dict = turn_out.zpm_report
    if turn_out.served_rung == _nl.RUNG_R2_OMEGA_ACTIVE:
        shadow_report = _lmo.OmegaShadowReport(
            status="ok",
            gate_source=_lmo.OMEGA_SHADOW_GATE_SOURCE,
            coherence_lane=coherence_lane,
            mode="active",
            available=True,
            served=True,
            served_text=turn_out.text,
            served_tokens=int(turn_out.tokens),
            hits=1,
            candidate_continuation_fnv="",
            demotion_reasons=(),
            wall_ms=float(r2_report_dict.get("wall_seconds", 0.0)) * 1000.0,
            omega_iterations=int(manifest.omega_iterations),
            sub_lattices_visited=0,
            zpm_distance_bits=int(r2_report_dict.get("distance_bits", -1)),
            zpm_threshold_bits=int(manifest.zpm_threshold_bits),
            stored_entry_index=int(r2_report_dict.get("stored_entry_index", -1)),
        )
    elif float(turn_out.r2_report.get("wall_seconds", 0.0)) > 0.0:
        shadow_report = _lmo.OmegaShadowReport(
            status="ok",
            gate_source=_lmo.OMEGA_SHADOW_GATE_SOURCE,
            coherence_lane=coherence_lane,
            mode="active",
            available=True,
            served=False,
            served_text="",
            served_tokens=0,
            hits=0,
            candidate_continuation_fnv="",
            demotion_reasons=("coherence_demoted",),
            wall_ms=float(r2_report_dict.get("wall_seconds", 0.0)) * 1000.0,
            omega_iterations=int(manifest.omega_iterations),
            sub_lattices_visited=0,
            zpm_distance_bits=int(r2_report_dict.get("distance_bits", -1)),
            zpm_threshold_bits=int(manifest.zpm_threshold_bits),
            stored_entry_index=int(r2_report_dict.get("stored_entry_index", -1)),
        )
    else:
        reason = (
            "r2_shadow_disabled"
            if not manifest.r2_shadow_enabled
            else "coherence_lane_disallows_r2_active"
            if not _nl.lane_allows_r2_active(coherence_lane)
            else "cache_hit_preempted_r2"
            if turn_out.served_rung in (_nl.RUNG_R0_MUSCLE_MEMORY, _nl.RUNG_R1_ZPM)
            else "r2_not_evaluated_yet"
        )
        shadow_report = _lmo.OmegaShadowReport.skipped(coherence_lane, reason)

    served_rung = turn_out.served_rung
    text = turn_out.text
    tokens = int(turn_out.tokens)
    cache_hit = served_rung in (_nl.RUNG_R0_MUSCLE_MEMORY, _nl.RUNG_R1_ZPM)

    gate_source: str | None = None
    gate_report_dict: dict[str, Any] = {}

    if served_rung == _nl.RUNG_R0_MUSCLE_MEMORY:
        tps.cache_tokens = tokens
        tps.cache_wall_s = max(turn_out.wall_seconds, 1e-9)
    elif served_rung == _nl.RUNG_R1_ZPM:
        tps.cache_tokens = tokens
        tps.cache_wall_s = max(turn_out.wall_seconds, 1e-9)
        gate_source = "zpm_nullspace"
        gate_report_dict = {
            "source": "zpm_nullspace",
            "distance_bits": int(zpm_report_dict.get("distance_bits", 0)),
            "threshold_bits": int(manifest.zpm_threshold_bits),
            "exact": bool(zpm_report_dict.get("exact", 0)),
            "stored_index": int(zpm_report_dict.get("stored_entry_index", -1)),
        }
    elif served_rung == _nl.RUNG_R2_OMEGA_ACTIVE:
        tps.executed_tokens = tokens
        tps.executed_wall_s = max(turn_out.wall_seconds, 1e-9)
        gate_source = _lmo.OMEGA_ACTIVE_GATE_SOURCE
        gate_report_dict = {
            "source": _lmo.OMEGA_ACTIVE_GATE_SOURCE,
            "coherence_lane": coherence_lane,
            "stored_entry_index": shadow_report.stored_entry_index,
            "zpm_distance_bits": shadow_report.zpm_distance_bits,
            "zpm_threshold_bits": shadow_report.zpm_threshold_bits,
            "wall_ms": shadow_report.wall_ms,
            "omega_iterations": shadow_report.omega_iterations,
            "sub_lattices_visited": shadow_report.sub_lattices_visited,
            "candidate_continuation_fnv": shadow_report.candidate_continuation_fnv,
        }
        # Phase 11 — promote n-gram rescues to R0 so follow-up asks of
        # the same rephrased prompt bypass the Omega probe entirely.
        if shadow_report.ngram_rescued:
            _promote_rescue_to_r0(
                manifest,
                prompt=prompt,
                text=text,
                tokens=tokens,
                model_sha=model_sha,
                wall_s=turn_out.wall_seconds,
            )
    else:
        # R5 (libllama) — mirror the Python path's post-decode
        # bookkeeping so banner numbers and evidence logs match.
        tps.executed_tokens = tokens
        tps.executed_wall_s = max(turn_out.wall_seconds, 1e-9)
        if manifest.muscle_memory == "on" and tokens > 0:
            muscle_memory_store(manifest, text, tokens)
        if (
            manifest.zpm_nullspace
            and manifest.muscle_memory == "on"
            and tokens > 0
        ):
            try:
                from . import zpm as _zpm  # noqa: PLC0415

                z_state = _zpm.anchor(intent_bytes)
                z_path = _zpm_index_path(model_sha)
                idx = _zpm_index_load(model_sha)
                existing_hit, existing_entry = idx.lookup(
                    z_state, threshold_bits=0
                )
                if existing_entry is None or not existing_hit.exact:
                    from . import zpm_persist as _zp  # noqa: PLC0415

                    ent = _zpm.ZpmEntry(
                        state=z_state,
                        reply_text=text,
                        tokens=tokens,
                        wall_s_at_write=float(tps.executed_wall_s),
                        metadata={
                            "model": Path(manifest.model).name,
                            "chat_format": manifest.chat_format,
                            "seed": str(manifest.seed),
                            "prompt_head": prompt[:256],
                        },
                    )
                    idx.add(ent)
                    _zp.persist_zpm_entry(model_sha, z_path, idx, ent)
            except Exception as _zpm_exc:  # noqa: BLE001
                print(
                    f"[nrl.gguf] ZPM store skipped ({type(_zpm_exc).__name__}): {_zpm_exc}",
                    file=sys.stderr,
                )

        if prefill_gate is not None and manifest.prefill_cache == "session":
            shaped = _build_chat_prompt(manifest, prompt)
            report = prefill_gate.compute_for(shaped)
            if report.skip_ratio > 0.0:
                tps.gate_skip_ratio = report.skip_ratio
                gate_source = report.source
                gate_report_dict = {
                    "source": report.source,
                    "skip_ratio": report.skip_ratio,
                    "shared_prefix_len": report.shared_prefix_len,
                    "prompt_token_count": report.prompt_token_count,
                }
            prefill_gate.observe(shaped)
        if gate_source is None and manifest.gate_skip_ratio_override > 0.0:
            tps.gate_skip_ratio = manifest.gate_skip_ratio_override
            gate_source = "override"
            gate_report_dict = {
                "source": "override",
                "skip_ratio": manifest.gate_skip_ratio_override,
            }

    tps.finalize()
    observation = (
        observation_thread.result(observation_timeout_s)
        if observation_thread is not None
        else NrlLatticeObservation(available=False, note="observation disabled")
    )

    if not isinstance(gate_report_dict, dict):
        gate_report_dict = {}
    gate_report_dict.setdefault("runner_backend", "native_full")

    _phase15_post_turn_native(preloaded_llm, served_rung, tokens)

    result = GgufRunResult(
        text=text,
        tokens=tokens,
        tps=tps,
        cache_hit=cache_hit,
        nrl_attestation=attestation,
        manifest=manifest,
        model_sha256=model_sha,
        lattice_observation=observation,
        gate_source=gate_source,
        gate_report=gate_report_dict,
        word_rates=_compute_word_rates(text, tokens, tps),
        omega_shadow=shadow_report,
        drift_reprime_count=int(drift_reprime),
    )
    _record_evidence(result)
    return result


def run_gguf(
    manifest: GgufManifest,
    *,
    stream_to: Any = None,
    attest_profile: str = "",
    observation_profile: str = "",
    observation_timeout_s: float = 5.0,
    prefill_gate: PrefillGate | None = None,
    preloaded_llm: Any = None,
    trust_model_sha: bool = False,
) -> GgufRunResult:
    """End-to-end P1 run: muscle-memory probe, libllama stream, TPS report, evidence log.

    ``stream_to`` is an optional writable (``sys.stdout``-like) for live token output;
    pass ``None`` to suppress streaming and only receive the final text.

    ``attest_profile`` controls the one-shot ``nrl bench`` engine probe at run start.
    Default ``""`` skips it (decode throughput); pass e.g. ``"omega"`` to opt in.

    ``observation_profile`` controls the background NRL-lattice probe (P2-Shadow).
    Default ``""`` skips it so decode is not competing with ``nrl bench`` on every
    token. Set to e.g. ``"omega-hybrid"`` for an advisory preview. The observation
    never inflates :class:`TpsReport` values — see :class:`NrlLatticeObservation`.

    ``manifest.runner_backend`` selects the §4.2 dispatcher (Phase 7-EG):

    * ``"python"`` (default) — pure Python ladder, unchanged from Phase 6-EG.
    * ``"native"`` — route the rung-selection decision through the C
      dispatcher (``engine/src/ladder_native.c``). Falls back to the
      Python ladder with a stderr warning if ``nrlpy._core`` was not
      built with Phase 7-EG bindings.
    * ``"native_strict"`` — same as ``"native"`` but raises
      :class:`RuntimeError` instead of falling back; used by CI gates.
    * ``"native_full"`` (Phase 8-EG) — drive the full-native hot path
      (R0 + R1 in C, R2 + R5 through bridge callbacks). Falls back to
      the Phase 7-EG native dispatcher, then the Python ladder, with
      a stderr warning when the full-native bindings aren't compiled
      into ``nrlpy._core``.
    * ``"native_full_strict"`` — same as ``"native_full"`` but raises
      :class:`RuntimeError` instead of falling back; used by CI gates.
    """
    if manifest.runner_backend in ("native_full", "native_full_strict"):
        try:
            return _run_gguf_native_full(
                manifest,
                stream_to=stream_to,
                attest_profile=attest_profile,
                observation_profile=observation_profile,
                observation_timeout_s=observation_timeout_s,
                prefill_gate=prefill_gate,
                preloaded_llm=preloaded_llm,
                trust_model_sha=trust_model_sha,
            )
        except _NativeLadderUnavailable as exc:
            if manifest.runner_backend == "native_full_strict":
                raise RuntimeError(
                    f"runner_backend='native_full_strict' but full-native ladder "
                    f"unavailable: {exc}"
                ) from exc
            print(
                f"[nrl.gguf] full-native ladder unavailable ({exc}); "
                "falling back to hybrid native ladder",
                file=sys.stderr,
            )
            # Fall through to the Phase 7-EG hybrid path.
            try:
                return _run_gguf_native(
                    manifest,
                    stream_to=stream_to,
                    attest_profile=attest_profile,
                    observation_profile=observation_profile,
                    observation_timeout_s=observation_timeout_s,
                    prefill_gate=prefill_gate,
                    preloaded_llm=preloaded_llm,
                    trust_model_sha=trust_model_sha,
                )
            except _NativeLadderUnavailable as exc2:
                print(
                    f"[nrl.gguf] hybrid native ladder also unavailable ({exc2}); "
                    "falling back to Python ladder",
                    file=sys.stderr,
                )
                # Fall through to the Python path.

    if manifest.runner_backend in ("native", "native_strict"):
        try:
            return _run_gguf_native(
                manifest,
                stream_to=stream_to,
                attest_profile=attest_profile,
                observation_profile=observation_profile,
                observation_timeout_s=observation_timeout_s,
                prefill_gate=prefill_gate,
                preloaded_llm=preloaded_llm,
                trust_model_sha=trust_model_sha,
            )
        except _NativeLadderUnavailable as exc:
            if manifest.runner_backend == "native_strict":
                raise RuntimeError(
                    f"runner_backend='native_strict' but native ladder unavailable: {exc}"
                ) from exc
            print(
                f"[nrl.gguf] native ladder unavailable ({exc}); "
                "falling back to Python ladder",
                file=sys.stderr,
            )
            # Fall through to the Python path.

    _resolve_manifest_paths(manifest)
    _apply_env_overrides(manifest)
    _apply_control_hints(manifest)

    model_path = Path(manifest.model)
    if not model_path.is_file():
        _diagnose_bad_model(model_path)
        raise FileNotFoundError(f"model not found: {manifest.model}")
    declared_sha = manifest.model_sha256
    if trust_model_sha and declared_sha:
        model_sha = declared_sha
    else:
        actual_sha = sha256_file(model_path)
        if declared_sha and declared_sha != actual_sha:
            raise RuntimeError(
                f"model_sha256 mismatch: manifest={declared_sha} actual={actual_sha}"
            )
        manifest.model_sha256 = actual_sha
        model_sha = actual_sha

    prompt = _resolve_prompt(manifest)
    manifest.prompt = prompt  # normalize so the cache key is stable even with prompt_file

    tps = TpsReport()
    attestation = (
        nrl_attest(profile=attest_profile)
        if attest_profile
        else NrlAttestation(profile="", available=False)
    )

    # Start the advisory lattice probe before decode so it overlaps with real work.
    observation_thread: _LatticeObservationThread | None = None
    if observation_profile:
        observation_thread = _LatticeObservationThread(
            profile=observation_profile,
            neurons=65_536,
            iterations=64,
        )
        observation_thread.start()

    # Phase 6-EG — Rung R2 (Omega Native Resolve). The ladder context
    # bundles everything the §4.3 probe needs so it stays independent of
    # the decode hot loop. R2 runs *synchronously* and *inline* on
    # cache-miss turns when the coherence lane permits; a hit emits the
    # stored reply and skips libllama entirely. On a demotion we record
    # the reason and fall through to R5. See nrlpy/ladder.py and
    # Final_NRL_Architecture_GGUF.MD §4.3 / §Phase 6-EG.
    intent_anchor_b = _zpm_anchor_bytes(manifest, prompt)
    drift_reprime = _phase15_pre_turn(
        preloaded_llm, manifest, prompt, model_sha, bytes(intent_anchor_b)
    )
    shadow_ctx = _ladder.LadderShadowContext(
        model_sha256=model_sha,
        intent_anchor_bytes=intent_anchor_b,
        coherence_lane=_ladder.resolve_coherence_lane(manifest.coherence_lane),
        omega_budget_ms=manifest.omega_budget_ms,
        omega_candidates=manifest.omega_candidates,
        omega_iterations=manifest.omega_iterations,
        zpm_threshold_bits=manifest.zpm_threshold_bits,
        r2_shadow_enabled=manifest.r2_shadow_enabled,
        prompt_text=prompt,
    )
    # Default shadow_report — overwritten below based on which branch R2
    # actually takes this turn. Always well-typed so evidence logs and
    # banners see a stable schema.
    shadow_report: _lmo.OmegaShadowReport = _lmo.OmegaShadowReport.skipped(
        shadow_ctx.coherence_lane, "r2_not_evaluated_yet"
    )

    hit = muscle_memory_lookup(manifest)
    zpm_hit_meta: dict[str, Any] = {}
    if hit is None and manifest.zpm_nullspace:
        # Plane-A.5 ZPM identity resolver. Exact match always served; a
        # near-match is served only if ``zpm_threshold_bits > 0`` AND the
        # recovered Hamming distance is ≤ that threshold. See nrlpy.zpm.
        try:
            from . import zpm as _zpm  # noqa: PLC0415

            z_state = _zpm.anchor(_zpm_anchor_bytes(manifest, prompt))
            idx = _zpm_index_load(model_sha)
            z_hit, z_entry = idx.lookup(z_state, threshold_bits=manifest.zpm_threshold_bits)
            if z_entry is not None:
                t_read = time.perf_counter()
                text_out = z_entry.reply_text
                tokens_out = int(z_entry.tokens) or max(1, len(text_out.split()))
                elapsed = max(time.perf_counter() - t_read, 1e-9)
                hit = MuscleMemoryHit(
                    text=text_out,
                    tokens=tokens_out,
                    cache_read_s=elapsed,
                    key_fnv1a64=0,
                )
                zpm_hit_meta = {
                    "source": "zpm_nullspace",
                    "distance_bits": z_hit.distance_bits,
                    "threshold_bits": z_hit.threshold_bits,
                    "exact": z_hit.exact,
                    "stored_index": z_hit.entry_index,
                }
                try:
                    from .lmo_disk_manager import bump_access_stat

                    bump_access_stat(model_sha, z_state)
                except Exception:
                    pass
        except Exception as _zpm_exc:  # noqa: BLE001
            # ZPM is opt-in and advisory — never take down a decode if the
            # on-disk index is corrupt or the module errors. Log on stderr
            # and fall through to libllama.
            print(
                f"[nrl.gguf] ZPM lookup skipped ({type(_zpm_exc).__name__}): {_zpm_exc}",
                file=sys.stderr,
            )
    if hit is not None:
        tps.cache_tokens = hit.tokens
        tps.cache_wall_s = max(hit.cache_read_s, 1e-9)
        if stream_to is not None:
            stream_to.write(hit.text)
            stream_to.flush()
        # Cache hits have no libllama work to skip; the simulation override
        # would be nonsensical here and is intentionally not applied.
        tps.finalize()
        observation = (
            observation_thread.result(observation_timeout_s)
            if observation_thread is not None
            else NrlLatticeObservation(available=False, note="observation disabled")
        )
        # R0/R1 already served this turn → R2 is not attempted. Record a
        # typed skipped report so evidence-log consumers see a stable
        # schema on cache-hit turns too.
        shadow_report = _lmo.OmegaShadowReport.skipped(
            shadow_ctx.coherence_lane, "cache_hit_preempted_r2"
        )
        _phase15_post_turn_python(preloaded_llm, lattice=True, tokens=0)
        result = GgufRunResult(
            text=hit.text,
            tokens=hit.tokens,
            tps=tps,
            cache_hit=True,
            nrl_attestation=attestation,
            manifest=manifest,
            model_sha256=model_sha,
            lattice_observation=observation,
            gate_source=("zpm_nullspace" if zpm_hit_meta else None),
            gate_report=zpm_hit_meta,
            word_rates=_compute_word_rates(hit.text, hit.tokens, tps),
            omega_shadow=shadow_report,
            drift_reprime_count=int(drift_reprime),
        )
        _record_evidence(result)
        return result

    # -------------------------- Phase 6-EG R2 ACTIVE -------------------------- #
    # Cache miss. If the coherence lane permits, attempt Rung R2 before
    # spinning up libllama. On Stage-VI pass we serve the stored reply
    # at lattice rate and skip R5 entirely; on any demotion we record
    # the reason and fall through.
    if _ladder.lane_allows_r2_active_in_ctx(shadow_ctx):
        r2_rung, r2_report = _ladder.execute_r2_active(shadow_ctx)
        shadow_report = r2_report
        if not r2_rung.coherence_demoted and r2_report.served:
            if stream_to is not None:
                stream_to.write(r2_report.served_text)
                stream_to.flush()
            # Honest accounting per §7.3: R2 contributes to
            # ``executed_tokens`` at lattice rate (wall_s bounded by the
            # R2 probe, not by a libllama decode).
            tps.executed_tokens = int(r2_report.served_tokens)
            tps.executed_wall_s = max(r2_rung.wall_s, 1e-9)
            # Phase 11 — promote n-gram rescues to R0 so follow-up asks
            # of the same rephrased prompt bypass the Omega probe.
            if r2_report.ngram_rescued:
                _promote_rescue_to_r0(
                    manifest,
                    prompt=prompt,
                    text=r2_report.served_text,
                    tokens=int(r2_report.served_tokens),
                    model_sha=model_sha,
                    wall_s=r2_rung.wall_s,
                )
            tps.finalize()
            observation = (
                observation_thread.result(observation_timeout_s)
                if observation_thread is not None
                else NrlLatticeObservation(
                    available=False, note="observation disabled"
                )
            )
            r2_gate_report: dict[str, Any] = {
                "source": _lmo.OMEGA_ACTIVE_GATE_SOURCE,
                "coherence_lane": shadow_ctx.coherence_lane,
                "stored_entry_index": r2_report.stored_entry_index,
                "zpm_distance_bits": r2_report.zpm_distance_bits,
                "zpm_threshold_bits": r2_report.zpm_threshold_bits,
                "wall_ms": r2_report.wall_ms,
                "omega_iterations": r2_report.omega_iterations,
                "sub_lattices_visited": r2_report.sub_lattices_visited,
                "candidate_continuation_fnv": (
                    r2_report.candidate_continuation_fnv
                ),
            }
            _phase15_post_turn_python(preloaded_llm, lattice=True, tokens=0)
            result = GgufRunResult(
                text=r2_report.served_text,
                tokens=int(r2_report.served_tokens),
                tps=tps,
                cache_hit=False,
                nrl_attestation=attestation,
                manifest=manifest,
                model_sha256=model_sha,
                lattice_observation=observation,
                gate_source=_lmo.OMEGA_ACTIVE_GATE_SOURCE,
                gate_report=r2_gate_report,
                word_rates=_compute_word_rates(
                    r2_report.served_text,
                    int(r2_report.served_tokens),
                    tps,
                ),
                omega_shadow=r2_report,
                drift_reprime_count=int(drift_reprime),
            )
            _record_evidence(result)
            return result
    else:
        # Lane disallows R2 active (fast-stable) or flag disabled.
        shadow_report = _lmo.OmegaShadowReport.skipped(
            shadow_ctx.coherence_lane,
            "r2_shadow_disabled"
            if not shadow_ctx.r2_shadow_enabled
            else "coherence_lane_disallows_r2_active",
        )
    # -------------------------------------------------------------------------- #

    llm = preloaded_llm if preloaded_llm is not None else _load_llm(manifest)
    shaped = _build_chat_prompt(manifest, prompt)
    pieces: list[str] = []
    token_count = 0
    chunk_pacing_s = _stream_pacing_s()
    tps.stream_chunk_ms = chunk_pacing_s * 1000.0
    t0 = time.perf_counter()
    try:
        for piece in _stream_tokens(llm, manifest, shaped):
            pieces.append(piece)
            token_count += 1
            if stream_to is not None:
                stream_to.write(piece)
                stream_to.flush()
            if chunk_pacing_s > 0:
                time.sleep(chunk_pacing_s)
    finally:
        tps.executed_wall_s = max(time.perf_counter() - t0, 1e-9)
    tps.executed_tokens = token_count
    text = "".join(pieces)

    if manifest.muscle_memory == "on" and token_count > 0:
        muscle_memory_store(manifest, text, token_count)

    # ZPM Plane-A.5: record the fresh reply into the topological index so
    # future near-matches (within ``zpm_threshold_bits``) collapse to this
    # stored state at memory-I/O speed. Only writes when the manifest opts
    # in AND muscle-memory is also on (closed write discipline — we never
    # persist something MM didn't already accept).
    if (
        manifest.zpm_nullspace
        and manifest.muscle_memory == "on"
        and token_count > 0
    ):
        try:
            from . import zpm as _zpm  # noqa: PLC0415

            z_state = _zpm.anchor(_zpm_anchor_bytes(manifest, prompt))
            z_path = _zpm_index_path(model_sha)
            idx = _zpm_index_load(model_sha)
            # Skip if we already have an exact state (keeps the index O(unique)).
            existing_hit, existing_entry = idx.lookup(z_state, threshold_bits=0)
            if existing_entry is None or not existing_hit.exact:
                from . import zpm_persist as _zp  # noqa: PLC0415

                ent = _zpm.ZpmEntry(
                    state=z_state,
                    reply_text=text,
                    tokens=token_count,
                    wall_s_at_write=float(tps.executed_wall_s),
                    metadata={
                        "model": Path(manifest.model).name,
                        "chat_format": manifest.chat_format,
                        "seed": str(manifest.seed),
                        "prompt_head": prompt[:256],
                    },
                )
                idx.add(ent)
                _zp.persist_zpm_entry(model_sha, z_path, idx, ent)
        except Exception as _zpm_exc:  # noqa: BLE001
            print(
                f"[nrl.gguf] ZPM store skipped ({type(_zpm_exc).__name__}): {_zpm_exc}",
                file=sys.stderr,
            )

    # P2-Active gate resolution: structural prefill-cache gate wins over the
    # numeric override. Resolution order:
    #   1. caller-supplied PrefillGate (structural, on real libllama this maps
    #      to n_past carry-over / free KV cache reuse);
    #   2. gate_skip_ratio_override from manifest or env (numeric fixture,
    #      labeled simulation in banner + evidence log).
    # Both are applied AFTER decode and BEFORE TpsReport.finalize(), so
    # virtual_tps = executed_tps / (1 - gate_skip_ratio) is the single source
    # of truth regardless of which gate fired.
    gate_source: str | None = None
    gate_report_dict: dict[str, Any] = {}
    if prefill_gate is not None and manifest.prefill_cache == "session":
        report = prefill_gate.compute_for(shaped)
        if report.skip_ratio > 0.0:
            tps.gate_skip_ratio = report.skip_ratio
            gate_source = report.source
            gate_report_dict = {
                "source": report.source,
                "skip_ratio": report.skip_ratio,
                "shared_prefix_len": report.shared_prefix_len,
                "prompt_token_count": report.prompt_token_count,
            }
        prefill_gate.observe(shaped)
    if gate_source is None and manifest.gate_skip_ratio_override > 0.0:
        tps.gate_skip_ratio = manifest.gate_skip_ratio_override
        gate_source = "override"
        gate_report_dict = {
            "source": "override",
            "skip_ratio": manifest.gate_skip_ratio_override,
        }
    tps.finalize()
    observation = (
        observation_thread.result(observation_timeout_s)
        if observation_thread is not None
        else NrlLatticeObservation(available=False, note="observation disabled")
    )
    # R2 ran inline above; ``shadow_report`` already captures its
    # outcome (whether it demoted to R5 or was lane-disabled). Falling
    # through here means libllama served the turn.
    _phase15_post_turn_python(
        preloaded_llm, lattice=False, tokens=int(token_count)
    )
    result = GgufRunResult(
        text=text,
        tokens=token_count,
        tps=tps,
        cache_hit=False,
        nrl_attestation=attestation,
        manifest=manifest,
        model_sha256=model_sha,
        lattice_observation=observation,
        gate_source=gate_source,
        gate_report=gate_report_dict,
        word_rates=_compute_word_rates(text, token_count, tps),
        omega_shadow=shadow_report,
        drift_reprime_count=int(drift_reprime),
    )
    _record_evidence(result)
    return result


# --------------------------------------------------------------------------- #
# Banner + evidence
# --------------------------------------------------------------------------- #


def format_banner(result: GgufRunResult) -> str:
    """Human-readable banner. Three independent blocks, never cross-cited.

    * **decode TPS** — libllama's real materialized tokens per second.
    * **NRL attestation** — proof the NRL engine ran on this host (sanity).
    * **NRL lattice observation** — advisory-only ``omega-hybrid`` skip preview.
      In P1 / P2-Shadow this is *not* wired into the TPS math; the banner says so.
    """
    name, value = result.tps.headline()
    att = result.nrl_attestation
    obs = result.lattice_observation
    paced = (
        f", paced={result.tps.stream_chunk_ms:.1f}ms (demo pacing, not native throughput)"
        if result.tps.stream_chunk_ms > 0
        else ""
    )
    gate_source = result.gate_source
    sim_active = gate_source == "override"
    prefill_active = gate_source == "prefill_cache"
    if result.tps.gate_skip_ratio > 0.0:
        if prefill_active:
            source_label = "P2-Active (prefill cache)"
        elif sim_active:
            source_label = "P2-Active simulation (override)"
        else:
            source_label = "P2-Active gate"
        gate_note = f"gate_skip_ratio={result.tps.gate_skip_ratio:.3f} [{source_label}]"
    else:
        gate_note = "gate_skip_ratio=0.000 (P1/P2-Shadow: virtual_tps == executed_tps)"
    headline_note = f"{gate_note}{paced}"
    lines = [
        "",
        "NRL gguf_run",
        f"  model         {Path(result.manifest.model).name}",
        f"  model_sha256  {result.model_sha256[:16]}...",
        f"  profile       {result.manifest.profile}",
        f"  bench_class   {result.manifest.benchmark_class}",
        f"  runner        {result.manifest.runner_backend}",
        f"  cache_hit     {'yes' if result.cache_hit else 'no'}",
        "",
        "decode TPS",
        f"  headline      {name:<14} {value:>10.2f}   ({headline_note})",
        f"  executed_tps  {result.tps.executed_tps:>10.2f}   (materialized, fresh tokens)",
        f"  virtual_tps   {result.tps.virtual_tps:>10.2f}   (executed / (1 - gate_skip_ratio))",
        f"  cache_tps     {result.tps.cache_tps:>10.2f}   (muscle-memory replays)",
        f"  effective_tps {result.tps.effective_tps:>10.2f}   (executed + cache)",
        "",
        "decode WPS (estimated from emitted text)",
        f"  words         {result.word_rates.word_count:>10}",
        f"  words/token   {result.word_rates.words_per_token:>10.3f}",
        f"  executed_wps  {result.word_rates.executed_wps:>10.2f}",
        f"  virtual_wps   {result.word_rates.virtual_wps:>10.2f}",
        f"  cache_wps     {result.word_rates.cache_wps:>10.2f}",
        f"  effective_wps {result.word_rates.effective_wps:>10.2f}",
        "",
        "NRL attestation (engine-sanity probe, not decode TPS)",
        f"  available     {'yes' if att.available else 'no'}",
        f"  variant       {att.variant or 'n/a'}",
        f"  profile       {att.profile}",
        f"  executed_gops {att.executed_gops:>10.3f}",
        f"  virtual_gops  {att.virtual_gops:>10.3f}   (lattice §15)",
        f"  skip_ratio    {att.skip_ratio:>10.6f}   (lattice, not libllama)",
        "",
        "NRL lattice observation (advisory; NOT applied to decode TPS until P2-Active)",
        f"  available     {'yes' if obs.available else 'no'}",
        f"  profile       {obs.profile}",
        f"  skip_ratio    {obs.skip_ratio:>10.6f}   (gate preview, lattice work)",
        f"  virtual_gops  {obs.virtual_gops:>10.3f}",
        f"  note          {obs.note}",
        "",
    ]
    shadow = result.omega_shadow
    banner_title = (
        "R2 Omega Native Resolve (Phase 6-EG ACTIVE — coherence-gated)"
        if shadow.mode == "active"
        else (
            "R2 Omega Native Resolve (Phase 5-EG SHADOW — advisory only)"
            if shadow.mode == "shadow"
            else "R2 Omega Native Resolve (Phase 6-EG — not evaluated this turn)"
        )
    )
    lines.extend([
        banner_title,
        f"  coherence_lane    {shadow.coherence_lane}",
        f"  mode              {shadow.mode}",
        f"  status            {shadow.status}",
        f"  gate_source       {shadow.gate_source}",
        f"  served            {'yes' if shadow.served else 'no'}",
        f"  served_tokens     {shadow.served_tokens}",
        f"  hits              {shadow.hits}",
        f"  candidate_fnv     "
        f"{shadow.candidate_continuation_fnv or '(none)'}",
        f"  demotion_reasons  {','.join(shadow.demotion_reasons) or '(none)'}",
        f"  zpm_distance_bits {shadow.zpm_distance_bits}",
        f"  iterations        {shadow.omega_iterations}",
        f"  sub_lattices      {shadow.sub_lattices_visited}",
        f"  wall_ms           {shadow.wall_ms:>10.3f}",
        "",
    ])
    if shadow.served:
        # Phase 6-EG transparency: when R2 actually emits tokens, surface
        # a loud warning so operators see it in logs. R2 is still marked
        # speculative in §9 of the architecture document.
        lines.extend([
            "!! R2 ACTIVE SERVED TOKENS THIS TURN !!",
            "   Rung R2 (Omega Native Resolve) emitted the stored reply",
            "   without running libllama. Stage-VI verify passed. This is",
            "   the §4.3 speculative native-lattice path; review §9 of",
            "   Final_NRL_Architecture_GGUF.MD for active-mode caveats.",
            "",
        ])
    if sim_active:
        formula_match = abs(
            result.tps.virtual_tps * (1.0 - result.tps.gate_skip_ratio)
            - result.tps.executed_tps
        ) < 1e-6
        lines.extend([
            "P2-Active simulation (numeric override — NOT a libllama elision)",
            f"  gate_skip_ratio_override  {result.manifest.gate_skip_ratio_override:.6f}",
            f"  virtual_tps_formula_ok    "
            f"{'yes' if formula_match else 'no'} "
            f"(virtual_tps * (1 - gate_skip_ratio) == executed_tps)",
            "  note                      real libllama elision lands in P3; this override",
            "                            exists so harnesses can exercise the flipped hinge.",
            "",
        ])
    if prefill_active:
        formula_match = abs(
            result.tps.virtual_tps * (1.0 - result.tps.gate_skip_ratio)
            - result.tps.executed_tps
        ) < 1e-6
        shared = result.gate_report.get("shared_prefix_len", 0)
        total = result.gate_report.get("prompt_token_count", 0)
        lines.extend([
            "P2-Active gate (prefill cache — shared-prefix policy)",
            f"  shared_prefix_len         {shared}",
            f"  prompt_token_count        {total}",
            f"  skip_ratio                {result.tps.gate_skip_ratio:.6f}",
            f"  virtual_tps_formula_ok    "
            f"{'yes' if formula_match else 'no'} "
            f"(virtual_tps * (1 - gate_skip_ratio) == executed_tps)",
            "  note                      on native libllama this is real KV-cache reuse;",
            "                            on stub/cli backends the accounting is structural.",
            "",
        ])
    return "\n".join(lines)


def _record_evidence(result: GgufRunResult) -> None:
    try:
        path = _evidence_log_path(result.manifest)
        event: dict[str, Any] = {
            "schema_id": EVIDENCE_SCHEMA,
            "event": "gguf_run",
            "model": Path(result.manifest.model).name,
            "model_sha256": result.model_sha256,
            "profile": result.manifest.profile,
            "benchmark_class": result.manifest.benchmark_class,
            "cache_hit": result.cache_hit,
            "tokens": result.tokens,
            "tps": result.tps.to_dict(),
            "nrl_attestation": asdict(result.nrl_attestation),
            "nrl_lattice_observation": asdict(result.lattice_observation),
            "gate_skip_ratio_override": result.manifest.gate_skip_ratio_override,
            "gate_simulation_active": result.gate_source == "override",
            "gate_source": result.gate_source,
            "gate_report": result.gate_report,
            "word_rates": asdict(result.word_rates),
            "prefill_cache": result.manifest.prefill_cache,
            "manifest_path": result.manifest.manifest_path,
            "coherence_lane": result.manifest.coherence_lane,
            "r2_shadow_enabled": result.manifest.r2_shadow_enabled,
            "runner_backend": result.manifest.runner_backend,
            # Phase 5-EG R2 shadow — top-level hit/demotion/wall fields
            # surface on the event for the Phase-6-EG activation gate's
            # aggregate analysis (10k-shadow-turns requirement).
            "omega_shadow_gate_source": result.omega_shadow.gate_source,
            "omega_shadow_status": result.omega_shadow.status,
            "omega_shadow_hits": result.omega_shadow.hits,
            "omega_shadow_demotion_reasons": list(
                result.omega_shadow.demotion_reasons
            ),
            "omega_shadow_wall_ms": result.omega_shadow.wall_ms,
            "omega_shadow_sub_lattices_visited": (
                result.omega_shadow.sub_lattices_visited
            ),
            "omega_shadow_iterations": result.omega_shadow.omega_iterations,
            "omega_shadow_zpm_distance_bits": (
                result.omega_shadow.zpm_distance_bits
            ),
            "omega_shadow_candidate_fnv": (
                result.omega_shadow.candidate_continuation_fnv
            ),
            # Phase 6-EG R2 active — aggregate counters for the Phase-7-EG
            # release-gate ("demotion rate < 5% on allowed lanes").
            # ``r2_active_hits`` is 1 when R2 actually served this turn;
            # ``r2_active_demotions`` is 1 when R2 ran in active mode but
            # demoted to R5; ``r2_served_tokens`` is the tokens R2
            # emitted (0 on demotion).
            "r2_active_hits": int(
                1
                if (
                    result.omega_shadow.mode == "active"
                    and result.omega_shadow.served
                )
                else 0
            ),
            "r2_active_demotions": int(
                1
                if (
                    result.omega_shadow.mode == "active"
                    and not result.omega_shadow.served
                    and result.omega_shadow.status == "ok"
                )
                else 0
            ),
            "r2_served_tokens": int(result.omega_shadow.served_tokens),
            "omega_shadow": asdict(result.omega_shadow),
            # Phase 15-EG — hinge-collapse warm-restart (evidence-only).
            "drift_reprime_count": int(getattr(result, "drift_reprime_count", 0)),
        }
        append_jsonl(path, event)
        result.evidence_path = str(path)
    except OSError:
        # Evidence emission must never break a run.
        result.evidence_path = ""


# --------------------------------------------------------------------------- #
# Convenience: build a manifest from argv-style kwargs
# --------------------------------------------------------------------------- #


def manifest_from_args(
    model: str,
    *,
    prompt: str = "",
    prompt_file: str = "",
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    seed: int = 0,
    n_ctx: int = 2048,
    n_threads: int = 0,
    n_batch: int = 512,
    chat_format: str = "none",
    profile: str = "sovereign",
    muscle_memory: str = "on",
    respect_control_hints: bool = True,
    benchmark_class: str = "B",
    kv_cache_dtype: str = "",
    no_repack: bool = False,
    gate_skip_ratio_override: float = 0.0,
    prefill_cache: str = "off",
    coherence_lane: str = "fast-stable",
    r2_shadow_enabled: bool = True,
    omega_budget_ms: float = 2.0,
    omega_candidates: int = 4,
    omega_iterations: int = 3,
    runner_backend: str = "python",
) -> GgufManifest:
    """Construct an in-memory ``GgufManifest`` (for CLI `nrl run <model>.gguf --prompt "..."`)."""
    if prompt and prompt_file:
        raise ManifestError("prompt and prompt_file are mutually exclusive")
    if benchmark_class == "A" and seed == 0:
        raise ManifestError("benchmark_class=A requires a non-zero seed")
    if kv_cache_dtype and kv_cache_dtype not in _VALID_KV_CACHE_DTYPES:
        raise ManifestError(f"invalid kv_cache_dtype {kv_cache_dtype!r}")
    if not (0.0 <= gate_skip_ratio_override < 1.0):
        raise ManifestError(
            f"gate_skip_ratio_override must be in [0.0, 1.0); got {gate_skip_ratio_override!r}"
        )
    if prefill_cache not in _VALID_PREFILL_CACHE:
        raise ManifestError(
            f"prefill_cache must be one of {sorted(_VALID_PREFILL_CACHE)}; "
            f"got {prefill_cache!r}"
        )
    if coherence_lane not in _VALID_COHERENCE_LANES:
        raise ManifestError(
            f"coherence_lane must be one of {sorted(_VALID_COHERENCE_LANES)}; "
            f"got {coherence_lane!r}"
        )
    if omega_budget_ms < 0.0:
        raise ManifestError("omega_budget_ms must be >= 0.0")
    if omega_candidates < 1:
        raise ManifestError("omega_candidates must be >= 1")
    if omega_iterations < 1:
        raise ManifestError("omega_iterations must be >= 1")
    if runner_backend not in _VALID_RUNNER_BACKENDS:
        raise ManifestError(
            f"runner_backend must be one of {sorted(_VALID_RUNNER_BACKENDS)}; "
            f"got {runner_backend!r}"
        )
    return GgufManifest(
        schema=MANIFEST_SCHEMA_V1,
        mode="gguf_run",
        profile=profile,
        model=model,
        prompt=prompt,
        prompt_file=prompt_file,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        seed=seed,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        chat_format=chat_format,
        muscle_memory=muscle_memory,
        respect_control_hints=respect_control_hints,
        benchmark_class=benchmark_class,
        kv_cache_dtype=kv_cache_dtype,
        no_repack=no_repack,
        gate_skip_ratio_override=gate_skip_ratio_override,
        prefill_cache=prefill_cache,
        coherence_lane=coherence_lane,
        r2_shadow_enabled=r2_shadow_enabled,
        omega_budget_ms=omega_budget_ms,
        omega_candidates=omega_candidates,
        omega_iterations=omega_iterations,
        runner_backend=runner_backend,
    )


__all__ = [
    "EVIDENCE_SCHEMA",
    "MANIFEST_SCHEMA_V1",
    "GateReport",
    "GgufManifest",
    "GgufRunResult",
    "ManifestError",
    "MuscleMemoryHit",
    "NrlAttestation",
    "NrlLatticeObservation",
    "PrefillGate",
    "TpsReport",
    "WordRateReport",
    "format_banner",
    "load_manifest",
    "manifest_from_args",
    "muscle_memory_lookup",
    "muscle_memory_store",
    "nrl_attest",
    "parse_manifest_text",
    "run_gguf",
    "sha256_file",
]
