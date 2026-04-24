# Copyright (c) 2026 Daniel Harding - RomanAILabs
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 7-EG native Resolution Ladder driver.

This module is the thin Python surface around the C ladder dispatcher
(:c:func:`nrl_v1_ladder_resolve` in ``engine/src/ladder_native.c``). It
exposes the same per-turn contract the Python ladder produces — the
``GgufRunResult`` shape, the ``OmegaShadowReport`` shape, the evidence
event keys, and the banner text — so the GGUF runner can swap between
the Python and native ladders by flipping a single ``runner_backend``
flag without changing any downstream consumer.

What moves to native code in Phase 7-EG:

* The §4.2 Resolution Ladder dispatch decision (R0 / R1 / R2 / R5).
* The libllama bridge call (today via a registered Python callback;
  Phase 8-EG rebinds the same bridge to direct ``libllama.dll`` linkage
  with no Python-side change).
* Per-rung wall-clock measurement.

What stays in Python during the parity-gate window:

* R0/R1/R2 deterministic candidate computation (cache lookups, ZPM
  index walk, omega evolution, Stage-VI verify). Keeping these in Python
  makes byte-identical evidence trivially observable: the Python and
  native paths share the same hashing, the same FNV-1a64 anchors, and
  the same Stage-VI gate. Tests assert equality, not approximation.
* Manifest parsing, prompt resolution, sha256 attestation, ZPM index
  persistence, evidence emission, banner formatting.

Honest accounting: the native ladder reports the same
``executed_tokens`` / ``cache_tokens`` split, with the same wall-clock
contributions, so :class:`nrlpy.gguf.TpsReport` produces identical
banner numbers across both backends.

Availability is best-effort: if the ``nrlpy._core`` extension wasn't
built, or was built without Phase 7-EG bindings, :func:`is_available`
returns ``False`` and callers should fall back to the Python ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "BACKEND_CALLBACK",
    "BACKEND_STUB",
    "FullTurnRequest",
    "FullTurnResult",
    "LadderCandidate",
    "MmLookupRequest",
    "MmLookupResult",
    "NativeLadderResult",
    "NativeLadderUnavailable",
    "R2ProbeRequest",
    "R2ProbeResponse",
    "RUNG_R0_MUSCLE_MEMORY",
    "RUNG_R1_ZPM",
    "RUNG_R2_OMEGA_ACTIVE",
    "RUNG_R5_LIBLLAMA",
    "ZpmLookupRequest",
    "ZpmLookupResult",
    "is_available",
    "is_full_native_available",
    "lane_allows_r2_active",
    "mm_lookup",
    "r2_has_callback",
    "register_libllama_callback",
    "register_r2_callback",
    "resolve_turn",
    "run_turn_full",
    "rung_name",
    "set_backend",
    "set_callback_to_python_libllama",
    "zpm_lookup",
]

RUNG_R0_MUSCLE_MEMORY = 0
RUNG_R1_ZPM = 1
RUNG_R2_OMEGA_ACTIVE = 2
RUNG_R3_PREFILL = 3
RUNG_R4_LAYER_SKIP = 4
RUNG_R5_LIBLLAMA = 5

BACKEND_STUB = "stub"
BACKEND_CALLBACK = "callback"


class NativeLadderUnavailable(RuntimeError):
    """Raised when the native ladder is requested but ``nrlpy._core``
    wasn't built with Phase 7-EG bindings.

    The runner catches this and degrades to the Python ladder so that
    hosts without a build toolchain (CI clones without zig) keep
    working unchanged.
    """


@dataclass(frozen=True)
class LadderCandidate:
    """Pre-computed candidate for a single rung.

    ``available`` is False on a miss / lane-disallow / Stage-VI fail.
    ``text`` is the would-be served reply (UTF-8 string).
    ``wall_s`` is the time the Python side already paid producing this
    candidate; the native ladder reports it as the rung's wall when this
    candidate is selected, so :class:`TpsReport` matches the Python path
    exactly.
    """

    available: bool = False
    text: str = ""
    tokens: int = 0
    wall_s: float = 0.0


@dataclass(frozen=True)
class NativeLadderResult:
    """Native ladder outcome for a single turn.

    Mirrors what :func:`nrlpy.gguf.run_gguf` would record per turn under
    the Python ladder, with the addition of ``served_class`` (the C ABI
    class used by ``nrl_v1_ladder_resolve``).
    """

    served_rung: int
    served_rung_name: str
    text: str
    tokens: int
    wall_seconds: float
    served_class: int
    text_truncated: bool


def _try_import_core() -> Any | None:
    """Import the C extension, returning ``None`` when unavailable.

    The extension is optional: hosts may have built only the static
    library, may be running on platforms the build doesn't target yet,
    or may simply not have run ``build.ps1``. The Python ladder remains
    the fallback.
    """
    try:
        from . import _core  # type: ignore[attr-defined]
    except ImportError:
        return None
    needed = (
        "ladder_resolve",
        "llama_set_backend",
        "llama_set_callback",
        "llama_get_backend",
        "lane_allows_r2_active",
    )
    for name in needed:
        if not hasattr(_core, name):
            return None
    return _core


_FULL_NATIVE_SYMBOLS: tuple[str, ...] = (
    "mm_lookup",
    "zpm_lookup",
    "r2_set_callback",
    "r2_has_callback",
    "ladder_run_turn_full",
)


def is_available() -> bool:
    """True iff ``nrlpy._core`` exposes the Phase 7-EG ladder bindings."""
    return _try_import_core() is not None


def is_full_native_available() -> bool:
    """True iff ``nrlpy._core`` also exposes the Phase 8-EG full-native bindings.

    Phase 7-EG availability is a strict prerequisite; Phase 8-EG adds
    the full-turn orchestrator, native R0/R1 probes, and R2 callback
    registration on top.
    """
    core = _try_import_core()
    if core is None:
        return False
    for name in _FULL_NATIVE_SYMBOLS:
        if not hasattr(core, name):
            return False
    return True


def _require_core() -> Any:
    core = _try_import_core()
    if core is None:
        raise NativeLadderUnavailable(
            "nrlpy._core does not expose Phase 7-EG ladder bindings; "
            "rebuild via build.ps1 / build.sh or pass runner_backend='python'"
        )
    return core


def set_backend(backend: str) -> None:
    """Select the bridge backend (``'stub'`` or ``'callback'``)."""
    if backend not in (BACKEND_STUB, BACKEND_CALLBACK):
        raise ValueError(
            f"backend must be 'stub' or 'callback'; got {backend!r}"
        )
    _require_core().llama_set_backend(backend)


def register_libllama_callback(
    cb: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> None:
    """Install a Python callable as the libllama bridge backend.

    The callback receives a dict with the request shape (model, prompt,
    sampler, ctx params) and must return a dict with keys ``"text"``
    (str) and ``"tokens"`` (int). The C bridge handles timing.

    Pass ``None`` to clear. Setting a callback does not switch the
    active backend; call :func:`set_backend` to engage it.
    """
    _require_core().llama_set_callback(cb)


def lane_allows_r2_active(lane: str) -> bool:
    """Native lane gate for Rung R2 active mode."""
    return bool(_require_core().lane_allows_r2_active(lane))


def rung_name(rung: int) -> str:
    """Name for an integer rung id (0..5)."""
    return str(_require_core().ladder_rung_name(int(rung)))


def resolve_turn(
    *,
    coherence_lane: str,
    r2_shadow_enabled: bool,
    r0: LadderCandidate | None = None,
    r1: LadderCandidate | None = None,
    r2_active: LadderCandidate | None = None,
    r5_request: dict[str, Any] | None = None,
    text_buf_capacity: int = 16384,
) -> NativeLadderResult:
    """Drive one ladder turn through the C dispatcher.

    Decision order is enforced in C and mirrors :func:`nrlpy.gguf.run_gguf`
    exactly: R0 → R1 → R2 (active, lane-gated) → R5 (libllama bridge).

    The R5 request is used only when no earlier rung serves. When the
    bridge backend is ``'callback'``, the registered Python callable is
    invoked with the R5 request dict.
    """
    core = _require_core()
    spec: dict[str, Any] = {
        "coherence_lane": coherence_lane,
        "r2_shadow_enabled": bool(r2_shadow_enabled),
        "text_buf_capacity": int(text_buf_capacity),
    }
    if r0 is not None and r0.available:
        spec["r0"] = {
            "available": True,
            "text": r0.text,
            "tokens": int(r0.tokens),
            "wall_s": float(r0.wall_s),
        }
    if r1 is not None and r1.available:
        spec["r1"] = {
            "available": True,
            "text": r1.text,
            "tokens": int(r1.tokens),
            "wall_s": float(r1.wall_s),
        }
    if r2_active is not None and r2_active.available:
        spec["r2_active"] = {
            "available": True,
            "text": r2_active.text,
            "tokens": int(r2_active.tokens),
            "wall_s": float(r2_active.wall_s),
        }
    if r5_request is not None:
        # Coerce to the exact shape the C bridge expects. Defaults stay
        # safe so missing keys don't blow up the C dispatcher.
        spec["r5_request"] = {
            "model":          str(r5_request.get("model", "")),
            "prompt":         str(r5_request.get("prompt", "")),
            "max_tokens":     int(r5_request.get("max_tokens", 128)),
            "seed":           int(r5_request.get("seed", 0)),
            "n_ctx":          int(r5_request.get("n_ctx", 2048)),
            "n_threads":      int(r5_request.get("n_threads", 0)),
            "n_batch":        int(r5_request.get("n_batch", 512)),
            "temperature":    float(r5_request.get("temperature", 0.7)),
            "top_p":          float(r5_request.get("top_p", 0.9)),
            "top_k":          int(r5_request.get("top_k", 40)),
            "repeat_penalty": float(r5_request.get("repeat_penalty", 1.1)),
        }
    out = core.ladder_resolve(spec)
    return NativeLadderResult(
        served_rung=int(out.get("served_rung", -1)),
        served_rung_name=str(out.get("served_rung_name", "unknown")),
        text=str(out.get("text", "")),
        tokens=int(out.get("tokens_emitted", 0)),
        wall_seconds=float(out.get("wall_seconds", 0.0)),
        served_class=int(out.get("served_class", 0)),
        text_truncated=bool(out.get("text_truncated", 0)),
    )


def set_callback_to_python_libllama() -> None:
    """Default wiring: install a callback that drives the existing
    :mod:`nrlpy.gguf` libllama path (``llama-cpp-python``).

    This keeps the native ladder fully functional without requiring the
    Phase 8-EG direct linkage. The callback loads the model on first
    use, caches it for the process lifetime, and emits ``"text"`` /
    ``"tokens"`` mirroring what :func:`nrlpy.gguf._stream_tokens`
    produces today.
    """
    from . import gguf as _gguf  # noqa: PLC0415 — break import cycle

    _llm_cache: dict[str, Any] = {}

    def _callback(req: dict[str, Any]) -> dict[str, Any]:
        try:
            model = str(req.get("model", ""))
            prompt = str(req.get("prompt", ""))
            if not model:
                return {"text": "", "tokens": 0}
            # Build a transient manifest mirroring the request. We keep
            # muscle_memory off so the bridge call never re-enters the
            # cache (the ladder already consulted R0/R1 in Python).
            mf = _gguf.manifest_from_args(
                model=model,
                prompt=prompt,
                max_tokens=int(req.get("max_tokens", 128)),
                temperature=float(req.get("temperature", 0.7)),
                top_p=float(req.get("top_p", 0.9)),
                top_k=int(req.get("top_k", 40)),
                repeat_penalty=float(req.get("repeat_penalty", 1.1)),
                seed=int(req.get("seed", 0)),
                n_ctx=int(req.get("n_ctx", 2048)),
                n_threads=int(req.get("n_threads", 0)),
                n_batch=int(req.get("n_batch", 512)),
                muscle_memory="off",
            )
            cache_key = f"{model}|{mf.n_ctx}|{mf.n_threads}|{mf.n_batch}"
            llm = _llm_cache.get(cache_key)
            if llm is None:
                llm = _gguf._load_llm(mf)  # type: ignore[attr-defined]
                _llm_cache[cache_key] = llm
            shaped = _gguf._build_chat_prompt(mf, prompt)  # type: ignore[attr-defined]
            pieces: list[str] = []
            tokens = 0
            for piece in _gguf._stream_tokens(llm, mf, shaped):  # type: ignore[attr-defined]
                pieces.append(piece)
                tokens += 1
            return {"text": "".join(pieces), "tokens": int(tokens)}
        except Exception:  # noqa: BLE001 — bridge must never raise into C
            return {"text": "", "tokens": 0}

    register_libllama_callback(_callback)


# --------------------------------------------------------------------------- #
# Phase 8-EG: full-native hot path surface
# --------------------------------------------------------------------------- #


def _require_full_core() -> Any:
    """Return the Phase 8-EG ``_core`` module or raise :class:`NativeLadderUnavailable`."""
    if not is_full_native_available():
        raise NativeLadderUnavailable(
            "nrlpy._core does not expose Phase 8-EG full-native bindings; "
            "rebuild via build.ps1 / build.sh or pass runner_backend='python' "
            "or runner_backend='native'"
        )
    core = _try_import_core()
    assert core is not None  # checked by is_full_native_available
    return core


@dataclass(frozen=True)
class MmLookupRequest:
    """Request for a native muscle-memory probe."""

    root_dir: str
    model_sha256: str
    prompt: str
    sampler_fingerprint: str
    seed: int
    max_tokens: int
    muscle_memory_on: bool


@dataclass(frozen=True)
class MmLookupResult:
    """Result of a native muscle-memory probe (R0)."""

    hit: bool
    tokens_emitted: int
    key_fnv1a64: int
    wall_seconds: float
    text: str
    text_truncated: bool
    text_byte_len: int


def mm_lookup(req: MmLookupRequest, *, text_buf_capacity: int = 65536) -> MmLookupResult:
    """Run a native muscle-memory probe.

    The native implementation matches :func:`nrlpy.gguf.muscle_memory_lookup`
    byte-for-byte: same FNV-1a64 key derivation, same 16-byte header layout,
    same UTF-8 body decoding. The key is always returned even on a miss so
    callers can record it in their evidence logs.
    """
    core = _require_full_core()
    spec = {
        "root_dir": str(req.root_dir),
        "model_sha256": str(req.model_sha256 or "unknown"),
        "prompt": str(req.prompt or ""),
        "sampler_fingerprint": str(req.sampler_fingerprint or ""),
        "seed": int(req.seed),
        "max_tokens": int(req.max_tokens),
        "muscle_memory_on": 1 if req.muscle_memory_on else 0,
        "text_buf_capacity": int(text_buf_capacity),
    }
    out = core.mm_lookup(spec)
    return MmLookupResult(
        hit=bool(out.get("hit", 0)),
        tokens_emitted=int(out.get("tokens_emitted", 0)),
        key_fnv1a64=int(out.get("key_fnv1a64", 0)),
        wall_seconds=float(out.get("wall_seconds", 0.0)),
        text=str(out.get("text", "")),
        text_truncated=bool(out.get("text_truncated", 0)),
        text_byte_len=int(out.get("text_byte_len", 0)),
    )


@dataclass(frozen=True)
class ZpmLookupRequest:
    """Request for a native ZPM nullspace probe."""

    index_path: str
    model_sha256: str
    prompt: str
    sampler_fingerprint: str
    seed: int
    max_tokens: int
    threshold_bits: int
    enabled: bool


@dataclass(frozen=True)
class ZpmLookupResult:
    """Result of a native ZPM nullspace probe (R1)."""

    hit: bool
    tokens_emitted: int
    exact: bool
    within_threshold: bool
    distance_bits: int
    stored_entry_index: int
    state: tuple[int, int, int, int]
    wall_seconds: float
    text: str
    text_truncated: bool
    text_byte_len: int


def zpm_lookup(
    req: ZpmLookupRequest, *, text_buf_capacity: int = 65536
) -> ZpmLookupResult:
    """Run a native ZPM nullspace probe.

    Computes the 256-bit anchor byte-for-byte identically to
    :func:`nrlpy.zpm.anchor`, scans the on-disk index without entering
    Python, and returns the served reply (exact hit or within-threshold
    near-match) plus audit numbers for the evidence log.
    """
    core = _require_full_core()
    spec = {
        "index_path": str(req.index_path),
        "model_sha256": str(req.model_sha256 or "unknown"),
        "prompt": str(req.prompt or ""),
        "sampler_fingerprint": str(req.sampler_fingerprint or ""),
        "seed": int(req.seed),
        "max_tokens": int(req.max_tokens),
        "threshold_bits": int(req.threshold_bits),
        "enabled": 1 if req.enabled else 0,
        "text_buf_capacity": int(text_buf_capacity),
    }
    out = core.zpm_lookup(spec)
    st = out.get("state", (0, 0, 0, 0))
    state_tuple = (int(st[0]), int(st[1]), int(st[2]), int(st[3]))
    return ZpmLookupResult(
        hit=bool(out.get("hit", 0)),
        tokens_emitted=int(out.get("tokens_emitted", 0)),
        exact=bool(out.get("exact", 0)),
        within_threshold=bool(out.get("within_threshold", 0)),
        distance_bits=int(out.get("distance_bits", 256)),
        stored_entry_index=int(out.get("stored_entry_index", -1)),
        state=state_tuple,
        wall_seconds=float(out.get("wall_seconds", 0.0)),
        text=str(out.get("text", "")),
        text_truncated=bool(out.get("text_truncated", 0)),
        text_byte_len=int(out.get("text_byte_len", 0)),
    )


@dataclass(frozen=True)
class R2ProbeRequest:
    """Request passed to the registered R2 probe callback."""

    coherence_lane: int
    r2_shadow_enabled: bool
    zpm_threshold_bits: int
    omega_iterations: int
    omega_candidates: int
    omega_budget_ms: float
    model_sha256: str
    intent_anchor_bytes: bytes


@dataclass(frozen=True)
class R2ProbeResponse:
    """Response returned by the R2 probe callback."""

    available: bool
    tokens_emitted: int
    stored_entry_index: int
    distance_bits: int
    wall_seconds: float
    text: str


def register_r2_callback(
    cb: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> None:
    """Install a Python callable as the R2 probe bridge for the native hot path.

    The callback receives a dict mirroring :class:`R2ProbeRequest` and
    must return a dict with the :class:`R2ProbeResponse` fields. Passing
    ``None`` clears the binding and the native ladder will always treat
    R2 as a miss and fall through to R5.
    """
    _require_full_core().r2_set_callback(cb)


def r2_has_callback() -> bool:
    """True iff an R2 probe callback is currently registered."""
    return bool(_require_full_core().r2_has_callback())


@dataclass(frozen=True)
class FullTurnRequest:
    """Single full-native turn request.

    All deterministic candidate computation for R0 (muscle memory) and R1
    (ZPM nullspace) is derived from these fields inside the C runtime; no
    Python code runs between calling :func:`run_turn_full` and getting
    the :class:`FullTurnResult` back (apart from the optional R2 callback
    and the R5 libllama callback).
    """

    # R0 muscle memory.
    mm_root_dir: str
    model_sha256: str
    prompt: str
    sampler_fingerprint: str
    seed: int
    max_tokens: int
    muscle_memory_on: bool
    # R1 ZPM nullspace.
    zpm_index_path: str
    zpm_enabled: bool
    zpm_threshold_bits: int
    # Lane + R2.
    coherence_lane: str
    r2_shadow_enabled: bool
    omega_iterations: int
    omega_candidates: int
    omega_budget_ms: float
    intent_anchor_bytes: bytes
    # R5 libllama bridge request.
    r5_request: dict[str, Any]


@dataclass(frozen=True)
class FullTurnResult:
    """Outcome of a single full-native turn."""

    served_rung: int
    served_rung_name: str
    text: str
    tokens: int
    wall_seconds: float
    text_truncated: bool
    text_byte_len: int
    mm_report: dict[str, Any]
    zpm_report: dict[str, Any]
    r2_report: dict[str, Any]
    r5_report: dict[str, Any]


def run_turn_full(
    req: FullTurnRequest, *, text_buf_capacity: int = 65536
) -> FullTurnResult:
    """Drive a single turn through the Phase 8-EG full-native hot path.

    Decision order (identical to :func:`nrlpy.gguf.run_gguf`):

    1. R0 muscle memory hit  → serve, skip R1..R5.
    2. R1 ZPM nullspace hit  → serve, skip R2..R5.
    3. R2 active (if eligible and the callback is registered).
    4. R5 libllama bridge call.

    Raises :class:`NativeLadderUnavailable` if the Phase 8-EG bindings
    were not compiled into ``nrlpy._core``.
    """
    core = _require_full_core()
    r5_req = dict(req.r5_request)
    spec = {
        "mm": {
            "root_dir": str(req.mm_root_dir),
            "model_sha256": str(req.model_sha256 or "unknown"),
            "prompt": str(req.prompt or ""),
            "sampler_fingerprint": str(req.sampler_fingerprint or ""),
            "seed": int(req.seed),
            "max_tokens": int(req.max_tokens),
            "muscle_memory_on": 1 if req.muscle_memory_on else 0,
        },
        "zpm": {
            "index_path": str(req.zpm_index_path),
            "model_sha256": str(req.model_sha256 or "unknown"),
            "prompt": str(req.prompt or ""),
            "sampler_fingerprint": str(req.sampler_fingerprint or ""),
            "seed": int(req.seed),
            "max_tokens": int(req.max_tokens),
            "threshold_bits": int(req.zpm_threshold_bits),
            "enabled": 1 if req.zpm_enabled else 0,
        },
        "coherence_lane": str(req.coherence_lane or "fast-stable"),
        "r2_shadow_enabled": 1 if req.r2_shadow_enabled else 0,
        "zpm_threshold_bits": int(req.zpm_threshold_bits),
        "omega_iterations": int(req.omega_iterations),
        "omega_candidates": int(req.omega_candidates),
        "omega_budget_ms": float(req.omega_budget_ms),
        "intent_anchor_bytes": bytes(req.intent_anchor_bytes or b""),
        "r5_request": {
            "model":          str(r5_req.get("model", "")),
            "prompt":         str(r5_req.get("prompt", "")),
            "max_tokens":     int(r5_req.get("max_tokens", 128)),
            "seed":           int(r5_req.get("seed", 0)),
            "n_ctx":          int(r5_req.get("n_ctx", 2048)),
            "n_threads":      int(r5_req.get("n_threads", 0)),
            "n_batch":        int(r5_req.get("n_batch", 512)),
            "temperature":    float(r5_req.get("temperature", 0.7)),
            "top_p":          float(r5_req.get("top_p", 0.9)),
            "top_k":          int(r5_req.get("top_k", 40)),
            "repeat_penalty": float(r5_req.get("repeat_penalty", 1.1)),
        },
        "text_buf_capacity": int(text_buf_capacity),
    }
    out = core.ladder_run_turn_full(spec)
    return FullTurnResult(
        served_rung=int(out.get("served_rung", -1)),
        served_rung_name=str(out.get("served_rung_name", "unknown")),
        text=str(out.get("text", "")),
        tokens=int(out.get("tokens_emitted", 0)),
        wall_seconds=float(out.get("wall_seconds", 0.0)),
        text_truncated=bool(out.get("text_truncated", 0)),
        text_byte_len=int(out.get("text_byte_len", 0)),
        mm_report=dict(out.get("mm_report", {})),
        zpm_report=dict(out.get("zpm_report", {})),
        r2_report=dict(out.get("r2_report", {})),
        r5_report=dict(out.get("r5_report", {})),
    )
