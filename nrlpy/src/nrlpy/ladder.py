# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Resolution Ladder orchestration (Final_NRL_Architecture_GGUF.MD §4.2).

This module hosts the per-turn ladder plumbing that the shipped GGUF
runner ( :mod:`nrlpy.gguf` ) consumes. It exists so ladder logic does
not bloat the runner file and so Phase 6-EG can promote R2 from
*shadow* to *active* by editing a single small surface.

Scope (current):

* **Rung R2 shadow** — background probe, advisory only (Phase 5-EG).
  Runs on a daemon thread concurrent with libllama decode; result is
  recorded in ``nrl.gguf_run.v1`` evidence events with
  ``gate_source = "omega_resolve_shadow"``.
* **Rung R2 active** — synchronous inline execution (Phase 6-EG).
  Invoked by the runner *after* R0/R1 cache lookups and *before*
  libllama decode when the coherence lane permits token service. On a
  ZPM hit that passes Stage-VI verify, the stored reply is returned
  and no libllama call is made. On any failure R2 demotes cleanly to
  R5. All other rungs (R0, R1, R3, R5) continue to live in
  :mod:`nrlpy.gguf` unchanged.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from . import lmo as _lmo
from . import zpm as _zpm

if TYPE_CHECKING:  # pragma: no cover
    from .lmo import LmoHandle, OmegaShadowReport, RungResult

__all__ = [
    "LadderShadowContext",
    "OmegaShadowThread",
    "R2_ACTIVE_DEMOTION_REASONS",
    "execute_r2_active",
    "lane_allows_r2_active_in_ctx",
    "lmo_dir_for",
    "load_zpm_index_if_any",
    "open_lmo_for_shadow",
    "resolve_coherence_lane",
    "zpm_index_path",
]


# --------------------------------------------------------------------------- #
# Lane resolution
# --------------------------------------------------------------------------- #


def resolve_coherence_lane(raw: str | None) -> str:
    """Normalize a user-supplied coherence-lane string.

    Empty or ``None`` → default ``"fast-stable"`` (the Phase 4-EG
    behaviour, Class-A legal). Unknown lanes fall back to
    ``"fast-stable"`` with a conservative default: unknown input never
    silently upgrades the ladder to R2/R4.
    """
    if not raw:
        return "fast-stable"
    v = str(raw).strip().lower()
    if v in _lmo.COHERENCE_LANES:
        return v
    return "fast-stable"


# --------------------------------------------------------------------------- #
# Cache discovery (mirrors nrlpy.gguf path conventions, without the import cycle)
# --------------------------------------------------------------------------- #


def lmo_dir_for(model_sha256: str, *, nrl_root: Path | None = None) -> Path:
    """Return the canonical LMO directory for ``model_sha256``.

    Matches the path :func:`nrlpy.lmo.absorb_gguf` writes to when
    ``out_root`` is not overridden.
    """
    import os  # local import: only used in this single function

    if nrl_root is None:
        env = os.environ.get("NRL_ROOT")
        nrl_root = Path(env) if env else Path.cwd()
    return nrl_root / "cache" / "lmo" / (model_sha256 or "unknown")


def zpm_index_path(model_sha256: str, *, nrl_root: Path | None = None) -> Path:
    """Return the canonical ZPM index path for ``model_sha256``.

    Mirrors the ``_zpm_index_path`` helper in :mod:`nrlpy.gguf`; kept
    here so the ladder module does not force a circular import.
    """
    import os

    if nrl_root is None:
        env = os.environ.get("NRL_ROOT")
        nrl_root = Path(env) if env else Path.cwd()
    return nrl_root / "cache" / "zpm" / (model_sha256 or "unknown") / "index.bin"


def open_lmo_for_shadow(model_sha256: str) -> "LmoHandle | None":
    """Best-effort open of the absorbed LMO for ``model_sha256``.

    Returns ``None`` when no LMO exists (Phase 4-EG absorption hasn't
    been run yet, or the user is running an old model). Shadow probes
    that reach this path will degrade to
    ``OmegaShadowReport.skipped("lmo_not_absorbed")`` without taking
    the runner down.
    """
    d = lmo_dir_for(model_sha256)
    if not d.is_dir():
        return None
    try:
        return _lmo.LmoHandle.open(d)
    except _lmo.LmoError:
        return None


def load_zpm_index_if_any(model_sha256: str) -> "_zpm.ZpmIndex | None":
    """Best-effort load of the on-disk ZPM index for ``model_sha256``.

    Returns ``None`` if the index file does not exist or cannot be
    parsed. Callers must tolerate ``None`` — an empty or missing index
    is a normal cold-start condition, not an error.
    """
    try:
        from . import zpm_persist as _zp  # noqa: PLC0415

        zpath = zpm_index_path(model_sha256)
        _zp.recover_zpm_for_model(model_sha256 or "unknown", zpath)
        if not zpath.is_file():
            return None
        return _zpm.ZpmIndex.load(zpath)
    except Exception:  # noqa: BLE001 — corrupt index must never crash the runner
        return None


# --------------------------------------------------------------------------- #
# Shadow-thread orchestrator
# --------------------------------------------------------------------------- #


class LadderShadowContext:
    """Bundled inputs for Rung R2 shadow, independent of the decoder.

    The runner builds this once per turn and hands it to
    :class:`OmegaShadowThread`. Keeping it pure data means the thread
    has no access to the decode hot loop, `stream_to`, or the
    ``llm.create_completion`` iterator — a structural guarantee that
    R2 cannot interfere with token flow.
    """

    def __init__(
        self,
        *,
        model_sha256: str,
        intent_anchor_bytes: bytes,
        coherence_lane: str,
        omega_budget_ms: float,
        omega_candidates: int,
        omega_iterations: int,
        zpm_threshold_bits: int,
        r2_shadow_enabled: bool,
        prompt_text: str = "",
    ) -> None:
        self.model_sha256 = model_sha256
        self.intent_anchor_bytes = intent_anchor_bytes
        self.coherence_lane = coherence_lane
        self.omega_budget_ms = float(omega_budget_ms)
        self.omega_candidates = int(omega_candidates)
        self.omega_iterations = int(omega_iterations)
        self.zpm_threshold_bits = int(zpm_threshold_bits)
        self.r2_shadow_enabled = bool(r2_shadow_enabled)
        # Phase 11 — carries the raw user prompt (or rendered chat
        # template, whichever the runner hands in) so the R2 resolver
        # can run the n-gram rescue scan against stored ``prompt_head``
        # metadata. Optional; R2 behaves exactly as pre-Phase-11 when
        # empty.
        self.prompt_text = str(prompt_text)

    def should_run_shadow(self) -> bool:
        """True iff the lane and enable-flag both permit a shadow probe.

        Does **not** check LMO availability — that happens inside the
        thread so a missing LMO is visible to the operator as
        ``demotion_reason = "lmo_not_absorbed"`` rather than a silent
        no-op.
        """
        if not self.r2_shadow_enabled:
            return False
        return _lmo.lane_allows_r2_shadow(self.coherence_lane)


class OmegaShadowThread(threading.Thread):
    """Background thread that runs :func:`try_omega_native_resolve`.

    Usage::

        th = OmegaShadowThread(ctx)
        th.start()
        # ... decode proceeds on the main thread ...
        report, rung_result = th.result(timeout_s=0.5)

    The thread is marked daemon so it never outlives the process. If
    :meth:`result` is called before the thread has finished, it joins
    with the given timeout; on timeout the result degrades to
    :meth:`_lmo.OmegaShadowReport.skipped` with the reason
    ``"shadow_thread_timeout"``.
    """

    def __init__(self, ctx: LadderShadowContext) -> None:
        super().__init__(daemon=True, name="nrl-omega-r2-shadow")
        self._ctx = ctx
        self._lock = threading.Lock()
        self._report: _lmo.OmegaShadowReport | None = None
        self._rung: _lmo.RungResult | None = None

    def run(self) -> None:  # noqa: D401 - Thread.run override
        ctx = self._ctx
        if not ctx.should_run_shadow():
            rep = _lmo.OmegaShadowReport.skipped(
                ctx.coherence_lane, "r2_shadow_disabled"
            )
            rung = _lmo.RungResult(
                rung="r2",
                gate_source=None,
                tokens_emitted=0,
                wall_s=0.0,
                coherence_demoted=True,
                stage_vi_reason="r2_shadow_disabled",
            )
            with self._lock:
                self._report, self._rung = rep, rung
            return
        handle = open_lmo_for_shadow(ctx.model_sha256)
        if handle is None:
            rep = _lmo.OmegaShadowReport.skipped(
                ctx.coherence_lane, "lmo_not_absorbed"
            )
            rung = _lmo.RungResult(
                rung="r2",
                gate_source=None,
                tokens_emitted=0,
                wall_s=0.0,
                coherence_demoted=True,
                stage_vi_reason="lmo_not_absorbed",
            )
            with self._lock:
                self._report, self._rung = rep, rung
            return
        zidx: _zpm.ZpmIndex | None = None
        try:
            zpath = zpm_index_path(ctx.model_sha256)
            if zpath.is_file():
                zidx = _zpm.ZpmIndex.load(zpath)
        except Exception:
            zidx = None
        rung, rep = _lmo.try_omega_native_resolve(
            handle,
            intent_anchor_bytes=ctx.intent_anchor_bytes,
            coherence_lane=ctx.coherence_lane,
            zpm_index=zidx,
            omega_budget_ms=ctx.omega_budget_ms,
            omega_candidates=ctx.omega_candidates,
            omega_iterations=ctx.omega_iterations,
            zpm_threshold_bits=ctx.zpm_threshold_bits,
            prompt_text=ctx.prompt_text,
        )
        with self._lock:
            self._report, self._rung = rep, rung

    def result(
        self, timeout_s: float = 0.5
    ) -> tuple["OmegaShadowReport", "RungResult"]:
        """Join with timeout and return ``(report, rung_result)``.

        On timeout the pair degrades to ``skipped`` + a demoted
        :class:`RungResult`. Never raises.
        """
        self.join(timeout=timeout_s)
        with self._lock:
            if self._report is not None and self._rung is not None:
                return self._report, self._rung
        rep = _lmo.OmegaShadowReport.skipped(
            self._ctx.coherence_lane, "shadow_thread_timeout"
        )
        rung = _lmo.RungResult(
            rung="r2",
            gate_source=None,
            tokens_emitted=0,
            wall_s=0.0,
            coherence_demoted=True,
            stage_vi_reason="shadow_thread_timeout",
        )
        return rep, rung


# --------------------------------------------------------------------------- #
# Phase 6-EG — synchronous R2-active path
# --------------------------------------------------------------------------- #

R2_ACTIVE_DEMOTION_REASONS: frozenset[str] = frozenset({
    "coherence_lane_disallows_r2_active",
    "r2_shadow_disabled",
    "lmo_not_absorbed",
    "no_sub_lattices",
    "no_candidate",
    "no_zpm_index",
    "zpm_no_match",
    "budget_exceeded",
    "stage_vi_empty_candidate",
    "stage_vi_invalid_utf8",
    "stage_vi_no_tokenizer_blob",
    "stage_vi_symmetry_drift",
})
"""Closed set of reasons R2 active mode may demote to R5.

Exported so the Phase 7-EG release-gate auditor can verify the runtime
only produces demotion reasons from this known-good set. Any unknown
reason in production evidence must be treated as a regression.
"""


def lane_allows_r2_active_in_ctx(ctx: "LadderShadowContext") -> bool:
    """True iff the ladder context permits Rung R2 *active* token service.

    Convenience predicate combining lane gating and the operator's
    ``r2_shadow_enabled`` manifest/CLI toggle. Used by the runner to
    decide whether to call :func:`execute_r2_active` inline.
    """
    if not ctx.r2_shadow_enabled:
        return False
    return _lmo.lane_allows_r2_active(ctx.coherence_lane)


def execute_r2_active(
    ctx: "LadderShadowContext",
) -> tuple["RungResult", "OmegaShadowReport"]:
    """Run R2 in *active* mode on the calling thread, per §4.3.

    Contract:

    * Returns ``(rung_result, report)``. On service,
      ``rung_result.coherence_demoted is False`` and
      ``report.served is True``; the caller must emit
      ``report.served_text`` as the turn's output and skip R5.
    * On any miss or Stage-VI failure, ``rung_result.coherence_demoted
      is True`` and ``report.served is False``; the caller must fall
      through to R5 (libllama) as usual.
    * Never raises. Corrupt LMOs / indexes degrade to
      ``skipped`` / ``error`` reports.
    * Synchronous, bounded by ``ctx.omega_budget_ms`` + Stage-VI verify
      (sub-millisecond in practice).
    """
    if not ctx.r2_shadow_enabled:
        rep = _lmo.OmegaShadowReport.skipped(
            ctx.coherence_lane, "r2_shadow_disabled"
        )
        rung = _lmo.RungResult(
            rung="r2",
            gate_source=None,
            tokens_emitted=0,
            wall_s=0.0,
            coherence_demoted=True,
            stage_vi_reason="r2_shadow_disabled",
        )
        return rung, rep
    handle = open_lmo_for_shadow(ctx.model_sha256)
    if handle is None:
        rep = _lmo.OmegaShadowReport.skipped(
            ctx.coherence_lane, "lmo_not_absorbed"
        )
        rung = _lmo.RungResult(
            rung="r2",
            gate_source=None,
            tokens_emitted=0,
            wall_s=0.0,
            coherence_demoted=True,
            stage_vi_reason="lmo_not_absorbed",
        )
        return rung, rep
    zidx = load_zpm_index_if_any(ctx.model_sha256)
    rung, rep = _lmo.try_omega_native_resolve(
        handle,
        intent_anchor_bytes=ctx.intent_anchor_bytes,
        coherence_lane=ctx.coherence_lane,
        zpm_index=zidx,
        omega_budget_ms=ctx.omega_budget_ms,
        omega_candidates=ctx.omega_candidates,
        omega_iterations=ctx.omega_iterations,
        zpm_threshold_bits=ctx.zpm_threshold_bits,
        mode="active",
        prompt_text=ctx.prompt_text,
    )
    return rung, rep
