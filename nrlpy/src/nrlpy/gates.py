# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""P2-Active gate sources for the GGUF runner.

A **gate** is anything that produces a ``TpsReport.gate_skip_ratio`` from a
structural policy that corresponds to real libllama work elision. Today this
module ships one implementation:

* :class:`PrefillGate` — tracks the longest common token prefix with the
  previous prompt in the same session. On a real libllama backend this is
  automatically elided by llama.cpp's KV cache (``n_past`` carry-over); on
  the stub backend it is structural-policy accounting (clearly labeled).

Design notes
------------

The override hinge shipped in prompt #6 wrote a raw float into
``gate_skip_ratio``; it has no structural meaning and the banner labels it
*simulation*. This module replaces that pattern with a typed report carrying
both the ratio **and its source**, so downstream surfaces (banner, evidence
log, golden harness) can distinguish:

* ``gate_source = "prefill_cache"`` — structural shared-prefix policy.
* ``gate_source = "override"``      — numeric fixture (dev / CI only).
* ``gate_source = None``            — P1 / P2-Shadow, no gate.

The real libllama layer-skip callback landing in a future phase will be a
third ``GateSource`` with ``gate_source = "layer_skip"``. The math contract,
banner layout, evidence-log schema, and golden-harness invariants are all
locked in CI **now** so that future wiring only swaps the source, not the
surfaces.

Honesty invariants (hard):

* ``skip_ratio`` is always in ``[0.0, 1.0)``. A gate that wants to report
  ``>= 1.0`` is clamped to ``1.0 - _SKIP_RATIO_EPSILON``; the ``virtual_tps``
  formula ``executed_tps / (1 - skip_ratio)`` must never divide by zero.
* Muscle-memory cache hits **never** call the gate. On a cache replay there
  is no libllama work to elide; any non-zero gate_skip_ratio there would be
  dishonest. ``run_gguf`` enforces this ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

_SKIP_RATIO_EPSILON = 1e-6
_MAX_SKIP_RATIO = 1.0 - _SKIP_RATIO_EPSILON

GateSourceName = Literal["prefill_cache", "override", "layer_skip"]


@dataclass(frozen=True)
class GateReport:
    """Result of asking a gate what it would elide for the current prompt.

    Attributes:
        skip_ratio: In ``[0.0, 1.0 - epsilon]``. ``0.0`` means the gate saw
            nothing to skip (e.g., no shared prefix); non-zero values set
            ``TpsReport.gate_skip_ratio`` directly.
        source: Stable identifier used in the banner and evidence log.
        shared_prefix_len: Convenience metric for telemetry (prefill gate
            only; zero for other sources).
        prompt_token_count: The gate's view of how many tokens the prompt
            represents. Used for self-diagnostic banner lines.
    """

    skip_ratio: float
    source: GateSourceName
    shared_prefix_len: int = 0
    prompt_token_count: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.skip_ratio <= _MAX_SKIP_RATIO):
            raise ValueError(
                f"GateReport.skip_ratio must be in [0.0, {_MAX_SKIP_RATIO}]; "
                f"got {self.skip_ratio!r} (source={self.source!r})"
            )
        if self.shared_prefix_len < 0 or self.prompt_token_count < 0:
            raise ValueError("GateReport counts must be non-negative")


def _clamp_skip(skip: float) -> float:
    if skip <= 0.0:
        return 0.0
    if skip >= _MAX_SKIP_RATIO:
        return _MAX_SKIP_RATIO
    return skip


def _tokenize_whitespace(text: str) -> list[str]:
    """Reference tokenizer for the stub / accounting paths.

    NOT used for the native backend (libllama owns tokenization there).
    Split on whitespace with a 1-char fallback so very short prompts still
    produce measurable skip ratios in tests.
    """
    parts = text.split()
    if len(parts) >= 2:
        return parts
    return list(text)  # fall back to per-character for short prompts


# --------------------------------------------------------------------------- #
# Prefill-cache gate
# --------------------------------------------------------------------------- #


@dataclass
class PrefillGate:
    """Tracks the previous prompt's tokens and reports shared-prefix skip.

    Lifecycle:

    1. Caller constructs ``PrefillGate()`` once per session.
    2. For each prompt, caller invokes :meth:`compute_for` **before** running
       libllama. The gate returns a :class:`GateReport`; if
       ``report.skip_ratio > 0`` it means the current prompt shares a
       non-empty token prefix with the previous one, and on a real backend
       libllama's internal KV cache will elide that prefix's attention/FFN
       work automatically.
    3. After the run completes (successfully, including a cache miss path),
       the caller invokes :meth:`observe` with the tokens actually fed to
       libllama so the next turn's :meth:`compute_for` can diff against them.

    The gate is intentionally session-scoped (no cross-process persistence);
    persistent shared-prefix caching is a separate phase (tracked as #8+).
    """

    previous_tokens: list[str] = field(default_factory=list)
    last_report: GateReport | None = None

    def compute_for(self, prompt_tokens: Sequence[str] | str) -> GateReport:
        """Report the skip ratio for the next run's prompt.

        ``prompt_tokens`` may be a pre-tokenized sequence (from the native
        backend's tokenizer) or a raw string (stub / accounting path — falls
        through to :func:`_tokenize_whitespace`).
        """
        if isinstance(prompt_tokens, str):
            tokens: list[str] = _tokenize_whitespace(prompt_tokens)
        else:
            tokens = list(prompt_tokens)

        n = len(tokens)
        if n == 0:
            rep = GateReport(skip_ratio=0.0, source="prefill_cache")
            self.last_report = rep
            return rep

        shared = _common_prefix_len(self.previous_tokens, tokens)
        if shared <= 0:
            rep = GateReport(
                skip_ratio=0.0,
                source="prefill_cache",
                shared_prefix_len=0,
                prompt_token_count=n,
            )
            self.last_report = rep
            return rep

        raw = shared / n
        rep = GateReport(
            skip_ratio=_clamp_skip(raw),
            source="prefill_cache",
            shared_prefix_len=shared,
            prompt_token_count=n,
        )
        self.last_report = rep
        return rep

    def observe(self, prompt_tokens: Sequence[str] | str) -> None:
        """Record what was actually fed to libllama on this turn."""
        if isinstance(prompt_tokens, str):
            self.previous_tokens = _tokenize_whitespace(prompt_tokens)
        else:
            self.previous_tokens = list(prompt_tokens)

    def reset(self) -> None:
        self.previous_tokens = []
        self.last_report = None


def _common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


__all__ = [
    "GateReport",
    "GateSourceName",
    "PrefillGate",
]
