# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Unit tests for :mod:`nrlpy.gates` — the typed gate source layer.

These tests are intentionally ``run_gguf``-free so regressions in the gate
semantics surface before they ripple through the whole runner. The end-to-end
integration is covered in ``test_gguf.py``.
"""

from __future__ import annotations

import pytest

from nrlpy.gates import GateReport, PrefillGate


# --------------------------------------------------------------------------- #
# GateReport invariants
# --------------------------------------------------------------------------- #


def test_gate_report_accepts_zero_skip() -> None:
    r = GateReport(skip_ratio=0.0, source="prefill_cache")
    assert r.skip_ratio == 0.0
    assert r.source == "prefill_cache"


def test_gate_report_accepts_max_skip() -> None:
    r = GateReport(skip_ratio=1.0 - 1e-6, source="prefill_cache")
    assert r.skip_ratio == pytest.approx(1.0 - 1e-6)


def test_gate_report_rejects_skip_at_or_above_one() -> None:
    with pytest.raises(ValueError, match="skip_ratio"):
        GateReport(skip_ratio=1.0, source="prefill_cache")
    with pytest.raises(ValueError, match="skip_ratio"):
        GateReport(skip_ratio=1.5, source="prefill_cache")


def test_gate_report_rejects_negative_skip() -> None:
    with pytest.raises(ValueError, match="skip_ratio"):
        GateReport(skip_ratio=-0.1, source="prefill_cache")


def test_gate_report_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="counts"):
        GateReport(skip_ratio=0.0, source="prefill_cache", shared_prefix_len=-1)
    with pytest.raises(ValueError, match="counts"):
        GateReport(
            skip_ratio=0.0, source="prefill_cache", prompt_token_count=-1,
        )


# --------------------------------------------------------------------------- #
# PrefillGate behaviour
# --------------------------------------------------------------------------- #


def test_prefill_gate_first_call_has_no_skip() -> None:
    gate = PrefillGate()
    rep = gate.compute_for("hello world")
    assert rep.skip_ratio == 0.0
    assert rep.source == "prefill_cache"
    assert rep.shared_prefix_len == 0
    assert rep.prompt_token_count == 2


def test_prefill_gate_empty_prompt_has_no_skip() -> None:
    gate = PrefillGate()
    gate.observe("hello world")
    rep = gate.compute_for("")
    assert rep.skip_ratio == 0.0
    assert rep.shared_prefix_len == 0


def test_prefill_gate_identical_prompts_clamps_below_one() -> None:
    """Back-to-back identical prompts: shared == total → ratio clamps below 1.0."""
    gate = PrefillGate()
    gate.observe("alpha beta gamma delta")
    rep = gate.compute_for("alpha beta gamma delta")
    assert rep.shared_prefix_len == 4
    assert rep.prompt_token_count == 4
    # The gate must clamp to avoid div-by-zero in virtual_tps math.
    assert rep.skip_ratio < 1.0
    assert rep.skip_ratio > 0.999


def test_prefill_gate_reports_half_for_half_shared() -> None:
    gate = PrefillGate()
    gate.observe("alpha beta")
    rep = gate.compute_for("alpha beta gamma delta")
    assert rep.shared_prefix_len == 2
    assert rep.prompt_token_count == 4
    assert rep.skip_ratio == pytest.approx(0.5)


def test_prefill_gate_no_shared_prefix_is_zero() -> None:
    gate = PrefillGate()
    gate.observe("alpha beta")
    rep = gate.compute_for("delta gamma beta alpha")
    assert rep.skip_ratio == 0.0
    assert rep.shared_prefix_len == 0


def test_prefill_gate_observe_sequence_updates_history() -> None:
    gate = PrefillGate()
    gate.observe(["one", "two", "three"])
    rep = gate.compute_for(["one", "two", "five"])
    assert rep.shared_prefix_len == 2
    assert rep.prompt_token_count == 3
    assert rep.skip_ratio == pytest.approx(2 / 3)


def test_prefill_gate_reset_clears_history() -> None:
    gate = PrefillGate()
    gate.observe("alpha beta gamma")
    gate.reset()
    rep = gate.compute_for("alpha beta gamma")
    assert rep.shared_prefix_len == 0
    assert rep.skip_ratio == 0.0


def test_prefill_gate_last_report_tracks_compute_calls() -> None:
    gate = PrefillGate()
    assert gate.last_report is None
    gate.compute_for("alpha beta")
    assert gate.last_report is not None
    assert gate.last_report.source == "prefill_cache"


def test_prefill_gate_string_tokenization_falls_back_for_short_prompts() -> None:
    """A prompt with <2 whitespace tokens should still produce a sensible skip metric."""
    gate = PrefillGate()
    gate.observe("abcdef")
    rep = gate.compute_for("abcxyz")
    assert rep.prompt_token_count == 6
    assert rep.shared_prefix_len == 3
    assert rep.skip_ratio == pytest.approx(0.5)
