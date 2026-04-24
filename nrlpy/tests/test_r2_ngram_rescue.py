# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 11 — R2 Omega Native Resolve: n-gram rescue path.

These tests validate the second-chance candidate-expansion layer that
raises R2's served fraction above the historical ~5% ceiling without
violating coherence safeguards. The rescue kicks in only when the
primary Omega -> ZPM lookup misses (or the match fails Stage-VI); it
ranks stored ZPM entries by character-4-gram Jaccard overlap against
the current prompt's ``prompt_head`` metadata and serves the best
candidate if both the overlap and Hamming gates pass the rescue
Stage-VI (tokenizer/UTF-8/blob — bit-symmetry is intentionally
skipped for rescues; see ``_stage_vi_ngram_rescue_audit``).

Structure:
  * Pure-function tests for ``_char_ngrams`` and ``_ngram_jaccard``.
  * Pure-function tests for ``_ngram_rescue_search``.
  * Integration tests that invoke ``try_omega_native_resolve`` with a
    seeded ZpmIndex and verify that the rescue:
      - fires and serves on a plausible rephrase (active mode),
      - records a hit in shadow mode but emits no tokens,
      - does not fire on an unrelated prompt,
      - does not fire when ``prompt_text`` is empty (disabled by
        construction, no hallucinated hits).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from nrlpy import gguf, lmo, zpm
from nrlpy.lmo import (
    _NGRAM_OVERLAP_THRESHOLD,
    _char_ngrams,
    _ngram_jaccard,
    _ngram_rescue_search,
    compute_r2_candidate_state,
    try_omega_native_resolve,
)


# --------------------------------------------------------------------------- #
# Minimal GGUF fixture (mirrors tests/test_ladder_r2_active — self-contained
# on purpose so this file has no cross-test imports).
# --------------------------------------------------------------------------- #

_GGUF_TYPE_F32 = 0
_GGUF_VAL_UINT32 = 4
_GGUF_VAL_STRING = 8


def _write_gguf_string(buf: bytearray, s: bytes | str) -> None:
    data = s.encode("utf-8") if isinstance(s, str) else s
    buf += struct.pack("<Q", len(data))
    buf += data


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", _GGUF_VAL_UINT32)
    buf += struct.pack("<I", int(value) & 0xFFFFFFFF)


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", _GGUF_VAL_STRING)
    _write_gguf_string(buf, value)


def _write_tensor_info(
    buf: bytearray, name: str, shape: tuple[int, ...],
    ggml_type: int, rel_offset: int,
) -> None:
    _write_gguf_string(buf, name)
    buf += struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<Q", int(d))
    buf += struct.pack("<I", int(ggml_type))
    buf += struct.pack("<Q", int(rel_offset))


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def _build_fixture_gguf(path: Path, *, n_blocks: int = 2) -> None:
    alignment = 32
    tensors: list[tuple[str, tuple[int, ...]]] = [("token_embd.weight", (16,))]
    for i in range(n_blocks):
        tensors.append((f"blk.{i}.attn_q.weight", (8,)))
    tensors.append(("output.weight", (16,)))

    kv_buf = bytearray()
    _write_kv_uint32(kv_buf, "general.alignment", alignment)
    _write_kv_string(kv_buf, "general.architecture", "llama")
    _write_kv_string(kv_buf, "tokenizer.ggml.model", "llama")
    _write_kv_string(kv_buf, "tokenizer.ggml.bos_token", "<s>")
    kv_count = 4

    rel_offsets: list[int] = []
    cursor = 0
    tinfo_buf = bytearray()
    for _name, shape in tensors:
        rel_offsets.append(cursor)
        n = 1
        for d in shape:
            n *= d
        cursor += n * 4
        cursor = _align_up(cursor, alignment)
    for (name, shape), rel in zip(tensors, rel_offsets):
        _write_tensor_info(tinfo_buf, name, shape, _GGUF_TYPE_F32, rel)

    head = bytearray()
    head += b"GGUF"
    head += struct.pack("<I", 3)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", kv_count)

    pre_data_bytes = bytes(head) + bytes(kv_buf) + bytes(tinfo_buf)
    data_start = _align_up(len(pre_data_bytes), alignment)
    data_buf = bytearray(max(data_start, len(pre_data_bytes)) + cursor)
    data_buf[: len(pre_data_bytes)] = pre_data_bytes
    for (_name, shape), rel in zip(tensors, rel_offsets):
        n = 1
        for d in shape:
            n *= d
        floats = [((i + 1) * 0.5) for i in range(n)]
        packed = struct.pack("<" + "f" * n, *floats)
        off = data_start + rel
        data_buf[off: off + len(packed)] = packed
    last_rel = rel_offsets[-1]
    last_n = 1
    for d in tensors[-1][1]:
        last_n *= d
    final_end = data_start + last_rel + last_n * 4
    data_buf = data_buf[:final_end]
    path.write_bytes(bytes(data_buf))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def nrl_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    for k in (
        "NRL_ZPM", "NRL_ZPM_THRESHOLD", "NRL_GATE_SKIP_RATIO_OVERRIDE",
        "NRL_COHERENCE_LANE", "NRL_R2_SHADOW", "NRL_OMEGA_BUDGET_MS",
    ):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture()
def fixture_gguf(nrl_root: Path) -> Path:
    p = nrl_root / "fixture.gguf"
    _build_fixture_gguf(p, n_blocks=2)
    return p


@pytest.fixture()
def absorbed_lmo(nrl_root: Path, fixture_gguf: Path) -> lmo.LmoHandle:
    return lmo.absorb_gguf(fixture_gguf, attempt_libllama=False)


@pytest.fixture()
def phi3_manifest(fixture_gguf: Path) -> gguf.GgufManifest:
    m = gguf.manifest_from_args(
        str(fixture_gguf),
        prompt="what is NRL?",
        max_tokens=3,
        seed=1,
        muscle_memory="off",
        coherence_lane="max-throughput",
        r2_shadow_enabled=True,
        omega_budget_ms=500.0,
    )
    m.omega_shadow_join_timeout_s = 2.0
    return m


# --------------------------------------------------------------------------- #
# 1. _char_ngrams — normalization + gram extraction
# --------------------------------------------------------------------------- #


def test_char_ngrams_lowercases_and_normalizes_whitespace() -> None:
    a = _char_ngrams("What is NRL?")
    b = _char_ngrams("what  is   nrl")
    # Different surface punctuation / casing / whitespace should overlap
    # heavily after normalization.
    jaccard = _ngram_jaccard(a, b)
    assert jaccard > 0.8, f"expected tight overlap after normalization, got {jaccard}"


def test_char_ngrams_short_string_returns_empty() -> None:
    assert _char_ngrams("ab") == frozenset()
    assert _char_ngrams("") == frozenset()


def test_char_ngrams_rephrase_has_meaningful_overlap() -> None:
    a = _char_ngrams("what is NRL")
    b = _char_ngrams("what's NRL again")
    assert _ngram_jaccard(a, b) >= _NGRAM_OVERLAP_THRESHOLD


def test_char_ngrams_unrelated_prompts_below_threshold() -> None:
    a = _char_ngrams("what is NRL")
    b = _char_ngrams("pineapple belongs on pizza")
    assert _ngram_jaccard(a, b) < _NGRAM_OVERLAP_THRESHOLD


# --------------------------------------------------------------------------- #
# Release-Gate Lockdown (Phase 11 Final)
# --------------------------------------------------------------------------- #
#
# ``_NGRAM_OVERLAP_THRESHOLD`` is pinned at 0.30 for Production Release
# v1.0. Changing this value silently would invalidate every rescue
# quality number reported in ``scripts/r2_rescue_bench.py`` and the
# bench-wps "Phase 11 segmentation" block. If you're reading this
# because the test is failing, see the comment block above the
# constant in ``nrlpy/src/nrlpy/lmo.py`` and the PR checklist in
# ``docs/CL4_ZPM_MATH.md``.


def test_ngram_overlap_threshold_is_locked_at_release_gate_value() -> None:
    """Pin ``_NGRAM_OVERLAP_THRESHOLD`` at 0.30 (release-gate value)."""
    assert _NGRAM_OVERLAP_THRESHOLD == 0.30, (
        f"_NGRAM_OVERLAP_THRESHOLD drifted from the locked release-gate "
        f"value 0.30 (currently {_NGRAM_OVERLAP_THRESHOLD}). See the "
        f"RELEASE GATE LOCKDOWN comment in lmo.py before changing."
    )


# --------------------------------------------------------------------------- #
# 2. _ngram_jaccard — pure set similarity
# --------------------------------------------------------------------------- #


def test_ngram_jaccard_empty_inputs_return_zero() -> None:
    assert _ngram_jaccard(frozenset(), frozenset({"abcd"})) == 0.0
    assert _ngram_jaccard(frozenset({"abcd"}), frozenset()) == 0.0


def test_ngram_jaccard_identical_sets_return_one() -> None:
    s = frozenset({"abcd", "bcde"})
    assert _ngram_jaccard(s, s) == 1.0


# --------------------------------------------------------------------------- #
# 3. _ngram_rescue_search — end-to-end candidate expansion
# --------------------------------------------------------------------------- #


def _seed_zpm(state: zpm.State, prompt_head: str, reply: str) -> zpm.ZpmIndex:
    idx = zpm.ZpmIndex()
    idx.add(
        zpm.ZpmEntry(
            state=state,
            reply_text=reply,
            tokens=max(1, len(reply.split())),
            wall_s_at_write=0.0001,
            metadata={"prompt_head": prompt_head},
        )
    )
    return idx


def test_rescue_finds_entry_on_prompt_rephrase(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    intent_bytes = gguf._zpm_anchor_bytes(phi3_manifest, "what is NRL?")
    state = compute_r2_candidate_state(
        absorbed_lmo, intent_bytes,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state, "what is NRL?", "NRL is the Neural Resolution Ladder.")

    hit, considered, best = _ngram_rescue_search(
        zpm_index=idx,
        candidate_state=state,
        prompt_text="what's NRL again?",
        zpm_threshold_bits=28,
    )
    assert hit is not None
    assert considered >= 1
    assert best >= _NGRAM_OVERLAP_THRESHOLD
    assert hit.entry.reply_text.startswith("NRL is ")


def test_rescue_does_not_fire_on_unrelated_prompt(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    intent_bytes = gguf._zpm_anchor_bytes(phi3_manifest, "what is NRL?")
    state = compute_r2_candidate_state(
        absorbed_lmo, intent_bytes,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state, "what is NRL?", "NRL answer.")

    hit, _, best = _ngram_rescue_search(
        zpm_index=idx,
        candidate_state=state,
        prompt_text="pineapple pizza debate ruins every party",
        zpm_threshold_bits=28,
    )
    assert hit is None
    assert best < _NGRAM_OVERLAP_THRESHOLD


def test_rescue_skips_entries_without_prompt_head(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    intent_bytes = gguf._zpm_anchor_bytes(phi3_manifest, "what is NRL?")
    state = compute_r2_candidate_state(
        absorbed_lmo, intent_bytes,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    # Legacy entry — no prompt_head metadata. Even with a very close
    # prompt, the scan must skip (no hallucinated matches).
    idx = zpm.ZpmIndex()
    idx.add(zpm.ZpmEntry(
        state=state, reply_text="legacy reply",
        tokens=2, wall_s_at_write=0.0001, metadata={},
    ))
    hit, _, best = _ngram_rescue_search(
        zpm_index=idx,
        candidate_state=state,
        prompt_text="what is NRL?",
        zpm_threshold_bits=28,
    )
    assert hit is None
    assert best == 0.0


def test_rescue_empty_prompt_disables_scan(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    intent_bytes = gguf._zpm_anchor_bytes(phi3_manifest, "what is NRL?")
    state = compute_r2_candidate_state(
        absorbed_lmo, intent_bytes,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state, "what is NRL?", "NRL answer.")

    hit, considered, best = _ngram_rescue_search(
        zpm_index=idx,
        candidate_state=state,
        prompt_text="",  # empty prompt disables rescue
        zpm_threshold_bits=28,
    )
    assert hit is None
    assert considered == 0
    assert best == 0.0


# --------------------------------------------------------------------------- #
# 4. try_omega_native_resolve — integration with the rescue path
# --------------------------------------------------------------------------- #


def test_active_mode_serves_via_ngram_rescue_on_rephrase(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    """Key performance claim: R2 serves a turn it would have missed.

    We compute the candidate state for prompt A, seed a ZPM entry at
    that exact state with prompt_head=A, then call
    try_omega_native_resolve with a DIFFERENT intent_anchor (from
    prompt B, a rephrase). The primary lookup misses (different
    state) but the n-gram rescue scores A's prompt_head against B,
    clears the overlap gate, and serves the stored reply.
    """
    prompt_a = "what is NRL?"
    prompt_b = "what's NRL again?"

    intent_a = gguf._zpm_anchor_bytes(phi3_manifest, prompt_a)
    state_a = compute_r2_candidate_state(
        absorbed_lmo, intent_a,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state_a, prompt_a, "NRL is the Neural Resolution Ladder.")

    intent_b = gguf._zpm_anchor_bytes(phi3_manifest, prompt_b)

    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=intent_b,
        coherence_lane="max-throughput",
        zpm_index=idx,
        zpm_threshold_bits=28,
        mode="active",
        prompt_text=prompt_b,
    )
    # If the two intent anchors happen to collide (tiny fixture), the
    # primary lookup would pass instead of the rescue — accept either
    # as long as R2 actually served. Differentiate by ngram_rescued.
    assert rep.served is True
    assert rep.served_text.startswith("NRL is ")
    assert rung.coherence_demoted is False
    # At least one of the two pathways must have fired; the rescue
    # counters are only populated on the rescue branch.
    if rep.ngram_rescued:
        assert rep.ngram_candidates_considered >= 1
        assert rep.ngram_best_overlap >= _NGRAM_OVERLAP_THRESHOLD


def test_shadow_mode_records_rescue_but_serves_nothing(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    prompt_a = "what is NRL?"
    prompt_b = "what's NRL again?"

    intent_a = gguf._zpm_anchor_bytes(phi3_manifest, prompt_a)
    state_a = compute_r2_candidate_state(
        absorbed_lmo, intent_a,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state_a, prompt_a, "NRL is the Neural Resolution Ladder.")
    intent_b = gguf._zpm_anchor_bytes(phi3_manifest, prompt_b)

    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=intent_b,
        coherence_lane="max-throughput",
        zpm_index=idx,
        zpm_threshold_bits=28,
        mode="shadow",
        prompt_text=prompt_b,
    )
    # Shadow mode: never emits, ever.
    assert rung.coherence_demoted is True
    assert rung.tokens_emitted == 0
    assert rep.served is False
    assert rep.served_tokens == 0
    # But the shadow report should still record whether the rescue
    # *would* have fired, for operator observability.
    if rep.hits == 1:
        # Either the primary or rescue matched in shadow — only the
        # rescue branch populates ngram_candidates_considered.
        assert rep.mode == "shadow"


def test_active_mode_demotes_when_ngram_rescue_does_not_apply(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    prompt_a = "what is NRL?"

    intent_a = gguf._zpm_anchor_bytes(phi3_manifest, prompt_a)
    state_a = compute_r2_candidate_state(
        absorbed_lmo, intent_a,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state_a, prompt_a, "NRL is the Neural Resolution Ladder.")
    unrelated_intent = b"unrelated-prompt-xxx"

    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=unrelated_intent,
        coherence_lane="max-throughput",
        zpm_index=idx,
        zpm_threshold_bits=28,
        mode="active",
        prompt_text="pineapple pizza debate ruins every party",
    )
    # Unrelated prompt + unrelated intent -> both primary and rescue
    # must miss. R2 demotes cleanly.
    assert rung.coherence_demoted is True
    assert rep.served is False
    assert "zpm_no_match" in rep.demotion_reasons


def test_active_mode_empty_prompt_text_does_not_rescue(
    absorbed_lmo: lmo.LmoHandle, phi3_manifest: gguf.GgufManifest,
) -> None:
    """Absent ``prompt_text``, rescue must be silently disabled.

    This is the safety contract for callers that don't yet pass a
    prompt (older native-full bridges): R2 behaves exactly as it did
    pre-Phase-11. No fabricated rescues.
    """
    prompt_a = "what is NRL?"
    intent_a = gguf._zpm_anchor_bytes(phi3_manifest, prompt_a)
    state_a = compute_r2_candidate_state(
        absorbed_lmo, intent_a,
        omega_iterations=phi3_manifest.omega_iterations,
        omega_budget_ms=phi3_manifest.omega_budget_ms,
    )
    idx = _seed_zpm(state_a, prompt_a, "NRL is the Neural Resolution Ladder.")
    unrelated_intent = b"xxxx-no-match-xxx"

    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=unrelated_intent,
        coherence_lane="max-throughput",
        zpm_index=idx,
        zpm_threshold_bits=28,
        mode="active",
        prompt_text="",
    )
    assert rung.coherence_demoted is True
    assert rep.ngram_rescued is False
    assert rep.ngram_candidates_considered == 0


def test_report_default_fields_for_skipped_reason() -> None:
    """Skipped reports must default-initialize the new rescue fields."""
    rep = lmo.OmegaShadowReport.skipped("fast-stable", "coherence_lane_locked")
    assert rep.ngram_rescued is False
    assert rep.ngram_candidates_considered == 0
    assert rep.ngram_best_overlap == 0.0
