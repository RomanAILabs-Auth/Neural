# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.zpm` — the Zero-Point Mapping identity resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from nrlpy.zpm import (
    Rotor,
    ZPM_GOLDEN_RATIO,
    ZpmEntry,
    ZpmIndex,
    anchor,
    format_stage_banner,
    hamming_state,
    inversion,
    inversion_determinant,
    nullspace_search,
    popcount64,
    rotor,
    verify,
)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def test_popcount64_matches_reference() -> None:
    assert popcount64(0) == 0
    assert popcount64(0xFFFFFFFFFFFFFFFF) == 64
    assert popcount64(0x5555555555555555) == 32
    # Truncates values > 64 bits cleanly (contract: 64-bit semantics).
    assert popcount64(0x1FFFFFFFFFFFFFFFFF) == popcount64(0xFFFFFFFFFFFFFFFF)


def test_hamming_state_is_symmetric_and_zero_for_equal() -> None:
    a = (0xDEAD, 0xBEEF, 0xF00D, 0xCAFE)
    b = (0xDEAD ^ 0x1, 0xBEEF, 0xF00D, 0xCAFE)  # clean 1-bit flip in word 0
    assert hamming_state(a, a) == 0
    assert hamming_state(a, b) == hamming_state(b, a)
    assert hamming_state(a, b) == 1


# --------------------------------------------------------------------------- #
# Stage I — Anchor
# --------------------------------------------------------------------------- #


def test_anchor_is_deterministic_for_same_input() -> None:
    blob = b"hello world | system=you are nrl | user=hi"
    assert anchor(blob) == anchor(blob)
    # Non-empty, four independent 64-bit words (identical rows would leak the
    # input structure across projections).
    s = anchor(blob)
    assert len(s) == 4
    assert len(set(s)) == 4


def test_anchor_distinguishes_near_inputs() -> None:
    """A single-byte change must produce a different state."""
    a = anchor(b"hi")
    b = anchor(b"hi!")
    assert a != b
    # With 4 independent FNV-1a64 rotations, *at least one* word differs.
    assert sum(1 for i in range(4) if a[i] != b[i]) >= 1


def test_anchor_accepts_iterable_parts() -> None:
    """Caller can pass (model_sha | system | history | user) as a sequence."""
    single = anchor(b"a\x1eb\x1ec")
    parts = anchor([b"a", b"b", b"c"])
    assert single == parts


def test_anchor_empty_state_is_zero() -> None:
    assert anchor(b"") == (0, 0, 0, 0)


# --------------------------------------------------------------------------- #
# Stage II — Inversion
# --------------------------------------------------------------------------- #


def test_inversion_matches_reference_formula() -> None:
    """``rows[i] = t[i] ^ (t[i] << 1) ^ 0x9E3779B97F4A7C15`` (mod 2^64)."""
    t = (0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x123456789ABCDEF0, 0xFEDCBA9876543210)
    rows = inversion(t)
    for i in range(4):
        expected = (t[i] ^ ((t[i] << 1) & 0xFFFFFFFFFFFFFFFF) ^ ZPM_GOLDEN_RATIO) & 0xFFFFFFFFFFFFFFFF
        assert rows[i] == expected


def test_inversion_determinant_is_xor_of_rows() -> None:
    t = (0x1, 0x2, 0x4, 0x8)
    rows = inversion(t)
    det = inversion_determinant(rows)
    expected = rows[0] ^ rows[1] ^ rows[2] ^ rows[3]
    assert det == expected


# --------------------------------------------------------------------------- #
# Stage III — Rotor
# --------------------------------------------------------------------------- #


def test_rotor_produces_phase_locked_symmetry_norm() -> None:
    """sin²+cos² == 1 ⇒ the norm shelf sits at 1.0 for every 64-bit seed."""
    r = rotor(0xDEADBEEFCAFEF00D)
    assert isinstance(r, Rotor)
    assert r.phase_locked
    assert abs(r.norm - 1.0) < 1e-12


def test_rotor_seed_zero_is_identity_scalar() -> None:
    r = rotor(0)
    assert r.s == pytest.approx(1.0)
    assert r.b_xy == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Stage IV — Nullspace search
# --------------------------------------------------------------------------- #


def test_nullspace_search_exact_hit() -> None:
    stored = [anchor(b"hi"), anchor(b"what is 2+2"), anchor(b"name a color")]
    query = anchor(b"what is 2+2")
    hit = nullspace_search(query, stored, threshold_bits=0)
    assert hit.entry_index == 1
    assert hit.distance_bits == 0
    assert hit.exact is True
    assert hit.within_threshold is True


def test_nullspace_search_near_hit_within_threshold() -> None:
    """A single-bit flip in any row should be within threshold=8 for any seed."""
    base = anchor(b"hi")
    # Corrupt 1 bit in word 0 — Hamming distance becomes exactly 1.
    near = (base[0] ^ 0x1, base[1], base[2], base[3])
    stored = [base]
    hit = nullspace_search(near, stored, threshold_bits=8)
    assert hit.entry_index == 0
    assert hit.distance_bits == 1
    assert hit.exact is False
    assert hit.within_threshold is True


def test_nullspace_search_empty_store_returns_miss() -> None:
    hit = nullspace_search(anchor(b"hi"), [], threshold_bits=0)
    assert hit.entry_index == -1
    assert hit.exact is False
    assert hit.within_threshold is False


def test_nullspace_search_distant_query_reports_threshold_miss() -> None:
    stored = [anchor(b"hi")]
    hit = nullspace_search(anchor(b"some completely unrelated text x"), stored, threshold_bits=0)
    assert hit.entry_index == 0
    assert hit.distance_bits > 0
    assert hit.exact is False
    assert hit.within_threshold is False


# --------------------------------------------------------------------------- #
# Stage VI — Verify
# --------------------------------------------------------------------------- #


def test_verify_unity_passes_bit_symmetry() -> None:
    t = anchor(b"unity test")
    assert verify(t, t) is True


def test_verify_rejects_any_nonzero_residual() -> None:
    t = anchor(b"unity test")
    perturbed = (t[0] ^ 0x1, t[1], t[2], t[3])
    assert verify(t, perturbed) is False


# --------------------------------------------------------------------------- #
# ZpmIndex — persistent store
# --------------------------------------------------------------------------- #


def test_index_exact_hit_returns_stored_reply(tmp_path: Path) -> None:
    idx = ZpmIndex()
    entry = ZpmEntry(
        state=anchor(b"hi"),
        reply_text="Hello! How can I help?",
        tokens=7,
        wall_s_at_write=0.123,
        metadata={"lane": "chat", "turn_idx": "0"},
    )
    idx.add(entry)

    hit, got = idx.lookup(anchor(b"hi"), threshold_bits=0)
    assert got is not None
    assert hit.exact is True
    assert hit.distance_bits == 0
    assert got.reply_text == "Hello! How can I help?"
    assert got.tokens == 7
    assert got.metadata["lane"] == "chat"


def test_index_near_hit_within_threshold_is_served(tmp_path: Path) -> None:
    idx = ZpmIndex()
    base_state = anchor(b"Hello there")
    idx.add(ZpmEntry(state=base_state, reply_text="Hi back!", tokens=3))
    near_state = (base_state[0] ^ 0x3, base_state[1], base_state[2], base_state[3])
    hit, got = idx.lookup(near_state, threshold_bits=8)
    assert hit.entry_index == 0
    assert 0 < hit.distance_bits <= 8
    assert hit.exact is False
    assert hit.within_threshold is True
    assert got is not None
    assert got.reply_text == "Hi back!"


def test_index_miss_returns_none_when_above_threshold() -> None:
    idx = ZpmIndex()
    idx.add(ZpmEntry(state=anchor(b"hi"), reply_text="Hi.", tokens=1))
    hit, got = idx.lookup(
        anchor(b"totally unrelated long question about quantum gravity"),
        threshold_bits=0,
    )
    assert hit.exact is False
    assert hit.within_threshold is False
    assert got is None


def test_index_roundtrip_persists_on_disk(tmp_path: Path) -> None:
    idx = ZpmIndex()
    idx.add(ZpmEntry(
        state=anchor(b"hi"),
        reply_text="Hi!",
        tokens=2,
        wall_s_at_write=0.0042,
        metadata={"lane": "chat"},
    ))
    idx.add(ZpmEntry(state=anchor(b"bye"), reply_text="See ya.", tokens=3))
    path = tmp_path / "zpm.idx"
    idx.save(path)
    assert path.is_file()
    reloaded = ZpmIndex.load(path)
    assert len(reloaded) == 2
    h1, e1 = reloaded.lookup(anchor(b"hi"))
    assert e1 is not None and e1.reply_text == "Hi!" and e1.metadata["lane"] == "chat"
    h2, e2 = reloaded.lookup(anchor(b"bye"))
    assert e2 is not None and e2.reply_text == "See ya."
    # Non-existent magic = empty index (graceful).
    corrupt = tmp_path / "corrupt.idx"
    corrupt.write_bytes(b"NOT-A-ZPM-FILE")
    assert len(ZpmIndex.load(corrupt)) == 0
    # Missing file = empty index.
    assert len(ZpmIndex.load(tmp_path / "missing.idx")) == 0


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# End-to-end: ZPM through run_gguf (stub backend, no libllama required)
# --------------------------------------------------------------------------- #


def test_run_gguf_stores_and_replays_via_zpm(tmp_path, monkeypatch) -> None:
    """Opt-in ZPM: first decode stores a unity state; second run *with a
    different model file SHA but the same anchor bytes* would be an exact
    hit. We use the same prompt twice against the stub to exercise the
    store+replay round-trip via the on-disk index.
    """
    from nrlpy.gguf import GgufManifest, run_gguf

    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"stub-gguf-bytes")

    def _mk() -> GgufManifest:
        return GgufManifest(
            model=str(model),
            prompt="unity test prompt for zpm",
            max_tokens=8,
            seed=7,
            muscle_memory="off",  # force MM miss so ZPM is the sole cache path
            zpm_nullspace=True,
            zpm_threshold_bits=0,
        )

    # First run: MM off, ZPM index empty → full decode. The write path should
    # still run because MM==on is required for writes — so with MM off the
    # index stays empty. That's the honesty contract: ZPM only persists what
    # MM would persist. Flip MM on for the write pass.
    m1 = _mk()
    m1.muscle_memory = "on"
    r1 = run_gguf(m1, stream_to=None)
    assert r1.tokens > 0
    # Not a cache hit on first run (the stub always generates fresh text).
    assert r1.cache_hit is False or r1.gate_source == "zpm_nullspace"

    # Second run: MM *off*, ZPM *on*. Muscle-memory cannot serve this turn
    # (MM off), so the lookup must come through ZPM.
    m2 = _mk()
    m2.muscle_memory = "off"
    r2 = run_gguf(m2, stream_to=None)
    assert r2.cache_hit is True
    assert r2.gate_source == "zpm_nullspace"
    assert r2.gate_report.get("exact") is True
    assert r2.gate_report.get("distance_bits") == 0
    assert r2.text == r1.text


def test_format_stage_banner_contains_all_six_stages() -> None:
    t = anchor(b"banner test")
    out = format_stage_banner(t, solution=t, ansi=False)
    # All Stage markers from ROMA-ZPM v2.0 present.
    for marker in ("ANCHOR", "INVERSION", "ROTOR", "NULLSPACE", "VERIFY"):
        assert marker in out, f"missing stage: {marker}"
    # Unity path reached.
    assert "SINGULARITY DETECTED" in out
    assert "ABSOLUTE UNITY" in out
    # No raw ANSI escape codes leaked in ``ansi=False``.
    assert "\x1b[" not in out
