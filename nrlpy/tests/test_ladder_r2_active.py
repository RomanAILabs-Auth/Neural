# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Phase 6-EG — Rung R2 (Omega Native Resolve) ACTIVE token-serving tests.

Phase 6-EG promotes R2 from the advisory shadow contract to active
token-service under strict coherence gating. This suite verifies:

* R2 serves tokens when the lane allows AND Stage-VI passes.
* R2 never serves on ``fast-stable`` (hard-locked).
* Stage-VI failures cause clean demotion to R5.
* The ``--no-r2-shadow`` / ``r2_shadow_enabled=False`` manifest flag
  disables R2 service on every lane.
* Token output remains coherent (byte-identical to the baseline when R2
  is disabled, since the stub libllama path is deterministic).
* Demotion rate stays under 5% on synthetic warm workloads (future
  release-gate in Phase 7-EG).
* The CLI ``--coherence-lane max-throughput`` switch activates R2.
* Evidence events carry the new ``r2_active_hits`` /
  ``r2_active_demotions`` / ``r2_served_tokens`` counters.
* The banner surfaces a clear warning line when R2 actually serves.

The suite is CPU-only and deterministic: it runs against the tiny GGUF
fixture used by :mod:`tests.test_lmo` and :mod:`tests.test_ladder_r2_shadow`
through the ``NRL_INFERENCE=stub`` backend.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import time
from pathlib import Path

import pytest

from nrlpy import gguf, ladder, lmo, zpm
from nrlpy.lmo import (
    LANES_ALLOWING_R2_ACTIVE,
    OMEGA_ACTIVE_GATE_SOURCE,
    OMEGA_SHADOW_GATE_SOURCE,
    OmegaShadowReport,
    compute_r2_candidate_state,
    lane_allows_r2_active,
    try_omega_native_resolve,
)


# --------------------------------------------------------------------------- #
# Minimal GGUF fixture (same shape as tests/test_lmo.py — self-contained)
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
    buf: bytearray,
    name: str,
    shape: tuple[int, ...],
    ggml_type: int,
    rel_offset: int,
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
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def nrl_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate NRL_ROOT so cache/lmo/, cache/zpm/, and evidence logs are scoped."""
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
    """Absorb the fixture GGUF into NRL_ROOT/cache/lmo/<sha>/ once per test."""
    return lmo.absorb_gguf(fixture_gguf, attempt_libllama=False)


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #

_PHASE_6_PROMPT = "hello from R2 active test"


def _make_manifest(
    fixture_gguf: Path,
    *,
    coherence_lane: str = "max-throughput",
    r2_shadow_enabled: bool = True,
    muscle_memory: str = "off",
    max_tokens: int = 3,
    omega_budget_ms: float = 500.0,
    prompt: str = _PHASE_6_PROMPT,
) -> gguf.GgufManifest:
    m = gguf.manifest_from_args(
        str(fixture_gguf),
        prompt=prompt,
        max_tokens=max_tokens,
        seed=1,
        muscle_memory=muscle_memory,
        coherence_lane=coherence_lane,
        r2_shadow_enabled=r2_shadow_enabled,
        omega_budget_ms=omega_budget_ms,
    )
    m.omega_shadow_join_timeout_s = 2.0
    return m


def _prime_zpm_index_for_r2(
    manifest: gguf.GgufManifest,
    absorbed: lmo.LmoHandle,
    *,
    reply_text: str,
    reply_tokens: int,
) -> Path:
    """Seed the on-disk ZPM index with a unity state R2 will match.

    Computes the exact candidate state R2 would synthesise for the
    ``(model, prompt)`` pair, wraps it in a :class:`zpm.ZpmEntry` with
    the supplied reply, saves the index, and returns the path. Used by
    active-mode tests to force a deterministic R2 hit without relying
    on libllama to populate the cache.
    """
    intent_bytes = gguf._zpm_anchor_bytes(manifest, manifest.prompt)
    state = compute_r2_candidate_state(
        absorbed,
        intent_bytes,
        omega_iterations=manifest.omega_iterations,
        omega_budget_ms=manifest.omega_budget_ms,
    )
    idx = zpm.ZpmIndex()
    idx.add(
        zpm.ZpmEntry(
            state=state,
            reply_text=reply_text,
            tokens=reply_tokens,
            wall_s_at_write=0.0001,
            metadata={
                "model": Path(manifest.model).name,
                "seed": str(manifest.seed),
                "source": "phase6_test_prime",
            },
        )
    )
    path = gguf._zpm_index_path(manifest.model_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(path)
    return path


# --------------------------------------------------------------------------- #
# 1. R2 serves tokens on max-throughput when Stage-VI passes
# --------------------------------------------------------------------------- #


def test_r2_active_serves_on_max_throughput(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
) -> None:
    # Baseline run to let the runner normalise model_sha256 on the manifest.
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False
    )
    gguf.run_gguf(base_manifest)

    served_reply = "<<<R2-ACTIVE-MAX-THROUGHPUT-RESPONSE>>>"
    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    # The baseline already populated ``model_sha256`` on a fresh manifest
    # via ``run_gguf`` — mirror that normalisation so the index path lines up.
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo, reply_text=served_reply, reply_tokens=9,
    )

    result = gguf.run_gguf(manifest)
    # R2 served this turn.
    assert result.text == served_reply
    assert result.tokens == 9
    assert result.gate_source == OMEGA_ACTIVE_GATE_SOURCE
    assert result.omega_shadow.served is True
    assert result.omega_shadow.mode == "active"
    assert result.omega_shadow.served_tokens == 9
    assert result.omega_shadow.hits == 1
    assert not result.omega_shadow.demotion_reasons
    # No libllama work was done → executed_wall_s is bounded by the R2 probe.
    assert result.tps.executed_wall_s < 1.0


# --------------------------------------------------------------------------- #
# 2. R2 serves on fast-balanced lane
# --------------------------------------------------------------------------- #


def test_r2_active_serves_on_fast_balanced(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
) -> None:
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False
    )
    gguf.run_gguf(base_manifest)

    manifest = _make_manifest(fixture_gguf, coherence_lane="fast-balanced")
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo,
        reply_text="fast-balanced served this turn",
        reply_tokens=5,
    )
    result = gguf.run_gguf(manifest)
    assert result.omega_shadow.served is True
    assert result.omega_shadow.mode == "active"
    assert result.gate_source == OMEGA_ACTIVE_GATE_SOURCE
    assert result.text == "fast-balanced served this turn"


# --------------------------------------------------------------------------- #
# 3. R2 must NEVER serve on fast-stable, even with a matching index
# --------------------------------------------------------------------------- #


def test_r2_active_never_serves_on_fast_stable(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
) -> None:
    # Prime the index against a max-throughput run so the state would
    # match if fast-stable ever consulted the lattice.
    mt_manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    gguf.run_gguf(mt_manifest)
    _prime_zpm_index_for_r2(
        mt_manifest, absorbed_lmo,
        reply_text="!!! SHOULD NEVER LEAK TO FAST-STABLE !!!",
        reply_tokens=4,
    )

    manifest = _make_manifest(fixture_gguf, coherence_lane="fast-stable")
    manifest.model_sha256 = mt_manifest.model_sha256
    result = gguf.run_gguf(manifest)
    # R2 was not consulted: the stored reply must not appear in the output.
    assert "!!! SHOULD NEVER LEAK TO FAST-STABLE !!!" not in result.text
    assert result.omega_shadow.served is False
    assert result.gate_source != OMEGA_ACTIVE_GATE_SOURCE
    # Report is skipped with the correct reason.
    assert result.omega_shadow.status == "skipped"
    assert "coherence_lane_disallows_r2_active" in (
        result.omega_shadow.demotion_reasons
    )


# --------------------------------------------------------------------------- #
# 4. Stage-VI failure → clean demotion to R5
# --------------------------------------------------------------------------- #


def test_r2_active_stage_vi_failure_demotes_to_r5(
    fixture_gguf: Path,
    absorbed_lmo: lmo.LmoHandle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prime the ZPM index so R2 would otherwise succeed.
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False
    )
    base_result = gguf.run_gguf(base_manifest)

    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo,
        reply_text="!!! WOULD HAVE BEEN SERVED !!!",
        reply_tokens=6,
    )

    # Force every Stage-VI audit to fail with a typed reason.
    monkeypatch.setattr(
        lmo, "_stage_vi_shadow_audit",
        lambda *a, **kw: "stage_vi_symmetry_drift",
    )

    result = gguf.run_gguf(manifest)
    # R2 must NOT have served → libllama output takes over, identical to
    # the baseline.
    assert result.text == base_result.text
    assert "!!! WOULD HAVE BEEN SERVED !!!" not in result.text
    assert result.omega_shadow.served is False
    assert result.omega_shadow.mode == "active"
    assert "stage_vi_symmetry_drift" in result.omega_shadow.demotion_reasons
    assert result.gate_source != OMEGA_ACTIVE_GATE_SOURCE


# --------------------------------------------------------------------------- #
# 5. R2 disabled via r2_shadow_enabled=False → identical baseline output
# --------------------------------------------------------------------------- #


def test_r2_disabled_flag_produces_byte_identical_baseline(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    # Even with a primed ZPM index, disabling R2 must bypass it entirely.
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False
    )
    base_result = gguf.run_gguf(base_manifest)
    _prime_zpm_index_for_r2(
        base_manifest, absorbed_lmo,
        reply_text="!!! FLAG-OFF MUST NEVER SEE THIS !!!",
        reply_tokens=4,
    )

    disabled_manifest = _make_manifest(
        fixture_gguf, coherence_lane="max-throughput", r2_shadow_enabled=False,
    )
    disabled_manifest.model_sha256 = base_manifest.model_sha256
    disabled_result = gguf.run_gguf(disabled_manifest)

    assert disabled_result.text == base_result.text
    assert "!!! FLAG-OFF MUST NEVER SEE THIS !!!" not in disabled_result.text
    assert disabled_result.omega_shadow.served is False
    assert disabled_result.omega_shadow.status == "skipped"
    assert "r2_shadow_disabled" in (
        disabled_result.omega_shadow.demotion_reasons
    )


# --------------------------------------------------------------------------- #
# 6. RungResult contract on active service
# --------------------------------------------------------------------------- #


def test_try_omega_native_resolve_active_serves_when_stage_vi_passes(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    # Normalise model_sha256.
    gguf.run_gguf(manifest)

    intent_bytes = b"phase6-direct-invocation"
    state = compute_r2_candidate_state(
        absorbed_lmo, intent_bytes,
        omega_iterations=manifest.omega_iterations,
        omega_budget_ms=manifest.omega_budget_ms,
    )
    idx = zpm.ZpmIndex()
    idx.add(zpm.ZpmEntry(
        state=state, reply_text="direct-served",
        tokens=7, wall_s_at_write=0.0001, metadata={},
    ))

    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=intent_bytes,
        coherence_lane="max-throughput",
        zpm_index=idx,
        mode="active",
    )
    assert rung.coherence_demoted is False
    assert rung.gate_source == OMEGA_ACTIVE_GATE_SOURCE
    assert rung.tokens_emitted == 7
    assert rep.served is True
    assert rep.served_tokens == 7
    assert rep.served_text == "direct-served"
    assert rep.mode == "active"
    # Report gate_source remains the shadow constant — it identifies the
    # report, not the rung decision.
    assert rep.gate_source == OMEGA_SHADOW_GATE_SOURCE


def test_try_omega_native_resolve_active_demotes_when_no_match(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    empty_idx = zpm.ZpmIndex()
    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=b"no-match-prompt",
        coherence_lane="max-throughput",
        zpm_index=empty_idx,
        mode="active",
    )
    assert rung.coherence_demoted is True
    assert rung.gate_source == OMEGA_SHADOW_GATE_SOURCE  # demoted → shadow label
    assert rep.served is False
    assert rep.served_tokens == 0
    assert rep.mode == "active"
    assert "no_zpm_index" in rep.demotion_reasons


def test_try_omega_native_resolve_active_rejects_fast_stable(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    rung, rep = try_omega_native_resolve(
        absorbed_lmo,
        intent_anchor_bytes=b"fast-stable-must-reject",
        coherence_lane="fast-stable",
        zpm_index=zpm.ZpmIndex(),
        mode="active",
    )
    assert rung.coherence_demoted is True
    assert rep.status == "skipped"
    assert "coherence_lane_disallows_r2_active" in rep.demotion_reasons
    assert rep.served is False


# --------------------------------------------------------------------------- #
# 7. Lane predicate symmetry
# --------------------------------------------------------------------------- #


def test_lane_allows_r2_active_matches_expected_lanes() -> None:
    assert lane_allows_r2_active("max-throughput") is True
    assert lane_allows_r2_active("fast-balanced") is True
    assert lane_allows_r2_active("fast-stable") is False
    assert lane_allows_r2_active("definitely-not-a-lane") is False
    assert LANES_ALLOWING_R2_ACTIVE == frozenset(
        {"fast-balanced", "max-throughput"}
    )


def test_ladder_lane_allows_r2_active_in_ctx_respects_flag(
    fixture_gguf: Path,
) -> None:
    ctx_on = ladder.LadderShadowContext(
        model_sha256="abc",
        intent_anchor_bytes=b"x",
        coherence_lane="max-throughput",
        omega_budget_ms=2.0,
        omega_candidates=4,
        omega_iterations=3,
        zpm_threshold_bits=0,
        r2_shadow_enabled=True,
    )
    assert ladder.lane_allows_r2_active_in_ctx(ctx_on) is True
    ctx_off = ladder.LadderShadowContext(
        model_sha256="abc",
        intent_anchor_bytes=b"x",
        coherence_lane="max-throughput",
        omega_budget_ms=2.0,
        omega_candidates=4,
        omega_iterations=3,
        zpm_threshold_bits=0,
        r2_shadow_enabled=False,
    )
    assert ladder.lane_allows_r2_active_in_ctx(ctx_off) is False
    ctx_stable = ladder.LadderShadowContext(
        model_sha256="abc",
        intent_anchor_bytes=b"x",
        coherence_lane="fast-stable",
        omega_budget_ms=2.0,
        omega_candidates=4,
        omega_iterations=3,
        zpm_threshold_bits=0,
        r2_shadow_enabled=True,
    )
    assert ladder.lane_allows_r2_active_in_ctx(ctx_stable) is False


# --------------------------------------------------------------------------- #
# 8. Evidence log: r2_active_hits / r2_active_demotions / r2_served_tokens
# --------------------------------------------------------------------------- #


def test_evidence_log_records_r2_active_fields_on_hit(
    nrl_root: Path, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False,
    )
    gguf.run_gguf(base_manifest)

    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    manifest.model_sha256 = base_manifest.model_sha256
    evidence = nrl_root / "evidence.jsonl"
    manifest.evidence_log = str(evidence)
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo, reply_text="active-hit-payload", reply_tokens=12,
    )
    gguf.run_gguf(manifest)

    lines = [
        json.loads(l) for l in evidence.read_text().splitlines() if l.strip()
    ]
    ev = lines[-1]
    assert ev["r2_active_hits"] == 1
    assert ev["r2_active_demotions"] == 0
    assert ev["r2_served_tokens"] == 12
    assert ev["gate_source"] == OMEGA_ACTIVE_GATE_SOURCE


def test_evidence_log_records_r2_active_demotion(
    nrl_root: Path, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    # No ZPM priming → R2 active runs and demotes (no_zpm_index).
    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    evidence = nrl_root / "evidence.jsonl"
    manifest.evidence_log = str(evidence)
    gguf.run_gguf(manifest)

    lines = [
        json.loads(l) for l in evidence.read_text().splitlines() if l.strip()
    ]
    ev = lines[-1]
    assert ev["r2_active_hits"] == 0
    assert ev["r2_active_demotions"] == 1
    assert ev["r2_served_tokens"] == 0
    # Gate source is the libllama fallback path (None or prefill/override).
    assert ev["gate_source"] != OMEGA_ACTIVE_GATE_SOURCE


# --------------------------------------------------------------------------- #
# 9. Banner carries the ACTIVE block and the served-warning line
# --------------------------------------------------------------------------- #


def test_banner_shows_r2_active_warning_when_served(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False,
    )
    gguf.run_gguf(base_manifest)

    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo,
        reply_text="banner-warning-reply", reply_tokens=5,
    )
    result = gguf.run_gguf(manifest)
    banner = gguf.format_banner(result)
    assert "Phase 6-EG ACTIVE" in banner
    assert "R2 ACTIVE SERVED TOKENS THIS TURN" in banner
    assert "served            yes" in banner
    assert "mode              active" in banner


# --------------------------------------------------------------------------- #
# 10. Demotion-rate tracking across a batch of allowed-lane turns
# --------------------------------------------------------------------------- #


def test_demotion_rate_trackable_across_turns(
    nrl_root: Path, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    """Demotion-rate aggregation shape (Phase-7-EG release-gate precursor).

    We run one R2-served turn (after priming the index) and several
    R2-demoted turns with varying prompts to get the counters wiggling.
    The assertion is shape-based: we verify each evidence event carries
    exactly one of ``r2_active_hits==1`` or ``r2_active_demotions==1``
    (mutually exclusive per §4.3) so a downstream aggregator can
    compute ``demotion_rate = sum(dem) / (sum(hit) + sum(dem))``
    unambiguously.
    """
    evidence = nrl_root / "evidence.jsonl"
    # Turn 1: prime and serve.
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False,
    )
    gguf.run_gguf(base_manifest)
    served_manifest = _make_manifest(
        fixture_gguf, coherence_lane="max-throughput",
    )
    served_manifest.model_sha256 = base_manifest.model_sha256
    served_manifest.evidence_log = str(evidence)
    _prime_zpm_index_for_r2(
        served_manifest, absorbed_lmo,
        reply_text="rate-tracked", reply_tokens=3,
    )
    gguf.run_gguf(served_manifest)

    # Turns 2-4: different prompts, same lane — R2 will miss each time.
    for i in range(3):
        m = _make_manifest(
            fixture_gguf,
            coherence_lane="max-throughput",
            prompt=f"unrelated prompt #{i}",
        )
        m.model_sha256 = base_manifest.model_sha256
        m.evidence_log = str(evidence)
        gguf.run_gguf(m)

    lines = [
        json.loads(l) for l in evidence.read_text().splitlines() if l.strip()
    ]
    # Filter to the max-throughput events only.
    mt = [e for e in lines if e.get("coherence_lane") == "max-throughput"]
    assert len(mt) == 4
    hits = sum(e["r2_active_hits"] for e in mt)
    dems = sum(e["r2_active_demotions"] for e in mt)
    # The primed turn served; the other three demoted.
    assert hits == 1
    assert dems == 3
    # Mutual-exclusion invariant (required for clean rate math).
    for e in mt:
        assert e["r2_active_hits"] + e["r2_active_demotions"] <= 1


# --------------------------------------------------------------------------- #
# 11. CLI --coherence-lane max-throughput activates R2 and serves
# --------------------------------------------------------------------------- #


def test_cli_coherence_lane_activates_r2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI integration: ``--coherence-lane max-throughput`` + a
    primed ZPM index must make R2 emit the stored reply via ``nrlpy run``.
    """
    # Set up isolated NRL_ROOT for the subprocess.
    env_root = tmp_path
    fixture = env_root / "fixture.gguf"
    _build_fixture_gguf(fixture, n_blocks=2)

    monkeypatch.setenv("NRL_ROOT", str(env_root))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.delenv("NRL_COHERENCE_LANE", raising=False)
    monkeypatch.delenv("NRL_R2_SHADOW", raising=False)

    # 1) Absorb the fixture (populates cache/lmo/<sha>/).
    absorbed = lmo.absorb_gguf(fixture, attempt_libllama=False)

    # 2) Prime the ZPM index via an in-process call so we know the state
    #    matches what R2 will compute in the subprocess.
    m = _make_manifest(fixture, coherence_lane="max-throughput")
    gguf.run_gguf(m)  # normalises model_sha256 on the manifest
    _prime_zpm_index_for_r2(
        m, absorbed,
        reply_text="CLI-ACTIVE-R2-PAYLOAD", reply_tokens=8,
    )

    # 3) Run the CLI with the lane flag.
    from nrlpy.cli import main as cli_main

    # Capture stdout via a monkeypatched stream.
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        rc = cli_main([
            "run", str(fixture),
            "--prompt", _PHASE_6_PROMPT,
            "--coherence-lane", "max-throughput",
            "--max-tokens", "3",
            "--seed", "1",
            "--no-stream",
        ])
    finally:
        monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    output = buf.getvalue()

    assert rc == 0
    # The banner includes the Phase 6-EG ACTIVE block and the served warning.
    assert "Phase 6-EG ACTIVE" in output
    assert "R2 ACTIVE SERVED TOKENS THIS TURN" in output
    # The stored reply text is referenced in the banner via the served_tokens
    # line (actual text goes to stream_to, which we suppressed with --no-stream).
    assert "gate_source       omega_resolve_shadow" in output


# --------------------------------------------------------------------------- #
# 12. CLI --no-r2-shadow disables R2 service even on max-throughput
# --------------------------------------------------------------------------- #


def test_cli_no_r2_shadow_disables_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_root = tmp_path
    fixture = env_root / "fixture.gguf"
    _build_fixture_gguf(fixture, n_blocks=2)
    monkeypatch.setenv("NRL_ROOT", str(env_root))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.delenv("NRL_COHERENCE_LANE", raising=False)
    monkeypatch.delenv("NRL_R2_SHADOW", raising=False)

    absorbed = lmo.absorb_gguf(fixture, attempt_libllama=False)
    m = _make_manifest(fixture, coherence_lane="max-throughput")
    gguf.run_gguf(m)
    _prime_zpm_index_for_r2(
        m, absorbed,
        reply_text="SHOULD-NOT-BE-SERVED-FLAG-OFF", reply_tokens=5,
    )

    from nrlpy.cli import main as cli_main

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        rc = cli_main([
            "run", str(fixture),
            "--prompt", _PHASE_6_PROMPT,
            "--coherence-lane", "max-throughput",
            "--no-r2-shadow",
            "--max-tokens", "3",
            "--seed", "1",
            "--no-stream",
        ])
    finally:
        monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    output = buf.getvalue()

    assert rc == 0
    assert "R2 ACTIVE SERVED TOKENS THIS TURN" not in output
    assert "SHOULD-NOT-BE-SERVED-FLAG-OFF" not in output
    # Banner still shows the R2 section but as skipped.
    assert "served            no" in output


# --------------------------------------------------------------------------- #
# 13. Executed-TPS accounting on R2 service
# --------------------------------------------------------------------------- #


def test_r2_active_accounts_tokens_at_lattice_rate(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    """§7.3 contract — R2 contributes to ``executed_tokens`` (not cache),
    with wall_s bounded by the R2 probe rather than a libllama decode."""
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False,
    )
    gguf.run_gguf(base_manifest)
    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo,
        reply_text="timing-contract-reply", reply_tokens=17,
    )

    result = gguf.run_gguf(manifest)
    assert result.omega_shadow.served is True
    assert result.tps.executed_tokens == 17
    assert result.tps.cache_tokens == 0
    # Lattice rate: the wall-clock spent on R2 should be <<< 1s for the
    # fixture. A 2 s cap gives plenty of headroom for loaded CI machines.
    assert result.tps.executed_wall_s < 2.0
    assert result.tps.executed_tps > 0.0


# --------------------------------------------------------------------------- #
# 14. Overhead bound — running R2 active on the decode hot path stays cheap
# --------------------------------------------------------------------------- #


def test_r2_active_overhead_is_bounded(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    """R2 active must not explode the hot path when it demotes.

    Baseline: fast-stable (no R2) vs max-throughput (R2 runs, demotes).
    We assert the *delta* stays within a generous ceiling; R2's
    ``omega_budget_ms`` default is 2 ms so the added latency should be
    small relative to the stub-backend decode.
    """
    base = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable",
        r2_shadow_enabled=False, max_tokens=2,
    )
    t = time.perf_counter()
    gguf.run_gguf(base)
    base_wall = time.perf_counter() - t

    probe = _make_manifest(
        fixture_gguf, coherence_lane="max-throughput",
        r2_shadow_enabled=True, max_tokens=2,
        omega_budget_ms=1.0,
    )
    t = time.perf_counter()
    gguf.run_gguf(probe)
    probe_wall = time.perf_counter() - t

    delta = probe_wall - base_wall
    # CI machines are noisy — allow up to 250 ms of delta. The contract
    # we actually care about is that R2 does not dominate decode time.
    assert delta < 0.25, f"R2 active overhead too high: delta={delta:.4f}s"


# --------------------------------------------------------------------------- #
# 15. OmegaShadowReport shape correctness on served path
# --------------------------------------------------------------------------- #


def test_omega_shadow_report_served_shape(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    base_manifest = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable", r2_shadow_enabled=False,
    )
    gguf.run_gguf(base_manifest)

    manifest = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    manifest.model_sha256 = base_manifest.model_sha256
    _prime_zpm_index_for_r2(
        manifest, absorbed_lmo,
        reply_text="shape-check", reply_tokens=2,
    )
    result = gguf.run_gguf(manifest)
    rep = result.omega_shadow
    # Invariants on served shape.
    assert rep.served is True
    assert rep.mode == "active"
    assert rep.served_text == "shape-check"
    assert rep.served_tokens == 2
    assert rep.stored_entry_index >= 0
    assert rep.hits == 1
    assert rep.zpm_distance_bits == 0  # exact match
    assert rep.candidate_continuation_fnv  # non-empty hex digest
    # Gate source label stays the *report* label (constant).
    assert rep.gate_source == OMEGA_SHADOW_GATE_SOURCE
    # The demotion_reasons tuple is empty on a successful service.
    assert rep.demotion_reasons == ()


# --------------------------------------------------------------------------- #
# 16. Demotion reasons all come from the closed set (release-gate precursor)
# --------------------------------------------------------------------------- #


def test_r2_active_demotion_reasons_are_in_closed_set(
    fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    # Scenario A: no ZPM index → "no_zpm_index".
    manifest_a = _make_manifest(fixture_gguf, coherence_lane="max-throughput")
    result_a = gguf.run_gguf(manifest_a)
    for r in result_a.omega_shadow.demotion_reasons:
        assert r in ladder.R2_ACTIVE_DEMOTION_REASONS, (
            f"unknown demotion reason leaked: {r!r}"
        )

    # Scenario B: lane disallows → "coherence_lane_disallows_r2_active".
    manifest_b = _make_manifest(fixture_gguf, coherence_lane="fast-stable")
    result_b = gguf.run_gguf(manifest_b)
    for r in result_b.omega_shadow.demotion_reasons:
        assert r in ladder.R2_ACTIVE_DEMOTION_REASONS


# --------------------------------------------------------------------------- #
# 17. Cache-hit path preempts R2 (skipped with a typed reason)
# --------------------------------------------------------------------------- #


def test_cache_hit_preempts_r2_and_records_skipped(
    nrl_root: Path, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle,
) -> None:
    # Warm muscle memory so the second call is a cache hit.
    warm = _make_manifest(
        fixture_gguf, coherence_lane="fast-stable",
        r2_shadow_enabled=False, muscle_memory="on",
    )
    gguf.run_gguf(warm)

    hot = _make_manifest(
        fixture_gguf, coherence_lane="max-throughput", muscle_memory="on",
    )
    hot.model_sha256 = warm.model_sha256
    result = gguf.run_gguf(hot)
    assert result.cache_hit is True
    assert result.omega_shadow.status == "skipped"
    assert "cache_hit_preempted_r2" in result.omega_shadow.demotion_reasons
    assert result.omega_shadow.served is False
