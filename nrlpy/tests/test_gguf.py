# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Unit tests for ``nrlpy.gguf`` — manifest, muscle memory, TPS math, CLI sugar.

These tests do not touch ``llama-cpp-python``. They mock the LLM via a stub that
satisfies the ``create_completion`` stream contract.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nrlpy import gguf

# --------------------------------------------------------------------------- #
# Manifest parser
# --------------------------------------------------------------------------- #


def test_manifest_minimal_v1(tmp_path: Path) -> None:
    p = tmp_path / "ok.nrl"
    p.write_text(
        """
        schema = nrl.manifest.v1
        mode = gguf_run
        model = test.gguf
        prompt = "hi"
        max_tokens = 8
        """,
        encoding="utf-8",
    )
    m = gguf.load_manifest(p)
    assert m.schema == gguf.MANIFEST_SCHEMA_V1
    assert m.mode == "gguf_run"
    assert m.model.endswith("test.gguf")
    assert m.prompt == "hi"
    assert m.max_tokens == 8
    assert m.benchmark_class == "B"


def test_manifest_missing_schema_fails(tmp_path: Path) -> None:
    p = tmp_path / "bad.nrl"
    p.write_text("mode = gguf_run\nmodel = test.gguf\n", encoding="utf-8")
    with pytest.raises(gguf.ManifestError, match="missing 'schema"):
        gguf.load_manifest(p)


def test_manifest_unknown_key_fails(tmp_path: Path) -> None:
    p = tmp_path / "bad.nrl"
    p.write_text(
        "schema = nrl.manifest.v1\nmode = gguf_run\nmodel = x.gguf\nbogus = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(gguf.ManifestError, match="unknown key"):
        gguf.load_manifest(p)


def test_manifest_class_a_requires_seed() -> None:
    with pytest.raises(gguf.ManifestError, match="benchmark_class=A"):
        gguf.parse_manifest_text(
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = m.gguf\n"
            "benchmark_class = A\n"
            "seed = 0\n"
        )


def test_manifest_prompt_and_prompt_file_exclusive() -> None:
    with pytest.raises(gguf.ManifestError, match="mutually exclusive"):
        gguf.parse_manifest_text(
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = m.gguf\n"
            "prompt = hi\n"
            "prompt_file = p.txt\n"
        )


def test_manifest_model_sha256_validation() -> None:
    with pytest.raises(gguf.ManifestError, match="64 hex chars"):
        gguf.parse_manifest_text(
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = m.gguf\n"
            "model_sha256 = abc\n"
        )


def test_manifest_from_args_class_a_requires_seed() -> None:
    with pytest.raises(gguf.ManifestError):
        gguf.manifest_from_args(model="x.gguf", benchmark_class="A", seed=0)


def test_sampler_fingerprint_stability() -> None:
    m = gguf.manifest_from_args(model="x.gguf", seed=1, temperature=0.7, top_p=0.9)
    fp1 = m.sampler_fingerprint()
    m2 = gguf.manifest_from_args(model="x.gguf", seed=1, temperature=0.7, top_p=0.9)
    assert fp1 == m2.sampler_fingerprint()
    m2.temperature = 0.8
    assert fp1 != m2.sampler_fingerprint()


# --------------------------------------------------------------------------- #
# TPS math
# --------------------------------------------------------------------------- #


def test_tps_report_finalize_basic() -> None:
    r = gguf.TpsReport(executed_tokens=60, executed_wall_s=1.0)
    r.finalize()
    assert r.executed_tps == pytest.approx(60.0)
    assert r.virtual_tps == pytest.approx(60.0)  # no skipping -> same
    assert r.cache_tps == 0.0
    assert r.effective_tps == pytest.approx(60.0)


def test_tps_report_virtual_scales_with_gate_skip() -> None:
    """virtual_tps scales with gate_skip_ratio. In P1, this field is always 0 —
    the test proves the math is correct *when* P2-Active sets it."""
    r = gguf.TpsReport(
        executed_tokens=60, executed_wall_s=1.0, gate_skip_ratio=0.95
    )
    r.finalize()
    assert r.executed_tps == pytest.approx(60.0)
    # 60 / (1 - 0.95) = 1200
    assert r.virtual_tps == pytest.approx(1200.0)
    name, value = r.headline()
    assert name == "virtual_tps"
    assert value == pytest.approx(1200.0)


def test_tps_report_identity_when_gate_skip_zero() -> None:
    """P1 honesty: gate_skip_ratio = 0 ⇒ virtual_tps == executed_tps."""
    r = gguf.TpsReport(executed_tokens=100, executed_wall_s=2.0)
    r.finalize()
    assert r.gate_skip_ratio == 0.0
    assert r.executed_tps == pytest.approx(50.0)
    assert r.virtual_tps == pytest.approx(r.executed_tps)


def test_tps_report_cache_tps() -> None:
    r = gguf.TpsReport(cache_tokens=100, cache_wall_s=0.001)
    r.finalize()
    assert r.cache_tps == pytest.approx(100_000.0)
    name, value = r.headline()
    assert name in {"cache_tps", "effective_tps"}
    assert value >= 100_000.0


# --------------------------------------------------------------------------- #
# Muscle memory
# --------------------------------------------------------------------------- #


def test_muscle_memory_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    m = gguf.manifest_from_args(
        model="x.gguf", prompt="hi", seed=7, max_tokens=32
    )
    m.model_sha256 = "a" * 64
    assert gguf.muscle_memory_lookup(m) is None
    path = gguf.muscle_memory_store(m, "hello world", 5)
    assert path is not None and path.is_file()

    hit = gguf.muscle_memory_lookup(m)
    assert hit is not None
    assert hit.text == "hello world"
    assert hit.tokens == 5
    assert hit.cache_read_s >= 0.0


def test_muscle_memory_respects_off_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    m = gguf.manifest_from_args(model="x.gguf", prompt="hi", muscle_memory="off")
    m.model_sha256 = "b" * 64
    # Force a cached file to exist even so
    (tmp_path / "cache" / "mm" / m.model_sha256).mkdir(parents=True)
    p = tmp_path / "cache" / "mm" / m.model_sha256 / "deadbeef.mm"
    p.write_bytes(
        gguf.MUSCLE_MEMORY_MAGIC + struct.pack("<II", 3, 3) + b"abc"
    )
    assert gguf.muscle_memory_lookup(m) is None


def test_muscle_memory_key_differs_across_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    m1 = gguf.manifest_from_args(model="x.gguf", prompt="a", seed=1)
    m1.model_sha256 = "c" * 64
    m2 = gguf.manifest_from_args(model="x.gguf", prompt="b", seed=1)
    m2.model_sha256 = "c" * 64
    gguf.muscle_memory_store(m1, "aaa", 3)
    assert gguf.muscle_memory_lookup(m2) is None
    hit = gguf.muscle_memory_lookup(m1)
    assert hit is not None and hit.text == "aaa"


# --------------------------------------------------------------------------- #
# End-to-end run with a mocked libllama
# --------------------------------------------------------------------------- #


class _StubCompletion:
    """Minimal stand-in for ``Llama`` used by :func:`run_gguf`."""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def create_completion(
        self, *_args: Any, **_kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        for p in self._pieces:
            yield {"choices": [{"text": p}]}


def _make_dummy_gguf(path: Path) -> None:
    # gguf magic "GGUF" + minimal garbage — the hasher just reads bytes.
    path.write_bytes(b"GGUF" + b"\x00" * 64)


def test_run_gguf_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)

    pieces = ["Hel", "lo", " world", "."]
    stub = _StubCompletion(pieces)
    monkeypatch.setattr(gguf, "_load_llm", lambda _m: stub)
    # Skip the real NRL bench probe in unit tests.
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )

    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=16, seed=123
    )
    buf = io.StringIO()
    result = gguf.run_gguf(m, stream_to=buf)

    assert result.text == "".join(pieces)
    assert result.tokens == len(pieces)
    assert result.tps.executed_tokens == len(pieces)
    assert result.tps.executed_tps > 0
    assert result.tps.virtual_tps == pytest.approx(result.tps.executed_tps)
    assert buf.getvalue() == "".join(pieces)
    assert result.model_sha256  # populated by sha256_file
    assert not result.cache_hit
    assert result.word_rates.word_count > 0
    assert result.word_rates.words_per_token > 0.0
    assert result.word_rates.executed_wps > 0.0

    # Second run: should be a muscle-memory cache hit, not a decode.
    monkeypatch.setattr(
        gguf,
        "_load_llm",
        lambda _m: pytest.fail("muscle memory hit must not reload libllama"),
    )
    result2 = gguf.run_gguf(m, stream_to=io.StringIO())
    assert result2.cache_hit is True
    assert result2.text == result.text


def test_run_gguf_missing_model(tmp_path: Path) -> None:
    m = gguf.manifest_from_args(model=str(tmp_path / "nope.gguf"), prompt="hi")
    with pytest.raises(FileNotFoundError):
        gguf.run_gguf(m)


def test_run_gguf_sha_mismatch_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    m = gguf.manifest_from_args(model=str(model), prompt="hi")
    m.model_sha256 = "0" * 64
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    with pytest.raises(RuntimeError, match="model_sha256 mismatch"):
        gguf.run_gguf(m)


def test_nrl_inference_stub_backend_runs_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NRL_INFERENCE=stub uses the bundled deterministic backend (no llama-cpp needed)."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=4, seed=42
    )
    result = gguf.run_gguf(m, stream_to=io.StringIO())
    assert result.tokens == 4
    assert result.text != ""
    assert not result.cache_hit


def test_nrl_inference_unknown_backend_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "warp-drive")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(model=str(model), prompt="hi", max_tokens=2)
    with pytest.raises(RuntimeError, match="unknown NRL_INFERENCE"):
        gguf.run_gguf(m)


def test_stream_pacing_populates_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NRL_STREAM_CHUNK_MS > 0 flags the banner with 'paced=Xms' (honest accounting)."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_STREAM_CHUNK_MS", "1")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=2, seed=5
    )
    result = gguf.run_gguf(m)
    assert result.tps.stream_chunk_ms == pytest.approx(1.0)
    banner = gguf.format_banner(result)
    assert "paced=1.0ms" in banner


def test_stream_pacing_invalid_value_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NRL_STREAM_CHUNK_MS", "not-a-number")
    assert gguf._stream_pacing_s() == 0.0


def test_env_overrides_fill_unset_kv_and_no_repack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NRL_KV_CACHE / NRL_NO_REPACK / NRL_CTX fill *unset* manifest fields only."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_KV_CACHE", "q8_0")
    monkeypatch.setenv("NRL_NO_REPACK", "1")
    monkeypatch.setenv("NRL_CTX", "1024")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(model=str(model), prompt="hi", max_tokens=1)
    assert m.kv_cache_dtype == ""
    assert m.no_repack is False
    assert m.n_ctx == 2048
    result = gguf.run_gguf(m)
    assert result.manifest.kv_cache_dtype == "q8_0"
    assert result.manifest.no_repack is True
    assert result.manifest.n_ctx == 1024


def test_kv_cache_dtype_validation_in_manifest() -> None:
    with pytest.raises(gguf.ManifestError, match="kv_cache_dtype"):
        gguf.parse_manifest_text(
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = m.gguf\n"
            "kv_cache_dtype = bogus\n"
        )


def test_diagnose_bad_model_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(model=str(tmp_path / "does-not-exist.gguf"), prompt="hi")
    with pytest.raises(FileNotFoundError):
        gguf.run_gguf(m)


def test_banner_contains_four_metrics_and_three_sections() -> None:
    r = gguf.TpsReport(executed_tokens=40, executed_wall_s=1.0, gate_skip_ratio=0.5)
    r.finalize()
    result = gguf.GgufRunResult(
        text="x",
        tokens=40,
        tps=r,
        cache_hit=False,
        nrl_attestation=gguf.NrlAttestation(),
        manifest=gguf.manifest_from_args(model="x.gguf", prompt="p"),
        model_sha256="f" * 64,
    )
    banner = gguf.format_banner(result)
    # Four TPS metrics.
    assert "executed_tps" in banner
    assert "virtual_tps" in banner
    assert "cache_tps" in banner
    assert "effective_tps" in banner
    # WPS section is present and labeled.
    assert "decode WPS" in banner
    assert "executed_wps" in banner
    assert "virtual_wps" in banner
    assert "cache_wps" in banner
    assert "effective_wps" in banner
    # Three clearly-labeled sections (§15 honesty).
    assert "decode TPS" in banner
    assert "NRL attestation" in banner
    assert "NRL lattice observation" in banner
    assert "NOT applied to decode TPS" in banner


def test_banner_p1_honesty_note_when_gate_skip_zero() -> None:
    """Banner explicitly states P1/P2-Shadow identity when gate_skip_ratio = 0."""
    r = gguf.TpsReport(executed_tokens=40, executed_wall_s=1.0)
    r.finalize()
    result = gguf.GgufRunResult(
        text="x",
        tokens=40,
        tps=r,
        cache_hit=False,
        nrl_attestation=gguf.NrlAttestation(),
        manifest=gguf.manifest_from_args(model="x.gguf", prompt="p"),
        model_sha256="f" * 64,
    )
    banner = gguf.format_banner(result)
    assert "gate_skip_ratio=0.000" in banner
    assert "virtual_tps == executed_tps" in banner


def test_lattice_observation_is_advisory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The P2-Shadow lattice observation must NEVER inflate virtual_tps.

    We inject a fake 0.9 skip_ratio observation and verify executed_tps and
    virtual_tps remain equal (because gate_skip_ratio, the applied value, is 0).
    """
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)

    def fake_bench_cli(**kwargs: Any) -> dict[str, Any]:
        return {
            "skip_ratio": 0.9,
            "executed_gops": 1.5,
            "virtual_gops": 15.0,
            "elapsed_s": 0.01,
            "variant": "avx2",
        }

    monkeypatch.setattr(gguf.runtime, "bench_cli", fake_bench_cli)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )

    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=2, seed=1
    )
    result = gguf.run_gguf(
        m,
        observation_profile="omega-hybrid",
        observation_timeout_s=2.0,
    )

    assert result.lattice_observation.available is True
    assert result.lattice_observation.skip_ratio == pytest.approx(0.9)
    # The hinge: virtual_tps must ignore the advisory observation.
    assert result.tps.gate_skip_ratio == 0.0
    assert result.tps.virtual_tps == pytest.approx(result.tps.executed_tps)


def test_lattice_observation_disabled_when_profile_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=1, seed=1
    )
    result = gguf.run_gguf(m, observation_profile="", observation_timeout_s=0.1)
    assert result.lattice_observation.available is False
    assert "disabled" in result.lattice_observation.note


# --------------------------------------------------------------------------- #
# P2-Active simulation hinge
# --------------------------------------------------------------------------- #


def test_manifest_rejects_out_of_range_gate_override(tmp_path: Path) -> None:
    p = tmp_path / "bad.nrl"
    p.write_text(
        """
        schema = nrl.manifest.v1
        mode = gguf_run
        model = test.gguf
        prompt = "hi"
        gate_skip_ratio_override = 1.5
        """,
        encoding="utf-8",
    )
    with pytest.raises(gguf.ManifestError, match="gate_skip_ratio_override"):
        gguf.load_manifest(p)


def test_manifest_from_args_rejects_out_of_range_gate_override() -> None:
    with pytest.raises(gguf.ManifestError, match="gate_skip_ratio_override"):
        gguf.manifest_from_args(model="x.gguf", gate_skip_ratio_override=1.0)


def test_manifest_accepts_zero_gate_override_default() -> None:
    m = gguf.manifest_from_args(model="x.gguf")
    assert m.gate_skip_ratio_override == 0.0


def test_env_override_populates_gate_skip_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_GATE_SKIP_RATIO_OVERRIDE", "0.25")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(model=str(model), prompt="hi", max_tokens=4, seed=7)
    result = gguf.run_gguf(m, observation_profile="", stream_to=io.StringIO())
    assert result.tps.gate_skip_ratio == pytest.approx(0.25)
    assert result.tps.virtual_tps > result.tps.executed_tps


def test_manifest_override_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setenv("NRL_GATE_SKIP_RATIO_OVERRIDE", "0.9")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=4, seed=7, gate_skip_ratio_override=0.4
    )
    result = gguf.run_gguf(m, observation_profile="", stream_to=io.StringIO())
    assert result.tps.gate_skip_ratio == pytest.approx(0.4)


def test_p2active_sim_virtual_tps_formula_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """virtual_tps * (1 - gate_skip_ratio) == executed_tps, exactly, under simulation."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    for override in (0.1, 0.25, 0.5, 0.75):
        m = gguf.manifest_from_args(
            model=str(model), prompt="hi", max_tokens=4, seed=1,
            gate_skip_ratio_override=override,
            muscle_memory="off",  # don't hit cache across iterations; we want fresh math each time
        )
        result = gguf.run_gguf(m, observation_profile="", stream_to=io.StringIO())
        tps = result.tps
        assert tps.gate_skip_ratio == pytest.approx(override)
        assert tps.virtual_tps > tps.executed_tps
        assert tps.virtual_tps * (1.0 - tps.gate_skip_ratio) == pytest.approx(
            tps.executed_tps, rel=1e-9
        )


def test_banner_labels_p2active_simulation_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=2, seed=1, gate_skip_ratio_override=0.5,
    )
    result = gguf.run_gguf(m, observation_profile="", stream_to=io.StringIO())
    banner = gguf.format_banner(result)
    assert "P2-Active simulation (override)" in banner
    assert "gate_skip_ratio_override  0.500000" in banner
    assert "virtual_tps_formula_ok    yes" in banner
    # And the non-simulation path must not leak the label.
    m2 = gguf.manifest_from_args(model=str(model), prompt="hi", max_tokens=2, seed=1)
    result2 = gguf.run_gguf(m2, observation_profile="", stream_to=io.StringIO())
    assert "P2-Active simulation" not in gguf.format_banner(result2)


def test_cache_hit_ignores_gate_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muscle-memory replays have no libllama work to skip; override is inert on hits."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    seed_kwargs: dict[str, Any] = dict(model=str(model), prompt="hi", max_tokens=3, seed=42)
    # First run without override populates muscle memory.
    first = gguf.run_gguf(
        gguf.manifest_from_args(**seed_kwargs),
        observation_profile="",
        stream_to=io.StringIO(),
    )
    assert not first.cache_hit
    # Second run with override should hit the cache (override isn't in the key fields).
    second = gguf.run_gguf(
        gguf.manifest_from_args(gate_skip_ratio_override=0.5, **seed_kwargs),
        observation_profile="",
        stream_to=io.StringIO(),
    )
    assert second.cache_hit is True
    # Critical: cache-hit path must NOT apply the override — executed_wall_s == 0.
    assert second.tps.gate_skip_ratio == 0.0
    assert second.tps.virtual_tps == pytest.approx(second.tps.executed_tps)


def test_evidence_log_records_gate_simulation_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    evidence = tmp_path / "events.jsonl"
    monkeypatch.setenv("NRL_EVIDENCE_LOG", str(evidence))
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model), prompt="hi", max_tokens=2, seed=1, gate_skip_ratio_override=0.3,
    )
    gguf.run_gguf(m, observation_profile="", stream_to=io.StringIO())
    assert evidence.is_file()
    event = _json.loads(evidence.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert event["gate_simulation_active"] is True
    assert event["gate_source"] == "override"
    assert event["gate_skip_ratio_override"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# P2-Active prefill-cache gate (structural)
# --------------------------------------------------------------------------- #


def test_manifest_rejects_unknown_prefill_cache_value(tmp_path: Path) -> None:
    p = tmp_path / "bad.nrl"
    p.write_text(
        """
        schema = nrl.manifest.v1
        mode = gguf_run
        model = test.gguf
        prompt = "hi"
        prefill_cache = persistent
        """,
        encoding="utf-8",
    )
    with pytest.raises(gguf.ManifestError, match="prefill_cache"):
        gguf.load_manifest(p)


def test_manifest_accepts_prefill_cache_session(tmp_path: Path) -> None:
    p = tmp_path / "ok.nrl"
    p.write_text(
        """
        schema = nrl.manifest.v1
        mode = gguf_run
        model = test.gguf
        prompt = "hi"
        prefill_cache = session
        """,
        encoding="utf-8",
    )
    m = gguf.load_manifest(p)
    assert m.prefill_cache == "session"


def _run_with_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    gate: gguf.PrefillGate | None,
    *,
    prefill_cache: str = "session",
    gate_skip_ratio_override: float = 0.0,
) -> gguf.GgufRunResult:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    if not model.is_file():
        _make_dummy_gguf(model)
    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile)
    )
    m = gguf.manifest_from_args(
        model=str(model),
        prompt=prompt,
        max_tokens=4,
        seed=1,
        prefill_cache=prefill_cache,
        muscle_memory="off",
        gate_skip_ratio_override=gate_skip_ratio_override,
    )
    return gguf.run_gguf(
        m, observation_profile="", stream_to=io.StringIO(), prefill_gate=gate,
    )


def test_prefill_gate_first_turn_honors_hinge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gguf.PrefillGate()
    result = _run_with_gate(tmp_path, monkeypatch, "hello world foo bar", gate)
    assert result.gate_source is None
    assert result.tps.gate_skip_ratio == 0.0
    assert result.tps.virtual_tps == pytest.approx(result.tps.executed_tps)


def test_prefill_gate_second_turn_flips_hinge_from_structural_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma", gate)
    r2 = _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma delta epsilon", gate)
    assert r2.gate_source == "prefill_cache"
    assert r2.tps.gate_skip_ratio > 0.0
    assert r2.tps.virtual_tps > r2.tps.executed_tps
    # Formula invariant — identical to the override path.
    assert r2.tps.virtual_tps * (1.0 - r2.tps.gate_skip_ratio) == pytest.approx(
        r2.tps.executed_tps, rel=1e-9
    )


def test_prefill_gate_wins_over_override_when_both_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural gate must win over numeric fixture."""
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta", gate, gate_skip_ratio_override=0.9)
    r2 = _run_with_gate(
        tmp_path, monkeypatch, "alpha beta gamma", gate, gate_skip_ratio_override=0.9,
    )
    assert r2.gate_source == "prefill_cache"
    # Structural gate says 2/3, not the override's 0.9.
    assert r2.tps.gate_skip_ratio == pytest.approx(2 / 3)


def test_prefill_gate_inactive_when_manifest_says_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prefill_cache=off must skip the gate even if caller passes one."""
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta", gate, prefill_cache="off")
    r2 = _run_with_gate(
        tmp_path, monkeypatch, "alpha beta gamma", gate, prefill_cache="off",
    )
    assert r2.gate_source is None
    assert r2.tps.gate_skip_ratio == 0.0


def test_prefill_gate_falls_back_to_override_when_no_shared_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate returns zero skip → override is allowed to take effect."""
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma", gate)
    r2 = _run_with_gate(
        tmp_path, monkeypatch, "zeta eta theta", gate, gate_skip_ratio_override=0.4,
    )
    assert r2.gate_source == "override"
    assert r2.tps.gate_skip_ratio == pytest.approx(0.4)


def test_banner_labels_prefill_cache_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma delta", gate)
    r2 = _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma zeta", gate)
    banner = gguf.format_banner(r2)
    assert "P2-Active (prefill cache)" in banner
    assert "shared_prefix_len" in banner
    assert "P2-Active simulation" not in banner


def test_evidence_log_records_prefill_gate_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    evidence = tmp_path / "events.jsonl"
    monkeypatch.setenv("NRL_EVIDENCE_LOG", str(evidence))
    gate = gguf.PrefillGate()
    _run_with_gate(tmp_path, monkeypatch, "alpha beta", gate)
    _run_with_gate(tmp_path, monkeypatch, "alpha beta gamma", gate)
    assert evidence.is_file()
    events = [
        _json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines() if line
    ]
    assert events[0]["gate_source"] is None
    assert events[0]["prefill_cache"] == "session"
    assert events[1]["gate_source"] == "prefill_cache"
    assert events[1]["gate_report"]["shared_prefix_len"] == 2
    assert events[1]["gate_report"]["prompt_token_count"] == 3
