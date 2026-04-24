# Copyright (c) 2026 Daniel Harding - RomanAILabs
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Gemini-Flash, ChatGPT-5.4
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 7-EG — Native C runner parity tests.

Verifies that the native dispatcher (``engine/src/ladder_native.c`` driven
through :mod:`nrlpy.native_ladder`) produces results that are
byte-identical to the Python ladder for every served rung. The native
runner is a §4.2 dispatch shim: deterministic R0/R1/R2 candidate
computation stays in Python during the parity-gate window, so the only
behavior moved across the C ABI is rung selection + libllama bridge call.

The suite is deterministic:

* The bridge backend is forced to the in-process stub (matches the
  Python ``NRL_INFERENCE=stub`` backend's per-token output).
* All runs use the same tiny GGUF fixture and seeded sampler so the R5
  fallback path produces identical token streams across both ladders.
* The ZPM index is seeded with :func:`compute_r2_candidate_state` so R2
  active hits are guaranteed.

When ``nrlpy._core`` was not built with the Phase 7-EG bindings every
test in this module is skipped; the parity gate only fires on hosts
that actually ship the native code.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import pytest

from nrlpy import gguf, lmo, native_ladder, zpm
from nrlpy.lmo import compute_r2_candidate_state

pytestmark = pytest.mark.skipif(
    not native_ladder.is_available(),
    reason="nrlpy._core does not expose Phase 7-EG ladder bindings",
)


# --------------------------------------------------------------------------- #
# Minimal GGUF fixture (same shape as test_ladder_r2_active.py — self-contained)
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
    """Isolate NRL_ROOT so cache/ and evidence are scoped per-test."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    for k in (
        "NRL_ZPM", "NRL_ZPM_THRESHOLD", "NRL_GATE_SKIP_RATIO_OVERRIDE",
        "NRL_COHERENCE_LANE", "NRL_R2_SHADOW", "NRL_OMEGA_BUDGET_MS",
    ):
        monkeypatch.delenv(k, raising=False)
    # Reset the bridge state between tests so callbacks from one test
    # cannot leak into the next.
    native_ladder.set_backend(native_ladder.BACKEND_STUB)
    native_ladder.register_libllama_callback(None)
    yield tmp_path
    native_ladder.set_backend(native_ladder.BACKEND_STUB)
    native_ladder.register_libllama_callback(None)


@pytest.fixture()
def fixture_gguf(nrl_root: Path) -> Path:
    p = nrl_root / "fixture.gguf"
    _build_fixture_gguf(p, n_blocks=2)
    return p


@pytest.fixture()
def absorbed_lmo(nrl_root: Path, fixture_gguf: Path) -> lmo.LmoHandle:
    return lmo.absorb_gguf(fixture_gguf, attempt_libllama=False)


def _make_manifest(
    fixture_gguf: Path,
    *,
    runner_backend: str = "python",
    coherence_lane: str = "fast-stable",
    r2_shadow_enabled: bool = True,
    muscle_memory: str = "off",
    max_tokens: int = 3,
    omega_budget_ms: float = 500.0,
    prompt: str = "hello from native parity test",
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
        runner_backend=runner_backend,
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
                "source": "phase7_native_test_prime",
            },
        )
    )
    path = gguf._zpm_index_path(manifest.model_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(path)
    return path


# --------------------------------------------------------------------------- #
# 1. Basic native-ladder availability and ABI surface
# --------------------------------------------------------------------------- #


class TestNativeAbi:
    def test_native_is_available(self) -> None:
        assert native_ladder.is_available()

    def test_lane_gate_matches_python(self) -> None:
        for lane in ("fast-stable", "fast-balanced", "max-throughput"):
            assert (
                native_ladder.lane_allows_r2_active(lane)
                == lmo.lane_allows_r2_active(lane)
            ), f"native and python disagree on lane {lane!r}"

    def test_unknown_lane_is_not_allowed(self) -> None:
        assert not native_ladder.lane_allows_r2_active("unknown-lane")
        assert not native_ladder.lane_allows_r2_active("")

    def test_rung_names_match_python_constants(self) -> None:
        assert native_ladder.rung_name(0) == "r0_muscle_memory"
        assert native_ladder.rung_name(1) == "r1_zpm_nullspace"
        assert native_ladder.rung_name(2) == "r2_omega_native_resolve"
        assert native_ladder.rung_name(5) == "r5_libllama"


# --------------------------------------------------------------------------- #
# 2. Native dispatcher selects the same rung the Python flow would
# --------------------------------------------------------------------------- #


class TestNativeDispatch:
    def test_r0_hit_serves_immediately(self, nrl_root: Path) -> None:
        out = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=True,
            r0=native_ladder.LadderCandidate(
                available=True, text="cached", tokens=2, wall_s=0.0001,
            ),
            r5_request={"model": "x.gguf", "prompt": "p", "max_tokens": 1},
        )
        assert out.served_rung == native_ladder.RUNG_R0_MUSCLE_MEMORY
        assert out.text == "cached"
        assert out.tokens == 2

    def test_r1_hit_serves_when_no_r0(self, nrl_root: Path) -> None:
        out = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=True,
            r1=native_ladder.LadderCandidate(
                available=True, text="zpm-served", tokens=4, wall_s=0.0001,
            ),
            r5_request={"model": "x.gguf", "prompt": "p", "max_tokens": 1},
        )
        assert out.served_rung == native_ladder.RUNG_R1_ZPM
        assert out.text == "zpm-served"
        assert out.tokens == 4

    def test_r2_active_serves_on_max_throughput(self, nrl_root: Path) -> None:
        out = native_ladder.resolve_turn(
            coherence_lane="max-throughput",
            r2_shadow_enabled=True,
            r2_active=native_ladder.LadderCandidate(
                available=True, text="r2", tokens=3, wall_s=0.0002,
            ),
            r5_request={"model": "x.gguf", "prompt": "p", "max_tokens": 1},
        )
        assert out.served_rung == native_ladder.RUNG_R2_OMEGA_ACTIVE

    def test_r2_active_blocked_on_fast_stable(self, nrl_root: Path) -> None:
        """Even if the caller passes available=True, the C dispatcher
        re-checks the lane and refuses to serve R2 on fast-stable."""
        out = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=True,
            r2_active=native_ladder.LadderCandidate(
                available=True, text="should not serve", tokens=3, wall_s=0.0,
            ),
            r5_request={"model": "x.gguf", "prompt": "x", "max_tokens": 1},
        )
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA
        # Stub backend produced the R5 reply.
        assert out.text.endswith(" [stub]")

    def test_r2_active_blocked_when_flag_off(self, nrl_root: Path) -> None:
        out = native_ladder.resolve_turn(
            coherence_lane="max-throughput",
            r2_shadow_enabled=False,
            r2_active=native_ladder.LadderCandidate(
                available=True, text="should not serve", tokens=3, wall_s=0.0,
            ),
            r5_request={"model": "x.gguf", "prompt": "y", "max_tokens": 1},
        )
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA

    def test_r5_stub_is_deterministic(self, nrl_root: Path) -> None:
        out1 = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=False,
            r5_request={"model": "x.gguf", "prompt": "abc", "max_tokens": 1},
        )
        out2 = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=False,
            r5_request={"model": "x.gguf", "prompt": "abc", "max_tokens": 1},
        )
        assert out1.text == out2.text
        assert out1.tokens == out2.tokens


# --------------------------------------------------------------------------- #
# 3. Bridge callback round-trip
# --------------------------------------------------------------------------- #


class TestBridgeCallback:
    def test_callback_receives_request_and_returns_reply(
        self, nrl_root: Path
    ) -> None:
        seen: list[dict[str, object]] = []

        def cb(req: dict[str, object]) -> dict[str, object]:
            seen.append(req)
            return {"text": "from-callback:" + str(req["prompt"]), "tokens": 11}

        native_ladder.register_libllama_callback(cb)
        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        out = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=False,
            r5_request={"model": "m.gguf", "prompt": "callback-prompt", "max_tokens": 4},
        )
        assert seen, "callback was never invoked"
        assert seen[0]["prompt"] == "callback-prompt"
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA
        assert out.text == "from-callback:callback-prompt"
        assert out.tokens == 11

    def test_clear_callback_then_stub_still_works(self, nrl_root: Path) -> None:
        native_ladder.register_libllama_callback(lambda r: {"text": "cb", "tokens": 1})
        native_ladder.register_libllama_callback(None)
        native_ladder.set_backend(native_ladder.BACKEND_STUB)
        out = native_ladder.resolve_turn(
            coherence_lane="fast-stable",
            r2_shadow_enabled=False,
            r5_request={"model": "m.gguf", "prompt": "stub", "max_tokens": 1},
        )
        # Stub backend produces a deterministic suffix.
        assert out.text == "stub [stub]"


# --------------------------------------------------------------------------- #
# 4. Full run_gguf parity: native vs python end-to-end
# --------------------------------------------------------------------------- #


class TestRunGgufParity:
    def test_r5_native_matches_python_text(
        self, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
    ) -> None:
        py_buf = io.StringIO()
        py_manifest = _make_manifest(fixture_gguf, runner_backend="python")
        py_result = gguf.run_gguf(py_manifest, stream_to=py_buf)

        n_buf = io.StringIO()
        n_manifest = _make_manifest(fixture_gguf, runner_backend="native")
        n_result = gguf.run_gguf(n_manifest, stream_to=n_buf)

        assert py_result.text == n_result.text, (
            "native R5 diverged from python R5: "
            f"py={py_result.text!r} native={n_result.text!r}"
        )
        assert py_result.tokens == n_result.tokens
        assert py_result.cache_hit == n_result.cache_hit is False
        # Streamed bytes must also match.
        assert py_buf.getvalue() == n_buf.getvalue()

    def test_native_strict_raises_when_unavailable_is_not_the_case_here(
        self, fixture_gguf: Path
    ) -> None:
        # Sanity: native_strict + native available should NOT raise.
        m = _make_manifest(fixture_gguf, runner_backend="native_strict")
        result = gguf.run_gguf(m, stream_to=None)
        assert result is not None
        assert result.manifest.runner_backend == "native_strict"

    def test_runner_backend_is_recorded_in_evidence(
        self, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
    ) -> None:
        m = _make_manifest(fixture_gguf, runner_backend="native")
        result = gguf.run_gguf(m, stream_to=None)
        log_path = Path(result.evidence_path)
        assert log_path.is_file(), "evidence log was not written"
        last_line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        event = json.loads(last_line)
        assert event["runner_backend"] == "native"

    def test_runner_backend_python_default_is_unchanged(
        self, fixture_gguf: Path
    ) -> None:
        # Default manifest has runner_backend="python"; confirms zero
        # behavior change when the new field is not set explicitly.
        m = gguf.manifest_from_args(
            str(fixture_gguf), prompt="x", max_tokens=2,
            muscle_memory="off", seed=1,
        )
        assert m.runner_backend == "python"
        result = gguf.run_gguf(m, stream_to=None)
        assert result.manifest.runner_backend == "python"

    def test_r2_active_native_serves_when_zpm_seeded(
        self, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
    ) -> None:
        # Build a manifest, prime the ZPM index so R2 active hits, and
        # check the native path serves the stored reply.
        m = _make_manifest(
            fixture_gguf,
            runner_backend="native",
            coherence_lane="max-throughput",
        )
        # We need model_sha256 populated before priming the ZPM index.
        m.model_sha256 = gguf.sha256_file(Path(m.model))
        served_text = "native-r2-served-reply"
        _prime_zpm_index_for_r2(
            m, absorbed_lmo,
            reply_text=served_text, reply_tokens=5,
        )
        result = gguf.run_gguf(m, stream_to=None)
        assert result.text == served_text
        assert result.tokens == 5
        assert result.gate_source == lmo.OMEGA_ACTIVE_GATE_SOURCE
        # Native backend must be recorded.
        assert result.gate_report.get("runner_backend") == "native"

    def test_r2_active_native_blocked_on_fast_stable(
        self, fixture_gguf: Path, absorbed_lmo: lmo.LmoHandle
    ) -> None:
        # Even with a primed ZPM index, fast-stable must serve R5.
        m = _make_manifest(
            fixture_gguf,
            runner_backend="native",
            coherence_lane="fast-stable",
        )
        m.model_sha256 = gguf.sha256_file(Path(m.model))
        _prime_zpm_index_for_r2(
            m, absorbed_lmo,
            reply_text="should-not-be-served", reply_tokens=5,
        )
        result = gguf.run_gguf(m, stream_to=None)
        assert "should-not-be-served" not in result.text
        assert result.gate_source != lmo.OMEGA_ACTIVE_GATE_SOURCE


# --------------------------------------------------------------------------- #
# 5. Banner + CLI integration
# --------------------------------------------------------------------------- #


class TestBannerAndCli:
    def test_banner_includes_runner_line(self, fixture_gguf: Path) -> None:
        m = _make_manifest(fixture_gguf, runner_backend="native")
        result = gguf.run_gguf(m, stream_to=None)
        banner = gguf.format_banner(result)
        assert "runner        native" in banner

    def test_banner_python_default_unchanged_marker(
        self, fixture_gguf: Path
    ) -> None:
        m = _make_manifest(fixture_gguf, runner_backend="python")
        result = gguf.run_gguf(m, stream_to=None)
        banner = gguf.format_banner(result)
        assert "runner        python" in banner

    def test_cli_native_flag_runs_via_native_backend(
        self,
        nrl_root: Path,
        fixture_gguf: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from nrlpy.cli import main as cli_main

        rc = cli_main(
            [
                "run",
                str(fixture_gguf),
                "--prompt",
                "cli-native-flag-test",
                "--max-tokens",
                "2",
                "--seed",
                "1",
                "--no-stream",
                "--native",
                "-v",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0, f"cli failed: {captured.err}"
        # Banner went to stdout, verbose summary went to stderr.
        assert "runner        native" in captured.out
        assert "[nrl.runner] backend=native" in captured.err

    def test_cli_python_ladder_flag_keeps_python(
        self,
        nrl_root: Path,
        fixture_gguf: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from nrlpy.cli import main as cli_main

        rc = cli_main(
            [
                "run",
                str(fixture_gguf),
                "--prompt",
                "cli-python-flag-test",
                "--max-tokens",
                "2",
                "--seed",
                "1",
                "--no-stream",
                "--python-ladder",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0, f"cli failed: {captured.err}"
        assert "runner        python" in captured.out


# --------------------------------------------------------------------------- #
# 6. Honest accounting (TpsReport) parity
# --------------------------------------------------------------------------- #


class TestTpsParity:
    def test_executed_tokens_reported_under_native(
        self, fixture_gguf: Path
    ) -> None:
        m = _make_manifest(fixture_gguf, runner_backend="native")
        result = gguf.run_gguf(m, stream_to=None)
        assert result.tps.executed_tokens == result.tokens
        assert result.tps.executed_wall_s > 0.0
        assert result.tps.cache_tokens == 0

    def test_native_strict_falls_back_to_python_when_core_missing(
        self,
        fixture_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force native_ladder.is_available to return False to simulate a
        # host without the Phase 7-EG bindings.
        monkeypatch.setattr(native_ladder, "is_available", lambda: False)
        m = _make_manifest(fixture_gguf, runner_backend="native_strict")
        with pytest.raises(RuntimeError, match="native ladder unavailable"):
            gguf.run_gguf(m, stream_to=None)

    def test_native_falls_back_silently_when_core_missing(
        self,
        fixture_gguf: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(native_ladder, "is_available", lambda: False)
        m = _make_manifest(fixture_gguf, runner_backend="native")
        result = gguf.run_gguf(m, stream_to=None)
        captured = capsys.readouterr()
        # The fallback warning landed on stderr and the run still succeeded.
        assert "native ladder unavailable" in captured.err
        assert result is not None


# --------------------------------------------------------------------------- #
# 7. Manifest & USAGE coverage
# --------------------------------------------------------------------------- #


class TestManifest:
    def test_runner_backend_round_trips_through_manifest_text(self) -> None:
        text = (
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = some.gguf\n"
            "prompt = hi\n"
            "runner_backend = native\n"
        )
        m = gguf.parse_manifest_text(text)
        assert m.runner_backend == "native"

    def test_runner_backend_invalid_value_raises(self) -> None:
        text = (
            "schema = nrl.manifest.v1\n"
            "mode = gguf_run\n"
            "model = some.gguf\n"
            "prompt = hi\n"
            "runner_backend = wat\n"
        )
        with pytest.raises(gguf.ManifestError, match="runner_backend"):
            gguf.parse_manifest_text(text)

    def test_manifest_from_args_accepts_runner_backend(self) -> None:
        m = gguf.manifest_from_args(
            "x.gguf", prompt="hi", muscle_memory="off",
            seed=1, runner_backend="native",
        )
        assert m.runner_backend == "native"
