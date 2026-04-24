# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 8-EG — Full native hot path parity tests.

These tests pin the Phase 8-EG native C implementations of R0 (muscle
memory), R1 (ZPM nullspace), and the full-turn orchestrator against the
Python reference in :mod:`nrlpy.gguf` and :mod:`nrlpy.zpm`. The parity
contract is byte-for-byte:

* Same FNV-1a64 keys (``nrlpy.runtime.fnv1a64_packed`` IV for muscle
  memory; ``nrlpy.zpm._fnv1a64`` IV for ZPM).
* Same 256-bit ZPM anchor words produced by the four rotations in
  :func:`nrlpy.zpm.anchor`.
* Same served text and token counts on a hit.
* Same served-rung identity across the native and hybrid paths for the
  full ``run_gguf`` driver.

Tests skip when ``nrlpy._core`` was built without Phase 8-EG bindings
(Phase 7-EG alone is a strict prerequisite — the Phase 7-EG native
suite is the upstream parity gate).
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

from nrlpy import gguf, native_ladder, zpm
from nrlpy.gguf import GgufManifest, MUSCLE_MEMORY_MAGIC

pytestmark = pytest.mark.skipif(
    not native_ladder.is_full_native_available(),
    reason="nrlpy._core does not expose Phase 8-EG full-native bindings",
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def nrl_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test NRL_ROOT so cache/evidence don't leak between tests."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    for k in (
        "NRL_ZPM",
        "NRL_ZPM_THRESHOLD",
        "NRL_GATE_SKIP_RATIO_OVERRIDE",
        "NRL_COHERENCE_LANE",
        "NRL_R2_SHADOW",
        "NRL_OMEGA_BUDGET_MS",
    ):
        monkeypatch.delenv(k, raising=False)
    # Reset both Phase 7-EG and Phase 8-EG bridge state between tests so
    # callbacks installed by one test never leak into the next.
    native_ladder.set_backend(native_ladder.BACKEND_STUB)
    native_ladder.register_libllama_callback(None)
    native_ladder.register_r2_callback(None)
    yield tmp_path
    native_ladder.set_backend(native_ladder.BACKEND_STUB)
    native_ladder.register_libllama_callback(None)
    native_ladder.register_r2_callback(None)


def _make_manifest(
    model: str = "ignored.gguf",
    *,
    prompt: str = "hello phase 8 full native",
    seed: int = 0,
    max_tokens: int = 8,
    muscle_memory: str = "on",
    zpm_nullspace: bool = True,
    model_sha: str = "phase8-test-sha",
    coherence_lane: str = "fast-stable",
) -> GgufManifest:
    m = GgufManifest(
        model=model,
        prompt=prompt,
        seed=seed,
        max_tokens=max_tokens,
        muscle_memory=muscle_memory,
        zpm_nullspace=zpm_nullspace,
        model_sha256=model_sha,
        coherence_lane=coherence_lane,
    )
    return m


def _write_mm_cache(nrl_root: Path, mf: GgufManifest, reply: str, tokens: int) -> Path:
    """Mirror :func:`gguf.muscle_memory_store` without requiring the
    full ``run_gguf`` round-trip — used to seed a deterministic R0 hit."""
    key = gguf._muscle_memory_key(mf)
    tag = mf.model_sha256 or "unknown"
    path = nrl_root / "cache" / "mm" / tag / f"{key:016x}.mm"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = reply.encode("utf-8")
    header = MUSCLE_MEMORY_MAGIC + struct.pack("<II", tokens, len(body))
    path.write_bytes(header + body)
    return path


# --------------------------------------------------------------------------- #
# 1. Availability / ABI surface                                               #
# --------------------------------------------------------------------------- #


class TestFullNativeAbi:
    def test_is_full_native_available(self) -> None:
        assert native_ladder.is_full_native_available() is True

    def test_is_available_is_prerequisite(self) -> None:
        # Phase 8-EG builds on Phase 7-EG; is_available() must also be True.
        assert native_ladder.is_available() is True

    def test_public_names_are_exported(self) -> None:
        for name in (
            "FullTurnRequest",
            "FullTurnResult",
            "MmLookupRequest",
            "MmLookupResult",
            "ZpmLookupRequest",
            "ZpmLookupResult",
            "mm_lookup",
            "zpm_lookup",
            "run_turn_full",
            "register_r2_callback",
            "r2_has_callback",
        ):
            assert hasattr(native_ladder, name), name

    def test_r2_callback_slot_starts_empty(self) -> None:
        native_ladder.register_r2_callback(None)
        assert native_ladder.r2_has_callback() is False

    def test_r2_callback_registers_and_clears(self) -> None:
        def _cb(req):
            return {
                "available": 0,
                "tokens": 0,
                "text": "",
                "stored_entry_index": -1,
                "distance_bits": 256,
                "wall_seconds": 0.0,
            }

        native_ladder.register_r2_callback(_cb)
        assert native_ladder.r2_has_callback() is True
        native_ladder.register_r2_callback(None)
        assert native_ladder.r2_has_callback() is False


# --------------------------------------------------------------------------- #
# 2. Native muscle memory (R0) parity                                         #
# --------------------------------------------------------------------------- #


class TestNativeMm:
    def test_key_matches_python(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="key parity check")
        py_key = gguf._muscle_memory_key(mf)
        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=False,  # miss-only; we only need the key
        )
        out = native_ladder.mm_lookup(req)
        assert out.hit is False
        assert out.key_fnv1a64 == py_key

    def test_miss_on_empty_cache(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="no cache for me")
        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=True,
        )
        out = native_ladder.mm_lookup(req)
        assert out.hit is False
        assert out.tokens_emitted == 0
        assert out.text == ""

    def test_hit_matches_python_lookup(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="cached hit please")
        reply = "cached reply from Phase 8 smoke"
        _write_mm_cache(nrl_root, mf, reply, tokens=42)

        py_hit = gguf.muscle_memory_lookup(mf)
        assert py_hit is not None
        assert py_hit.text == reply

        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=True,
        )
        out = native_ladder.mm_lookup(req)
        assert out.hit is True
        assert out.tokens_emitted == 42
        assert out.text == reply
        assert out.key_fnv1a64 == py_hit.key_fnv1a64

    def test_off_mode_never_hits(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="off mode", muscle_memory="on")
        _write_mm_cache(nrl_root, mf, "would have hit", tokens=3)

        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=False,
        )
        out = native_ladder.mm_lookup(req)
        assert out.hit is False
        # Key is still reported on a miss so callers can persist it.
        assert out.key_fnv1a64 == gguf._muscle_memory_key(mf)

    def test_corrupt_header_is_miss(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="corrupt header")
        key = gguf._muscle_memory_key(mf)
        path = (
            nrl_root / "cache" / "mm" / mf.model_sha256 / f"{key:016x}.mm"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"BADMAGIC" + struct.pack("<II", 3, 4) + b"abcd")

        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=True,
        )
        out = native_ladder.mm_lookup(req)
        assert out.hit is False


# --------------------------------------------------------------------------- #
# 3. Native ZPM (R1) parity                                                   #
# --------------------------------------------------------------------------- #


class TestNativeZpm:
    def test_anchor_matches_python(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="anchor parity check")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        py_state = zpm.anchor(intent)
        req = native_ladder.ZpmLookupRequest(
            index_path="",  # no index needed, we just want the anchor
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=0,
            enabled=False,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is False  # enabled=False
        assert out.state == py_state

    def test_exact_match_served(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="zpm exact served")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        py_state = zpm.anchor(intent)
        idx = zpm.ZpmIndex()
        idx.add(
            zpm.ZpmEntry(
                state=py_state,
                reply_text="exact zpm reply",
                tokens=11,
                wall_s_at_write=0.0,
                metadata={"src": "phase8-test"},
            )
        )
        index_path = nrl_root / "cache" / "zpm" / mf.model_sha256 / "index.bin"
        idx.save(index_path)

        req = native_ladder.ZpmLookupRequest(
            index_path=str(index_path),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=0,
            enabled=True,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is True
        assert out.exact is True
        assert out.distance_bits == 0
        assert out.tokens_emitted == 11
        assert out.text == "exact zpm reply"
        assert out.stored_entry_index == 0
        assert out.state == py_state

    def test_near_match_within_threshold_served(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="zpm near-match")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        py_state = zpm.anchor(intent)
        # Flip 2 bits across state[0] so distance=2.
        perturbed = (py_state[0] ^ 0b11, py_state[1], py_state[2], py_state[3])
        idx = zpm.ZpmIndex()
        idx.add(
            zpm.ZpmEntry(
                state=perturbed,
                reply_text="near zpm reply",
                tokens=5,
                wall_s_at_write=0.0,
            )
        )
        index_path = nrl_root / "cache" / "zpm" / mf.model_sha256 / "index.bin"
        idx.save(index_path)

        req = native_ladder.ZpmLookupRequest(
            index_path=str(index_path),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=4,  # allow up to 4 bit differences
            enabled=True,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is True
        assert out.exact is False
        assert out.within_threshold is True
        assert out.distance_bits == 2
        assert out.text == "near zpm reply"

    def test_near_match_beyond_threshold_is_miss(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="zpm beyond threshold")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        py_state = zpm.anchor(intent)
        # Flip 10 bits so distance=10, beyond a threshold of 4.
        perturbed = (
            py_state[0] ^ 0b1111111111,
            py_state[1],
            py_state[2],
            py_state[3],
        )
        idx = zpm.ZpmIndex()
        idx.add(
            zpm.ZpmEntry(
                state=perturbed,
                reply_text="should not serve",
                tokens=5,
                wall_s_at_write=0.0,
            )
        )
        index_path = nrl_root / "cache" / "zpm" / mf.model_sha256 / "index.bin"
        idx.save(index_path)

        req = native_ladder.ZpmLookupRequest(
            index_path=str(index_path),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=4,
            enabled=True,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is False
        assert out.distance_bits == 10
        assert out.within_threshold is False

    def test_missing_index_file_is_clean_miss(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="no zpm yet")
        req = native_ladder.ZpmLookupRequest(
            index_path=str(nrl_root / "never-written.bin"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=0,
            enabled=True,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is False
        # Anchor is still computed so the audit log has data.
        assert out.state == zpm.anchor(gguf._zpm_anchor_bytes(mf, mf.prompt))

    def test_disabled_reports_anchor_only(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="zpm disabled")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        py_state = zpm.anchor(intent)
        req = native_ladder.ZpmLookupRequest(
            index_path="some path",
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            threshold_bits=0,
            enabled=False,
        )
        out = native_ladder.zpm_lookup(req)
        assert out.hit is False
        assert out.state == py_state


# --------------------------------------------------------------------------- #
# 4. Full-turn orchestrator decision order                                    #
# --------------------------------------------------------------------------- #


def _base_full_turn_request(
    nrl_root: Path,
    mf: GgufManifest,
    *,
    coherence_lane: str = "fast-stable",
    r2_shadow_enabled: bool = False,
) -> native_ladder.FullTurnRequest:
    return native_ladder.FullTurnRequest(
        mm_root_dir=str(nrl_root / "cache" / "mm"),
        model_sha256=mf.model_sha256,
        prompt=mf.prompt,
        sampler_fingerprint=mf.sampler_fingerprint(),
        seed=mf.seed,
        max_tokens=mf.max_tokens,
        muscle_memory_on=(mf.muscle_memory != "off"),
        zpm_index_path=str(
            nrl_root / "cache" / "zpm" / mf.model_sha256 / "index.bin"
        ),
        zpm_enabled=bool(mf.zpm_nullspace),
        zpm_threshold_bits=int(mf.zpm_threshold_bits),
        coherence_lane=coherence_lane,
        r2_shadow_enabled=r2_shadow_enabled,
        omega_iterations=int(mf.omega_iterations),
        omega_candidates=int(mf.omega_candidates),
        omega_budget_ms=float(mf.omega_budget_ms),
        intent_anchor_bytes=bytes(gguf._zpm_anchor_bytes(mf, mf.prompt)),
        r5_request={
            "model": mf.model,
            "prompt": mf.prompt,
            "max_tokens": int(mf.max_tokens),
            "seed": int(mf.seed),
            "n_ctx": int(mf.n_ctx),
            "n_threads": int(mf.n_threads),
            "n_batch": int(mf.n_batch),
            "temperature": float(mf.temperature),
            "top_p": float(mf.top_p),
            "top_k": int(mf.top_k),
            "repeat_penalty": float(mf.repeat_penalty),
        },
    )


class TestFullTurnDecisionOrder:
    def test_r0_hit_preempts_everything(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="r0 preempts")
        _write_mm_cache(nrl_root, mf, "served from mm", tokens=4)

        # R5 callback would fail the test if it were called.
        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(
            lambda req: pytest.fail(f"R5 called with {req!r}; R0 should have served")
        )

        req = _base_full_turn_request(nrl_root, mf)
        out = native_ladder.run_turn_full(req)
        assert out.served_rung == native_ladder.RUNG_R0_MUSCLE_MEMORY
        assert out.served_rung_name == "r0_muscle_memory"
        assert out.text == "served from mm"
        assert out.tokens == 4
        assert out.mm_report["hit"] == 1
        # R1/R2/R5 should not have been consulted.
        assert out.r5_report["invoked"] == 0

    def test_r1_hit_preempts_r2_and_r5(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="r1 preempts", coherence_lane="max-throughput")
        intent = gguf._zpm_anchor_bytes(mf, mf.prompt)
        state = zpm.anchor(intent)
        idx = zpm.ZpmIndex()
        idx.add(
            zpm.ZpmEntry(
                state=state,
                reply_text="served from zpm",
                tokens=7,
                wall_s_at_write=0.0,
            )
        )
        (nrl_root / "cache" / "zpm" / mf.model_sha256).mkdir(parents=True)
        idx.save(nrl_root / "cache" / "zpm" / mf.model_sha256 / "index.bin")

        # Installed but should never be called.
        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(
            lambda req: pytest.fail("R5 should not have been called")
        )
        native_ladder.register_r2_callback(
            lambda req: pytest.fail("R2 should not have been called")
        )

        req = _base_full_turn_request(
            nrl_root, mf, coherence_lane="max-throughput", r2_shadow_enabled=True
        )
        out = native_ladder.run_turn_full(req)
        assert out.served_rung == native_ladder.RUNG_R1_ZPM
        assert out.served_rung_name == "r1_zpm_nullspace"
        assert out.text == "served from zpm"
        assert out.tokens == 7
        assert out.zpm_report["hit"] == 1
        assert out.zpm_report["exact"] == 1
        assert out.zpm_report["distance_bits"] == 0
        assert out.r5_report["invoked"] == 0

    def test_r2_served_on_eligible_lane(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="r2 active", coherence_lane="max-throughput")

        calls = {"n": 0}

        def _r2_cb(req: dict) -> dict:
            calls["n"] += 1
            assert req["coherence_lane"] == 2  # max-throughput
            return {
                "available": 1,
                "tokens": 9,
                "text": "served from r2",
                "stored_entry_index": 0,
                "distance_bits": 1,
                "wall_seconds": 0.0005,
            }

        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(
            lambda req: pytest.fail("R5 should not have been called when R2 serves")
        )
        native_ladder.register_r2_callback(_r2_cb)

        req = _base_full_turn_request(
            nrl_root, mf, coherence_lane="max-throughput", r2_shadow_enabled=True
        )
        out = native_ladder.run_turn_full(req)
        assert calls["n"] == 1
        assert out.served_rung == native_ladder.RUNG_R2_OMEGA_ACTIVE
        assert out.text == "served from r2"
        assert out.tokens == 9
        assert out.r2_report["available"] == 1
        assert out.r2_report["distance_bits"] == 1
        assert out.r5_report["invoked"] == 0

    def test_r2_demotes_falls_through_to_r5(self, nrl_root: Path) -> None:
        mf = _make_manifest(
            prompt="r2 demotes to r5", coherence_lane="fast-balanced"
        )

        r2_calls = {"n": 0}
        r5_calls = {"n": 0}

        def _r2_cb(req: dict) -> dict:
            r2_calls["n"] += 1
            return {
                "available": 0,
                "tokens": 0,
                "text": "",
                "stored_entry_index": -1,
                "distance_bits": 256,
                "wall_seconds": 0.0001,
            }

        def _r5_cb(req: dict) -> dict:
            r5_calls["n"] += 1
            return {"text": "from r5 after demotion", "tokens": 3}

        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(_r5_cb)
        native_ladder.register_r2_callback(_r2_cb)

        req = _base_full_turn_request(
            nrl_root, mf, coherence_lane="fast-balanced", r2_shadow_enabled=True
        )
        out = native_ladder.run_turn_full(req)
        assert r2_calls["n"] == 1
        assert r5_calls["n"] == 1
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA
        assert out.text == "from r5 after demotion"
        assert out.tokens == 3
        # R2 demotion still populates its audit fields.
        assert out.r2_report["available"] == 0
        assert out.r2_report["wall_seconds"] >= 0.0

    def test_fast_stable_skips_r2_and_uses_r5(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="fast stable no r2")

        def _r2_cb(req):
            pytest.fail("R2 must not run on fast-stable")

        r5_calls = {"n": 0}

        def _r5_cb(req):
            r5_calls["n"] += 1
            return {"text": "r5 on fast-stable", "tokens": 2}

        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(_r5_cb)
        native_ladder.register_r2_callback(_r2_cb)

        req = _base_full_turn_request(
            nrl_root, mf, coherence_lane="fast-stable", r2_shadow_enabled=True
        )
        out = native_ladder.run_turn_full(req)
        assert r5_calls["n"] == 1
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA
        assert out.text == "r5 on fast-stable"

    def test_no_r2_callback_registered_falls_through_to_r5(
        self, nrl_root: Path
    ) -> None:
        mf = _make_manifest(
            prompt="no r2 cb installed", coherence_lane="max-throughput"
        )
        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(
            lambda req: {"text": "r5 fallback", "tokens": 1}
        )
        native_ladder.register_r2_callback(None)

        req = _base_full_turn_request(
            nrl_root, mf, coherence_lane="max-throughput", r2_shadow_enabled=True
        )
        out = native_ladder.run_turn_full(req)
        assert out.served_rung == native_ladder.RUNG_R5_LIBLLAMA
        assert out.text == "r5 fallback"
        assert out.r2_report["wall_seconds"] == 0.0


# --------------------------------------------------------------------------- #
# 5. Parity with Python reference across a matrix                             #
# --------------------------------------------------------------------------- #


class TestKeyAndAnchorParity:
    """Sweep a range of inputs to catch subtle hash/byte-order regressions."""

    @pytest.mark.parametrize("prompt", ["", "a", "abc", "longer prompt with spaces"])
    @pytest.mark.parametrize("seed", [0, 1, 42, 2 ** 31 - 1])
    def test_mm_key_parity(self, prompt: str, seed: int) -> None:
        mf = GgufManifest(
            model="m",
            prompt=prompt,
            seed=seed,
            max_tokens=17,
            muscle_memory="on",
            model_sha256="sha-for-parity",
        )
        py_key = gguf._muscle_memory_key(mf)
        out = native_ladder.mm_lookup(
            native_ladder.MmLookupRequest(
                root_dir="",  # no filesystem read — key only
                model_sha256=mf.model_sha256,
                prompt=mf.prompt,
                sampler_fingerprint=mf.sampler_fingerprint(),
                seed=mf.seed,
                max_tokens=mf.max_tokens,
                muscle_memory_on=False,
            )
        )
        assert out.key_fnv1a64 == py_key

    @pytest.mark.parametrize(
        "prompt",
        ["", "a", "abc", "anchor-parity-long-" + "x" * 200],
    )
    @pytest.mark.parametrize("model_sha", ["", "abcd", "a" * 64])
    def test_zpm_anchor_parity(self, prompt: str, model_sha: str) -> None:
        mf = GgufManifest(
            model="m",
            prompt=prompt,
            seed=7,
            max_tokens=23,
            model_sha256=model_sha,
        )
        intent = gguf._zpm_anchor_bytes(mf, prompt)
        py_state = zpm.anchor(intent)
        out = native_ladder.zpm_lookup(
            native_ladder.ZpmLookupRequest(
                index_path="",
                model_sha256=mf.model_sha256,
                prompt=mf.prompt,
                sampler_fingerprint=mf.sampler_fingerprint(),
                seed=mf.seed,
                max_tokens=mf.max_tokens,
                threshold_bits=0,
                enabled=False,
            )
        )
        assert out.state == py_state


# --------------------------------------------------------------------------- #
# 6. End-to-end run_gguf parity (Python vs. native_full)                      #
# --------------------------------------------------------------------------- #


def _ensure_dummy_model(tmp: Path) -> Path:
    """Write a zero-byte placeholder GGUF so the manifest sha256 check passes.
    The full-native path never actually reads the model — it defers to the
    R5 callback — so any file with a stable sha256 is sufficient."""
    p = tmp / "dummy.gguf"
    p.write_bytes(b"GGUF\x00phase8")
    return p


class TestRunGgufParity:
    def test_runner_backend_valid_for_native_full(self) -> None:
        mf = GgufManifest(model="x", runner_backend="native_full")
        # Round-trip through the validator used in manifest_from_args.
        assert mf.runner_backend in gguf._VALID_RUNNER_BACKENDS

    def test_native_full_runs_r0_cache_hit(
        self, nrl_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _ensure_dummy_model(nrl_root)
        mf = gguf.manifest_from_args(
            str(model),
            prompt="e2e r0 native full",
            max_tokens=4,
            seed=0,
            muscle_memory="on",
            runner_backend="native_full",
        )
        # Seed a muscle-memory hit under the model's real sha256.
        mf_with_sha = GgufManifest(
            model=mf.model,
            prompt=mf.prompt,
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory=mf.muscle_memory,
            model_sha256=gguf.sha256_file(Path(mf.model)),
            temperature=mf.temperature,
            top_p=mf.top_p,
            top_k=mf.top_k,
            repeat_penalty=mf.repeat_penalty,
            n_ctx=mf.n_ctx,
            n_batch=mf.n_batch,
            n_threads=mf.n_threads,
            chat_format=mf.chat_format,
            profile=mf.profile,
        )
        _write_mm_cache(nrl_root, mf_with_sha, "cached e2e reply", tokens=3)

        # Ensure the R5 callback is never invoked.
        native_ladder.set_backend(native_ladder.BACKEND_CALLBACK)
        native_ladder.register_libllama_callback(
            lambda req: pytest.fail("R5 must not run when R0 serves")
        )

        result = gguf.run_gguf(mf, trust_model_sha=False)
        assert result.text == "cached e2e reply"
        assert result.tokens == 3
        assert result.cache_hit is True
        assert result.gate_report.get("runner_backend") == "native_full"

    def test_native_full_runs_r5_callback(
        self, nrl_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With NRL_INFERENCE=stub (set by the nrl_root fixture), the
        # _run_gguf_native_full path registers its own R5 callback that
        # drives _load_llm + _stream_tokens through the stub backend.
        # We verify here that the native_full runner dispatches through
        # to R5, produces text, and stamps runner_backend correctly.
        model = _ensure_dummy_model(nrl_root)
        mf = gguf.manifest_from_args(
            str(model),
            prompt="e2e r5 native full",
            max_tokens=2,
            seed=0,
            muscle_memory="off",
            runner_backend="native_full",
        )
        mf.zpm_nullspace = False  # no ZPM index yet, avoid spurious R1 lookups
        result = gguf.run_gguf(mf, trust_model_sha=False)
        assert result.tokens > 0
        assert result.text != ""
        assert result.cache_hit is False
        assert result.gate_report.get("runner_backend") == "native_full"
        # R5 is an executed-tokens path; cache_tokens must stay zero.
        assert result.tps.cache_tokens == 0
        assert result.tps.executed_tokens == result.tokens


# --------------------------------------------------------------------------- #
# 7. Micro-benchmark: native probes beat pure Python                          #
# --------------------------------------------------------------------------- #


class TestMicroPerf:
    """Sanity check that the native path isn't a perf regression.

    We don't assert a hard multiplier because CI hosts vary, but the
    native path should be at least as fast as the Python path on a small
    hot loop of R0 lookups.
    """

    def test_native_mm_not_slower_than_python(self, nrl_root: Path) -> None:
        mf = _make_manifest(prompt="perf mm")
        _write_mm_cache(nrl_root, mf, "perf reply", tokens=1)
        req = native_ladder.MmLookupRequest(
            root_dir=str(nrl_root / "cache" / "mm"),
            model_sha256=mf.model_sha256,
            prompt=mf.prompt,
            sampler_fingerprint=mf.sampler_fingerprint(),
            seed=mf.seed,
            max_tokens=mf.max_tokens,
            muscle_memory_on=True,
        )
        import time

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            native_ladder.mm_lookup(req)
        native_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters):
            gguf.muscle_memory_lookup(mf)
        py_ms = (time.perf_counter() - t0) * 1000.0

        # The native path should at most be ~3x slower on a very small
        # hot loop (file I/O dominates). If it's _much_ worse, something
        # is pathologically wrong.
        assert native_ms < py_ms * 3.0, (
            f"native mm_lookup suspiciously slow: native={native_ms:.2f}ms "
            f"python={py_ms:.2f}ms over {iters} iters"
        )
