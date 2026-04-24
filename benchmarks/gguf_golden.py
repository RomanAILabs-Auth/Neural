# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""GGUF runner golden smoke + real-model claim lock.

Five modes:

* ``--mode stub`` — create a synthetic ``.gguf``, run with
  ``NRL_INFERENCE=stub``, verify determinism and four-metric TPS invariants.
  Always runnable, zero external deps. Used as the always-on CI gate.

* ``--mode p2active-sim`` — same stub setup, but with
  ``gate_skip_ratio_override > 0``. Locks the flipped-hinge invariants:
  ``virtual_tps > executed_tps``, the ``virtual_tps = executed_tps / (1 − gate_skip_ratio)``
  formula holds exactly, and the banner carries the "P2-Active simulation"
  label (numeric fixture).

* ``--mode p2active-prefill`` — two sequential stub runs sharing a
  :class:`nrlpy.gates.PrefillGate`. Turn 1 with a fresh gate must show
  ``gate_skip_ratio == 0`` (nothing to skip yet). Turn 2 with a prompt that
  shares a prefix with turn 1 must show ``gate_skip_ratio > 0`` sourced from
  the structural gate (``gate_source == "prefill_cache"``), the formula still
  holds exactly, and the banner carries the "P2-Active (prefill cache)"
  label. This is the first **structural** gate crossing the hinge — on the
  native backend it corresponds to real libllama KV-cache reuse.

* ``--mode real`` — run against a real GGUF (``--model PATH`` or env
  ``NRL_GGUF_GOLDEN_MODEL``). Verifies the completion is non-empty, tokens
  > 0, and the honesty-hinge invariant ``virtual_tps == executed_tps`` while
  the project is in P1 / P2-Shadow. Records the completion SHA-256 and the
  four-metric TPS into ``build/gguf_golden/gguf_golden.{json,md}`` for
  claim-governance diffs.

* ``--mode mm-replay`` — two identical stub runs with ``muscle_memory=on``.
  Turn 2 must be a cache hit. If ``--mm-min-wps`` > 0, also require
  ``effective_wps >= --mm-min-wps`` on that replay (default ``0`` = measure
  only, no floor — use ``1000`` in CI when you want an explicit gate). This is
  **replay** throughput, not a cap on how fast fresh decode can go.

* ``--mode auto`` — stub, p2active-sim, p2active-prefill, and mm-replay always
  run; real only runs if ``NRL_GGUF_GOLDEN_MODEL`` is set and the path resolves.

Exit codes:

* ``0`` — all selected modes passed.
* ``1`` — an assertion failed (regression).
* ``2`` — configuration error (e.g. ``--mode real`` without a model path).

See ``docs/nrl_gguf_runner_architecture.md`` §7 for the honest-accounting
checklist this harness enforces.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _ensure_repo_nrlpy_on_path(root: Path) -> None:
    src = root / "nrlpy" / "src"
    if src.is_dir():
        sp = str(src)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ROOT = Path(__file__).resolve().parent.parent
_ensure_repo_nrlpy_on_path(_ROOT)

from nrlpy import gguf  # noqa: E402 — path rewrite above

_P2ACTIVE_SIM_OVERRIDE = 0.5


# --------------------------------------------------------------------------- #
# Structured results
# --------------------------------------------------------------------------- #


@dataclass
class ModeResult:
    mode: str
    passed: bool = False
    skipped: bool = False
    skip_reason: str = ""
    model: str = ""
    model_sha256: str = ""
    tokens: int = 0
    completion_sha256: str = ""
    executed_tps: float = 0.0
    virtual_tps: float = 0.0
    cache_tps: float = 0.0
    effective_tps: float = 0.0
    gate_skip_ratio: float = 0.0
    stream_chunk_ms: float = 0.0
    cache_hit: bool = False
    attestation_variant: str = ""
    lattice_observation_skip_ratio: float = 0.0
    words: int = 0
    words_per_token: float = 0.0
    executed_wps: float = 0.0
    virtual_wps: float = 0.0
    effective_wps: float = 0.0
    assertions: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    error: str = ""
    gate_source: str | None = None
    shared_prefix_len: int = 0
    prompt_token_count: int = 0


@dataclass
class Report:
    version: str = ""
    timestamp: str = ""
    platform: str = ""
    python: str = ""
    results: list[ModeResult] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "python": self.python,
            "results": [asdict(r) for r in self.results],
        }


# --------------------------------------------------------------------------- #
# Assertion helpers (recorded, not raised — we want the JSON artifact either way)
# --------------------------------------------------------------------------- #


def _record(result: ModeResult, name: str, ok: bool, detail: str) -> None:
    result.assertions.append({"name": name, "ok": ok, "detail": detail})


def _evaluate(result: ModeResult) -> None:
    result.passed = all(a["ok"] for a in result.assertions)


def _metric_wps(result: ModeResult, metric: str) -> float:
    if metric == "executed":
        return result.executed_wps
    if metric == "virtual":
        return result.virtual_wps
    return result.effective_wps


def _assert_min_wps(result: ModeResult, *, min_wps: float, metric: str) -> None:
    if result.skipped or min_wps <= 0.0:
        return
    value = _metric_wps(result, metric)
    _record(
        result,
        f"min_{metric}_wps_gate",
        value >= min_wps,
        f"{metric}_wps={value:.3f} min_required={min_wps:.3f}",
    )


# --------------------------------------------------------------------------- #
# Stub mode
# --------------------------------------------------------------------------- #


def _make_dummy_gguf(path: Path) -> None:
    """Synthetic GGUF blob. The stub backend ignores weights; hasher reads bytes."""
    path.write_bytes(b"GGUF" + b"\x00" * 64)


def run_stub_mode(*, prompt: str, max_tokens: int, seed: int) -> ModeResult:
    """Deterministic smoke: no llama-cpp, no real model, no network."""
    r = ModeResult(mode="stub")
    with tempfile.TemporaryDirectory(prefix="nrl_gguf_golden_") as tmp:
        tmp_path = Path(tmp)
        os.environ["NRL_ROOT"] = str(tmp_path)
        os.environ["NRL_INFERENCE"] = "stub"
        model = tmp_path / "stub.gguf"
        _make_dummy_gguf(model)
        r.model = str(model)

        manifest = gguf.manifest_from_args(
            model=str(model), prompt=prompt, max_tokens=max_tokens, seed=seed
        )
        t0 = time.perf_counter()
        try:
            result = gguf.run_gguf(manifest, stream_to=io.StringIO(), observation_profile="")
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"
            _record(r, "run_gguf_does_not_raise", False, r.error)
            _evaluate(r)
            return r
        r.wall_s = time.perf_counter() - t0

        tps = result.tps
        r.tokens = result.tokens
        r.completion_sha256 = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        r.model_sha256 = result.model_sha256
        r.executed_tps = tps.executed_tps
        r.virtual_tps = tps.virtual_tps
        r.cache_tps = tps.cache_tps
        r.effective_tps = tps.effective_tps
        r.gate_skip_ratio = tps.gate_skip_ratio
        r.stream_chunk_ms = tps.stream_chunk_ms
        r.cache_hit = result.cache_hit
        r.attestation_variant = result.nrl_attestation.variant
        r.words = result.word_rates.word_count
        r.words_per_token = result.word_rates.words_per_token
        r.executed_wps = result.word_rates.executed_wps
        r.virtual_wps = result.word_rates.virtual_wps
        r.effective_wps = result.word_rates.effective_wps

        # Stub is deterministic in max_tokens and seed. _StubLlm yields exactly
        # max_tokens pieces: " token_<(seed+i)%997>" for i in range(max_tokens).
        expected_pieces = [f" token_{(seed + i) % 997}" for i in range(max_tokens)]
        expected_text = "".join(expected_pieces)
        expected_sha = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()

        _record(r, "token_count_matches_max_tokens",
                result.tokens == max_tokens,
                f"tokens={result.tokens} max_tokens={max_tokens}")
        _record(r, "completion_is_deterministic_for_seed",
                r.completion_sha256 == expected_sha,
                f"got_sha={r.completion_sha256[:12]} expected_sha={expected_sha[:12]}")
        _record(r, "executed_tps_is_positive",
                tps.executed_tps > 0,
                f"executed_tps={tps.executed_tps:.3f}")
        _record(r, "virtual_tps_equals_executed_tps_in_p1_p2_shadow",
                abs(tps.virtual_tps - tps.executed_tps) < 1e-9,
                f"executed={tps.executed_tps:.6f} virtual={tps.virtual_tps:.6f} "
                f"(honesty hinge — see docs/nrl_gguf_runner_architecture.md §1.0)")
        _record(r, "gate_skip_ratio_is_zero_in_p1_p2_shadow",
                tps.gate_skip_ratio == 0.0,
                f"gate_skip_ratio={tps.gate_skip_ratio}")
        _record(r, "cache_miss_on_first_run",
                not result.cache_hit,
                f"cache_hit={result.cache_hit}")

    _evaluate(r)
    return r


# --------------------------------------------------------------------------- #
# P2-Active simulation mode (the flipped-hinge claim)
# --------------------------------------------------------------------------- #


def run_p2active_sim_mode(
    *, prompt: str, max_tokens: int, seed: int, override: float = _P2ACTIVE_SIM_OVERRIDE,
) -> ModeResult:
    """Stub run with an explicit ``gate_skip_ratio_override``.

    Locks the **first legitimate** ``virtual_tps > executed_tps`` measurement
    end-to-end (manifest → TpsReport → banner → evidence). The override is
    labeled "simulation" in the banner and evidence log — when P3 ships the
    real libllama elision, this mode will be repurposed to assert the same
    invariants against a gate callback instead of a fixed number.
    """
    r = ModeResult(mode="p2active-sim")
    with tempfile.TemporaryDirectory(prefix="nrl_gguf_golden_p2a_") as tmp:
        tmp_path = Path(tmp)
        os.environ["NRL_ROOT"] = str(tmp_path)
        os.environ["NRL_INFERENCE"] = "stub"
        # Isolate: don't let a stray host NRL_GATE_SKIP_RATIO_OVERRIDE skew this.
        os.environ.pop("NRL_GATE_SKIP_RATIO_OVERRIDE", None)
        model = tmp_path / "stub.gguf"
        _make_dummy_gguf(model)
        r.model = str(model)

        manifest = gguf.manifest_from_args(
            model=str(model),
            prompt=prompt,
            max_tokens=max_tokens,
            seed=seed,
            gate_skip_ratio_override=override,
        )
        t0 = time.perf_counter()
        try:
            result = gguf.run_gguf(manifest, stream_to=io.StringIO(), observation_profile="")
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"
            _record(r, "run_gguf_does_not_raise", False, r.error)
            _evaluate(r)
            return r
        r.wall_s = time.perf_counter() - t0

        tps = result.tps
        r.tokens = result.tokens
        r.completion_sha256 = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        r.model_sha256 = result.model_sha256
        r.executed_tps = tps.executed_tps
        r.virtual_tps = tps.virtual_tps
        r.cache_tps = tps.cache_tps
        r.effective_tps = tps.effective_tps
        r.gate_skip_ratio = tps.gate_skip_ratio
        r.stream_chunk_ms = tps.stream_chunk_ms
        r.cache_hit = result.cache_hit
        r.attestation_variant = result.nrl_attestation.variant
        r.words = result.word_rates.word_count
        r.words_per_token = result.word_rates.words_per_token
        r.executed_wps = result.word_rates.executed_wps
        r.virtual_wps = result.word_rates.virtual_wps
        r.effective_wps = result.word_rates.effective_wps

        banner = gguf.format_banner(result)

        _record(r, "gate_skip_ratio_equals_override",
                abs(tps.gate_skip_ratio - override) < 1e-9,
                f"gate_skip_ratio={tps.gate_skip_ratio:.6f} override={override}")
        _record(r, "virtual_tps_exceeds_executed_tps",
                tps.virtual_tps > tps.executed_tps,
                f"virtual={tps.virtual_tps:.3f} executed={tps.executed_tps:.3f}")
        formula_err = abs(tps.virtual_tps * (1.0 - tps.gate_skip_ratio) - tps.executed_tps)
        _record(r, "virtual_tps_formula_holds",
                formula_err < 1e-6,
                f"|virtual*(1-skip) - executed| = {formula_err:.2e}")
        _record(r, "banner_marks_simulation",
                "P2-Active simulation" in banner,
                "banner must explicitly label simulation (no stealth claim)")
        _record(r, "banner_formula_ok_yes",
                "virtual_tps_formula_ok    yes" in banner,
                "banner sanity-check line must self-report yes")

    _evaluate(r)
    return r


# --------------------------------------------------------------------------- #
# P2-Active structural gate (prefill cache) — the real-elision machinery
# --------------------------------------------------------------------------- #


def run_p2active_prefill_mode(
    *, prompt: str, max_tokens: int, seed: int,
) -> ModeResult:
    """Two sequential runs sharing a :class:`PrefillGate`.

    Turn 1 uses the shared prefix alone; the gate has no history so
    ``gate_skip_ratio`` must be ``0.0``. Turn 2 extends turn 1's prompt with
    a trailing novel suffix; the gate must now report a positive
    ``gate_skip_ratio`` sourced from ``prefill_cache``, and the
    ``virtual_tps = executed_tps / (1 − gate_skip_ratio)`` formula must hold.

    This is the **first structural gate crossing the honesty hinge**. On the
    native backend the same gate reading corresponds to libllama's free
    ``n_past`` carry-over (no per-layer patching required); on stub we
    exercise the full end-to-end path so regressions are caught in CI.
    """
    r = ModeResult(mode="p2active-prefill")
    with tempfile.TemporaryDirectory(prefix="nrl_gguf_golden_pfx_") as tmp:
        tmp_path = Path(tmp)
        os.environ["NRL_ROOT"] = str(tmp_path)
        os.environ["NRL_INFERENCE"] = "stub"
        os.environ.pop("NRL_GATE_SKIP_RATIO_OVERRIDE", None)
        model = tmp_path / "stub.gguf"
        _make_dummy_gguf(model)
        r.model = str(model)

        gate = gguf.PrefillGate()
        shared_prefix = prompt
        turn2_prompt = f"{shared_prefix} and now tell me what comes next please"

        base_kwargs: dict[str, Any] = dict(
            model=str(model),
            max_tokens=max_tokens,
            seed=seed,
            prefill_cache="session",
            muscle_memory="off",  # don't let muscle-memory short-circuit the gate path
        )

        t0 = time.perf_counter()
        try:
            turn1 = gguf.run_gguf(
                gguf.manifest_from_args(prompt=shared_prefix, **base_kwargs),
                stream_to=io.StringIO(),
                observation_profile="",
                prefill_gate=gate,
            )
            turn2 = gguf.run_gguf(
                gguf.manifest_from_args(prompt=turn2_prompt, **base_kwargs),
                stream_to=io.StringIO(),
                observation_profile="",
                prefill_gate=gate,
            )
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"
            _record(r, "two_turn_run_does_not_raise", False, r.error)
            _evaluate(r)
            return r
        r.wall_s = time.perf_counter() - t0

        # Snapshot turn 2 (the interesting one) on the ModeResult surface.
        tps2 = turn2.tps
        r.tokens = turn2.tokens
        r.completion_sha256 = hashlib.sha256(turn2.text.encode("utf-8")).hexdigest()
        r.model_sha256 = turn2.model_sha256
        r.executed_tps = tps2.executed_tps
        r.virtual_tps = tps2.virtual_tps
        r.cache_tps = tps2.cache_tps
        r.effective_tps = tps2.effective_tps
        r.gate_skip_ratio = tps2.gate_skip_ratio
        r.stream_chunk_ms = tps2.stream_chunk_ms
        r.cache_hit = turn2.cache_hit
        r.attestation_variant = turn2.nrl_attestation.variant
        r.words = turn2.word_rates.word_count
        r.words_per_token = turn2.word_rates.words_per_token
        r.executed_wps = turn2.word_rates.executed_wps
        r.virtual_wps = turn2.word_rates.virtual_wps
        r.effective_wps = turn2.word_rates.effective_wps
        r.gate_source = turn2.gate_source
        r.shared_prefix_len = int(turn2.gate_report.get("shared_prefix_len", 0) or 0)
        r.prompt_token_count = int(turn2.gate_report.get("prompt_token_count", 0) or 0)

        banner2 = gguf.format_banner(turn2)

        _record(r, "turn1_gate_source_is_none",
                turn1.gate_source is None,
                f"turn1.gate_source={turn1.gate_source!r} (gate had no history)")
        _record(r, "turn1_gate_skip_ratio_is_zero",
                turn1.tps.gate_skip_ratio == 0.0,
                f"turn1.tps.gate_skip_ratio={turn1.tps.gate_skip_ratio}")
        _record(r, "turn1_honesty_hinge_holds",
                turn1.tps.virtual_tps == turn1.tps.executed_tps,
                f"virtual={turn1.tps.virtual_tps:.3f} executed={turn1.tps.executed_tps:.3f}")
        _record(r, "turn2_gate_source_is_prefill_cache",
                turn2.gate_source == "prefill_cache",
                f"turn2.gate_source={turn2.gate_source!r}")
        _record(r, "turn2_gate_skip_ratio_positive",
                tps2.gate_skip_ratio > 0.0,
                f"gate_skip_ratio={tps2.gate_skip_ratio:.6f}")
        _record(r, "turn2_virtual_tps_exceeds_executed_tps",
                tps2.virtual_tps > tps2.executed_tps,
                f"virtual={tps2.virtual_tps:.3f} executed={tps2.executed_tps:.3f}")
        formula_err = abs(tps2.virtual_tps * (1.0 - tps2.gate_skip_ratio) - tps2.executed_tps)
        _record(r, "turn2_virtual_tps_formula_holds",
                formula_err < 1e-6,
                f"|virtual*(1-skip) - executed| = {formula_err:.2e}")
        _record(r, "banner_marks_prefill_cache_gate",
                "P2-Active (prefill cache)" in banner2,
                "banner must label the structural gate source")
        # Cross-check: sim label MUST NOT appear when the structural gate fired.
        _record(r, "banner_does_not_claim_simulation",
                "P2-Active simulation" not in banner2,
                "structural-gate banner must not wear the simulation label")

    _evaluate(r)
    return r


# --------------------------------------------------------------------------- #
# Muscle-memory replay (>=1k effective WPS gate)
# --------------------------------------------------------------------------- #


def run_mm_replay_mode(*, prompt: str, max_tokens: int, seed: int) -> ModeResult:
    """Two identical manifests: turn 1 fills muscle memory; turn 2 must replay.

    Uses ``NRL_INFERENCE=stub`` under a fresh ``NRL_ROOT`` so CI is hermetic.
    The gate is ``effective_wps`` on turn 2 (disk replay is effectively
    instantaneous, so words/sec explodes — that is the product contract for
    "muscle memory", distinct from executed_wps on a cold prompt).
    """
    r = ModeResult(mode="mm-replay")
    mt = max(64, int(max_tokens))
    with tempfile.TemporaryDirectory(prefix="nrl_gguf_mm_") as tmp:
        tmp_path = Path(tmp)
        os.environ["NRL_ROOT"] = str(tmp_path)
        os.environ["NRL_INFERENCE"] = "stub"
        os.environ.pop("NRL_STREAM_CHUNK_MS", None)
        os.environ.pop("NRL_GATE_SKIP_RATIO_OVERRIDE", None)
        model = tmp_path / "stub.gguf"
        _make_dummy_gguf(model)
        r.model = str(model)

        base_kwargs: dict[str, Any] = dict(
            model=str(model),
            prompt=prompt,
            max_tokens=mt,
            seed=seed,
            muscle_memory="on",
            chat_format="none",
        )

        t0 = time.perf_counter()
        try:
            turn1 = gguf.run_gguf(
                gguf.manifest_from_args(**base_kwargs),
                stream_to=io.StringIO(),
                observation_profile="",
            )
            turn2 = gguf.run_gguf(
                gguf.manifest_from_args(**base_kwargs),
                stream_to=io.StringIO(),
                observation_profile="",
            )
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"
            _record(r, "mm_replay_two_turn_run", False, r.error)
            _evaluate(r)
            return r
        r.wall_s = time.perf_counter() - t0

        tps2 = turn2.tps
        wr2 = turn2.word_rates
        r.tokens = turn2.tokens
        r.model_sha256 = turn2.model_sha256
        r.executed_tps = tps2.executed_tps
        r.virtual_tps = tps2.virtual_tps
        r.cache_tps = tps2.cache_tps
        r.effective_tps = tps2.effective_tps
        r.gate_skip_ratio = tps2.gate_skip_ratio
        r.stream_chunk_ms = tps2.stream_chunk_ms
        r.cache_hit = turn2.cache_hit
        r.attestation_variant = turn2.nrl_attestation.variant
        r.words = wr2.word_count
        r.words_per_token = wr2.words_per_token
        r.executed_wps = wr2.executed_wps
        r.virtual_wps = wr2.virtual_wps
        r.effective_wps = wr2.effective_wps
        r.completion_sha256 = hashlib.sha256(turn2.text.encode("utf-8")).hexdigest()

        _record(r, "turn1_not_cache_hit", not turn1.cache_hit, f"turn1.cache_hit={turn1.cache_hit}")
        _record(r, "turn2_is_cache_hit", turn2.cache_hit, f"turn2.cache_hit={turn2.cache_hit}")
        _record(
            r,
            "turn2_effective_wps_positive",
            wr2.effective_wps > 0.0,
            f"effective_wps={wr2.effective_wps:.3f}",
        )
        _record(
            r,
            "turn2_executed_wps_near_zero_on_replay",
            wr2.executed_wps < 1.0,
            f"executed_wps={wr2.executed_wps:.3f} (replay path, not fresh decode)",
        )

    _evaluate(r)
    return r


# --------------------------------------------------------------------------- #
# Real mode
# --------------------------------------------------------------------------- #


def _resolve_real_model(cli_path: str | None) -> Path | None:
    if cli_path:
        p = Path(cli_path).expanduser()
        if p.is_file():
            return p
        return None
    env = os.environ.get("NRL_GGUF_GOLDEN_MODEL", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    return None


def run_real_mode(
    *,
    model: Path,
    prompt: str,
    max_tokens: int,
    seed: int,
    chat_format: str,
    required: bool,
) -> ModeResult:
    r = ModeResult(mode="real", model=str(model))
    # Live inference must not be silently paced; harness is about raw behaviour.
    os.environ.pop("NRL_STREAM_CHUNK_MS", None)
    # Respect operator's NRL_INFERENCE. Default to native when unset so the
    # harness exercises llama-cpp-python, not the stub.
    os.environ.setdefault("NRL_INFERENCE", "native")
    try:
        manifest = gguf.manifest_from_args(
            model=str(model),
            prompt=prompt,
            max_tokens=max_tokens,
            seed=seed,
            chat_format=chat_format,
        )
    except Exception as exc:  # noqa: BLE001
        r.error = f"manifest_error: {type(exc).__name__}: {exc}"
        _record(r, "manifest_parses", False, r.error)
        if not required:
            r.skipped = True
            r.skip_reason = r.error
        _evaluate(r)
        return r

    t0 = time.perf_counter()
    try:
        result = gguf.run_gguf(manifest, stream_to=io.StringIO())
    except Exception as exc:  # noqa: BLE001
        r.error = f"{type(exc).__name__}: {exc}"
        _record(r, "run_gguf_does_not_raise", False, r.error)
        if not required:
            r.skipped = True
            r.skip_reason = r.error
        _evaluate(r)
        return r
    r.wall_s = time.perf_counter() - t0

    tps = result.tps
    r.tokens = result.tokens
    r.completion_sha256 = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
    r.model_sha256 = result.model_sha256
    r.executed_tps = tps.executed_tps
    r.virtual_tps = tps.virtual_tps
    r.cache_tps = tps.cache_tps
    r.effective_tps = tps.effective_tps
    r.gate_skip_ratio = tps.gate_skip_ratio
    r.stream_chunk_ms = tps.stream_chunk_ms
    r.cache_hit = result.cache_hit
    r.attestation_variant = result.nrl_attestation.variant
    r.words = result.word_rates.word_count
    r.words_per_token = result.word_rates.words_per_token
    r.executed_wps = result.word_rates.executed_wps
    r.virtual_wps = result.word_rates.virtual_wps
    r.effective_wps = result.word_rates.effective_wps
    obs = result.lattice_observation
    r.lattice_observation_skip_ratio = obs.skip_ratio if obs.available else 0.0

    _record(r, "completion_is_nonempty",
            len(result.text.strip()) > 0,
            f"len={len(result.text)} head={result.text[:32]!r}")
    _record(r, "token_count_is_positive",
            result.tokens > 0,
            f"tokens={result.tokens}")
    _record(r, "executed_tps_is_positive",
            tps.executed_tps > 0,
            f"executed_tps={tps.executed_tps:.3f}")
    # The honesty hinge applies in P1 / P2-Shadow. When a future phase activates
    # a real gate, this assertion will need relaxing alongside the runner doc.
    _record(r, "virtual_tps_equals_executed_tps_in_p1_p2_shadow",
            abs(tps.virtual_tps - tps.executed_tps) < 1e-9,
            f"executed={tps.executed_tps:.6f} virtual={tps.virtual_tps:.6f}")
    _record(r, "gate_skip_ratio_is_zero_in_p1_p2_shadow",
            tps.gate_skip_ratio == 0.0,
            f"gate_skip_ratio={tps.gate_skip_ratio}")

    _evaluate(r)
    return r


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _format_md(report: Report) -> str:
    lines: list[str] = []
    lines.append("# GGUF golden harness report")
    lines.append("")
    lines.append(f"- **nrl version:** `{report.version}`")
    lines.append(f"- **timestamp:** `{report.timestamp}`")
    lines.append(f"- **platform:** `{report.platform}`")
    lines.append(f"- **python:** `{report.python}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for r in report.results:
        status = "SKIPPED" if r.skipped else ("PASS" if r.passed else "FAIL")
        lines.append(f"### `{r.mode}` — **{status}**")
        lines.append("")
        if r.skipped:
            lines.append(f"_skip reason:_ `{r.skip_reason}`")
            lines.append("")
            continue
        if r.error and not r.passed:
            lines.append(f"_error:_ `{r.error}`")
            lines.append("")
        lines.append(f"- model: `{r.model}`")
        lines.append(f"- model_sha256: `{r.model_sha256[:16]}…`")
        lines.append(f"- completion_sha256: `{r.completion_sha256[:16]}…`")
        lines.append(f"- tokens: `{r.tokens}`")
        lines.append(f"- wall_s: `{r.wall_s:.3f}`")
        lines.append(f"- executed_tps: `{r.executed_tps:.3f}`")
        lines.append(f"- virtual_tps: `{r.virtual_tps:.3f}`")
        lines.append(f"- cache_tps: `{r.cache_tps:.3f}`")
        lines.append(f"- effective_tps: `{r.effective_tps:.3f}`")
        lines.append(f"- words: `{r.words}`")
        lines.append(f"- words_per_token: `{r.words_per_token:.3f}`")
        lines.append(f"- executed_wps: `{r.executed_wps:.3f}`")
        lines.append(f"- virtual_wps: `{r.virtual_wps:.3f}`")
        lines.append(f"- effective_wps: `{r.effective_wps:.3f}`")
        lines.append(f"- gate_skip_ratio: `{r.gate_skip_ratio:.3f}`")
        lines.append(f"- stream_chunk_ms: `{r.stream_chunk_ms:.1f}`")
        lines.append(f"- attestation_variant: `{r.attestation_variant or '<none>'}`")
        lines.append(f"- lattice_observation_skip_ratio: `{r.lattice_observation_skip_ratio:.3f}`")
        lines.append("")
        lines.append("| assertion | ok | detail |")
        lines.append("|---|---|---|")
        for a in r.assertions:
            lines.append(f"| `{a['name']}` | `{a['ok']}` | {a['detail']} |")
        lines.append("")
    lines.append("## Honesty contract")
    lines.append("")
    lines.append(
        "`virtual_tps == executed_tps` is the current honesty hinge "
        "(P1 / P2-Shadow). It will be relaxed when a layer / KV / expert gate "
        "is actually wired into libllama (P2-Active+). See "
        "`docs/nrl_gguf_runner_architecture.md` §1.0."
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("stub", "p2active-sim", "p2active-prefill", "mm-replay", "real", "auto"),
        default="auto",
    )
    p.add_argument("--model", default=None, help="GGUF path (overrides NRL_GGUF_GOLDEN_MODEL)")
    p.add_argument("--prompt", default="Reply with one short sentence about the number two.")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chat-format", default="phi3")
    p.add_argument("--min-wps", type=float, default=0.0)
    p.add_argument(
        "--wps-metric",
        choices=("executed", "virtual", "effective"),
        default="effective",
    )
    p.add_argument(
        "--mm-min-wps",
        type=float,
        default=0.0,
        help="If > 0: require effective_wps >= this on mm-replay turn 2 (e.g. 1000 in CI). "
        "Default 0 = report WPS without imposing a floor.",
    )
    p.add_argument("--output-dir", default=str(_ROOT / "build" / "gguf_golden"))
    return p.parse_args(argv)


def _read_nrl_version() -> str:
    try:
        from nrlpy import runtime  # noqa: PLC0415
        return str(runtime.version())
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = Report(
        version=_read_nrl_version(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        platform=f"{platform.system()} {platform.machine()}",
        python=sys.version.split()[0],
    )

    if args.mode in ("stub", "auto"):
        print("[gguf-golden] mode=stub …")
        stub_r = run_stub_mode(prompt=args.prompt, max_tokens=args.max_tokens, seed=args.seed)
        report.results.append(stub_r)
        _assert_min_wps(stub_r, min_wps=args.min_wps, metric=args.wps_metric)
        _evaluate(stub_r)
        print(f"[gguf-golden] stub: {'PASS' if stub_r.passed else 'FAIL'} ({len(stub_r.assertions)} checks)")

    if args.mode in ("p2active-sim", "auto"):
        print("[gguf-golden] mode=p2active-sim …")
        sim_r = run_p2active_sim_mode(
            prompt=args.prompt, max_tokens=args.max_tokens, seed=args.seed,
        )
        report.results.append(sim_r)
        _assert_min_wps(sim_r, min_wps=args.min_wps, metric=args.wps_metric)
        _evaluate(sim_r)
        print(
            f"[gguf-golden] p2active-sim: "
            f"{'PASS' if sim_r.passed else 'FAIL'} ({len(sim_r.assertions)} checks, "
            f"virtual_tps={sim_r.virtual_tps:.1f} > executed_tps={sim_r.executed_tps:.1f})"
        )

    if args.mode in ("p2active-prefill", "auto"):
        print("[gguf-golden] mode=p2active-prefill …")
        pfx_r = run_p2active_prefill_mode(
            prompt=args.prompt, max_tokens=args.max_tokens, seed=args.seed,
        )
        report.results.append(pfx_r)
        _assert_min_wps(pfx_r, min_wps=args.min_wps, metric=args.wps_metric)
        _evaluate(pfx_r)
        print(
            f"[gguf-golden] p2active-prefill: "
            f"{'PASS' if pfx_r.passed else 'FAIL'} ({len(pfx_r.assertions)} checks, "
            f"gate_source={pfx_r.gate_source}, skip={pfx_r.gate_skip_ratio:.3f}, "
            f"shared={pfx_r.shared_prefix_len}/{pfx_r.prompt_token_count})"
        )

    if args.mode in ("mm-replay", "auto"):
        print("[gguf-golden] mode=mm-replay …")
        mm_r = run_mm_replay_mode(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        report.results.append(mm_r)
        mm_thr = max(float(args.mm_min_wps), float(args.min_wps))
        _assert_min_wps(mm_r, min_wps=mm_thr, metric="effective")
        _evaluate(mm_r)
        print(
            f"[gguf-golden] mm-replay: "
            f"{'PASS' if mm_r.passed else 'FAIL'} "
            f"(cache_hit={mm_r.cache_hit}, effective_wps={mm_r.effective_wps:.1f}, "
            f"min={mm_thr:.1f})"
        )

    if args.mode in ("real", "auto"):
        model = _resolve_real_model(args.model)
        if model is None:
            skipped = ModeResult(mode="real", skipped=True,
                                 skip_reason="no --model and NRL_GGUF_GOLDEN_MODEL unset")
            if args.mode == "real":
                _record(skipped, "model_resolves", False, skipped.skip_reason)
                skipped.skipped = False  # explicit --mode real makes this a hard fail
                _evaluate(skipped)
                print("[gguf-golden] real: FAIL (no model path)")
                report.results.append(skipped)
                _write_artifacts(report, Path(args.output_dir))
                return 2
            print(f"[gguf-golden] real: SKIPPED ({skipped.skip_reason})")
            report.results.append(skipped)
        else:
            print(f"[gguf-golden] mode=real model={model}")
            real_r = run_real_mode(
                model=model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                seed=args.seed,
                chat_format=args.chat_format,
                required=(args.mode == "real"),
            )
            report.results.append(real_r)
            _assert_min_wps(real_r, min_wps=args.min_wps, metric=args.wps_metric)
            _evaluate(real_r)
            if real_r.skipped:
                print(f"[gguf-golden] real: SKIPPED ({real_r.skip_reason})")
            else:
                print(f"[gguf-golden] real: {'PASS' if real_r.passed else 'FAIL'} "
                      f"({len(real_r.assertions)} checks, wall={real_r.wall_s:.2f}s)")

    _write_artifacts(report, Path(args.output_dir))

    hard_fail = any((not r.passed) and (not r.skipped) for r in report.results)
    return 1 if hard_fail else 0


def _write_artifacts(report: Report, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gguf_golden.json"
    md_path = out_dir / "gguf_golden.md"
    json_path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_format_md(report), encoding="utf-8")
    print(f"[gguf-golden] wrote {json_path}")
    print(f"[gguf-golden] wrote {md_path}")


if __name__ == "__main__":
    sys.exit(main())
