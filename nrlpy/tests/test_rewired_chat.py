# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Unit tests for the ``--rewired`` chat preset.

Rewired mode is the strict superset of ``--fast-chat``: it unlocks R2
Omega Native Resolve as the primary fast path via ``coherence_lane =
max-throughput`` and a fuzzy ZPM Hamming threshold, and intentionally
skips the ``chat_prewarm`` seed corpus (rewired learns from the
conversation, not from canned generic prompts).

These tests verify

* ``_apply_rewired_defaults`` + ``apply_rewired_post_build`` produce
  the advertised manifest shape.
* User-supplied flags win over the rewired preset.
* ``--rewired`` is parseable on both ``nrlpy chat`` and
  ``nrlpy run`` and propagates through to the REPL with the
  ``rewired=True`` flag.
* Rewired mode does **not** call ``chat_prewarm.prewarm_chat_cache``
  (no pre-warmed generic prompts — user requirement).
"""

from __future__ import annotations

from typing import Any

import pytest

from nrlpy import gguf, gguf_chat
from nrlpy.cli import main as cli_main


# --------------------------------------------------------------------------- #
# _apply_rewired_defaults + apply_rewired_post_build
# --------------------------------------------------------------------------- #


def test_apply_rewired_defaults_populates_expected_kwargs() -> None:
    kwargs: dict[str, Any] = {}
    user_overrides: set[str] = set()
    gguf_chat._apply_rewired_defaults(kwargs, user_overrides)
    assert kwargs["runner_backend"] == "native_full"
    assert kwargs["muscle_memory"] == "on"
    assert kwargs["coherence_lane"] == "max-throughput"
    assert kwargs["prefill_cache"] == "session"
    assert kwargs["max_tokens"] == 192
    assert kwargs["temperature"] == pytest.approx(0.15)
    assert kwargs["repeat_penalty"] == pytest.approx(1.08)
    assert kwargs["omega_budget_ms"] == pytest.approx(12.0)
    assert kwargs["omega_candidates"] == 12


def test_apply_rewired_defaults_respects_user_overrides() -> None:
    kwargs: dict[str, Any] = {"temperature": 0.7, "max_tokens": 512}
    user_overrides = {"temperature", "max_tokens", "coherence_lane"}
    gguf_chat._apply_rewired_defaults(kwargs, user_overrides)
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["max_tokens"] == 512
    # coherence_lane was claimed as an override but the kwarg slot was
    # empty, so the preset must not fill it.
    assert "coherence_lane" not in kwargs


def test_apply_rewired_post_build_sets_zpm_knobs(tmp_path: Any) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 64)
    mf = gguf.manifest_from_args(
        model=str(model),
        seed=1,
        coherence_lane="max-throughput",
        muscle_memory="on",
    )
    assert bool(getattr(mf, "zpm_nullspace", False)) is False
    assert int(getattr(mf, "zpm_threshold_bits", 0)) == 0
    gguf_chat.apply_rewired_post_build(mf, user_overrides=set())
    assert mf.zpm_nullspace is True
    assert mf.zpm_threshold_bits == 28


def test_apply_rewired_post_build_respects_threshold_override(tmp_path: Any) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 64)
    mf = gguf.manifest_from_args(
        model=str(model), seed=1, coherence_lane="max-throughput"
    )
    mf.zpm_threshold_bits = 7
    gguf_chat.apply_rewired_post_build(
        mf, user_overrides={"zpm_threshold_bits"}
    )
    assert mf.zpm_threshold_bits == 7
    # zpm_nullspace was not overridden so the preset still turns it on.
    assert mf.zpm_nullspace is True


# --------------------------------------------------------------------------- #
# CLI wiring — nrlpy chat model.gguf --rewired
# --------------------------------------------------------------------------- #


def test_cli_chat_rewired_propagates_to_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nrlpy chat model.gguf --rewired`` must build a manifest with
    ``coherence_lane=max-throughput``, a 32-bit ZPM fuzzy threshold,
    and hand the REPL ``rewired=True`` without calling the prewarmer.
    """
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["coherence_lane"] = manifest.coherence_lane
        seen["zpm_nullspace"] = manifest.zpm_nullspace
        seen["zpm_threshold_bits"] = manifest.zpm_threshold_bits
        seen["muscle_memory"] = manifest.muscle_memory
        seen["omega_candidates"] = manifest.omega_candidates
        seen["omega_budget_ms"] = manifest.omega_budget_ms
        seen["rewired"] = kwargs.get("rewired", False)
        seen["fast_chat"] = kwargs.get("fast_chat", False)
        return object()

    def bomb(*a: Any, **k: Any) -> Any:
        pytest.fail("chat_prewarm must not run under --rewired")

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr("nrlpy.chat_prewarm.prewarm_chat_cache", bomb)

    assert cli_main(["chat", "model.gguf", "--rewired"]) == 0
    assert seen["runner_backend"] == "native_full"
    assert seen["coherence_lane"] == "max-throughput"
    assert seen["zpm_nullspace"] is True
    assert seen["zpm_threshold_bits"] == 28
    assert seen["muscle_memory"] == "on"
    assert seen["omega_candidates"] == 12
    assert seen["omega_budget_ms"] == pytest.approx(12.0)
    assert seen["rewired"] is True
    # rewired is the superset -- fast_chat must not be doubly-activated.
    assert seen["fast_chat"] is False


def test_cli_chat_rewired_honors_user_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit flags must win over the rewired preset."""
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["max_tokens"] = manifest.max_tokens
        seen["coherence_lane"] = manifest.coherence_lane
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)

    rc = cli_main(
        [
            "chat",
            "model.gguf",
            "--rewired",
            "--python-ladder",
            "--max-tokens",
            "1024",
        ]
    )
    assert rc == 0
    assert seen["runner_backend"] == "python"
    assert seen["max_tokens"] == 1024
    # coherence_lane is not settable through the chat parser today, so
    # the rewired preset wins here -- this test just pins the contract.
    assert seen["coherence_lane"] == "max-throughput"


# --------------------------------------------------------------------------- #
# CLI wiring — nrlpy run model.gguf --rewired
# --------------------------------------------------------------------------- #


def test_cli_run_rewired_forces_chat_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nrlpy run model.gguf --rewired`` must start the chat REPL
    with rewired defaults, even without an explicit ``--chat``.
    """
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["coherence_lane"] = manifest.coherence_lane
        seen["zpm_threshold_bits"] = manifest.zpm_threshold_bits
        seen["zpm_nullspace"] = manifest.zpm_nullspace
        seen["rewired"] = kwargs.get("rewired", False)
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *a, **k: pytest.fail("run_gguf must not fire in --rewired"),
    )

    assert cli_main(["run", "model.gguf", "--rewired"]) == 0
    assert seen["runner_backend"] == "native_full"
    assert seen["coherence_lane"] == "max-throughput"
    assert seen["zpm_threshold_bits"] == 28
    assert seen["zpm_nullspace"] is True
    assert seen["rewired"] is True


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #


def test_rewired_banner_includes_badge() -> None:
    banner = gguf_chat._boot_banner(
        model_name="phi3.gguf",
        chat_format="phi3",
        seed=1,
        zpm_on=True,
        mm_on=True,
        use_color=False,
        runner_backend="native_full",
        fast_chat=False,
        prewarm_summary="",
        rewired=True,
        zpm_threshold_bits=28,
        coherence_lane="max-throughput",
    )
    assert "REWIRED" in banner
    assert "ZPM + Omega primary path" in banner
    assert "rewired" in banner
    assert "max-throughput" in banner
    assert "zpm_thresh=28b" in banner
    # Must NOT advertise the fast-chat prewarm row.
    assert "cache prewarmed" not in banner


# --------------------------------------------------------------------------- #
# REPL: "learning this prompt..." hint on R5 decode in rewired mode
# --------------------------------------------------------------------------- #


def test_rewired_repl_prints_learning_hint_on_decode_turn(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In rewired mode, every R5 (``[Decode]``) turn must print the
    'learning this prompt... next similar question will be instant.'
    hint so the user understands the latency they just paid is
    amortized over future rephrasings.
    """
    import io  # noqa: PLC0415

    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    monkeypatch.setattr(
        gguf,
        "_collect_lattice_observation",
        lambda profile, neurons, iterations: gguf.NrlLatticeObservation(
            profile=profile, available=False
        ),
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    # P3 default rewired enables response recall; that can lattice-serve turn 0
    # in CI. Force an R5 decode here so the learning hint contract stays stable.
    manifest = gguf.manifest_from_args(
        model=str(model),
        prompt="",
        max_tokens=4,
        seed=1,
        chat_format="none",
        runner_backend="python",
        coherence_lane="max-throughput",
        muscle_memory="off",
    )
    manifest.zpm_nullspace = False
    apply_rewired_post_build = gguf_chat.apply_rewired_post_build  # noqa: N806
    apply_rewired_post_build(
        manifest, user_overrides={"muscle_memory", "zpm_nullspace"}
    )

    stdin = io.StringIO("hello\n/quit\n")
    stdout = io.StringIO()
    gguf_chat.run_gguf_chat_repl(
        manifest,
        stdin=stdin,
        stdout=stdout,
        rewired=True,
        observation_profile="",
    )
    text = stdout.getvalue()
    low = text.lower()
    assert "learning this prompt" in low
    assert "next similar question will be instant" in low


def test_non_rewired_repl_does_not_print_learning_hint(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint is a rewired-mode UX; plain chat must stay silent."""
    import io  # noqa: PLC0415

    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    monkeypatch.setattr(
        gguf,
        "_collect_lattice_observation",
        lambda profile, neurons, iterations: gguf.NrlLatticeObservation(
            profile=profile, available=False
        ),
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    manifest = gguf.manifest_from_args(
        model=str(model), prompt="", max_tokens=4, seed=1, chat_format="none"
    )

    stdin = io.StringIO("hello\n/quit\n")
    stdout = io.StringIO()
    gguf_chat.run_gguf_chat_repl(
        manifest, stdin=stdin, stdout=stdout, observation_profile=""
    )
    text = stdout.getvalue()
    assert "learning this prompt" not in text


def test_non_rewired_banner_has_no_rewired_badge() -> None:
    banner = gguf_chat._boot_banner(
        model_name="phi3.gguf",
        chat_format="phi3",
        seed=1,
        zpm_on=True,
        mm_on=True,
        use_color=False,
        runner_backend="native_full",
    )
    assert "REWIRED" not in banner
    assert "ZPM + Omega primary path" not in banner
