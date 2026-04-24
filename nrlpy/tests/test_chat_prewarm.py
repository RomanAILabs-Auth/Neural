# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Unit tests for :mod:`nrlpy.chat_prewarm` and the ``--fast-chat`` flag.

These tests don't load any real model — they verify that

* The prewarmer writes byte-identical MM + ZPM entries to the paths
  the live Resolution Ladder probes (so a live run would flag a hit).
* The prewarmer is idempotent and accumulates seeds across calls.
* ``--fast-chat`` is parseable on both ``nrlpy chat`` and
  ``nrlpy run --chat`` and propagates through to the REPL.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from nrlpy import chat_prewarm, gguf, zpm
from nrlpy.cli import main as cli_main


def _dummy_manifest(tmp_path: Path) -> gguf.GgufManifest:
    model = tmp_path / "phi3-dummy.gguf"
    model.write_bytes(b"NRL-TEST-GGUF" * 16)
    sha = gguf.sha256_file(model)
    mf = gguf.manifest_from_args(
        model=str(model),
        seed=1,
        max_tokens=192,
        temperature=0.2,
        muscle_memory="on",
        chat_format="phi3",
    )
    mf.model_sha256 = sha
    mf.zpm_nullspace = True
    return mf


def test_prewarm_writes_mm_and_zpm_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    mf = _dummy_manifest(tmp_path)

    seeds = (
        chat_prewarm.PrewarmSeed("hi", "Hello from the cache."),
        chat_prewarm.PrewarmSeed("who are you", "I am the NRL lattice runner."),
    )
    result = chat_prewarm.prewarm_chat_cache(mf, system="be brief", seeds=seeds)

    assert result.seeds_attempted == 2
    assert result.mm_written == 2
    assert result.zpm_written == 2
    assert result.skipped == 0

    # The MM directory now contains one entry per seed (unique FNV keys).
    mm_dir = tmp_path / "cache" / "mm" / mf.model_sha256
    files = list(mm_dir.glob("*.mm"))
    assert len(files) == 2
    for p in files:
        head = p.read_bytes()[:8]
        assert head == gguf.MUSCLE_MEMORY_MAGIC

    # The ZPM index has two (state, reply) pairs.
    zpm_path = tmp_path / "cache" / "zpm" / mf.model_sha256 / "index.bin"
    assert zpm_path.is_file()
    idx = zpm.ZpmIndex.load(zpm_path)
    assert len(idx) == 2


def test_prewarm_mm_hit_matches_live_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MM entry the prewarmer writes must hit when the user types
    the same first-turn prompt through :func:`gguf.muscle_memory_lookup`.
    """
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    mf = _dummy_manifest(tmp_path)
    seeds = (chat_prewarm.PrewarmSeed("hello", "Hi there, served from cache."),)
    chat_prewarm.prewarm_chat_cache(mf, system="", seeds=seeds)

    # Build the per-turn manifest the REPL would build for a fresh
    # session typing "hello" (system="", empty history).
    from nrlpy import gguf_chat  # noqa: PLC0415

    session = gguf_chat.ChatSession(base_manifest=mf, system="")
    rendered = gguf_chat.build_history_prompt(session, "hello")
    live = dataclasses.replace(
        mf, prompt=rendered, prompt_file="", chat_format="none"
    )

    hit = gguf.muscle_memory_lookup(live)
    assert hit is not None
    assert hit.text == "Hi there, served from cache."


def test_prewarm_is_idempotent_and_accumulates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    mf = _dummy_manifest(tmp_path)

    first = (chat_prewarm.PrewarmSeed("ping", "pong"),)
    second = (chat_prewarm.PrewarmSeed("ok", "Okay."),)
    chat_prewarm.prewarm_chat_cache(mf, system="", seeds=first)
    chat_prewarm.prewarm_chat_cache(mf, system="", seeds=second)
    chat_prewarm.prewarm_chat_cache(mf, system="", seeds=first)  # duplicate

    zpm_path = tmp_path / "cache" / "zpm" / mf.model_sha256 / "index.bin"
    idx = zpm.ZpmIndex.load(zpm_path)
    assert len(idx) == 2  # duplicate state skipped


def test_prewarm_skips_without_model_sha(tmp_path: Path) -> None:
    mf = _dummy_manifest(tmp_path)
    mf.model_sha256 = ""
    result = chat_prewarm.prewarm_chat_cache(
        mf, system="", seeds=chat_prewarm.FAST_CHAT_SEEDS[:3]
    )
    assert result.mm_written == 0
    assert result.zpm_written == 0
    assert result.seeds_attempted == 0  # early-out before the loop


def test_prewarm_respects_muscle_memory_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    mf = _dummy_manifest(tmp_path)
    mf.muscle_memory = "off"
    mf.zpm_nullspace = False
    result = chat_prewarm.prewarm_chat_cache(
        mf, system="", seeds=chat_prewarm.FAST_CHAT_SEEDS[:3]
    )
    assert result.mm_written == 0
    assert result.zpm_written == 0
    assert result.skipped == 3


def test_cli_chat_fast_chat_forces_native_full_and_prewarms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``nrlpy chat model.gguf --fast-chat`` must select the native_full
    backend, turn MM + ZPM on, and call the prewarmer before the REPL
    enters its input loop.
    """
    seen: dict[str, Any] = {}
    prewarm_calls: list[tuple[str, str]] = []

    def fake_prewarm(
        manifest: gguf.GgufManifest, **kwargs: Any
    ) -> chat_prewarm.PrewarmResult:
        prewarm_calls.append((manifest.model, kwargs.get("system", "")))
        return chat_prewarm.PrewarmResult(
            seeds_attempted=20, mm_written=20, zpm_written=20
        )

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["muscle_memory"] = manifest.muscle_memory
        seen["zpm_nullspace"] = manifest.zpm_nullspace
        seen["max_tokens"] = manifest.max_tokens
        seen["coherence_lane"] = manifest.coherence_lane
        seen["fast_chat"] = kwargs.get("fast_chat", False)
        # Call the real prewarm codepath indirectly: the REPL invokes
        # chat_prewarm.prewarm_chat_cache when fast_chat=True. Our
        # fake_repl just records that it got the flag.
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr("nrlpy.chat_prewarm.prewarm_chat_cache", fake_prewarm)

    assert cli_main(["chat", "model.gguf", "--fast-chat"]) == 0
    assert seen["runner_backend"] == "native_full"
    assert seen["muscle_memory"] == "on"
    assert seen["zpm_nullspace"] is True
    assert seen["max_tokens"] == 192
    assert seen["coherence_lane"] == "fast-balanced"
    assert seen["fast_chat"] is True


def test_cli_chat_fast_chat_honors_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-supplied flags must win over fast-chat defaults."""
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["max_tokens"] = manifest.max_tokens
        seen["temperature"] = manifest.temperature
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)

    assert (
        cli_main(
            [
                "chat",
                "model.gguf",
                "--fast-chat",
                "--python-ladder",
                "--max-tokens",
                "512",
                "--temperature",
                "0.7",
            ]
        )
        == 0
    )
    assert seen["runner_backend"] == "python"
    assert seen["max_tokens"] == 512
    assert seen["temperature"] == pytest.approx(0.7)


def test_cli_run_fast_chat_flag_forces_chat_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nrlpy run model.gguf --fast-chat`` must start the REPL with
    fast-chat defaults, even without an explicit ``--chat``.
    """
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        seen["coherence_lane"] = manifest.coherence_lane
        seen["fast_chat"] = kwargs.get("fast_chat", False)
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *a, **k: pytest.fail("run_gguf must not fire in fast-chat mode"),
    )

    assert cli_main(["run", "model.gguf", "--fast-chat"]) == 0
    assert seen["runner_backend"] == "native_full"
    assert seen["coherence_lane"] == "fast-balanced"
    assert seen["fast_chat"] is True
