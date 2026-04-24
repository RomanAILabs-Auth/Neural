# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""CLI dispatches ``nrlpy chat``."""

from __future__ import annotations

from typing import Any

import pytest

from nrlpy import gguf
from nrlpy.cli import main as cli_main


def test_cli_chat_one_shot(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["chat", "--one", "help"]) == 0
    out = capsys.readouterr().out
    assert "Commands" in out


def test_cli_talk_alias(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["talk", "--one", "version"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


def test_cli_run_gguf_without_prompt_starts_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["model"] = manifest.model
        seen["seed"] = manifest.seed
        seen["max_tokens"] = manifest.max_tokens
        seen["temperature"] = manifest.temperature
        seen["n_batch"] = manifest.n_batch
        seen["prefill_cache"] = manifest.prefill_cache
        seen["system"] = _kwargs.get("system", "")
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["run", "model.gguf", "--seed", "7"]) == 0
    assert seen["model"] == "model.gguf"
    assert seen["seed"] == 7
    assert seen["max_tokens"] == 256
    assert seen["temperature"] == pytest.approx(0.2)
    assert seen["n_batch"] == 1024
    assert seen["prefill_cache"] == "session"
    assert seen["system"] == ""


def test_cli_run_gguf_with_prompt_keeps_single_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "nrlpy.gguf_chat.run_gguf_chat_repl",
        lambda *args, **kwargs: pytest.fail("chat REPL should not start when --prompt is given"),
    )

    def fake_run(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["prompt"] = manifest.prompt
        seen["stream_to"] = kwargs.get("stream_to")
        return object()

    monkeypatch.setattr("nrlpy.gguf.run_gguf", fake_run)
    monkeypatch.setattr("nrlpy.gguf.format_banner", lambda _result: "ok-banner")

    assert cli_main(["run", "model.gguf", "--prompt", "hello", "--no-stream"]) == 0
    assert seen["prompt"] == "hello"
    assert seen["stream_to"] is None


def test_cli_bare_gguf_without_prompt_starts_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["model"] = manifest.model
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["model.gguf"]) == 0
    assert seen["model"] == "model.gguf"


def test_cli_run_split_gguf_path_without_quotes_starts_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["model"] = manifest.model
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert (
        cli_main(
            [
                "run",
                r"C:\Users\Asus\Desktop\Documents\RomaPy",
                r"Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf",
            ]
        )
        == 0
    )
    assert (
        seen["model"]
        == r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf"
    )


def test_cli_infers_phi3_chat_format_for_phi3_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["chat_format"] = manifest.chat_format
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["run", "phi-3-mini-4k-instruct.Q4_K_M.gguf"]) == 0
    assert seen["chat_format"] == "phi3"


def test_cli_chat_defaults_do_not_override_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["max_tokens"] = manifest.max_tokens
        seen["temperature"] = manifest.temperature
        seen["n_batch"] = manifest.n_batch
        seen["prefill_cache"] = manifest.prefill_cache
        seen["n_ctx"] = manifest.n_ctx
        seen["system"] = _kwargs.get("system", "")
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert (
        cli_main(
            [
                "run",
                "model.gguf",
                "--max-tokens",
                "32",
                "--temperature",
                "0.6",
                "--n-batch",
                "256",
            ]
        )
        == 0
    )
    assert seen["max_tokens"] == 32
    assert seen["temperature"] == pytest.approx(0.6)
    assert seen["n_batch"] == 256


def test_cli_specs_mode_applies_poc_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["max_tokens"] = manifest.max_tokens
        seen["temperature"] = manifest.temperature
        seen["n_batch"] = manifest.n_batch
        seen["prefill_cache"] = manifest.prefill_cache
        seen["n_ctx"] = manifest.n_ctx
        seen["system"] = _kwargs.get("system", "")
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["run", "-specs", "model.gguf"]) == 0
    assert seen["max_tokens"] == 256
    assert seen["temperature"] == pytest.approx(0.15)
    assert seen["n_batch"] >= 2048
    assert seen["prefill_cache"] == "session"
    assert seen["n_ctx"] >= 4096
    assert "NRL specs chat" in seen["system"]
    out = capsys.readouterr().out
    assert "NRL specs mode" in out


def test_cli_resolves_bare_gguf_filename_from_nrl_models_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    real_model = models_dir / "phi-3-mini-4k-instruct.Q4_K_M.gguf"
    real_model.write_bytes(b"fake-gguf-bytes")
    monkeypatch.setenv("NRL_MODELS_DIR", str(models_dir))

    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["model"] = manifest.model
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["run", "phi-3-mini-4k-instruct.Q4_K_M.gguf"]) == 0
    assert seen["model"] == str(real_model)


def test_cli_run_chat_flag_forces_repl_over_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nrlpy run model.gguf --chat --prompt hi`` must start the REPL
    (and ignore the prompt) — the --chat flag is an explicit opt-in to
    interactive mode regardless of what else is on the command line.
    """
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["model"] = manifest.model
        seen["prompt"] = manifest.prompt
        seen["prompt_file"] = manifest.prompt_file
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf must not run when --chat is set"),
    )

    assert (
        cli_main(
            [
                "run",
                "model.gguf",
                "--chat",
                "--prompt",
                "ignored",
                "--native-full",
            ]
        )
        == 0
    )
    assert seen["model"] == "model.gguf"
    assert seen["prompt"] == ""
    assert seen["prompt_file"] == ""


def test_cli_run_chat_flag_propagates_native_full_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf must not run when --chat is set"),
    )

    assert (
        cli_main(["run", "model.gguf", "--chat", "--native-full"])
        == 0
    )
    assert seen["runner_backend"] == "native_full"


def test_cli_chat_defaults_backend_when_not_specified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nrlpy chat model.gguf`` should pick a backend automatically —
    either ``native_full`` (if bindings are built) or ``python``. It
    must never leave ``runner_backend`` unset.
    """
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)

    assert cli_main(["chat", "model.gguf"]) == 0
    assert seen["runner_backend"] in {"native_full", "python"}


def test_cli_chat_honors_explicit_backend_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **_kwargs: Any) -> object:
        seen["runner_backend"] = manifest.runner_backend
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)

    assert cli_main(["chat", "model.gguf", "--python-ladder"]) == 0
    assert seen["runner_backend"] == "python"

    seen.clear()
    assert cli_main(["chat", "model.gguf", "--native-full"]) == 0
    assert seen["runner_backend"] == "native_full"


def test_cli_chat_rejects_unknown_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["chat", "model.gguf", "--bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown chat flag" in err
    assert "--bogus" in err


def test_cli_specs_mode_allows_explicit_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_repl(manifest: gguf.GgufManifest, **kwargs: Any) -> object:
        seen["system"] = kwargs.get("system", "")
        return object()

    monkeypatch.setattr("nrlpy.gguf_chat.run_gguf_chat_repl", fake_repl)
    monkeypatch.setattr(
        "nrlpy.gguf.run_gguf",
        lambda *args, **kwargs: pytest.fail("run_gguf should not be called without --prompt"),
    )

    assert cli_main(["run", "-specs", "model.gguf", "--system", "Be precise."]) == 0
    assert seen["system"] == "Be precise."
