# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.gguf_chat`.

Covers:
* History digest stability across copies + mutation invalidation.
* Prompt rendering per chat format threads full history correctly.
* Per-turn manifest has the rendered prompt and ``chat_format='none'``.
* Slash-command handlers (``/clear``, ``/system``, ``/save``, ``/load``,
  ``/seed``, ``/tps``, ``/history``, ``/quit``, unknown).
* REPL runs end-to-end with a stub inference backend.
* Muscle-memory hit on a repeated turn (same history + same user text).
* Session TPS aggregation preserves the four-metric contract.
* ``/load`` refuses a snapshot from a different ``model_sha256``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nrlpy import gguf, gguf_chat


def _make_dummy_gguf(path: Path) -> None:
    path.write_bytes(b"GGUF" + b"\x00" * 64)


def _base_manifest(model: Path, **overrides: object) -> gguf.GgufManifest:
    kwargs = dict(
        model=str(model),
        prompt="",
        max_tokens=4,
        seed=1,
        chat_format="none",
    )
    kwargs.update(overrides)
    return gguf.manifest_from_args(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# History digest / message manipulation
# --------------------------------------------------------------------------- #


def test_history_digest_stable_and_updates() -> None:
    session = gguf_chat.ChatSession(
        base_manifest=gguf.manifest_from_args(model="x.gguf")
    )
    d0 = session.history_digest()
    session.messages.append(gguf_chat.ChatMessage(role="user", text="hello"))
    d1 = session.history_digest()
    session.messages.append(gguf_chat.ChatMessage(role="assistant", text="hi"))
    d2 = session.history_digest()
    assert d0 != d1 != d2
    assert session.history_digest() == d2  # idempotent


def test_history_digest_system_prompt_matters() -> None:
    a = gguf_chat.ChatSession(
        base_manifest=gguf.manifest_from_args(model="x.gguf"), system="be concise"
    )
    b = gguf_chat.ChatSession(base_manifest=gguf.manifest_from_args(model="x.gguf"))
    assert a.history_digest() != b.history_digest()


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def test_render_plain_contains_all_turns() -> None:
    session = gguf_chat.ChatSession(
        base_manifest=gguf.manifest_from_args(model="x.gguf"), system="S"
    )
    session.messages.append(gguf_chat.ChatMessage(role="user", text="u1"))
    session.messages.append(gguf_chat.ChatMessage(role="assistant", text="a1"))
    rendered = gguf_chat.build_history_prompt(session, "u2", fmt="none")
    assert "System: S" in rendered
    assert "User: u1" in rendered
    assert "Assistant: a1" in rendered
    assert rendered.rstrip().endswith("Assistant:")
    assert "u2" in rendered


def test_render_chatml_turn_envelope() -> None:
    session = gguf_chat.ChatSession(
        base_manifest=gguf.manifest_from_args(model="x.gguf", chat_format="chatml"),
        system="S",
    )
    session.messages.append(gguf_chat.ChatMessage(role="user", text="u1"))
    session.messages.append(gguf_chat.ChatMessage(role="assistant", text="a1"))
    rendered = gguf_chat.build_history_prompt(session, "u2")
    assert "<|im_start|>system\nS<|im_end|>" in rendered
    assert "<|im_start|>user\nu1<|im_end|>" in rendered
    assert "<|im_start|>assistant\na1<|im_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_render_phi3_and_llama2_thread_history() -> None:
    base_phi = gguf.manifest_from_args(model="x.gguf", chat_format="phi3")
    sess_phi = gguf_chat.ChatSession(base_manifest=base_phi, system="S")
    sess_phi.messages.append(gguf_chat.ChatMessage(role="user", text="u1"))
    sess_phi.messages.append(gguf_chat.ChatMessage(role="assistant", text="a1"))
    p = gguf_chat.build_history_prompt(sess_phi, "u2")
    assert "<|system|>\nS<|end|>" in p
    assert "<|user|>\nu1<|end|>" in p
    assert "<|assistant|>\na1<|end|>" in p
    assert p.endswith("<|assistant|>\n")

    base_l2 = gguf.manifest_from_args(model="x.gguf", chat_format="llama2")
    sess_l2 = gguf_chat.ChatSession(base_manifest=base_l2, system="S")
    sess_l2.messages.append(gguf_chat.ChatMessage(role="user", text="u1"))
    sess_l2.messages.append(gguf_chat.ChatMessage(role="assistant", text="a1"))
    rendered = gguf_chat.build_history_prompt(sess_l2, "u2")
    assert "<<SYS>>\nS\n<</SYS>>" in rendered
    assert "[INST]" in rendered
    assert rendered.rstrip().endswith("[/INST]")
    # Second user turn must not re-emit the SYS block.
    assert rendered.count("<<SYS>>") == 1


# --------------------------------------------------------------------------- #
# chat_turn end-to-end with stub backend
# --------------------------------------------------------------------------- #


def test_chat_turn_populates_history_and_tps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)

    session = gguf_chat.ChatSession(base_manifest=_base_manifest(model))
    result = gguf_chat.chat_turn(session, "hello")
    assert result.tokens > 0
    assert len(session.messages) == 2
    assert session.messages[0].role == "user" and session.messages[0].text == "hello"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].text == result.text
    assert session.tps.turns == 1
    assert session.tps.executed_tokens == result.tps.executed_tokens
    # Honesty hinge: virtual == executed at the session level too.
    agg = session.tps.as_tps_report()
    assert agg.gate_skip_ratio == 0.0
    assert agg.virtual_tps == pytest.approx(agg.executed_tps)


def test_chat_turn_muscle_memory_hit_on_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same (history, user_text, sampler) on turn N+1 must hit muscle memory."""
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)

    # Session A: fresh decode, writes cache entry keyed on (model, rendered prompt).
    session_a = gguf_chat.ChatSession(base_manifest=_base_manifest(model))
    r1 = gguf_chat.chat_turn(session_a, "hello")
    assert r1.cache_hit is False

    # Session B: identical base manifest + empty history + same user text.
    # Rendered prompt is identical; muscle-memory key must match.
    session_b = gguf_chat.ChatSession(base_manifest=_base_manifest(model))
    r2 = gguf_chat.chat_turn(session_b, "hello")
    assert r2.cache_hit is True
    assert r2.text == r1.text


def test_session_tps_aggregation_preserves_four_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)

    session = gguf_chat.ChatSession(base_manifest=_base_manifest(model))
    # Three distinct turns; the stub's output depends on seed, not prompt.
    for i in range(3):
        gguf_chat.chat_turn(session, f"turn-{i}")

    agg = session.tps.as_tps_report()
    assert session.tps.turns == 3
    assert agg.executed_tokens > 0
    assert agg.executed_tps > 0
    assert agg.gate_skip_ratio == 0.0
    assert agg.virtual_tps == pytest.approx(agg.executed_tps)


def test_chat_session_prefill_gate_activates_on_second_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    monkeypatch.setattr(
        gguf,
        "nrl_attest",
        lambda profile="omega", **_k: gguf.NrlAttestation(profile=profile),
    )
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    manifest = _base_manifest(model, prefill_cache="session")
    session = gguf_chat.build_session(manifest)

    turn1 = gguf_chat.chat_turn(session, "alpha beta")
    turn2 = gguf_chat.chat_turn(session, "alpha beta gamma")

    assert session.prefill_gate is not None
    assert turn1.gate_source is None
    assert turn1.tps.gate_skip_ratio == 0.0
    assert turn2.gate_source == "prefill_cache"
    assert turn2.tps.gate_skip_ratio > 0.0


# --------------------------------------------------------------------------- #
# Slash commands (direct dispatch, no REPL needed)
# --------------------------------------------------------------------------- #


def _fresh_session(tmp_path: Path) -> gguf_chat.ChatSession:
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    return gguf_chat.ChatSession(
        base_manifest=_base_manifest(model),
        model_sha256="a" * 64,
    )


def test_slash_clear_resets_messages_keeps_system(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    session.system = "stay-put"
    session.messages.append(gguf_chat.ChatMessage("user", "x"))
    session.messages.append(gguf_chat.ChatMessage("assistant", "y"))
    buf = io.StringIO()
    verdict = gguf_chat._handle_slash("/clear", session, buf)
    assert verdict == "continue"
    assert session.messages == []
    assert session.system == "stay-put"


def test_slash_system_resets_history(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    session.messages.append(gguf_chat.ChatMessage("user", "x"))
    buf = io.StringIO()
    gguf_chat._handle_slash("/system be brief", session, buf)
    assert session.system == "be brief"
    assert session.messages == []


def test_slash_seed_validates_int(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    buf = io.StringIO()
    gguf_chat._handle_slash("/seed 42", session, buf)
    assert session.base_manifest.seed == 42
    gguf_chat._handle_slash("/seed bogus", session, buf)
    assert "requires an integer" in buf.getvalue()
    assert session.base_manifest.seed == 42  # unchanged on bad input


def test_slash_save_and_load_roundtrip(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    session.system = "sys"
    session.messages.append(gguf_chat.ChatMessage("user", "hi"))
    session.messages.append(gguf_chat.ChatMessage("assistant", "hello"))
    path = tmp_path / "session.json"
    buf = io.StringIO()

    gguf_chat._handle_slash(f"/save {path}", session, buf)
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_id"] == "nrl.gguf_chat_session.v1"
    assert saved["model_sha256"] == "a" * 64

    # New session with same model_sha256 loads cleanly.
    other = _fresh_session(tmp_path)
    gguf_chat._handle_slash(f"/load {path}", other, buf)
    assert other.system == "sys"
    assert [(m.role, m.text) for m in other.messages] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_slash_load_refuses_model_sha_mismatch(tmp_path: Path) -> None:
    session_src = _fresh_session(tmp_path)
    session_src.messages.append(gguf_chat.ChatMessage("user", "hi"))
    path = tmp_path / "s.json"
    buf = io.StringIO()
    gguf_chat._handle_slash(f"/save {path}", session_src, buf)

    other = _fresh_session(tmp_path)
    other.model_sha256 = "b" * 64  # different model
    out = io.StringIO()
    gguf_chat._handle_slash(f"/load {path}", other, out)
    assert "refused" in out.getvalue()
    assert other.messages == []  # unchanged


def test_slash_quit_and_unknown(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    buf = io.StringIO()
    assert gguf_chat._handle_slash("/quit", session, buf) == "quit"
    assert gguf_chat._handle_slash("/exit", session, buf) == "quit"
    assert gguf_chat._handle_slash("/nope", session, buf) == "continue"
    assert "unknown slash command" in buf.getvalue()


def test_non_slash_returns_none(tmp_path: Path) -> None:
    session = _fresh_session(tmp_path)
    buf = io.StringIO()
    assert gguf_chat._handle_slash("hello there", session, buf) is None


# --------------------------------------------------------------------------- #
# Full REPL end-to-end
# --------------------------------------------------------------------------- #


def test_repl_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    _make_dummy_gguf(model)

    manifest = _base_manifest(model)
    stdin = io.StringIO("hello\n/tps\n/quit\n")
    stdout = io.StringIO()

    session = gguf_chat.run_gguf_chat_repl(
        manifest,
        system="be brief",
        stdin=stdin,
        stdout=stdout,
        attest_profile="omega",
        observation_profile="",  # skip observation in tests (no real engine)
    )

    # One user turn was processed.
    assert session.tps.turns == 1
    assert session.turn_count() == 1
    assert session.system == "be brief"
    text = stdout.getvalue()
    # Polished boot banner is emitted and carries the Neural-AI brand line.
    assert "Neural AI" in text and "RomanAILabs" in text
    # Turn header uses the new "[you, turn 0]" / "> " prompt.
    assert "[you, turn 0]" in text
    assert "NRL gguf_chat" in text  # final session banner printed
    assert "session-aggregate" in text


def test_repl_preloads_llm_once_for_all_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    _make_dummy_gguf(model)
    manifest = _base_manifest(model)

    calls = {"n": 0}
    real_loader = gguf._load_llm

    def wrapped_loader(m: gguf.GgufManifest) -> object:
        calls["n"] += 1
        return real_loader(m)

    monkeypatch.setattr(gguf, "_load_llm", wrapped_loader)
    stdin = io.StringIO("hello\nagain\n/quit\n")
    stdout = io.StringIO()
    gguf_chat.run_gguf_chat_repl(manifest, stdin=stdin, stdout=stdout, observation_profile="")
    assert calls["n"] == 1


def test_repl_disables_attestation_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_ROOT", str(tmp_path))
    monkeypatch.setenv("NRL_INFERENCE", "stub")
    model = tmp_path / "m.gguf"
    _make_dummy_gguf(model)
    manifest = _base_manifest(model)

    monkeypatch.setattr(
        gguf, "nrl_attest", lambda profile="omega", **_k: pytest.fail("attestation should be disabled by default")
    )
    stdin = io.StringIO("/quit\n")
    stdout = io.StringIO()
    gguf_chat.run_gguf_chat_repl(manifest, stdin=stdin, stdout=stdout)


def test_repl_routes_from_main_chat_for_gguf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``nrlpy.chat.main_chat`` must delegate ``.gguf`` args to ``main_gguf_chat``."""
    from nrlpy import chat

    called: dict[str, object] = {}

    def fake(args: list[str]) -> int:
        called["args"] = args
        return 0

    monkeypatch.setattr(gguf_chat, "main_gguf_chat", fake)
    rc = chat.main_chat(["model.gguf", "--seed", "5"])
    assert rc == 0
    assert called["args"] == ["model.gguf", "--seed", "5"]
