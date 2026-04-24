# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for nrlpy.nrl_ai_chat (prompt #5 of the NRL-AI rebuild).

Covers the Neural-AI REPL surface that routes every turn through the
NRL-AI fast lane (no libllama decode):

  * banner is rendered with brand + fast_lane + threshold
  * hit turn streams the composed reply and increments ``stats.hits``
  * resolver miss emits the honest "not confident" surface and
    increments ``stats.misses`` (never synthesizes)
  * slash commands: /help, /stats, /reset, /status, /quit, /exit,
    unknown commands
  * blank line is ignored (REPL loops without emitting a reply)
  * EOF (stdin closed) cleanly exits and emits the session summary
  * missing index raises ``FileNotFoundError`` with a clean hint
  * CLI subprocess path: `nrlpy nrl-ai chat` boots against a real
    index, replies to a prompt, exits on /quit with exit code 0
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import (
    nrl_ai,
    nrl_ai_chat as chat_mod,
    nrl_ai_ingest as ingest,
)


SAMPLE_CORPUS = (
    "Hello there. How can I help you today?\n"
    "The weather is warm and sunny. I enjoy it very much.\n"
    "NRL-AI runs entirely on CPU. It uses SimHash anchors and ZPM routing.\n"
    "You can ask about speed, safety, or architecture.\n"
    "Thanks for chatting with Neural AI."
)


@pytest.fixture()
def sample_index(tmp_path: Path) -> nrl_ai.NrlAiPaths:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    result = ingest.ingest(corpus, out_dir=tmp_path / "idx")
    return result.paths


# ---------------------------------------------------------------------------
# Unit: banner + streaming
# ---------------------------------------------------------------------------


def test_repl_banner_contains_brand_and_fast_lane(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO("/quit\n")
    chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    banner = out.getvalue()
    assert "Neural AI" in banner
    assert "RomanAILabs" in banner
    assert "NRL-AI native runner" in banner
    assert "fast_lane" in banner
    assert "[NRL-AI]" in banner
    # Target WPS is visible so users know what we're gating on
    assert "1000" in banner
    # No libllama decode badge or chat_format leaks into the pure lane
    assert "chat_format" not in banner
    assert "[decode]" not in banner


def test_repl_exits_on_quit_command(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO("/quit\n")
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert stats.turns == 0
    assert stats.hits == 0
    assert stats.misses == 0
    assert "[session]" in out.getvalue()


def test_repl_exits_on_exit_command(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO("/exit\n")
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert stats.turns == 0


def test_repl_exits_on_eof(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO("")  # immediate EOF
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert stats.turns == 0
    assert "[session]" in out.getvalue()


def test_repl_blank_line_is_ignored(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO("\n   \n/quit\n")
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert stats.turns == 0


# ---------------------------------------------------------------------------
# Unit: hit / miss paths
# ---------------------------------------------------------------------------


def test_repl_hit_streams_reply_and_increments_hits(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO("NRL-AI runs on CPU\n/quit\n")
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    text = out.getvalue()
    assert stats.turns == 1
    assert stats.hits == 1
    assert stats.misses == 0
    assert stats.total_words >= 1
    assert "[NRL-AI]" in text
    assert "[NRL-AI hit" in text
    assert "wps" in text
    assert "stop=" in text
    # The resolver should have pulled a CPU-adjacent fragment
    assert "CPU" in text or "SimHash" in text or "ZPM" in text


def test_repl_miss_emits_honest_surface(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    # threshold=0 guarantees a miss unless the query perfectly matches a
    # fragment; an unrelated query will miss at any threshold <=~64
    inp = io.StringIO("zzzzzzzz unrelated banana mango\n/quit\n")
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index,
        threshold_bits=0,
        stdin=inp,
        stdout=out,
        use_color=False,
    )
    text = out.getvalue()
    assert stats.turns == 1
    assert stats.hits == 0
    assert stats.misses == 1
    assert "[resolve miss]" in text
    assert "not confident" in text.lower() or "indexed" in text.lower()


def test_repl_stats_reflects_multi_turn_accumulation(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO(
        "NRL-AI runs on CPU\n"
        "How can I help you today\n"
        "/quit\n"
    )
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert stats.turns == 2
    assert stats.hits + stats.misses == 2
    assert stats.total_words > 0
    assert stats.total_wall_s > 0
    assert stats.wps > 0


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_repl_help_prints_command_list(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO("/help\n/quit\n")
    chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    text = out.getvalue()
    assert "/help" in text
    assert "/quit" in text
    assert "/stats" in text
    assert "/reset" in text
    assert "/status" in text


def test_repl_stats_command_emits_cumulative_summary(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO(
        "NRL-AI runs on CPU\n"
        "/stats\n"
        "/quit\n"
    )
    chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    text = out.getvalue()
    assert "turns=1" in text
    assert "cumulative_wps=" in text


def test_repl_reset_zeros_counters(sample_index: nrl_ai.NrlAiPaths) -> None:
    out = io.StringIO()
    inp = io.StringIO(
        "NRL-AI runs on CPU\n"
        "/reset\n"
        "/stats\n"
        "/quit\n"
    )
    stats = chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    text = out.getvalue()
    assert "session stats reset" in text
    assert "turns=0" in text
    assert stats.turns == 0


def test_repl_status_command_reports_index(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO("/status\n/quit\n")
    chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    text = out.getvalue()
    assert "index=" in text
    assert "fragments=" in text
    assert "threshold=" in text


def test_repl_unknown_slash_command_is_rejected(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    inp = io.StringIO("/garbage\n/quit\n")
    chat_mod.run_nrl_ai_chat_repl(
        sample_index, stdin=inp, stdout=out, use_color=False
    )
    assert "unknown command" in out.getvalue()


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


def test_repl_missing_index_raises_filenotfound(tmp_path: Path) -> None:
    paths = nrl_ai.NrlAiPaths(tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError) as excinfo:
        chat_mod.run_nrl_ai_chat_repl(
            paths, stdin=io.StringIO(""), stdout=io.StringIO()
        )
    assert "nrl-ai ingest" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI subprocess: full boot / reply / quit loop
# ---------------------------------------------------------------------------


def test_cli_chat_subprocess_boots_and_replies(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    idx = tmp_path / "idx"

    ingest_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "ingest",
            str(corpus),
            "--out",
            str(idx),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ingest_proc.returncode == 0, ingest_proc.stderr

    chat_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "chat",
            "--index",
            str(idx),
        ],
        input="NRL-AI runs on CPU\n/quit\n",
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **_clean_env(),
            "NRL_NO_COLOR": "1",
        },
    )
    assert chat_proc.returncode == 0, chat_proc.stderr
    out = chat_proc.stdout
    assert "Neural AI" in out
    assert "[NRL-AI]" in out
    assert "[session]" in out
    assert "turns=1" in out


def test_cli_chat_missing_index_returns_exit_code_2(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "chat",
            "--index",
            str(tmp_path / "nope"),
        ],
        input="",
        capture_output=True,
        text=True,
        timeout=30,
        env=_clean_env(),
    )
    assert proc.returncode == 2
    assert "nrl-ai" in proc.stderr
    assert "ingest" in proc.stderr


def _clean_env() -> dict[str, str]:
    """Minimal env so pytest/subprocess doesn't inherit a tty/color config."""
    import os

    keep = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "PYTHONPATH"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join([src_path, existing])
    return env
