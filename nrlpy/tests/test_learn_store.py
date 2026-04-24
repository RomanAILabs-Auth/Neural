# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bounded learn store."""

from __future__ import annotations

from pathlib import Path

import pytest

from nrlpy.learn_store import LearnStore


def test_observe_and_stats(tmp_path: Path) -> None:
    s = LearnStore(tmp_path / "learn")
    toks, newu = s.observe_text("NRL nrlpy throughput NRL")
    assert toks >= 3
    assert newu >= 1
    st = s.stats()
    assert st.unique_words >= 2
    assert st.max_bytes == 4 * 1024**3


def test_cap_and_prune(tmp_path: Path) -> None:
    s = LearnStore(tmp_path / "learn2")
    with pytest.warns(UserWarning, match="1 MiB"):
        s.set_max_bytes(12_000)
    for i in range(400):
        s.observe_text(f"token{i:04d} " * 8)
    st = s.stats()
    assert st.unique_words >= 1
    vocab_path = tmp_path / "learn2" / "vocab.json"
    assert vocab_path.is_file()
    assert vocab_path.stat().st_size <= s.max_bytes + 500


def test_cli_learn_status(capsys: pytest.CaptureFixture[str]) -> None:
    from nrlpy.cli import main as cli_main

    assert cli_main(["learn", "status"]) == 0
    out = capsys.readouterr().out
    assert "Budget cap" in out
    assert "GiB" in out
