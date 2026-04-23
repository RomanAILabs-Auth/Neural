"""CLI dispatches ``nrlpy chat``."""

from __future__ import annotations

import pytest

from nrlpy.cli import main as cli_main


def test_cli_chat_one_shot(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["chat", "--one", "help"]) == 0
    out = capsys.readouterr().out
    assert "Commands" in out


def test_cli_talk_alias(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["talk", "--one", "version"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip()) > 0
