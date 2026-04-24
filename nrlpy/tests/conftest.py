# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Isolate learn-store and chat paths per test process."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_learn_dir(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path_factory.mktemp("iso")
    learn = base / "learn"
    learn.mkdir(parents=True)
    monkeypatch.setenv("NRL_LEARN_DIR", str(learn))
    chat = base / "nrlpy_chat"
    chat.mkdir(parents=True)
    monkeypatch.setenv("NRL_CHAT_SESSION_DIR", str(chat))
