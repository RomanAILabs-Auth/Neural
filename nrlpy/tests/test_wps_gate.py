# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Repo discovery for ``nrlpy wps-gate``."""

from __future__ import annotations

from nrlpy import wps_gate

import pytest


def test_find_nrl_repo_root_via_package_parents() -> None:
    r = wps_gate.find_nrl_repo_root()
    assert r is not None
    assert (r / "benchmarks" / "gguf_golden.py").is_file()


def test_find_nrl_repo_root_env_matches_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    real = wps_gate.find_nrl_repo_root()
    assert real is not None
    monkeypatch.setenv("NRL_REPO", str(real))
    assert wps_gate.find_nrl_repo_root() == real
