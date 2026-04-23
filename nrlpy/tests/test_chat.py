"""Chat intent mapping (no full REPL in CI)."""

from __future__ import annotations

from nrlpy.chat import interpret


def test_help() -> None:
    r = interpret("help")
    assert "temperature" in r.lower()
    assert "throughput" in r.lower()


def test_version_phrase() -> None:
    r = interpret("what version are you")
    assert "engine:" in r.lower() or "nrl" in r.lower()
    assert len(r) > 4


def test_learn_disclaimer() -> None:
    r = interpret("how do you learn and grow")
    assert "shadow" in r.lower() or "plasticity" in r.lower() or "bounded" in r.lower()
    assert "budget cap" in r.lower()


def test_growth_stats() -> None:
    r = interpret("how much have you grown")
    assert "budget cap" in r.lower()
    assert "GiB" in r


def test_unknown_steer() -> None:
    r = interpret("tell me a joke about quantum gravity")
    assert "help" in r.lower()
