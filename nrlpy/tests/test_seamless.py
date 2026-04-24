# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for seamless assimilated builtins."""

from __future__ import annotations

import pytest

from nrlpy.compat import llm_globals
from nrlpy.seamless import fabric_pulse, is_prime, next_prime


def test_is_prime_small() -> None:
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(4) is False
    assert is_prime(97) is True
    assert is_prime(121) is False


def test_is_prime_large_known() -> None:
    assert is_prime(1_000_000_000_039) is True


def test_next_prime_after_trillion() -> None:
    assert next_prime(10**12) == 1_000_000_000_039


def test_next_prime_small() -> None:
    assert next_prime(1) == 2
    assert next_prime(2) == 3
    assert next_prime(9) == 11


def test_llm_globals_includes_seamless() -> None:
    g = llm_globals()
    assert callable(g["next_prime"])
    assert callable(g["is_prime"])
    assert callable(g["fabric_pulse"])


def test_fabric_pulse_runs() -> None:
    out = fabric_pulse(neurons=2048, iterations=32, threshold=8)
    assert out["kernel"] == "braincore_int4"
    assert float(out["giga_neurons_per_sec"]) >= 0.0


def test_trial_against_bruteforce_small() -> None:
    def brute(n: int) -> bool:
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True

    for n in range(2, 500):
        assert is_prime(n) == brute(n), n
