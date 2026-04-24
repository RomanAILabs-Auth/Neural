# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for throughput math helper functions."""

from __future__ import annotations

import pytest

from nrlpy.throughput_math import (
    ThroughputProjection,
    calibrate_updates_per_token,
    required_gops_for_words_per_second,
    words_per_second,
    words_per_second_band,
)


def test_calibrate_updates_per_token_matches_known_example() -> None:
    # ~22.2 GOPS and ~8.9 TPS implies ~2.49e9 updates/token.
    u = calibrate_updates_per_token(executed_gops=22.2, executed_tps=8.9)
    assert u == pytest.approx(2.494e9, rel=1e-3)


def test_words_per_second_simple_case() -> None:
    wps = words_per_second(gops=30.0, updates_per_token=2.5e9, words_per_token=0.75)
    assert wps == pytest.approx(9.0)


def test_words_per_second_band_projection() -> None:
    proj = words_per_second_band(
        min_gops=20.0,
        max_gops=40.0,
        updates_per_token=2.5e9,
        min_words_per_token=0.6,
        max_words_per_token=0.8,
    )
    assert isinstance(proj, ThroughputProjection)
    assert proj.min_words_per_sec == pytest.approx(4.8)
    assert proj.max_words_per_sec == pytest.approx(12.8)


def test_required_gops_for_target_wps() -> None:
    gops = required_gops_for_words_per_second(
        target_words_per_second=1000.0,
        updates_per_token=2.5e9,
        words_per_token=0.75,
    )
    assert gops == pytest.approx(3333.3333333, rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"executed_gops": 0.0, "executed_tps": 1.0},
        {"executed_gops": 1.0, "executed_tps": 0.0},
    ],
)
def test_calibration_input_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calibrate_updates_per_token(**kwargs)


def test_band_input_validation() -> None:
    with pytest.raises(ValueError):
        words_per_second_band(
            min_gops=10.0,
            max_gops=5.0,
            updates_per_token=2.0,
        )

