# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Throughput math helpers for GGUF runner planning.

These functions are model-universal. They operate on measured/calibrated
constants rather than hardcoding assumptions to one model family.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputProjection:
    """Projected words/sec bounds under a GOPS band."""

    min_words_per_sec: float
    max_words_per_sec: float


def calibrate_updates_per_token(*, executed_gops: float, executed_tps: float) -> float:
    """Infer updates/token from an observed dense executed run."""
    if executed_gops <= 0.0:
        raise ValueError("executed_gops must be > 0")
    if executed_tps <= 0.0:
        raise ValueError("executed_tps must be > 0")
    return (executed_gops * 1e9) / executed_tps


def words_per_second(*, gops: float, updates_per_token: float, words_per_token: float) -> float:
    """Compute projected words/sec from GOPS and calibration constants."""
    if gops < 0.0:
        raise ValueError("gops must be >= 0")
    if updates_per_token <= 0.0:
        raise ValueError("updates_per_token must be > 0")
    if words_per_token <= 0.0:
        raise ValueError("words_per_token must be > 0")
    return ((gops * 1e9) / updates_per_token) * words_per_token


def words_per_second_band(
    *,
    min_gops: float,
    max_gops: float,
    updates_per_token: float,
    min_words_per_token: float = 0.6,
    max_words_per_token: float = 0.8,
) -> ThroughputProjection:
    """Project min/max words/sec over GOPS and words/token ranges."""
    if min_gops < 0.0 or max_gops < 0.0:
        raise ValueError("gops values must be >= 0")
    if min_gops > max_gops:
        raise ValueError("min_gops must be <= max_gops")
    if min_words_per_token <= 0.0 or max_words_per_token <= 0.0:
        raise ValueError("words/token bounds must be > 0")
    if min_words_per_token > max_words_per_token:
        raise ValueError("min_words_per_token must be <= max_words_per_token")
    min_wps = words_per_second(
        gops=min_gops,
        updates_per_token=updates_per_token,
        words_per_token=min_words_per_token,
    )
    max_wps = words_per_second(
        gops=max_gops,
        updates_per_token=updates_per_token,
        words_per_token=max_words_per_token,
    )
    return ThroughputProjection(min_words_per_sec=min_wps, max_words_per_sec=max_wps)


def required_gops_for_words_per_second(
    *,
    target_words_per_second: float,
    updates_per_token: float,
    words_per_token: float,
) -> float:
    """Compute GOPS required to reach a target words/sec."""
    if target_words_per_second <= 0.0:
        raise ValueError("target_words_per_second must be > 0")
    if updates_per_token <= 0.0:
        raise ValueError("updates_per_token must be > 0")
    if words_per_token <= 0.0:
        raise ValueError("words_per_token must be > 0")
    return (target_words_per_second * updates_per_token) / (words_per_token * 1e9)

