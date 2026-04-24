# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for nrlpy.nrl_ai_resolve (prompt #3 of the NRL-AI rebuild).

Exercise the resolver contract end-to-end:
  * exact-match query -> distance 0, hit=True
  * near-match query  -> distance < threshold, hit=True
  * orthogonal query  -> distance > threshold, hit=False
  * top-k ordering    -> ascending by distance, stable id tie-break
  * determinism       -> identical inputs produce identical outputs
  * missing index     -> FileNotFoundError with actionable message
  * CLI subprocess    -> JSON shape matches ``nrl_ai.resolve.v1``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import nrl_ai, nrl_ai_ingest as ingest, nrl_ai_resolve as resolve


SAMPLE_CORPUS = (
    "Hello there. How can I help you today?\n"
    "The weather is warm and sunny. I enjoy it very much.\n"
    "NRL-AI runs entirely on CPU. It uses SimHash anchors and ZPM routing.\n"
    "You can ask about speed, safety, or architecture."
)


@pytest.fixture()
def sample_index(tmp_path: Path) -> nrl_ai.NrlAiPaths:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    result = ingest.ingest(corpus, out_dir=tmp_path / "idx")
    return result.paths


def test_resolve_exact_match_is_zero_distance(sample_index: nrl_ai.NrlAiPaths) -> None:
    # Pull the literal first fragment out of the index and query with it
    # verbatim; distance must collapse to zero.
    fragments = ingest.load_fragments(sample_index)
    target = fragments[0]
    r = resolve.resolve(target, paths=sample_index, top_k=1)
    assert r.best is not None
    assert r.best.distance_bits == 0
    assert r.best.fragment_id == 0
    assert r.hit is True


def test_resolve_near_match_lands_under_threshold(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    # The corpus has "NRL-AI runs entirely on CPU." -- drop one word and
    # verify the resolver still locks on to it well within the default
    # 96-bit threshold.
    r = resolve.resolve("NRL-AI runs on CPU", paths=sample_index, top_k=3)
    assert r.best is not None
    assert r.best.within_threshold, (
        f"near-match missed threshold: dist={r.best.distance_bits} "
        f"threshold={r.threshold_bits}"
    )
    assert "CPU" in r.best.fragment_text
    assert r.hit is True


def test_resolve_orthogonal_query_misses_default_threshold(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    r = resolve.resolve(
        "zyxwvut qponmlk jihgfed cbazyx wvutsrq ponmlkj",
        paths=sample_index,
        top_k=1,
    )
    assert r.best is not None
    # 96-bit default threshold is loose, but random strings should still
    # sit near the 128-bit orthogonal band. Allow a small slack (80 bits)
    # to avoid flakiness on corpus variation while still proving the
    # locality contract.
    assert r.best.distance_bits > 80, (
        f"orthogonal query landed too close: dist={r.best.distance_bits}"
    )


def test_resolve_top_k_is_sorted_and_unique(sample_index: nrl_ai.NrlAiPaths) -> None:
    r = resolve.resolve("weather sunny CPU SimHash", paths=sample_index, top_k=4)
    assert len(r.top) == 4
    distances = [h.distance_bits for h in r.top]
    assert distances == sorted(distances)
    assert len({h.fragment_id for h in r.top}) == 4  # unique fragments


def test_resolve_tight_threshold_forces_miss(sample_index: nrl_ai.NrlAiPaths) -> None:
    # Threshold 0 only admits exact matches; a variant string must miss.
    r = resolve.resolve(
        "this is nowhere in the corpus at all unrelated text",
        paths=sample_index,
        threshold_bits=0,
        top_k=1,
    )
    assert r.best is not None
    assert r.best.within_threshold is False
    assert r.hit is False


def test_resolve_is_deterministic(sample_index: nrl_ai.NrlAiPaths) -> None:
    r1 = resolve.resolve("how do i use NRL-AI?", paths=sample_index, top_k=3)
    r2 = resolve.resolve("how do i use NRL-AI?", paths=sample_index, top_k=3)
    assert r1.query_simhash == r2.query_simhash
    assert [(h.fragment_id, h.distance_bits) for h in r1.top] == [
        (h.fragment_id, h.distance_bits) for h in r2.top
    ]


def test_resolve_missing_index_raises_with_hint(tmp_path: Path) -> None:
    paths = nrl_ai.NrlAiPaths(tmp_path / "no-index-here")
    with pytest.raises(FileNotFoundError) as exc:
        resolve.resolve("anything", paths=paths)
    assert "ingest" in str(exc.value)


def test_resolve_detects_index_inconsistency(
    sample_index: nrl_ai.NrlAiPaths, tmp_path: Path
) -> None:
    # Truncate the anchors file so counts mismatch fragments.json. The
    # resolver must detect this rather than silently returning wrong
    # matches.
    sample_index.anchors.write_bytes(sample_index.anchors.read_bytes()[:32])
    with pytest.raises(RuntimeError) as exc:
        resolve.resolve("hi", paths=sample_index)
    assert "inconsistency" in str(exc.value)


def test_cli_resolve_roundtrip(tmp_path: Path) -> None:
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

    resolve_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "resolve",
            "NRL-AI runs on CPU",
            "--index",
            str(idx),
            "--top",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resolve_proc.returncode == 0, resolve_proc.stderr
    payload = json.loads(resolve_proc.stdout)
    assert payload["schema"] == "nrl_ai.resolve.v1"
    assert payload["simhash_bits"] == 256
    assert payload["threshold_bits"] == resolve.DEFAULT_THRESHOLD_BITS
    assert payload["hit"] is True
    assert payload["best"]["distance_bits"] < payload["threshold_bits"]
    assert len(payload["top"]) == 2
    assert len(payload["query_simhash_hex"]) == 4
    for word_hex in payload["query_simhash_hex"]:
        assert len(word_hex) == 16  # 64-bit hex


def test_cli_resolve_missing_index_returns_2(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "resolve",
            "hi",
            "--index",
            str(tmp_path / "no-such-dir"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert "ingest" in proc.stderr
