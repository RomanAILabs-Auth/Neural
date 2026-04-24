# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for nrlpy.nrl_ai_ingest (prompt #2 of the NRL-AI rebuild).

Covers:
  * SimHash256 determinism, locality, orthogonality, emptiness.
  * Fragment splitter (sentence boundaries, whitespace collapse, long-frag
    chunking, minimum-length filtering).
  * On-disk packed readers/writers (anchors, transitions) and JSON
    fragments table.
  * End-to-end ingest + round-trip loaders.
  * CLI surface: ``nrl-ai ingest <corpus>`` followed by ``nrl-ai status``
    reports ``index_ready: True`` with the right manifest.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import nrl_ai
from nrlpy import nrl_ai_ingest as ingest


# ---------------------------------------------------------------------------
# SimHash256
# ---------------------------------------------------------------------------


def test_simhash_is_deterministic() -> None:
    a = ingest.simhash256("the quick brown fox jumps over the lazy dog")
    b = ingest.simhash256("the quick brown fox jumps over the lazy dog")
    assert a == b
    assert len(a) == 4
    for word in a:
        assert 0 <= word < (1 << 64)


def test_simhash_empty_string_is_zero() -> None:
    assert ingest.simhash256("") == (0, 0, 0, 0)


def test_simhash_locality_for_near_matches() -> None:
    a = ingest.simhash256("the quick brown fox jumps over the lazy dog")
    b = ingest.simhash256("the quick brown fox jumps over the lazy dogs")
    d = ingest.hamming_distance_simhash(a, b)
    # One char added -> only a handful of trigrams change; distance stays
    # well under half of the 256-bit space. 64 is a comfortable ceiling
    # for a single-char edit at this corpus size.
    assert d < 64, f"locality violated: dist={d}"


def test_simhash_orthogonal_for_unrelated_strings() -> None:
    a = ingest.simhash256("the quick brown fox jumps over the lazy dog")
    b = ingest.simhash256("zyxwvut qponmlk jihgfed cbazyx wvutsrq ponmlkj")
    d = ingest.hamming_distance_simhash(a, b)
    # Unrelated inputs should cluster near the 128-bit midpoint of the
    # 256-bit space. Allow generous margin; the test only needs to show
    # "clearly not locality-hit".
    assert d > 64, f"orthogonality violated: dist={d}"


def test_hamming_distance_is_symmetric_and_zero_for_equal() -> None:
    a = ingest.simhash256("hello world")
    assert ingest.hamming_distance_simhash(a, a) == 0
    b = ingest.simhash256("hello world!")
    assert ingest.hamming_distance_simhash(a, b) == ingest.hamming_distance_simhash(b, a)


# ---------------------------------------------------------------------------
# Fragment extraction
# ---------------------------------------------------------------------------


def test_iter_fragments_splits_simple_sentences() -> None:
    text = "Hi there. How are you? Fine, thanks."
    frags = list(ingest.iter_fragments(text))
    assert len(frags) == 3
    assert frags[0] == "Hi there."
    assert frags[1] == "How are you?"
    assert frags[2] == "Fine, thanks."


def test_iter_fragments_collapses_whitespace() -> None:
    text = "First.\n\n  Second\twith   tabs."
    frags = list(ingest.iter_fragments(text))
    assert frags == ["First.", "Second with tabs."]


def test_iter_fragments_skips_below_minimum_length() -> None:
    text = "A. This one counts."
    frags = list(ingest.iter_fragments(text))
    # ``A.`` is 2 chars -> below the 3-char floor; only the second stays.
    assert frags == ["This one counts."]


def test_iter_fragments_splits_overly_long_fragment() -> None:
    # 80 clauses separated by ", " => one sentence well over the 500-char
    # cap so the long-fragment splitter has to kick in.
    flat = ", ".join(f"clause number {i}" for i in range(80)) + "."
    frags = list(ingest.iter_fragments(flat))
    assert len(frags) > 1
    for piece in frags:
        assert len(piece) <= 520  # allow slack for the last comma boundary


def test_iter_fragments_empty_corpus_yields_nothing() -> None:
    assert list(ingest.iter_fragments("")) == []
    assert list(ingest.iter_fragments("   \n\n  ")) == []


# ---------------------------------------------------------------------------
# On-disk packed writers + readers
# ---------------------------------------------------------------------------


def test_anchors_roundtrip(tmp_path: Path) -> None:
    anchors = [
        (0xDEADBEEFDEADBEEF, 1, 2, 3),
        (0, 0, 0, 0),
        (0xFFFFFFFFFFFFFFFF, 0xAAAA, 0x5555, 0x1234),
    ]
    path = tmp_path / "a.bin"
    n = ingest._write_anchors(path, anchors)
    assert n == 3
    assert path.stat().st_size == 3 * struct.calcsize("<4Q")
    loaded = ingest._read_anchors(path)
    assert loaded == anchors


def test_transitions_roundtrip(tmp_path: Path) -> None:
    transitions = {(0, 1): 5, (1, 2): 3, (2, 3): 1}
    path = tmp_path / "t.bin"
    n = ingest._write_transitions(path, transitions)
    assert n == 3
    loaded = ingest._read_transitions(path)
    assert loaded == transitions


def test_anchors_reader_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00" * 7)  # not a multiple of 32
    with pytest.raises(ValueError):
        ingest._read_anchors(path)


def test_fragments_json_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "frags.json"
    path.write_text(json.dumps({"schema": "wrong.v9", "fragments": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        ingest._read_fragments_json(path)


# ---------------------------------------------------------------------------
# End-to-end ingest
# ---------------------------------------------------------------------------


SAMPLE_CORPUS = (
    "Hello there. How can I help you today?\n"
    "The weather is warm and sunny. I enjoy it very much.\n"
    "NRL-AI runs entirely on CPU. It uses SimHash anchors and ZPM routing."
)


def test_ingest_writes_full_index(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    out = tmp_path / "idx"

    result = ingest.ingest(corpus, out_dir=out)

    assert result.fragment_count >= 5
    assert result.transition_count == result.fragment_count - 1
    assert result.corpus_sha256
    assert (out / "manifest.json").is_file()
    assert (out / "fragments.json").is_file()
    assert (out / "anchors.bin").is_file()
    assert (out / "transitions.bin").is_file()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "nrl_ai.index.v1"
    assert manifest["fragment_count"] == result.fragment_count
    assert manifest["simhash_bits"] == 256


def test_ingest_roundtrip_via_loaders(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    out = tmp_path / "idx"
    result = ingest.ingest(corpus, out_dir=out)

    fragments = ingest.load_fragments(result.paths)
    anchors = ingest.load_anchors(result.paths)
    transitions = ingest.load_transitions(result.paths)

    assert len(fragments) == result.fragment_count
    assert len(anchors) == result.fragment_count
    assert len(transitions) == result.transition_count
    assert all(isinstance(a, tuple) and len(a) == 4 for a in anchors)
    # Transition keys should be consecutive (i, i+1) pairs.
    for i in range(result.fragment_count - 1):
        assert (i, i + 1) in transitions


def test_ingest_is_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")

    r1 = ingest.ingest(corpus, out_dir=tmp_path / "a")
    r2 = ingest.ingest(corpus, out_dir=tmp_path / "b")

    assert r1.corpus_sha256 == r2.corpus_sha256
    assert r1.fragment_count == r2.fragment_count
    assert ingest.load_anchors(r1.paths) == ingest.load_anchors(r2.paths)
    assert ingest.load_fragments(r1.paths) == ingest.load_fragments(r2.paths)


def test_ingest_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest.ingest(tmp_path / "nope.txt", out_dir=tmp_path / "idx")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_ingest_then_status_reports_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    idx = tmp_path / "idx"

    monkeypatch.setenv("NRL_AI_INDEX", str(idx))

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
    ingest_payload = json.loads(ingest_proc.stdout)
    assert ingest_payload["schema"] == "nrl_ai.ingest.v1"
    assert ingest_payload["status"] == "ok"
    assert ingest_payload["fragment_count"] >= 1
    assert ingest_payload["simhash_bits"] == 256
    assert ingest_payload["wps_target"] == nrl_ai.NRL_AI_WPS_TARGET

    status_proc = subprocess.run(
        [sys.executable, "-m", "nrlpy", "nrl-ai", "status", "--index", str(idx)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert status_proc.returncode == 0, status_proc.stderr
    status = json.loads(status_proc.stdout)
    assert status["index_ready"] is True
    assert status["manifest"]["fragment_count"] == ingest_payload["fragment_count"]
