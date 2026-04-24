# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for nrlpy.nrl_ai_compose (prompt #4 of the NRL-AI rebuild).

Covers the Omega-routed fragment walker and end-of-reply detection:

  * walks follow consecutive transitions in the default ranked order
  * each stop gate (sentence_cap, char_cap, soft_end, terminal,
    cycle_pruned) triggers with its correct reason
  * visited-set pruning prevents cycles
  * ``compose_stream`` emits the same fragments in the same order
  * ``compose_from_resolve`` honors the resolver's hit/miss contract
  * out-of-range fragment ids are rejected
  * CLI ``nrl-ai compose`` round-trips ingest -> resolve -> compose
    through the real subprocess entry point
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import (
    nrl_ai,
    nrl_ai_compose as compose_mod,
    nrl_ai_ingest as ingest,
    nrl_ai_resolve as resolve_mod,
)


SAMPLE_CORPUS = (
    "Hello there. How can I help you today?\n"
    "The weather is warm and sunny. I enjoy it very much.\n"
    "NRL-AI runs entirely on CPU. It uses SimHash anchors and ZPM routing.\n"
    "You can ask about speed, safety, or architecture.\n"
    "Thanks for chatting with Neural AI."
)


@pytest.fixture()
def sample_index(tmp_path: Path) -> nrl_ai.NrlAiPaths:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    result = ingest.ingest(corpus, out_dir=tmp_path / "idx")
    return result.paths


def test_compose_walks_consecutive_fragments(sample_index: nrl_ai.NrlAiPaths) -> None:
    # From fragment 0 with generous caps we expect the soft-end heuristic
    # to fire on the natural sentence boundary after >=2 sentences.
    result = compose_mod.compose(
        0,
        paths=sample_index,
        min_sentences=2,
        max_sentences=10,
        max_chars=2000,
    )
    ids = result.step_ids()
    assert ids[0] == 0
    assert ids[1] == 1
    assert result.reply.startswith("Hello there.")
    assert result.sentence_count >= 2


def test_compose_stops_at_sentence_cap(sample_index: nrl_ai.NrlAiPaths) -> None:
    result = compose_mod.compose(
        0,
        paths=sample_index,
        min_sentences=10,  # disable soft-end
        max_sentences=2,
        max_chars=2000,
    )
    assert result.sentence_count == 2
    assert result.stop_reason == "sentence_cap"
    assert result.steps[-1].stop_reason == "sentence_cap"


def test_compose_stops_at_char_cap(sample_index: nrl_ai.NrlAiPaths) -> None:
    result = compose_mod.compose(
        0,
        paths=sample_index,
        min_sentences=10,
        max_sentences=20,
        max_chars=1,  # trip on the very first fragment
    )
    assert result.sentence_count == 1
    assert result.stop_reason == "char_cap"


def test_compose_soft_end_on_terminator(sample_index: nrl_ai.NrlAiPaths) -> None:
    result = compose_mod.compose(
        0,
        paths=sample_index,
        min_sentences=2,
        max_sentences=20,
        max_chars=2000,
    )
    assert result.stop_reason == "soft_end"


def test_compose_terminal_at_end_of_chain(sample_index: nrl_ai.NrlAiPaths) -> None:
    fragments = ingest.load_fragments(sample_index)
    last_id = len(fragments) - 1
    result = compose_mod.compose(
        last_id,
        paths=sample_index,
        min_sentences=10,  # disable soft-end
        max_sentences=20,
        max_chars=5000,
    )
    assert result.start_fragment_id == last_id
    assert result.sentence_count == 1
    assert result.stop_reason == "terminal"


def test_compose_cycle_pruned_when_only_visited_candidates() -> None:
    # Hand-construct a tiny state with a self-loop so ``_omega_next``
    # must surface cycle_pruned rather than terminal.
    fragments = ["first sentence", "second sentence"]
    index = {
        0: [(1, 1)],
        1: [(0, 1)],  # only option is already visited after step 0 -> 1
    }
    steps = list(
        compose_mod._walk(
            0,
            fragments,
            index,
            min_sentences=10,
            max_sentences=10,
            max_chars=9999,
        )
    )
    assert [s.fragment_id for s in steps] == [0, 1]
    assert steps[-1].stop_reason == "cycle_pruned"


def test_compose_omega_prefers_highest_count() -> None:
    fragments = ["a", "b", "c", "d"]
    # From fragment 0 there are three candidates; count ordering should
    # pick (2, count=5) first.
    index = {
        0: [(1, 1), (2, 5), (3, 3)],
    }
    # Sort once the way _build_transition_index does so we exercise the
    # exact selection logic (desc count, asc dst).
    for lst in index.values():
        lst.sort(key=lambda x: (-x[1], x[0]))
    steps = list(
        compose_mod._walk(
            0,
            fragments,
            index,
            min_sentences=10,
            max_sentences=2,
            max_chars=9999,
        )
    )
    assert [s.fragment_id for s in steps] == [0, 2]


def test_compose_is_deterministic(sample_index: nrl_ai.NrlAiPaths) -> None:
    r1 = compose_mod.compose(1, paths=sample_index)
    r2 = compose_mod.compose(1, paths=sample_index)
    assert r1.step_ids() == r2.step_ids()
    assert r1.reply == r2.reply
    assert r1.stop_reason == r2.stop_reason


def test_compose_stream_matches_compose(sample_index: nrl_ai.NrlAiPaths) -> None:
    result = compose_mod.compose(
        2,
        paths=sample_index,
        min_sentences=2,
        max_sentences=5,
        max_chars=2000,
    )
    streamed = list(
        compose_mod.compose_stream(
            2,
            paths=sample_index,
            min_sentences=2,
            max_sentences=5,
            max_chars=2000,
        )
    )
    assert streamed == [s.fragment_text for s in result.steps]


def test_compose_rejects_out_of_range_id(sample_index: nrl_ai.NrlAiPaths) -> None:
    fragments = ingest.load_fragments(sample_index)
    with pytest.raises(ValueError):
        compose_mod.compose(len(fragments), paths=sample_index)
    with pytest.raises(ValueError):
        compose_mod.compose(-1, paths=sample_index)


def test_compose_from_resolve_miss_returns_none(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    rresult = resolve_mod.resolve(
        "zzzzzzzz unrelated banana mango",
        paths=sample_index,
        threshold_bits=0,  # force miss
    )
    assert rresult.hit is False
    assert compose_mod.compose_from_resolve(rresult, paths=sample_index) is None


def test_compose_from_resolve_hit_returns_reply(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    rresult = resolve_mod.resolve(
        "NRL-AI runs on CPU",
        paths=sample_index,
        top_k=1,
    )
    assert rresult.hit is True
    cresult = compose_mod.compose_from_resolve(rresult, paths=sample_index)
    assert cresult is not None
    assert cresult.hit is True
    assert "CPU" in cresult.reply
    assert cresult.sentence_count >= 1


def test_cli_compose_hit_roundtrip(tmp_path: Path) -> None:
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

    compose_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "compose",
            "NRL-AI runs on CPU",
            "--index",
            str(idx),
            "--max-sentences",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compose_proc.returncode == 0, compose_proc.stderr
    payload = json.loads(compose_proc.stdout)
    assert payload["schema"] == "nrl_ai.compose.v1"
    assert payload["hit"] is True
    assert payload["reply"]
    assert "CPU" in payload["reply"]
    assert payload["stop_reason"] in {
        "sentence_cap",
        "char_cap",
        "soft_end",
        "terminal",
        "cycle_pruned",
    }
    assert 1 <= payload["sentence_count"] <= 3
    assert len(payload["steps"]) == payload["sentence_count"]


def test_cli_compose_resolver_miss_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    idx = tmp_path / "idx"
    subprocess.run(
        [sys.executable, "-m", "nrlpy", "nrl-ai", "ingest", str(corpus), "--out", str(idx)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "compose",
            "banana mango zyxwvu unrelated",
            "--index",
            str(idx),
            "--threshold",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "nrl_ai.compose.v1"
    assert payload["hit"] is False
    assert payload["reply"] is None
    assert payload["stop_reason"] == "resolver_miss"
    assert payload["steps"] == []
