# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for nrlpy.nrl_ai_bench (prompt #6 of the NRL-AI rebuild).

Proves that the pure NRL lattice clears the 1,000 WPS gate on CPU
with a replay-locked contract.

Covers:

  * ``select_queries`` is a pure function of ``(corpus_sha256, turns)``:
    two calls with the same inputs return bit-identical query lists
    (that is the replay-locked property).
  * ``run_bench`` honors ``turns`` / ``warmup`` and excludes warmup
    turns from aggregate counters.
  * ``run_bench`` attaches corpus_sha256, fragment_count, threshold,
    env, and started_utc to the ``BenchResult``.
  * The gate trips correctly: artificially-high target -> gate_pass
    False; default target -> gate_pass True on a real corpus.
  * ``cli_main`` emits valid ``nrl_ai.bench.v1`` JSON on stdout, a
    human-readable summary on stderr, and returns exit code 0 on
    PASS, 4 on FAIL, 2 on missing index.
  * Full subprocess path: ``nrlpy nrl-ai bench`` on a real ingested
    corpus clears the 1,000 WPS gate on the bench host.
  * Queries file override loads cleanly and overrides the sampler.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import (
    nrl_ai,
    nrl_ai_bench as bench_mod,
    nrl_ai_ingest as ingest,
)


SAMPLE_CORPUS = (
    "Hello there. How can I help you today?\n"
    "The weather is warm and sunny. I enjoy it very much.\n"
    "NRL-AI runs entirely on CPU. It uses SimHash anchors and ZPM routing.\n"
    "You can ask about speed, safety, or architecture.\n"
    "Thanks for chatting with Neural AI.\n"
    "The lattice is deterministic. Every turn is replay-locked.\n"
    "Omega routes fragments through the transition map.\n"
    "SimHash256 anchors are locality sensitive. Near-match works.\n"
)


@pytest.fixture()
def sample_index(tmp_path: Path) -> nrl_ai.NrlAiPaths:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(SAMPLE_CORPUS, encoding="utf-8")
    return ingest.ingest(corpus, out_dir=tmp_path / "idx").paths


# ---------------------------------------------------------------------------
# Query selection is replay-locked
# ---------------------------------------------------------------------------


def test_select_queries_is_deterministic() -> None:
    frags = [f"fragment number {i}" for i in range(10)]
    sha = "a" * 64
    a = bench_mod.select_queries(frags, total_turns=20, corpus_sha256=sha)
    b = bench_mod.select_queries(frags, total_turns=20, corpus_sha256=sha)
    assert a == b
    assert len(a) == 20


def test_select_queries_changes_with_corpus_sha() -> None:
    frags = [
        f"w{i}a w{i}b w{i}c w{i}d w{i}e w{i}tail" for i in range(10)
    ]
    a = bench_mod.select_queries(frags, total_turns=8, corpus_sha256="a" * 64)
    b = bench_mod.select_queries(frags, total_turns=8, corpus_sha256="b" * 64)
    assert a != b


def test_select_queries_rejects_empty_fragments() -> None:
    with pytest.raises(ValueError):
        bench_mod.select_queries([], total_turns=4, corpus_sha256="x" * 64)


# ---------------------------------------------------------------------------
# run_bench: contract fields + warmup exclusion
# ---------------------------------------------------------------------------


def test_run_bench_populates_contract_fields(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    result = bench_mod.run_bench(
        sample_index, turns=4, warmup=1, target_wps=1
    )
    assert result.corpus_sha256  # non-empty
    assert result.fragment_count > 0
    assert result.threshold_bits > 0
    assert result.warmup_turns == 1
    assert result.measured_turns == 4
    assert result.total_turns == 5
    assert result.started_utc.endswith("Z")
    assert result.env["nrl_ai_version"] == nrl_ai.NRL_AI_VERSION
    assert result.env["nrl_ai_index_version"] == nrl_ai.NRL_AI_INDEX_VERSION
    assert result.env["cpu_count"] >= 0


def test_run_bench_warmup_turns_excluded_from_aggregates(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    result = bench_mod.run_bench(
        sample_index, turns=3, warmup=2, target_wps=1
    )
    measured_from_rows = [t for t in result.turns if not t.warmup]
    warmup_from_rows = [t for t in result.turns if t.warmup]
    assert len(measured_from_rows) == 3
    assert len(warmup_from_rows) == 2
    assert result.total_words == sum(t.words for t in measured_from_rows)
    assert result.total_wall_s == pytest.approx(
        sum(t.wall_s for t in measured_from_rows)
    )


def test_run_bench_rejects_bad_inputs(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    with pytest.raises(ValueError):
        bench_mod.run_bench(sample_index, turns=0, warmup=0)
    with pytest.raises(ValueError):
        bench_mod.run_bench(sample_index, turns=1, warmup=-1)


def test_run_bench_missing_index_raises(tmp_path: Path) -> None:
    paths = nrl_ai.NrlAiPaths(tmp_path / "not-there")
    with pytest.raises(FileNotFoundError):
        bench_mod.run_bench(paths, turns=4, warmup=0)


# ---------------------------------------------------------------------------
# Gate semantics
# ---------------------------------------------------------------------------


def test_run_bench_gate_passes_at_default_target(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    result = bench_mod.run_bench(sample_index, turns=8, warmup=1)
    assert result.target_wps == nrl_ai.NRL_AI_WPS_TARGET
    # Pure NRL should clear 1000 wps by a wide margin on this corpus.
    # Measure in-process (no subprocess overhead) so the assertion is
    # tight and honest.
    assert result.wps_mean >= nrl_ai.NRL_AI_WPS_TARGET
    assert result.gate_pass is True
    assert result.hits >= 1


def test_run_bench_gate_fails_when_target_is_absurd(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    # 10 billion wps is physically impossible on any CPU -> gate must fail.
    result = bench_mod.run_bench(
        sample_index, turns=4, warmup=1, target_wps=10_000_000_000
    )
    assert result.gate_pass is False
    assert result.wps_mean < result.target_wps


# ---------------------------------------------------------------------------
# Determinism: same index + same queries -> identical query + hit pattern
# ---------------------------------------------------------------------------


def test_run_bench_is_replay_locked_on_query_and_hit_pattern(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    a = bench_mod.run_bench(sample_index, turns=6, warmup=0, target_wps=1)
    b = bench_mod.run_bench(sample_index, turns=6, warmup=0, target_wps=1)
    assert [t.query for t in a.turns] == [t.query for t in b.turns]
    assert [t.hit for t in a.turns] == [t.hit for t in b.turns]
    assert [t.fragment_id for t in a.turns] == [t.fragment_id for t in b.turns]
    assert [t.stop_reason for t in a.turns] == [t.stop_reason for t in b.turns]
    # Content of the reply is independent of wall time
    assert a.total_words == b.total_words


# ---------------------------------------------------------------------------
# Queries-file override
# ---------------------------------------------------------------------------


def test_load_queries_file_strips_comments_and_blanks(tmp_path: Path) -> None:
    qfile = tmp_path / "q.txt"
    qfile.write_text(
        "# this is a comment\n"
        "\n"
        "NRL-AI runs on CPU\n"
        "  The weather is warm  \n"
        "# another comment\n"
        "Hello there\n",
        encoding="utf-8",
    )
    out = bench_mod.load_queries_file(qfile)
    assert out == ["NRL-AI runs on CPU", "The weather is warm", "Hello there"]


def test_load_queries_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bench_mod.load_queries_file(tmp_path / "nope.txt")


def test_load_queries_file_empty_raises(tmp_path: Path) -> None:
    qfile = tmp_path / "empty.txt"
    qfile.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        bench_mod.load_queries_file(qfile)


def test_run_bench_accepts_query_override(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    queries = ["NRL-AI runs on CPU", "The weather is warm", "Hello there"] * 4
    result = bench_mod.run_bench(
        sample_index, turns=6, warmup=1, queries=queries, target_wps=1
    )
    # Warmup uses queries[0..warmup); measured uses queries[warmup..total)
    assert [t.query for t in result.turns] == queries[:7]


def test_run_bench_query_override_must_be_long_enough(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    with pytest.raises(ValueError):
        bench_mod.run_bench(
            sample_index, turns=6, warmup=1, queries=["only", "two", "three"]
        )


# ---------------------------------------------------------------------------
# cli_main: json on stdout, summary on stderr, exit codes
# ---------------------------------------------------------------------------


def test_cli_main_emits_bench_v1_json_on_pass(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    err = io.StringIO()
    rc = bench_mod.cli_main(
        sample_index,
        turns=4,
        warmup=1,
        threshold_bits=None,
        target_wps=1,
        queries_file=None,
        min_sentences=2,
        max_sentences=4,
        max_chars=400,
        stdout=out,
        stderr=err,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["schema"] == bench_mod.BENCH_SCHEMA
    assert payload["gate_pass"] is True
    assert payload["measured_turns"] == 4
    assert payload["corpus_sha256"]
    assert "PASS" in err.getvalue()


def test_cli_main_returns_exit_code_4_on_gate_fail(
    sample_index: nrl_ai.NrlAiPaths,
) -> None:
    out = io.StringIO()
    err = io.StringIO()
    rc = bench_mod.cli_main(
        sample_index,
        turns=3,
        warmup=1,
        threshold_bits=None,
        target_wps=10_000_000_000,
        queries_file=None,
        min_sentences=2,
        max_sentences=4,
        max_chars=400,
        stdout=out,
        stderr=err,
    )
    assert rc == 4
    assert "FAIL" in err.getvalue()
    payload = json.loads(out.getvalue())
    assert payload["gate_pass"] is False


def test_cli_main_returns_exit_code_2_when_index_missing(
    tmp_path: Path,
) -> None:
    paths = nrl_ai.NrlAiPaths(tmp_path / "not-there")
    out = io.StringIO()
    err = io.StringIO()
    rc = bench_mod.cli_main(
        paths,
        turns=4,
        warmup=0,
        threshold_bits=None,
        target_wps=1,
        queries_file=None,
        min_sentences=2,
        max_sentences=4,
        max_chars=400,
        stdout=out,
        stderr=err,
    )
    assert rc == 2
    assert "nrl-ai ingest" in err.getvalue()


def test_cli_main_writes_out_file_when_requested(
    sample_index: nrl_ai.NrlAiPaths, tmp_path: Path
) -> None:
    dest = tmp_path / "bench.json"
    rc = bench_mod.cli_main(
        sample_index,
        turns=4,
        warmup=1,
        threshold_bits=None,
        target_wps=1,
        queries_file=None,
        min_sentences=2,
        max_sentences=4,
        max_chars=400,
        out_json=dest,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["schema"] == bench_mod.BENCH_SCHEMA


# ---------------------------------------------------------------------------
# End-to-end subprocess: prove 1000 wps on a real ingest + bench
# ---------------------------------------------------------------------------


def test_cli_bench_subprocess_clears_1000_wps_gate(tmp_path: Path) -> None:
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

    bench_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "bench",
            "--index",
            str(idx),
            "--turns",
            "8",
            "--warmup",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert bench_proc.returncode == 0, (
        f"bench failed (rc={bench_proc.returncode}):\n"
        f"stdout={bench_proc.stdout}\nstderr={bench_proc.stderr}"
    )
    payload = json.loads(bench_proc.stdout)
    assert payload["schema"] == bench_mod.BENCH_SCHEMA
    assert payload["target_wps"] == nrl_ai.NRL_AI_WPS_TARGET
    assert payload["gate_pass"] is True
    assert payload["wps_mean"] >= nrl_ai.NRL_AI_WPS_TARGET
    assert "PASS" in bench_proc.stderr


def test_cli_bench_subprocess_exits_2_when_index_missing(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "bench",
            "--index",
            str(tmp_path / "nope"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert "ingest" in proc.stderr
