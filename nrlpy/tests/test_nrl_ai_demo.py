# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for prompt #7 of the NRL-AI rebuild (docs + seed corpora + demo).

Covers:

  * Packaged seed corpus exists, is non-trivial, and round-trips
    through the ingest pipeline into a usable index.
  * ``seed_corpus_path()`` resolves to a real file under the installed
    package and points at the UTF-8 ``data/seed_corpus.txt`` fixture.
  * ``cmd_demo --ingest-only`` ingests the seed corpus into the
    requested demo index, emits a terse stderr summary, and exits 0
    without opening a chat loop.
  * ``cmd_demo --ingest-only`` reuses an existing index without
    re-ingesting (stderr log mentions "reusing").
  * End-to-end: the seed corpus compiles cleanly and clears the 1000
    wps bench gate in-process (proves the packaged corpus is a real
    NRL-AI out-of-box demo, not a placeholder).
  * CLI subprocess ``nrl-ai demo --ingest-only`` exits 0.
  * README advertises the NRL-AI section and the demo command.
  * Roadmap prompt #7 is marked shipped in the status payload.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Seed corpus on disk
# ---------------------------------------------------------------------------


def test_seed_corpus_path_points_at_real_utf8_file() -> None:
    p = nrl_ai.seed_corpus_path()
    assert p.is_file()
    assert p.name == nrl_ai.SEED_CORPUS_NAME
    text = p.read_text(encoding="utf-8")
    # Non-trivial corpus: enough to produce a usable fragment table
    assert len(text) > 2000
    # Conversational markers the chat REPL expects to anchor on
    assert "NRL-AI" in text
    assert "RomanAILabs" in text
    assert "words per second" in text.lower() or "wps" in text.lower()


def test_seed_corpus_ingests_cleanly(tmp_path: Path) -> None:
    idx = tmp_path / "seed_idx"
    result = ingest.ingest(nrl_ai.seed_corpus_path(), out_dir=idx)
    assert result.fragment_count >= 40
    assert result.transition_count >= result.fragment_count - 1
    assert (idx / "manifest.json").is_file()
    assert (idx / "anchors.bin").is_file()
    assert (idx / "transitions.bin").is_file()
    assert (idx / "fragments.json").is_file()


def test_seed_corpus_clears_wps_gate_in_process(tmp_path: Path) -> None:
    """The packaged corpus isn't a placeholder -- it must actually prove
    the NRL-AI >=1000 wps contract from a clean ingest on every host."""
    idx = tmp_path / "seed_idx"
    ingest.ingest(nrl_ai.seed_corpus_path(), out_dir=idx)
    result = bench_mod.run_bench(
        nrl_ai.NrlAiPaths(idx), turns=16, warmup=2
    )
    assert result.hits >= 8, (
        f"seed corpus produced too few hits: {result.hits}/{result.measured_turns}"
    )
    assert result.gate_pass is True, (
        f"seed corpus failed wps gate: mean={result.wps_mean:.1f} "
        f"target={result.target_wps}"
    )


# ---------------------------------------------------------------------------
# cmd_demo --ingest-only
# ---------------------------------------------------------------------------


def test_cmd_demo_ingest_only_builds_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_idx = tmp_path / "demo_idx"
    rc = nrl_ai.dispatch(
        ["demo", "--index", str(demo_idx), "--ingest-only"]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "ingested seed corpus" in err
    assert (demo_idx / "manifest.json").is_file()


def test_cmd_demo_ingest_only_reuses_existing_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_idx = tmp_path / "demo_idx"
    rc1 = nrl_ai.dispatch(
        ["demo", "--index", str(demo_idx), "--ingest-only"]
    )
    assert rc1 == 0
    capsys.readouterr()
    rc2 = nrl_ai.dispatch(
        ["demo", "--index", str(demo_idx), "--ingest-only"]
    )
    assert rc2 == 0
    err2 = capsys.readouterr().err
    assert "reusing existing index" in err2


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


def test_cli_demo_ingest_only_subprocess(tmp_path: Path) -> None:
    demo_idx = tmp_path / "demo_idx"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "demo",
            "--index",
            str(demo_idx),
            "--ingest-only",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "seed corpus" in proc.stderr
    assert (demo_idx / "manifest.json").is_file()


def test_cli_demo_then_bench_clears_gate(tmp_path: Path) -> None:
    demo_idx = tmp_path / "demo_idx"
    demo_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "demo",
            "--index",
            str(demo_idx),
            "--ingest-only",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert demo_proc.returncode == 0, demo_proc.stderr

    bench_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrlpy",
            "nrl-ai",
            "bench",
            "--index",
            str(demo_idx),
            "--turns",
            "12",
            "--warmup",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert bench_proc.returncode == 0, (
        f"demo corpus failed bench gate:\n{bench_proc.stdout}\n{bench_proc.stderr}"
    )
    payload = json.loads(bench_proc.stdout)
    assert payload["gate_pass"] is True
    assert payload["wps_mean"] >= nrl_ai.NRL_AI_WPS_TARGET


# ---------------------------------------------------------------------------
# Roadmap + README
# ---------------------------------------------------------------------------


def test_status_reports_prompt_7_shipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NRL_AI_INDEX", str(tmp_path / "empty"))
    rc = nrl_ai.dispatch(["status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    scopes = {item["prompt"]: item["status"] for item in payload["roadmap"]}
    assert all(v == "shipped" for v in scopes.values()), scopes


def test_readme_advertises_nrl_ai_and_demo() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = root / "README.md"
    assert readme.is_file(), f"README missing at {readme}"
    text = readme.read_text(encoding="utf-8")
    assert "NRL-AI" in text
    assert "nrl-ai demo" in text
    assert "1000" in text or "1,000" in text
    assert "no GPU" in text.lower() or "no gpu" in text.lower()
    # The seven commands table should reference every subcommand
    for cmd in ("ingest", "resolve", "compose", "chat", "bench", "status", "demo"):
        assert f"nrl-ai {cmd}" in text, f"README is missing `nrl-ai {cmd}`"
