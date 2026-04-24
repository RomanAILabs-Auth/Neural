# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for the NRL-AI scaffold (prompt #1 of the pivot).

These tests intentionally prove the *scaffold contract* only:

* the CLI surface is fully wired (status / ingest / chat / bench),
* status works without any index on disk,
* unimplemented stages raise ``NrlAiUnimplemented`` with the right prompt
  number,
* the on-disk manifest round-trips cleanly through JSON,
* the ``nrlpy nrl-ai`` dispatch is reachable from the top-level CLI,
* ``nrl run <gguf> --legacy-llama`` parses cleanly (forward-compat flag).

When later prompts ship real implementations, the ``NrlAiUnimplemented``
assertions will flip into behavioral tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nrlpy import nrl_ai
from nrlpy.cli import _parse_gguf_cli_args


def test_version_and_target_constants() -> None:
    assert nrl_ai.NRL_AI_VERSION == "0.1.0-scaffold"
    assert nrl_ai.NRL_AI_INDEX_VERSION == 1
    assert nrl_ai.NRL_AI_WPS_TARGET == 1000


def test_paths_default_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NRL_AI_INDEX", str(tmp_path / "override"))
    paths = nrl_ai.NrlAiPaths.default()
    assert paths.root == tmp_path / "override"
    assert paths.manifest == tmp_path / "override" / "manifest.json"
    assert not paths.exists()


def test_paths_default_uses_nrl_root_when_no_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NRL_AI_INDEX", raising=False)
    monkeypatch.setenv("NRL_ROOT", str(tmp_path / "nrlroot"))
    paths = nrl_ai.NrlAiPaths.default()
    assert paths.root == tmp_path / "nrlroot" / "cache" / "nrl_ai"


def test_paths_ensure_creates_root(tmp_path: Path) -> None:
    paths = nrl_ai.NrlAiPaths(tmp_path / "idx")
    assert not paths.root.exists()
    paths.ensure()
    assert paths.root.is_dir()


def test_index_manifest_roundtrip(tmp_path: Path) -> None:
    m = nrl_ai.NrlAiIndexManifest(
        corpus_sha256="deadbeef",
        fragment_count=7,
        transition_count=42,
        simhash_bits=256,
        created_utc="2026-04-23T17:00:00Z",
        source_path=str(tmp_path / "corpus.txt"),
    )
    dst = tmp_path / "idx" / "manifest.json"
    m.save(dst)
    round = nrl_ai.NrlAiIndexManifest.load(dst)
    assert round.corpus_sha256 == "deadbeef"
    assert round.fragment_count == 7
    assert round.transition_count == 42
    payload = json.loads(dst.read_text(encoding="utf-8"))
    assert payload["schema"] == "nrl_ai.index.v1"


def test_index_manifest_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError):
        nrl_ai.NrlAiIndexManifest.from_json({"schema": "not.ours.v7"})


def test_status_without_index_is_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NRL_AI_INDEX", str(tmp_path / "empty"))
    rc = nrl_ai.dispatch(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["schema"] == "nrl_ai.status.v1"
    assert payload["phase"] == "scaffold"
    assert payload["wps_target"] == nrl_ai.NRL_AI_WPS_TARGET
    assert payload["index_ready"] is False
    scopes = {item["prompt"]: item["status"] for item in payload["roadmap"]}
    assert scopes[1] == "shipped"
    assert scopes[2] == "shipped"
    assert scopes[3] == "shipped"
    assert scopes[4] == "shipped"
    assert scopes[5] == "shipped"
    assert scopes[6] == "shipped"
    assert scopes[7] == "shipped"


def test_status_picks_up_saved_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "idx"
    monkeypatch.setenv("NRL_AI_INDEX", str(root))
    paths = nrl_ai.NrlAiPaths(root)
    nrl_ai.NrlAiIndexManifest(
        corpus_sha256="abc123",
        fragment_count=3,
        created_utc="2026-04-23T17:00:00Z",
    ).save(paths.manifest)
    rc = nrl_ai.dispatch(["status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["index_ready"] is True
    assert payload["manifest"]["corpus_sha256"] == "abc123"
    assert payload["manifest"]["fragment_count"] == 3


def test_dispatch_ingest_missing_corpus_returns_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NRL_AI_INDEX", str(tmp_path / "idx"))
    rc = nrl_ai.dispatch(["ingest", str(tmp_path / "does_not_exist.txt")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "corpus not found" in err


def test_dispatch_no_subcommand_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = nrl_ai.dispatch([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NRL-AI" in out or "nrl-ai" in out


def test_parse_gguf_cli_accepts_legacy_llama_flag() -> None:
    model, kwargs = _parse_gguf_cli_args(["model.gguf", "--legacy-llama"])
    assert model == "model.gguf"
    assert kwargs["_legacy_llama"] is True


def test_parse_gguf_cli_legacy_llama_is_forward_compat_only() -> None:
    model, kwargs = _parse_gguf_cli_args(
        ["model.gguf", "--legacy-llama", "--max-tokens", "64"]
    )
    assert model == "model.gguf"
    assert kwargs["_legacy_llama"] is True
    assert kwargs["max_tokens"] == 64


def test_top_level_cli_routes_nrl_ai() -> None:
    # Shell out to guarantee the real entry point is wired; this catches any
    # accidental import-time regressions in cli.py / nrl_ai.py.
    result = subprocess.run(
        [sys.executable, "-m", "nrlpy", "nrl-ai", "status"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "nrl_ai.status.v1"
