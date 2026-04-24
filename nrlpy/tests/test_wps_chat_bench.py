# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Smoke tests for ``benchmarks/wps_chat_bench.py`` using the stub backend.

These tests run without ``llama-cpp-python`` and without any real GGUF file —
``NRL_INFERENCE=stub`` inside ``nrlpy.gguf`` returns a deterministic fake
stream, so the bench harness can be exercised end-to-end in CI.

What we prove here:

* The bench writes a JSON + Markdown artifact with the expected shape.
* Both ``phase_fill`` and ``phase_replay`` are populated when
  ``--replay-phase`` is on (default).
* The replay phase reports 100 % cache hits and a ``replay_session_effective_wps``
  floor well above the 1000 WPS architecture target — i.e. the muscle-memory
  read path (not libllama decode) drives that number, which is exactly the
  P1 claim we want CI to lock in.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks" / "wps_chat_bench.py"


def _fake_gguf(tmp_path: Path) -> Path:
    """Create a tiny file with a ``.gguf`` extension to satisfy path checks."""
    p = tmp_path / "stub-model.gguf"
    p.write_bytes(b"STUB-GGUF-FOR-TESTS")
    return p


@pytest.mark.skipif(not BENCH.is_file(), reason="bench script missing")
def test_wps_chat_bench_stub_produces_artifacts_and_hits_1000_wps(
    tmp_path: Path,
) -> None:
    model = _fake_gguf(tmp_path)
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"

    env = os.environ.copy()
    env["NRL_INFERENCE"] = "stub"
    env["NRL_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "nrlpy" / "src"), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable,
        str(BENCH),
        "--model", str(model),
        "--max-tokens", "16",
        "--n-ctx", "256",
        "--n-batch", "64",
        "--output", str(out_json),
        "--markdown", str(out_md),
        "--replay-fail-under", "1000",
        "--fail-under", "0",
        "--prompts", "hi", "what is 2 plus 2", "name a color",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, (
        f"bench exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert out_json.is_file(), "bench must write JSON artifact"
    assert out_md.is_file(), "bench must write Markdown artifact"

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "nrl.wps_chat_bench.v1"
    report = payload["report"]
    assert report["phase_fill"] is not None
    assert report["phase_replay"] is not None
    assert report["replay_cache_hit_rate"] == pytest.approx(1.0)
    assert report["replay_session_effective_wps"] >= 1000.0, (
        "muscle-memory replay must clear the 1k WPS architecture floor "
        f"(got {report['replay_session_effective_wps']:.2f})"
    )
    assert report["replay_min_effective_wps"] > 0.0


@pytest.mark.skipif(not BENCH.is_file(), reason="bench script missing")
def test_wps_chat_bench_honors_replay_fail_under(tmp_path: Path) -> None:
    """An impossibly-high ``--replay-fail-under`` must trip the CI gate."""
    model = _fake_gguf(tmp_path)
    out_json = tmp_path / "bench.json"

    env = os.environ.copy()
    env["NRL_INFERENCE"] = "stub"
    env["NRL_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "nrlpy" / "src"), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable,
        str(BENCH),
        "--model", str(model),
        "--max-tokens", "8",
        "--n-ctx", "128",
        "--n-batch", "32",
        "--output", str(out_json),
        "--replay-fail-under", "10000000000",
        "--fail-under", "0",
        "--prompts", "hi",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode != 0, "bench should FAIL when replay WPS floor is unreachable"
    assert "replay_session_effective_wps" in (result.stdout + result.stderr)


def test_wps_chat_bench_module_importable() -> None:
    """The bench module is importable via ``benchmarks.wps_chat_bench`` when the
    repo's benchmarks dir is on the path — this mirrors how
    :mod:`nrlpy.wps_chat_bench_cli` loads it after ``nrlpy wps-chat-bench``.
    """
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    try:
        mod = importlib.import_module("wps_chat_bench")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_run_transcript")
        assert hasattr(mod, "_clear_muscle_memory")
    finally:
        try:
            sys.path.remove(str(REPO_ROOT / "benchmarks"))
        except ValueError:
            pass
