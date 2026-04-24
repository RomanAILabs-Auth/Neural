# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Smoke-test the WPS autopilot harness (subprocess, repo-root relative)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_wps_autopilot_writes_json_and_meets_1k_floor() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "benchmarks" / "wps_autopilot.py"
    assert script.is_file()
    out = root / "build" / "wps_autopilot_test_out.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "nrlpy" / "src")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(out),
            "--min-mm-effective-wps",
            "1000",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mm_replay_stub"]["effective_wps_turn2"] >= 1000.0
