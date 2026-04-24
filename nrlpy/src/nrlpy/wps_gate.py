# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""WPS / GGUF full gate — runnable from any cwd when the NRL git repo is known.

Resolves the repo root in this order:

1. Environment variable ``NRL_REPO`` (absolute path to the NRL clone).
2. Walk parents of this package until ``benchmarks/gguf_golden.py`` exists
   (works for ``PYTHONPATH=.../nrlpy/src`` dev layouts).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_nrl_repo_root() -> Path | None:
    raw = os.environ.get("NRL_REPO", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if (p / "benchmarks" / "gguf_golden.py").is_file():
            return p
    here = Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "benchmarks" / "gguf_golden.py").is_file():
            return anc
    return None


def main_wps_gate(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    ci = "--ci" in args or "-ci" in args or (len(args) >= 1 and args[0].lower() == "ci")
    args = [a for a in args if a not in ("--ci", "-ci", "ci", "CI")]

    root = find_nrl_repo_root()
    if root is None:
        print(
            "error: cannot find NRL repo (need benchmarks/gguf_golden.py).\n"
            "  Set NRL_REPO to your clone, e.g.:\n"
            "    setx NRL_REPO \"C:\\Users\\Asus\\Desktop\\Documents\\NRL\"\n"
            "  Then open a new terminal and run: nrlpy wps-gate",
            file=sys.stderr,
        )
        return 2

    py_src = root / "nrlpy" / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(py_src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def run(cmd: list[str]) -> int:
        r = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            check=False,
        )
        return int(r.returncode)

    rc = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "nrlpy/tests/test_wps_autopilot.py",
            "nrlpy/tests/test_gguf.py",
            "nrlpy/tests/test_gguf_chat.py",
            "nrlpy/tests/test_cli_chat.py",
            "nrlpy/tests/test_wps_chat_bench.py",
            "-q",
            "--tb=line",
        ]
    )
    if rc != 0:
        return rc

    rc = run([sys.executable, str(root / "benchmarks" / "gguf_golden.py"), "--mode", "auto"])
    if rc != 0:
        return rc

    autopilot = [
        sys.executable,
        str(root / "benchmarks" / "wps_autopilot.py"),
        "--markdown",
        str(root / "build" / "wps_autopilot.md"),
    ]
    if ci:
        autopilot += ["--min-mm-effective-wps", "1000"]
    rc = run(autopilot)
    if rc != 0:
        return rc

    if ci:
        rc = run(
            [
                sys.executable,
                str(root / "benchmarks" / "gguf_golden.py"),
                "--mode",
                "mm-replay",
                "--mm-min-wps",
                "1000",
            ]
        )
        if rc != 0:
            return rc

    # Optional real-model chat bench. Opt-in via NRL_WPS_CHAT_MODEL so CI on
    # machines without the GGUF weights stays green, but power-users who have
    # a model locally get the real CPU-floor + muscle-memory-replay numbers
    # landed in build/wps_chat_bench.{json,md} as part of the gate.
    chat_model = os.environ.get("NRL_WPS_CHAT_MODEL", "").strip()
    if chat_model:
        chat_cmd = [
            sys.executable,
            str(root / "benchmarks" / "wps_chat_bench.py"),
            "--model", chat_model,
            "--n-ctx", os.environ.get("NRL_WPS_CHAT_N_CTX", "1024"),
            "--n-batch", os.environ.get("NRL_WPS_CHAT_N_BATCH", "256"),
            "--max-tokens", os.environ.get("NRL_WPS_CHAT_MAX_TOKENS", "48"),
            "--output", str(root / "build" / "wps_chat_bench.json"),
            "--markdown", str(root / "build" / "wps_chat_bench.md"),
        ]
        if ci:
            chat_cmd += [
                "--replay-fail-under",
                os.environ.get("NRL_WPS_CHAT_REPLAY_FLOOR", "1000"),
            ]
        else:
            chat_cmd += ["--replay-fail-under", "0"]
        rc = run(chat_cmd)
        if rc != 0:
            return rc

    print(f"[wps-gate] OK  repo={root}")
    print(f"  artifacts: {root / 'build' / 'gguf_golden'} , {root / 'build' / 'wps_autopilot.json'}")
    if chat_model:
        print(f"             {root / 'build' / 'wps_chat_bench.json'}")
    return 0


__all__ = ["find_nrl_repo_root", "main_wps_gate"]
