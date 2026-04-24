# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""``nrlpy wps-chat-bench`` dispatcher.

Resolves bare ``<name>.gguf`` filenames the same way ``nrl run`` does (cwd,
``NRL_MODELS_DIR``, the canonical Desktop/Documents/RomaPy Engine path, etc.),
then shells into :func:`benchmarks.wps_chat_bench.main`. Keeps the bench
runnable from any directory after ``install_nrl.ps1`` wires up ``NRL_REPO``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path | None:
    env = os.environ.get("NRL_REPO", "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "benchmarks" / "wps_chat_bench.py").is_file():
            return parent
    return None


def main_wps_chat_bench(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: nrlpy wps-chat-bench <model.gguf|bare-name.gguf> [options]\n"
            "\n"
            "Options forwarded to benchmarks/wps_chat_bench.py:\n"
            "  --auto-tune               grid-search n_threads/n_batch, pick best\n"
            "  --max-tokens N            default 48\n"
            "  --n-ctx N                 default 2048\n"
            "  --n-batch N               default 512\n"
            "  --n-threads N             default = physical cores\n"
            "  --output PATH             JSON artifact path (default build/wps_chat_bench.json)\n"
            "  --markdown PATH           optional human-readable MD summary\n"
            "  --fail-under FLOAT        CI gate on session_effective_wps\n",
            end="",
        )
        return 0

    # Resolve bare filenames the same way nrl run does.
    try:
        from .cli import _resolve_model_path
    except Exception:
        def _resolve_model_path(p: str) -> str:  # pragma: no cover
            return p

    # Find and rewrite the first positional / --model value.
    rewrote = False
    new_args: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i + 1 < len(args):
            new_args.append(a)
            new_args.append(_resolve_model_path(args[i + 1]))
            i += 2
            rewrote = True
            continue
        if i == 0 and not a.startswith("-"):
            # First positional: treat as model path.
            new_args.append("--model")
            new_args.append(_resolve_model_path(a))
            rewrote = True
            i += 1
            continue
        new_args.append(a)
        i += 1
    if not rewrote:
        print("error: wps-chat-bench needs a model path (bare .gguf or --model <path>)", file=sys.stderr)
        return 2

    repo = _find_repo_root()
    if repo is None:
        print(
            "error: could not locate NRL repo (set NRL_REPO or run from the git clone).",
            file=sys.stderr,
        )
        return 2
    sys.path.insert(0, str(repo / "benchmarks"))
    import importlib

    mod = importlib.import_module("wps_chat_bench")
    return int(mod.main(new_args))


if __name__ == "__main__":
    raise SystemExit(main_wps_chat_bench(sys.argv[1:]))
