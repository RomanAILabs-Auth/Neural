# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Wall-clock words/sec probe on a short coherent multi-turn dialogue.

This script measures **honest** ``wall_wps = assistant_words / wall_seconds``
per turn and for the whole session. It uses the same stack as interactive
chat: ``native_full`` when available, Phi-3 chat templating, and per-turn
``chat_kv_reuse`` (see ``nrlpy.gguf_chat._per_turn_manifest``) so later turns
reuse the KV prefix of the growing transcript where libllama allows it.

It is **not** a substitute for ``nrlpy bench-wps`` (release gate). It exists
to answer: "on this machine, with this GGUF, how fast does a realistic
back-and-forth run on the wall clock?"

Usage (PowerShell, from the NRL repo root)::

    python scripts/wall_wps.py --model \"C:\\Users\\Asus\\Desktop\\Documents\\RomaPy Engine\\phi-3-mini-4k-instruct.Q4_K_M.gguf\"

Optional::

    python scripts/wall_wps.py --model \"...\" --max-tokens 192 --temperature 0.55 --seed 1 --json-out build/wall_wps.json

**Interactive coherent chat** (same model, Phi-3 template, native hot path)::

    cd nrlpy
    python -m nrlpy chat \"C:\\Users\\Asus\\Desktop\\Documents\\RomaPy Engine\\phi-3-mini-4k-instruct.Q4_K_M.gguf\" --chat-format phi3 --native-full

Do **not** set ``NRL_INFERENCE=stub`` when measuring real hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_NRLPY_SRC = _REPO / "nrlpy" / "src"
if _NRLPY_SRC.is_dir() and str(_NRLPY_SRC) not in sys.path:
    sys.path.insert(0, str(_NRLPY_SRC))

from nrlpy.gguf_chat import build_session, chat_turn  # noqa: E402
from nrlpy import gguf  # noqa: E402


def _word_count(text: str) -> int:
    return len(text.split())


_DEFAULT_SCRIPT: tuple[str, ...] = (
    "Hello — we are going to discuss photosynthesis in a few short turns. "
    "Please keep answers concise but correct.",
    "What is chlorophyll?",
    "Why does chlorophyll make plants look green?",
    "In one sentence, how do plants turn sunlight into chemical energy?",
    "Thanks. Summarize our thread in two sentences.",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf",
        help="Path to the GGUF file (default: standard Phi-3 mini path on this workstation).",
    )
    ap.add_argument("--system", default="", help="Optional system string.")
    ap.add_argument("--seed", type=int, default=1, help="Sampler seed (default 1).")
    ap.add_argument("--max-tokens", type=int, default=160, help="Max new tokens per turn.")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.55,
        help="Sampling temperature (lower = steadier, more coherent).",
    )
    ap.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context length passed to libllama (default 2048; matches common Phi-3-mini bench runs).",
    )
    ap.add_argument(
        "--json-out",
        default="",
        help="If set, write a JSON report to this path.",
    )
    args = ap.parse_args()
    model = Path(args.model)
    if not model.is_file():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2
    if os.environ.get("NRL_INFERENCE", "").lower() == "stub":
        print(
            "error: NRL_INFERENCE=stub is set; wall_wps would not measure real decode.",
            file=sys.stderr,
        )
        return 2

    manifest = gguf.manifest_from_args(
        model=str(model),
        seed=int(args.seed),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        n_ctx=int(args.n_ctx),
        chat_format="phi3",
        runner_backend="native_full",
        muscle_memory="off",
        zpm_nullspace=False,
        coherence_lane="fast-stable",
        benchmark_class="B",
    )
    manifest.prompt = ""
    manifest.prompt_file = ""

    actual_sha = gguf.sha256_file(model)
    manifest.model_sha256 = actual_sha

    session = build_session(manifest, system=str(args.system))
    llm = gguf._load_llm(manifest)

    rows: list[dict[str, object]] = []
    total_words = 0
    t_session = time.perf_counter()

    for i, user_line in enumerate(_DEFAULT_SCRIPT):
        t0 = time.perf_counter()
        try:
            result = chat_turn(session, user_line, preloaded_llm=llm)
        except Exception as e:  # noqa: BLE001
            print(f"error on turn {i}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        wall = max(time.perf_counter() - t0, 1e-9)
        wc = _word_count(result.text)
        twps = wc / wall
        total_words += wc
        kv = getattr(llm, "_nrl_kv_reuse_info", None) or {}
        rows.append(
            {
                "turn": i,
                "user_chars": len(user_line),
                "assistant_words": wc,
                "assistant_tokens": int(result.tokens),
                "wall_s": round(wall, 4),
                "turn_wall_wps": round(twps, 2),
                "cache_hit": bool(result.cache_hit),
                "gate_source": str(result.gate_source or ""),
                "kv_reused_tokens": int(kv.get("reused", 0)),
                "kv_prompt_tokens": int(kv.get("total", 0)),
            }
        )
        print(
            f"turn {i:>2}  words={wc:>4}  wall={wall:>7.2f}s  "
            f"wall_wps={twps:>8.2f}  "
            f"kv={kv.get('reused', 0)}/{kv.get('total', 0)}  "
            f"hit={'Y' if result.cache_hit else 'N'}"
        )

    session_wall = max(time.perf_counter() - t_session, 1e-9)
    session_wps = total_words / session_wall

    print("-" * 72)
    print(
        f"SESSION  total_assistant_words={total_words}  "
        f"wall={session_wall:.2f}s  session_wall_wps={session_wps:.2f}"
    )
    print(
        "Note: session_wall_wps is total assistant words / total wall time "
        "(includes prompt processing). Per-turn wall_wps is the headline "
        "for each decode. KV reuse grows on later turns when libllama keeps "
        "the cache warm."
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(model),
            "model_sha256": actual_sha,
            "session_wall_s": round(session_wall, 4),
            "total_assistant_words": total_words,
            "session_wall_wps": round(session_wps, 2),
            "turns": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
