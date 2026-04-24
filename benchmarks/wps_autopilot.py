#!/usr/bin/env python3
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Automated WPS landing report (CPU-first, no GPU required).

Runs repeatable harnesses and optional real-model probes, writes JSON (+ optional
Markdown). This does **not** silently equate NRL lattice ``virtual_gops`` with
GGUF decode words/sec — those are different planes. What we report:

* ``mm_replay`` — muscle-memory turn-2 ``effective_wps`` (stub lane; can far
  exceed 1k; no ceiling applied).
* ``golden_subset`` — reuses ``gguf_golden`` modes you select (default: stub,
  p2active-sim, p2active-prefill, mm-replay) and copies headline WPS fields.
* ``real_cold_vs_replay`` — if ``--model`` or ``NRL_GGUF_GOLDEN_MODEL`` resolves
  and ``NRL_INFERENCE`` is not forced to stub, runs the same prompt twice with
  a fresh ``NRL_ROOT`` cache dir: turn1 ``executed_wps``, turn2 cache path
  ``effective_wps`` (honest "where it lands" on your box).

Optional **planning** math (same units as ``throughput_math``): pass
``--plan-gops`` and ``--plan-decode-tps`` from a *single* calibration story you
trust (e.g. dense decode observation), then we project a words/sec band and the
GOPS implied for 1k / 48k WPS targets — illustrative only unless those inputs
are measured together.

Exit code ``1`` only when ``--min-mm-effective-wps`` is > 0 and stub mm-replay
turn-2 ``effective_wps`` is **below** that minimum (CI / local gate). There is
no upper cap — higher is always better.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent


def _load_gguf_golden_module() -> ModuleType:
    path = _ROOT / "benchmarks" / "gguf_golden.py"
    name = "nrl_gguf_golden_autoload"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_repo_nrlpy_on_path() -> None:
    src = _ROOT / "nrlpy" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_repo_nrlpy_on_path()

from nrlpy import gguf  # noqa: E402
from nrlpy import throughput_math  # noqa: E402


def _run_mm_replay_stub(*, prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    gh = _load_gguf_golden_module()
    r = gh.run_mm_replay_mode(prompt=prompt, max_tokens=max_tokens, seed=seed)
    return {
        "mode": "mm-replay",
        "passed": r.passed,
        "cache_hit_turn2": r.cache_hit,
        "effective_wps_turn2": r.effective_wps,
        "executed_wps_turn2": r.executed_wps,
        "virtual_wps_turn2": r.virtual_wps,
        "tokens_turn2": r.tokens,
        "words_turn2": r.words,
        "wall_s_total": r.wall_s,
    }


def _golden_headlines(modes: list[str], *, prompt: str, max_tokens: int, seed: int) -> list[dict[str, Any]]:
    gh = _load_gguf_golden_module()
    out: list[dict[str, Any]] = []
    runners = {
        "stub": gh.run_stub_mode,
        "p2active-sim": gh.run_p2active_sim_mode,
        "p2active-prefill": gh.run_p2active_prefill_mode,
        "mm-replay": lambda **kw: gh.run_mm_replay_mode(
            prompt=kw["prompt"], max_tokens=kw["max_tokens"], seed=kw["seed"]
        ),
    }
    for name in modes:
        fn = runners.get(name)
        if fn is None:
            continue
        r = fn(prompt=prompt, max_tokens=max_tokens, seed=seed)
        out.append(
            {
                "mode": name,
                "passed": r.passed,
                "skipped": r.skipped,
                "executed_wps": r.executed_wps,
                "virtual_wps": r.virtual_wps,
                "effective_wps": r.effective_wps,
                "executed_tps": r.executed_tps,
                "virtual_tps": r.virtual_tps,
                "effective_tps": r.effective_tps,
                "gate_skip_ratio": r.gate_skip_ratio,
                "cache_hit": r.cache_hit,
            }
        )
    return out


def _resolve_model(p: str | None) -> Path | None:
    if p:
        pp = Path(p).expanduser()
        if pp.is_file():
            return pp
    env = os.environ.get("NRL_GGUF_GOLDEN_MODEL", "").strip()
    if env:
        pe = Path(env).expanduser()
        if pe.is_file():
            return pe
    return None


def _real_two_turn(
    model: Path,
    *,
    prompt: str,
    max_tokens: int,
    seed: int,
    chat_format: str,
) -> dict[str, Any] | None:
    if os.environ.get("NRL_INFERENCE", "").lower().strip() == "stub":
        return {"skipped": True, "reason": "NRL_INFERENCE=stub (skip real decode)"}
    # Tight RAM defaults: logits scratch is O(vocab × n_ctx). Default 512 ctx for probes.
    n_ctx = int(os.environ.get("NRL_WPS_AUTOPILOT_N_CTX", "512"))
    n_batch = int(os.environ.get("NRL_WPS_AUTOPILOT_N_BATCH", "256"))
    saved_root = os.environ.get("NRL_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="nrl_wps_autopilot_") as tmp:
            root = Path(tmp)
            os.environ["NRL_ROOT"] = str(root)
            base = dict(
                model=str(model),
                prompt=prompt,
                max_tokens=min(max_tokens, 128),
                seed=seed,
                chat_format=chat_format,
                muscle_memory="on",
                n_ctx=n_ctx,
                n_batch=n_batch,
            )
            t0 = time.perf_counter()
            a = gguf.run_gguf(gguf.manifest_from_args(**base), stream_to=None, observation_profile="")
            b = gguf.run_gguf(gguf.manifest_from_args(**base), stream_to=None, observation_profile="")
            wall = time.perf_counter() - t0
            return {
                "skipped": False,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "wall_s_both_turns": wall,
                "turn1_cache_hit": a.cache_hit,
                "turn2_cache_hit": b.cache_hit,
                "turn1_executed_wps": a.word_rates.executed_wps,
                "turn1_effective_wps": a.word_rates.effective_wps,
                "turn2_executed_wps": b.word_rates.executed_wps,
                "turn2_effective_wps": b.word_rates.effective_wps,
                "turn2_cache_wps": b.word_rates.cache_wps,
                "turn1_tokens": a.tokens,
                "turn2_tokens": b.tokens,
            }
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if saved_root is None:
            os.environ.pop("NRL_ROOT", None)
        else:
            os.environ["NRL_ROOT"] = saved_root


def _planning_block(gops: float, decode_tps: float) -> dict[str, Any]:
    upt = throughput_math.calibrate_updates_per_token(
        executed_gops=gops, executed_tps=decode_tps
    )
    band = throughput_math.words_per_second_band(
        min_gops=gops * 0.85,
        max_gops=gops * 1.15,
        updates_per_token=upt,
    )
    return {
        "inputs_note": "plan_gops and plan_decode_tps must describe the same workload for this to be meaningful",
        "calibrated_updates_per_token": upt,
        "projected_words_per_sec_min": band.min_words_per_sec,
        "projected_words_per_sec_max": band.max_words_per_sec,
        "required_gops_for_1000_wps": throughput_math.required_gops_for_words_per_second(
            target_words_per_second=1000.0,
            updates_per_token=upt,
            words_per_token=0.75,
        ),
        "required_gops_for_48000_wps": throughput_math.required_gops_for_words_per_second(
            target_words_per_second=48000.0,
            updates_per_token=upt,
            words_per_token=0.75,
        ),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default="Short benchmark prompt for WPS autopilot.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chat-format", default="phi3")
    p.add_argument(
        "--golden-modes",
        default="stub,p2active-sim,p2active-prefill,mm-replay",
        help="Comma-separated subset of gguf_golden modes to snapshot.",
    )
    p.add_argument("--model", default=None, help="Real GGUF path (else NRL_GGUF_GOLDEN_MODEL).")
    p.add_argument("--output", default=str(_ROOT / "build" / "wps_autopilot.json"))
    p.add_argument("--markdown", default=None, help="Optional path for a short .md summary.")
    p.add_argument("--plan-gops", type=float, default=0.0, help="If >0 with --plan-decode-tps, run projection block.")
    p.add_argument("--plan-decode-tps", type=float, default=0.0)
    p.add_argument(
        "--min-mm-effective-wps",
        type=float,
        default=0.0,
        dest="min_mm_effective_wps",
        help="If >0: exit 1 when stub mm-replay turn-2 effective_wps is below this floor.",
    )
    p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        dest="fail_under_legacy",
        help=argparse.SUPPRESS,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if getattr(args, "fail_under_legacy", None) is not None:
        args.min_mm_effective_wps = float(args.fail_under_legacy)
    modes = [m.strip() for m in args.golden_modes.split(",") if m.strip()]

    payload: dict[str, Any] = {
        "schema": "nrl.wps_autopilot.v1",
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "mm_replay_stub": _run_mm_replay_stub(
            prompt=args.prompt, max_tokens=args.max_tokens, seed=args.seed
        ),
        "golden_subset": _golden_headlines(
            modes, prompt=args.prompt, max_tokens=args.max_tokens, seed=args.seed
        ),
    }

    model = _resolve_model(args.model)
    if model is not None:
        payload["real_cold_vs_replay"] = _real_two_turn(
            model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            seed=args.seed,
            chat_format=args.chat_format,
        )
    else:
        payload["real_cold_vs_replay"] = {"skipped": True, "reason": "no model path"}

    if args.plan_gops > 0.0 and args.plan_decode_tps > 0.0:
        payload["projection_planning"] = _planning_block(args.plan_gops, args.plan_decode_tps)
    else:
        payload["projection_planning"] = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[wps-autopilot] wrote {out_path}")

    mm_eff = float(payload["mm_replay_stub"]["effective_wps_turn2"])
    floor = float(args.min_mm_effective_wps)
    if floor > 0.0 and mm_eff < floor:
        print(
            f"[wps-autopilot] FAIL: mm_replay effective_wps_turn2={mm_eff:.1f} "
            f"< --min-mm-effective-wps={floor}",
            file=sys.stderr,
        )
        return 1

    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# WPS autopilot",
            "",
            f"- **mm-replay (stub) effective_wps turn2:** `{mm_eff:.2f}`",
            f"- **min_mm_effective_wps floor:** `{floor}` (0 = disabled)",
            "",
            "## Golden subset",
            "",
        ]
        for row in payload["golden_subset"]:
            lines.append(
                f"- `{row['mode']}`: executed_wps={row['executed_wps']:.2f}, "
                f"virtual_wps={row['virtual_wps']:.2f}, effective_wps={row['effective_wps']:.2f}"
            )
        lines.append("")
        lines.append("## Real cold vs replay")
        lines.append("")
        lines.append(f"```json\n{json.dumps(payload.get('real_cold_vs_replay'), indent=2)}\n```\n")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[wps-autopilot] wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
