#!/usr/bin/env python3
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Real-model chat WPS bench — preload once, run scripted dialog + repeats.

Measures on **your** CPU (no GPU assumed):

* ``cold_tps`` / ``cold_wps`` — first turn, fresh decode (libllama materialization).
* ``warm_tps`` / ``warm_wps`` — second+ turns on **new** user text, same session.
* ``replay_wps``             — turn that is *identical* to a prior turn (muscle
  memory cache hit → disk replay).
* ``session_effective_wps``  — across the full scripted transcript.

Also auto-tunes ``n_threads`` and ``n_batch`` when ``--auto-tune`` is set:
grid-search a small set, pick the config with the highest **cold_wps** (most
important signal for user-perceived first-reply latency), re-run the full
transcript on that config and pin it to the artifact.

Usage::

    python benchmarks/wps_chat_bench.py --model "C:\\path\\to\\model.gguf"
    python benchmarks/wps_chat_bench.py --model ... --auto-tune
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "nrlpy" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nrlpy import gguf  # noqa: E402
from nrlpy.gguf_chat import ChatSession, build_session, chat_turn  # noqa: E402


DEFAULT_PROMPTS = [
    "hi",
    "What's 2+2?",
    "Name three primary colors.",
    "Briefly, what is the capital of France?",
    "hi",  # deliberate repeat of turn 0 → muscle memory hit
]


@dataclass
class TurnRecord:
    idx: int
    prompt: str
    is_repeat_of: int | None
    tokens: int
    wall_s: float
    executed_tps: float
    executed_wps: float
    cache_hit: bool
    words: int
    words_per_token: float
    text_preview: str = ""


@dataclass
class PhaseSummary:
    label: str
    turns: list[TurnRecord] = field(default_factory=list)
    wall_s: float = 0.0
    total_tokens: int = 0
    total_words: int = 0
    mean_executed_wps: float = 0.0
    mean_effective_wps: float = 0.0
    session_effective_wps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d


@dataclass
class BenchReport:
    model: str
    model_sha256: str
    config: dict[str, Any] = field(default_factory=dict)
    machine: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    cold_executed_tps: float = 0.0
    cold_executed_wps: float = 0.0
    warm_mean_executed_wps: float = 0.0
    replay_mean_effective_wps: float = 0.0
    session_executed_wps: float = 0.0
    session_effective_wps: float = 0.0
    session_wall_s: float = 0.0
    total_tokens: int = 0
    total_words: int = 0
    phase_fill: PhaseSummary | None = None
    phase_replay: PhaseSummary | None = None
    replay_session_effective_wps: float = 0.0
    replay_min_effective_wps: float = 0.0
    replay_cache_hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        if self.phase_fill is not None:
            d["phase_fill"] = self.phase_fill.to_dict()
        if self.phase_replay is not None:
            d["phase_replay"] = self.phase_replay.to_dict()
        return d


def _detect_chat_format(model_path: Path) -> str:
    low = model_path.name.lower()
    if "phi-3" in low or "phi3" in low:
        return "phi3"
    if "llama-2" in low or "llama2" in low:
        return "llama2"
    return "none"


def _machine_specs() -> dict[str, Any]:
    out: dict[str, Any] = {"platform": sys.platform}
    try:
        import psutil  # noqa: PLC0415

        out["physical_cores"] = psutil.cpu_count(logical=False)
        out["logical_cores"] = psutil.cpu_count(logical=True)
        out["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        out["physical_cores"] = os.cpu_count()
        out["logical_cores"] = os.cpu_count()
    try:
        import llama_cpp  # noqa: PLC0415

        out["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        out["llama_cpp_version"] = "not-installed"
    return out


def _run_phase(
    prompts: list[str],
    llm: Any,
    manifest: Any,
    system: str,
    label: str,
) -> tuple[PhaseSummary, ChatSession]:
    """Run ``prompts`` as a single chat session. Returns (summary, ending-session)."""
    session = build_session(manifest, system=system)
    session.model_sha256 = manifest.model_sha256
    summary = PhaseSummary(label=label)
    t_phase = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        result = chat_turn(session, prompt, preloaded_llm=llm, stream_to=None)
        wall = time.perf_counter() - t0
        summary.turns.append(
            TurnRecord(
                idx=idx,
                prompt=prompt,
                is_repeat_of=None,
                tokens=result.tokens,
                wall_s=wall,
                executed_tps=result.tps.executed_tps,
                executed_wps=result.word_rates.executed_wps,
                cache_hit=result.cache_hit,
                words=result.word_rates.word_count,
                words_per_token=result.word_rates.words_per_token,
                text_preview=result.text[:80].replace("\n", " ").strip(),
            )
        )
    summary.wall_s = time.perf_counter() - t_phase
    summary.total_tokens = sum(t.tokens for t in summary.turns)
    summary.total_words = sum(t.words for t in summary.turns)
    if summary.turns:
        summary.mean_executed_wps = sum(t.executed_wps for t in summary.turns) / len(summary.turns)
    if summary.wall_s > 0 and summary.total_words > 0:
        summary.session_effective_wps = summary.total_words / summary.wall_s
    tps = session.tps.as_tps_report()
    total_turn_wall = tps.executed_wall_s + tps.cache_wall_s
    total_turn_tokens = tps.executed_tokens + tps.cache_tokens
    if total_turn_wall > 0 and total_turn_tokens > 0:
        wpt = summary.total_words / max(total_turn_tokens, 1)
        summary.mean_effective_wps = tps.effective_tps * wpt
    return summary, session


def _clear_muscle_memory(model_sha256: str) -> int:
    """Wipe on-disk MM entries for this model so the fill phase is truly cold.

    Matches :func:`nrlpy.gguf._muscle_memory_root` — i.e. ``$NRL_ROOT/cache/mm``
    (``$NRL_ROOT`` falls back to the current working directory). Only touches
    the directory namespaced by this model's SHA256 so other models' entries
    are preserved.
    """
    env = os.environ.get("NRL_ROOT", "").strip()
    base = Path(env) if env else Path.cwd()
    mm_dir = base / "cache" / "mm" / model_sha256
    if not mm_dir.is_dir():
        return 0
    count = 0
    for p in mm_dir.glob("*.mm"):
        try:
            p.unlink()
            count += 1
        except OSError:
            pass
    return count


def _run_transcript(
    model: Path,
    prompts: list[str],
    *,
    n_ctx: int,
    n_batch: int,
    n_threads: int,
    max_tokens: int,
    chat_format: str,
    seed: int,
    system: str,
    replay_phase: bool = False,
    clear_mm: bool = True,
) -> BenchReport:
    manifest = gguf.manifest_from_args(
        model=str(model),
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.08,
        chat_format=chat_format,
        prefill_cache="session",
        seed=seed,
    )

    actual_sha = gguf.sha256_file(model)
    manifest.model_sha256 = actual_sha
    if clear_mm:
        removed = _clear_muscle_memory(actual_sha)
        if removed:
            print(f"    [wps-chat-bench] cleared {removed} muscle-memory entries for model")
    llm = gguf._load_llm(manifest)

    report = BenchReport(
        model=str(model),
        model_sha256=actual_sha,
        config={
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_threads": n_threads,
            "max_tokens": max_tokens,
            "chat_format": chat_format,
            "seed": seed,
            "system_chars": len(system),
            "prefill_cache": "session",
            "replay_phase": replay_phase,
        },
        machine=_machine_specs(),
    )

    # Phase fill — real decode.
    phase_fill, _sess = _run_phase(prompts, llm, manifest, system, label="fill")
    report.phase_fill = phase_fill
    if phase_fill.turns:
        report.cold_executed_tps = phase_fill.turns[0].executed_tps
        report.cold_executed_wps = phase_fill.turns[0].executed_wps
    report.warm_mean_executed_wps = (
        sum(t.executed_wps for t in phase_fill.turns[1:]) / max(len(phase_fill.turns) - 1, 1)
        if len(phase_fill.turns) > 1
        else 0.0
    )
    report.turns.extend(phase_fill.turns)
    report.total_tokens += phase_fill.total_tokens
    report.total_words += phase_fill.total_words
    report.session_wall_s += phase_fill.wall_s

    # Phase replay — re-run the identical transcript from a fresh session. Every
    # turn should hit muscle memory because (system, user, history) bytes match
    # exactly what was stored during the fill phase.
    if replay_phase:
        phase_replay, _sess2 = _run_phase(prompts, llm, manifest, system, label="replay")
        report.phase_replay = phase_replay
        report.total_tokens += phase_replay.total_tokens
        report.total_words += phase_replay.total_words
        report.session_wall_s += phase_replay.wall_s
        if phase_replay.turns:
            hits = sum(1 for t in phase_replay.turns if t.cache_hit)
            report.replay_cache_hit_rate = hits / len(phase_replay.turns)
            eff_list: list[float] = []
            for t in phase_replay.turns:
                if t.cache_hit and t.wall_s > 0:
                    eff_list.append(t.words / t.wall_s)
            if eff_list:
                report.replay_mean_effective_wps = sum(eff_list) / len(eff_list)
                report.replay_min_effective_wps = min(eff_list)
            if phase_replay.wall_s > 0:
                report.replay_session_effective_wps = (
                    phase_replay.total_words / phase_replay.wall_s
                )
        report.turns.extend(phase_replay.turns)

    if report.session_wall_s > 0 and report.total_words > 0:
        report.session_effective_wps = report.total_words / report.session_wall_s
        report.session_executed_wps = report.total_words / report.session_wall_s
    return report


def _grid_for_auto_tune(default_threads: int) -> list[tuple[int, int]]:
    """Return ``[(n_threads, n_batch), ...]`` to try during ``--auto-tune``.

    Keep this list deliberately small: on a 4-physical / 8-logical box each
    config still loads the model (slow); we want the tune to finish in a
    reasonable time (well under a minute for a cold bench per config).
    """
    try:
        import psutil  # noqa: PLC0415

        phys = int(psutil.cpu_count(logical=False) or default_threads)
        logi = int(psutil.cpu_count(logical=True) or default_threads)
    except Exception:
        phys, logi = default_threads, default_threads
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for t in (phys, max(phys - 1, 1), logi):
        for b in (512, 1024):
            k = (t, b)
            if k in seen:
                continue
            seen.add(k)
            candidates.append(k)
    return candidates


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-batch", type=int, default=512)
    p.add_argument("--n-threads", type=int, default=0, help="0 = physical-core auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chat-format", default="")
    p.add_argument(
        "--system",
        default="You are a terse assistant. Answer in one short sentence.",
    )
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--auto-tune", action="store_true")
    p.add_argument(
        "--replay-phase",
        action="store_true",
        default=True,
        help="Run a second 'replay' session after the fill session (default on).",
    )
    p.add_argument(
        "--no-replay-phase",
        dest="replay_phase",
        action="store_false",
        help="Disable the replay phase (executed-only bench).",
    )
    p.add_argument(
        "--replay-fail-under",
        type=float,
        default=1000.0,
        help="CI gate: min acceptable replay_session_effective_wps (default 1000).",
    )
    p.add_argument(
        "--clear-mm",
        dest="clear_mm",
        action="store_true",
        default=True,
        help="Wipe $NRL_ROOT/cache/mm for this model before fill (default on).",
    )
    p.add_argument(
        "--keep-mm",
        dest="clear_mm",
        action="store_false",
        help="Keep existing muscle-memory entries (fill phase may hit cache).",
    )
    p.add_argument("--output", default=str(_ROOT / "build" / "wps_chat_bench.json"))
    p.add_argument("--markdown", default=None)
    p.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="If >0: exit 1 when session_effective_wps is below this floor.",
    )
    return p.parse_args(argv)


def _print_specs(model: Path, args: argparse.Namespace, auto_tune_grid: list[tuple[int, int]] | None) -> None:
    specs = _machine_specs()
    print("=== [wps-chat-bench] specs ===")
    print(f"  model:              {model}")
    print(f"  platform:           {specs.get('platform')}")
    print(f"  physical / logical: {specs.get('physical_cores')} / {specs.get('logical_cores')}")
    print(f"  ram_gb:             {specs.get('ram_gb')}")
    print(f"  llama-cpp-python:   {specs.get('llama_cpp_version')}")
    print(f"  n_ctx / n_batch:    {args.n_ctx} / {args.n_batch}")
    print(f"  n_threads (manual): {args.n_threads if args.n_threads > 0 else 'auto (physical)'}")
    print(f"  max_tokens / seed:  {args.max_tokens} / {args.seed}")
    if auto_tune_grid:
        print(f"  auto_tune_grid:     {auto_tune_grid}")
    print("")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model = Path(args.model).expanduser()
    if not model.is_file():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2

    chat_format = args.chat_format or _detect_chat_format(model)
    prompts = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)

    # Default n_threads: physical cores if known, else min(4, logical).
    default_threads = args.n_threads
    if default_threads <= 0:
        try:
            import psutil  # noqa: PLC0415

            default_threads = int(psutil.cpu_count(logical=False) or 4)
        except Exception:
            default_threads = min(4, os.cpu_count() or 4)

    tune_grid = _grid_for_auto_tune(default_threads) if args.auto_tune else None
    _print_specs(model, args, tune_grid)

    def _one(threads: int, batch: int, *, label: str) -> BenchReport:
        print(f"--- running config: n_threads={threads} n_batch={batch} ({label})")
        r = _run_transcript(
            model,
            prompts,
            n_ctx=args.n_ctx,
            n_batch=batch,
            n_threads=threads,
            max_tokens=args.max_tokens,
            chat_format=chat_format,
            seed=args.seed,
            system=args.system,
            replay_phase=args.replay_phase,
            clear_mm=args.clear_mm,
        )
        print(
            f"    cold_wps={r.cold_executed_wps:.2f}  warm_wps_mean={r.warm_mean_executed_wps:.2f}  "
            f"replay_session_wps={r.replay_session_effective_wps:.2f}  "
            f"replay_min_wps={r.replay_min_effective_wps:.2f}  "
            f"replay_cache_hit_rate={r.replay_cache_hit_rate:.0%}  "
            f"wall={r.session_wall_s:.1f}s"
        )
        return r

    best: BenchReport | None = None
    attempts: list[dict[str, Any]] = []
    if tune_grid:
        for (t, b) in tune_grid:
            rep = _one(t, b, label="auto-tune")
            attempts.append({"n_threads": t, "n_batch": b, "cold_wps": rep.cold_executed_wps,
                             "warm_wps_mean": rep.warm_mean_executed_wps,
                             "session_effective_wps": rep.session_effective_wps})
            if best is None or rep.cold_executed_wps > best.cold_executed_wps:
                best = rep
        assert best is not None
        final = best
    else:
        final = _one(default_threads, args.n_batch, label="single-config")

    payload: dict[str, Any] = {
        "schema": "nrl.wps_chat_bench.v1",
        "best_config": final.config,
        "report": final.to_dict(),
        "auto_tune_attempts": attempts,
        "prompts": prompts,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[wps-chat-bench] wrote {out}")

    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# WPS chat bench",
            "",
            f"- model: `{final.model}`",
            f"- model_sha256: `{final.model_sha256[:16]}...`",
            f"- config: `{final.config}`",
            f"- machine: `{final.machine}`",
            "",
            "## Headline WPS",
            "",
            f"- cold_executed_tps: `{final.cold_executed_tps:.2f}`",
            f"- cold_executed_wps: `{final.cold_executed_wps:.2f}`    (raw libllama CPU floor)",
            f"- warm_mean_executed_wps: `{final.warm_mean_executed_wps:.2f}`",
            f"- replay_session_effective_wps: `{final.replay_session_effective_wps:.2f}`",
            f"- replay_min_effective_wps: `{final.replay_min_effective_wps:.2f}`",
            f"- replay_cache_hit_rate: `{final.replay_cache_hit_rate:.0%}`",
            f"- session_effective_wps (fill+replay combined): `{final.session_effective_wps:.2f}`",
            f"- session_wall_s: `{final.session_wall_s:.2f}`",
            "",
            "## Turns",
            "",
            "| phase | idx | prompt | tokens | wall_s | executed_wps | cache_hit |",
            "|---|---|---|---|---|---|---|",
        ]
        if final.phase_fill is not None:
            for t in final.phase_fill.turns:
                lines.append(
                    f"| fill | {t.idx} | `{t.prompt[:30]}` | {t.tokens} | "
                    f"{t.wall_s:.2f} | {t.executed_wps:.2f} | {t.cache_hit} |"
                )
        if final.phase_replay is not None:
            for t in final.phase_replay.turns:
                wps_cell = (t.words / t.wall_s) if t.wall_s > 0 else 0.0
                lines.append(
                    f"| replay | {t.idx} | `{t.prompt[:30]}` | {t.tokens} | "
                    f"{t.wall_s:.4f} | {wps_cell:.2f} | {t.cache_hit} |"
                )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[wps-chat-bench] wrote {md}")

    print(
        "\n[wps-chat-bench] summary:\n"
        f"  cold_executed_wps              = {final.cold_executed_wps:.2f}   (silicon floor)\n"
        f"  warm_mean_executed_wps         = {final.warm_mean_executed_wps:.2f}\n"
        f"  replay_session_effective_wps   = {final.replay_session_effective_wps:.2f}   (muscle memory)\n"
        f"  replay_min_effective_wps       = {final.replay_min_effective_wps:.2f}\n"
        f"  replay_cache_hit_rate          = {final.replay_cache_hit_rate:.0%}\n"
        f"  session_effective_wps (overall)= {final.session_effective_wps:.2f}"
    )

    failed = False
    if args.replay_phase and args.replay_fail_under > 0:
        if final.replay_session_effective_wps < args.replay_fail_under:
            print(
                f"[wps-chat-bench] FAIL: replay_session_effective_wps={final.replay_session_effective_wps:.2f} "
                f"< --replay-fail-under={args.replay_fail_under}",
                file=sys.stderr,
            )
            failed = True
    if args.fail_under > 0 and final.session_effective_wps < args.fail_under:
        print(
            f"[wps-chat-bench] FAIL: session_effective_wps={final.session_effective_wps:.2f} "
            f"< --fail-under={args.fail_under}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
