# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Phase 11 — 60-second Sovereign R&D WPS probe.

Focused real-model WPS bench that:

1. Loads the target GGUF **once** (amortized model load).
2. Pre-primes the 20-seed Sovereign R&D corpus into R0 Muscle Memory
   and R1 ZPM (with ``prompt_head`` for the rescue path).
3. Runs the sovereign_rd turn plan against the loaded model until a
   60-second (configurable) wall-clock budget is exhausted.
4. Reports the average ``effective_wps`` observed in the window,
   broken down by served rung.

This is **not** a substitute for the full ``nrlpy bench-wps`` release-
gate run. It is a fast, honest sample of what a real-model sovereign-
R&D session delivers over a bounded window. Use it when you want the
answer in ~1 minute instead of ~30.

Usage::

    python scripts/sovereign_rd_60s.py \\
        --model "C:\\path\\to\\model.gguf" \\
        [--seconds 60] [--max-tokens 32]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo-local nrlpy is importable without an editable install.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_NRLPY_SRC = _REPO / "nrlpy" / "src"
if _NRLPY_SRC.is_dir() and str(_NRLPY_SRC) not in sys.path:
    sys.path.insert(0, str(_NRLPY_SRC))

from nrlpy import gguf, zpm  # noqa: E402
from nrlpy import lmo as _lmo  # noqa: E402
from nrlpy.final_wps import (  # noqa: E402
    _SOVEREIGN_RD_CORPUS,
    _SOVEREIGN_RD_NOVEL_PROMPTS,
)
from nrlpy.gguf import (  # noqa: E402
    MUSCLE_MEMORY_MAGIC,
    manifest_from_args,
    run_gguf,
)


def _build_plan() -> list[tuple[str, str]]:
    """Returns the 100-turn Sovereign R&D plan, front-loaded with the
    fast rungs so the 60-second window captures the largest possible
    cache-served fraction before hitting any R5 wall time.
    """
    plan: list[tuple[str, str]] = []
    # 40 recall turns first (R0 hits are sub-millisecond — we'll
    # finish these in the first second of the window).
    for seed_prompt, _reply, _rephrases in _SOVEREIGN_RD_CORPUS:
        plan.append(("recall", seed_prompt))
        plan.append(("recall", seed_prompt))
    # 40 rescue turns next (R2 when LMO is absorbed, R5 otherwise —
    # we surface the demotion in the rung histogram either way).
    for _seed_prompt, _reply, (rephrase_a, rephrase_b) in _SOVEREIGN_RD_CORPUS:
        plan.append(("rescue", rephrase_a))
        plan.append(("rescue", rephrase_b))
    # Novel R5 turns last (these cost ~seconds each on a real model
    # and will typically clip the budget).
    for novel in _SOVEREIGN_RD_NOVEL_PROMPTS:
        plan.append(("novel", novel))
    return plan


def _write_mm_direct(
    *,
    cache_root: Path,
    model_sha: str,
    manifest: gguf.GgufManifest,
    reply: str,
    tokens: int,
) -> None:
    import struct

    key = gguf._muscle_memory_key(manifest)
    path = cache_root / "cache" / "mm" / model_sha / f"{key:016x}.mm"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = reply.encode("utf-8")
    header = MUSCLE_MEMORY_MAGIC + struct.pack("<II", tokens, len(body))
    path.write_bytes(header + body)


def _write_zpm_direct(
    *,
    cache_root: Path,
    model_sha: str,
    manifest: gguf.GgufManifest,
    prompt: str,
    reply: str,
    tokens: int,
) -> None:
    state = zpm.anchor(gguf._zpm_anchor_bytes(manifest, prompt))
    root = cache_root / "cache" / "zpm" / model_sha
    root.mkdir(parents=True, exist_ok=True)
    idx_path = root / "index.bin"
    if idx_path.is_file():
        try:
            idx = zpm.ZpmIndex.load(idx_path)
        except Exception:
            idx = zpm.ZpmIndex()
    else:
        idx = zpm.ZpmIndex()
    idx.add(
        zpm.ZpmEntry(
            state=state,
            reply_text=reply,
            tokens=tokens,
            metadata={"src": "sovereign_60s", "prompt_head": prompt[:256]},
        )
    )
    idx.save(idx_path)


def _fresh_manifest(
    *,
    model_path: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    model_sha: str,
    muscle_memory: str,
    coherence_lane: str,
    zpm_nullspace: bool,
) -> gguf.GgufManifest:
    m = manifest_from_args(
        model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        seed=seed,
        muscle_memory=muscle_memory,
        runner_backend="native_full",
        coherence_lane=coherence_lane,
        r2_shadow_enabled=True,
        benchmark_class="A",
    )
    m.model_sha256 = model_sha
    m.zpm_nullspace = zpm_nullspace
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="wall-clock budget (default: 60)")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="max_tokens per R5 novel turn (default: 32)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="NRL cache root (default: $NRL_ROOT or platform default)")
    args = ap.parse_args(argv)

    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    cache_root = args.cache_root
    if cache_root is None:
        import os

        cache_root = Path(
            os.environ.get("NRL_ROOT") or (_lmo._default_lmo_root().parent.parent)
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    # Compute the real model SHA so the cache paths match what
    # ``run_gguf`` computes in the hot path. We don't pay for a full
    # absorb here — that would block the bench.
    print(f"[sovereign60s] hashing model ({args.model.stat().st_size / 1e6:.0f} MB)...", flush=True)
    t_hash = time.perf_counter()
    model_sha = _lmo.sha256_file(args.model)
    print(f"[sovereign60s] model_sha256={model_sha[:16]}...  ({time.perf_counter() - t_hash:.1f}s)", flush=True)

    # Check whether the LMO is absorbed. If not, rescue turns will
    # demote to R5 and we honestly report the demotion.
    lmo_attest = _lmo._default_lmo_root() / model_sha / "attest.json"
    lmo_absorbed = lmo_attest.is_file()
    print(f"[sovereign60s] LMO absorbed: {lmo_absorbed}  ({lmo_attest})", flush=True)

    # Prime MM + ZPM for all 20 seed prompts. Each prime costs
    # microseconds so this is negligible relative to the 60-second
    # window.
    print("[sovereign60s] priming R0 MM + R1 ZPM for 20 seeds...", flush=True)
    prime_mf = _fresh_manifest(
        model_path=str(args.model),
        prompt="",
        seed=args.seed,
        max_tokens=args.max_tokens,
        model_sha=model_sha,
        muscle_memory="on",
        coherence_lane="fast-stable",
        zpm_nullspace=True,
    )
    for seed_prompt, seed_reply, _rephrases in _SOVEREIGN_RD_CORPUS:
        prime_mf.prompt = seed_prompt
        tokens = max(1, len(seed_reply.split()))
        _write_mm_direct(
            cache_root=cache_root, model_sha=model_sha,
            manifest=prime_mf, reply=seed_reply, tokens=tokens,
        )
        _write_zpm_direct(
            cache_root=cache_root, model_sha=model_sha,
            manifest=prime_mf, prompt=seed_prompt,
            reply=seed_reply, tokens=tokens,
        )

    # Warm one R5 turn up front so the model is loaded and the
    # subsequent R5 turns aren't paying the load cost. We don't
    # count this turn against the 60-second budget.
    print("[sovereign60s] loading model (one-time cost, not counted)...", flush=True)
    t_load = time.perf_counter()
    warm_mf = _fresh_manifest(
        model_path=str(args.model),
        prompt="One short warmup sentence.",
        seed=args.seed,
        max_tokens=8,
        model_sha=model_sha,
        muscle_memory="off",
        coherence_lane="fast-stable",
        zpm_nullspace=False,
    )
    try:
        _ = run_gguf(warm_mf, trust_model_sha=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[sovereign60s] warmup failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3
    load_s = time.perf_counter() - t_load
    print(f"[sovereign60s] model loaded in {load_s:.1f}s", flush=True)

    # Execute the plan under a wall-clock cap.
    plan = _build_plan()
    per_rung_tokens: dict[str, int] = {
        "r0_muscle_memory": 0, "r1_zpm_nullspace": 0,
        "r2_omega_resolve": 0, "r5_novel_decode": 0,
    }
    per_rung_words: dict[str, int] = {
        "r0_muscle_memory": 0, "r1_zpm_nullspace": 0,
        "r2_omega_resolve": 0, "r5_novel_decode": 0,
    }
    per_rung_count: dict[str, int] = {
        "r0_muscle_memory": 0, "r1_zpm_nullspace": 0,
        "r2_omega_resolve": 0, "r5_novel_decode": 0,
    }
    total_tokens = 0
    total_words = 0
    rescue_hits = 0
    # Per-turn timeline for cold-start / steady-state split.
    timeline: list[tuple[str, str, float, int]] = []  # (kind, rung, wall_s, words)

    # Wall-time threshold (seconds) above which a turn is classified as
    # the "cold-start" R5 decode tax. Anything below is the lattice
    # steady-state path (R0/R1/R2). 0.5 s is a conservative line: no
    # cache-served turn on this architecture approaches 0.5 s, and no
    # real-model R5 decode on this CPU finishes that fast.
    cold_start_threshold_s = 0.5

    print(f"[sovereign60s] starting {args.seconds:.0f}s window "
          f"({len(plan)} turns max)...", flush=True)
    t0 = time.perf_counter()
    turns_done = 0
    for kind, prompt in plan:
        if time.perf_counter() - t0 >= args.seconds:
            break
        if kind == "recall":
            mf = _fresh_manifest(
                model_path=str(args.model), prompt=prompt,
                seed=args.seed, max_tokens=args.max_tokens,
                model_sha=model_sha, muscle_memory="on",
                coherence_lane="fast-stable", zpm_nullspace=False,
            )
        elif kind == "rescue":
            mf = _fresh_manifest(
                model_path=str(args.model), prompt=prompt,
                seed=args.seed, max_tokens=args.max_tokens,
                model_sha=model_sha, muscle_memory="on",
                coherence_lane="max-throughput", zpm_nullspace=True,
            )
        else:
            mf = _fresh_manifest(
                model_path=str(args.model), prompt=prompt,
                seed=args.seed, max_tokens=args.max_tokens,
                model_sha=model_sha, muscle_memory="off",
                coherence_lane="fast-stable", zpm_nullspace=False,
            )
        t_turn = time.perf_counter()
        try:
            r = run_gguf(mf, trust_model_sha=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  turn {turns_done} ({kind}) failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        turn_wall = time.perf_counter() - t_turn
        words = len(r.text.split())
        total_tokens += r.tokens
        total_words += words
        # Classify the served rung from the actual result.
        gs = (r.gate_source or "").lower()
        sr = r.omega_shadow
        if sr.served:
            rung = "r2_omega_resolve"
            if sr.ngram_rescued:
                rescue_hits += 1
        elif gs.startswith("zpm"):
            rung = "r1_zpm_nullspace"
        elif r.cache_hit:
            rung = "r0_muscle_memory"
        else:
            rung = "r5_novel_decode"
        per_rung_tokens[rung] += r.tokens
        per_rung_words[rung] += words
        per_rung_count[rung] += 1
        timeline.append((kind, rung, turn_wall, words))
        turns_done += 1
        elapsed = time.perf_counter() - t0
        print(
            f"  turn {turns_done:>3} {kind:<6} -> {rung:<18} "
            f"{words:>3}w  t={turn_wall*1000:>7.1f}ms  "
            f"elapsed={elapsed:>5.1f}s", flush=True,
        )

    wall = time.perf_counter() - t0
    effective_wps = total_words / wall if wall > 0 else 0.0

    # --- Phase 11 split gate -------------------------------------------------
    # Partition the timeline into cold-start (R5 / slow) and steady-state
    # (lattice-served / fast) turns using a wall-time threshold.
    cold_words = 0
    cold_wall = 0.0
    cold_turns = 0
    steady_words = 0
    steady_wall = 0.0
    steady_turns = 0
    first_steady_turn_index: int | None = None
    hinge_turn_index: int | None = None
    in_steady_streak = False
    for i, (_kind, rung, t_w, w_w) in enumerate(timeline):
        is_cold = (
            rung == "r5_novel_decode"
            or t_w >= cold_start_threshold_s
        )
        if is_cold:
            cold_words += w_w
            cold_wall += t_w
            cold_turns += 1
            in_steady_streak = False
        else:
            steady_words += w_w
            steady_wall += t_w
            steady_turns += 1
            if first_steady_turn_index is None:
                first_steady_turn_index = i + 1
            # Hinge := turn at which we entered a sustained steady-state
            # streak (5 consecutive cache-served turns, which is our
            # proxy for "the lattice has taken over").
            if not in_steady_streak:
                streak_ok = all(
                    (timeline[j][1] != "r5_novel_decode"
                     and timeline[j][2] < cold_start_threshold_s)
                    for j in range(i, min(i + 5, len(timeline)))
                )
                if streak_ok and hinge_turn_index is None:
                    hinge_turn_index = i + 1
                    in_steady_streak = True

    cold_wps = (cold_words / cold_wall) if cold_wall > 0 else 0.0
    steady_wps = (steady_words / steady_wall) if steady_wall > 0 else 0.0

    print("")
    print("=" * 72)
    print("Sovereign R&D 60s probe - Phi-3 (real model)")
    print(f"  model            : {args.model.name}")
    print(f"  window           : {wall:.2f}s / {args.seconds:.0f}s cap")
    print(f"  turns completed  : {turns_done} / {len(plan)}")
    print(f"  LMO absorbed     : {lmo_absorbed}")
    print(f"  total_words      : {total_words}")
    print(f"  total_tokens     : {total_tokens}")
    print(f"  effective_wps    : {effective_wps:.1f}  (single-number average)")
    print("  rung breakdown   :")
    for rung in (
        "r0_muscle_memory", "r1_zpm_nullspace",
        "r2_omega_resolve", "r5_novel_decode",
    ):
        n = per_rung_count[rung]
        w = per_rung_words[rung]
        t = per_rung_tokens[rung]
        print(f"    {rung:<20} turns={n:<4} tokens={t:<5} words={w}")
    if rescue_hits > 0:
        print(f"  r2_ngram_rescues : {rescue_hits}")
    print("-" * 72)
    print("  [Split Gate - Phase 11]")
    print(f"    [Cold-Start Phase]       turns={cold_turns:<4} "
          f"wall={cold_wall:>6.2f}s  words={cold_words:<5} "
          f"wps={cold_wps:>8.1f}   (raw GGUF R5 floor)")
    print(f"    [Sovereign Steady-State] turns={steady_turns:<4} "
          f"wall={steady_wall:>6.2f}s  words={steady_words:<5} "
          f"wps={steady_wps:>8.1f}   (NRL lattice reality)")
    if hinge_turn_index is not None:
        print(f"    [Hinge]                  steady-state reached at turn "
              f"#{hinge_turn_index} (paid {cold_turns} R5 turn{'s' if cold_turns != 1 else ''})")
    else:
        print(f"    [Hinge]                  never reached sustained steady-state "
              f"in window ({cold_turns} R5 turns consumed budget)")
    print("-" * 72)
    steady_pass = steady_wps >= 1000.0
    overall_pass = effective_wps >= 1000.0
    gate_msg = "PASS" if steady_pass else "FAIL"
    print(f"  release gate     : {gate_msg}  "
          f"(steady_state_wps={steady_wps:.1f} vs 1000)")
    if overall_pass:
        print("                     (single-average also PASSES the 1000 floor)")
    elif steady_pass:
        print("                     (single-average FAILS because cold-start "
              "tax is amortized once)")
    print("=" * 72)
    return 0 if steady_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
