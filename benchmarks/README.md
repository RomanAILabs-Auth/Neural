<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL benchmarks

Benchmark governance for NRL: locked profiles, reproducible artifacts, and explicit separation of **executed** vs **virtual** throughput where System 1 lanes apply.

## Principles

| Rule | Rationale |
|------|-----------|
| Locked profiles for public claims | Avoid cherry-picked ad-hoc CLI snapshots. |
| Machine-readable artifacts | CI, reviewers, and partners can diff JSON/Markdown outputs. |
| Deterministic vs adaptive classes | Sovereign baselines must not be silently conflated with virtual lanes. |
| Median / repeat policy | Follow harness defaults; document changes in commit messages. |

## Harnesses

| Harness | Purpose | Artifacts (after run) |
|---------|---------|------------------------|
| Native `nrl bench` | Profile matrix on this host | Console; copy into `initial_results.md` only with context |
| [`nrl_vs_cpp.py`](./nrl_vs_cpp.py) | Apples-to-apples vs C++ `-O0` / `-O3` INT4 reference | `build/bench/nrl_vs_cpp.json`, `build/bench/nrl_vs_cpp.md` |
| [`gguf_golden.py`](./gguf_golden.py) | GGUF runner honesty-contract gate. Modes: `stub` (always; P1 / P2-Shadow invariants), `p2active-sim` (always; flipped hinge from numeric override, labeled simulation), `p2active-prefill` (always; flipped hinge from structural `PrefillGate` — 2-turn shared-prefix proof, 9 assertions), `real` (when `NRL_GGUF_GOLDEN_MODEL` is set). Stub verifies determinism + honesty; p2active-sim verifies `virtual_tps > executed_tps` + formula match + simulation label; p2active-prefill verifies the same plus `gate_source == "prefill_cache"` + non-claim of simulation label. | `build/gguf_golden/gguf_golden.json`, `build/gguf_golden/gguf_golden.md` |

Examples:

```bash
python benchmarks/nrl_vs_cpp.py --neurons 1048576 --iterations 4096 --reps 6 --threshold 8
python benchmarks/gguf_golden.py --mode stub                            # no deps, no weights
python benchmarks/gguf_golden.py --mode p2active-sim                    # flipped hinge via numeric override (simulation)
python benchmarks/gguf_golden.py --mode p2active-prefill                # flipped hinge via structural PrefillGate (2 turns)
python benchmarks/gguf_golden.py --mode real --model path/to/model.gguf # real libllama
```

The `build/bench/` and `build/gguf_golden/` paths are gitignored until generated — commit **policy and harness**, not stale numbers, unless you intentionally snapshot results for a release tag.

## Reference snapshots

- Narrative and tables: [`initial_results.md`](./initial_results.md)

## See also

- Root [README](../README.md) — benchmark table and governance summary  
- [`nrl-architecture.md`](../nrl-architecture.md) — ZPM / Omega accounting semantics  
