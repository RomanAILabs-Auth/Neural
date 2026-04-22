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

Example:

```bash
python benchmarks/nrl_vs_cpp.py --neurons 1048576 --iterations 4096 --reps 6 --threshold 8
```

The `build/bench/` paths are gitignored until generated — commit **policy and harness**, not stale numbers, unless you intentionally snapshot results for a release tag.

## Reference snapshots

- Narrative and tables: [`initial_results.md`](./initial_results.md)

## See also

- Root [README](../README.md) — benchmark table and governance summary  
- [`nrl-architecture.md`](../nrl-architecture.md) — ZPM / Omega accounting semantics  
