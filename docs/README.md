<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL documentation index

Start with the architecture doc, then follow the pointer for what you're doing.

## Core contracts

| Document | What it's for |
|---|---|
| [`../nrl-architecture.md`](../nrl-architecture.md) | Full ABI + lattice contracts (§15 accounting, execution profiles, dispatch rules). **Every other doc is downstream of this one.** |
| [`../language/spec/minimal_nrl_v0.md`](../language/spec/minimal_nrl_v0.md) | `.nrl` v0 manifest grammar. Ships in the native `nrl file` parser. |
| [`../language/spec/nrl_manifest_v1.md`](../language/spec/nrl_manifest_v1.md) | `.nrl` v1 schema-gated extension for GGUF inference (`mode = gguf_run`). Parsed in `nrlpy.gguf.load_manifest`. Runtime knobs: `NRL_INFERENCE`, `NRL_KV_CACHE`, `NRL_NO_REPACK`, `NRL_CTX`, `NRL_STREAM_CHUNK_MS`, `NRL_ROOT`. |

## Subsystem specs

| Document | What it covers |
|---|---|
| [`nrl_gguf_runner_architecture.md`](./nrl_gguf_runner_architecture.md) | `llama.nrl` — three-plane architecture (L0 numerics / L1 gating / L2 orchestrator), the four-metric TPS contract (`executed_tps` / `virtual_tps` / `cache_tps` / `effective_tps`), and phased delivery (P1 shipped, P2-Shadow shipped, P2-Active spike, P3 KV+expert gates, P4 native C runner). **§1.0 contains the honesty hinge** — `virtual_tps == executed_tps` until a gate actually elides libllama work. |
| [`nrl_immune_system_spec.md`](./nrl_immune_system_spec.md) | Governance + runtime safety contract. Evidence-log schemas live here. |
| [`nrl_alive_language_evolution_architecture.md`](./nrl_alive_language_evolution_architecture.md) | Long-horizon vision for language-driven adaptation and bounded plasticity. |
| [`PRIOR_ART.md`](./PRIOR_ART.md) | What we adopted from `RomanAI-4D-GGUF-Engine` and `Ghost_Compressor` (operational knobs, pacing, backend selector) and what we refused (claims inconsistent with §15 honest accounting). |
| [`PRODUCTION_READINESS.md`](./PRODUCTION_READINESS.md) | Pre-release checklist. Referenced by `scripts/live_readiness.*`. |

## Schemas

| Path | Purpose |
|---|---|
| [`schemas/`](./schemas/) | JSON schemas for evidence-log entries (`nrl.gguf_run.v1`, control audit, etc.). |

## Quick navigation

- **"I want to run a GGUF model"** → [`nrl_gguf_runner_architecture.md`](./nrl_gguf_runner_architecture.md) §P1, then `../README.md#run-a-gguf-model` for the command line.
- **"I want to chat multi-turn with a model"** → same runner architecture doc §P1, then `python -m nrlpy chat <model>.gguf` (`/help` inside the REPL for slash-commands).
- **"What do the four TPS metrics mean?"** → [`nrl_gguf_runner_architecture.md`](./nrl_gguf_runner_architecture.md) §1.
- **"When does `virtual_tps` exceed `executed_tps`?"** → [`nrl_gguf_runner_architecture.md`](./nrl_gguf_runner_architecture.md) §1.0 (the honesty hinge). Short answer: only under P2-Active or later.
- **"What about the `omega-hybrid skip_ratio` I see in the banner?"** → advisory lattice observation (P2-Shadow). See §P1 "P2-Shadow additions" and the `NrlLatticeObservation` section.
- **"I need to publish a TPS number"** → §7 of the runner doc is the honest-accounting checklist. Run [`../benchmarks/gguf_golden.py`](../benchmarks/gguf_golden.py) (stub or `--mode real`) or [`../scripts/live_readiness_gguf.ps1`](../scripts/live_readiness_gguf.ps1) / [`.sh`](../scripts/live_readiness_gguf.sh) — both emit `build/gguf_golden/gguf_golden.{json,md}` and assert the hinge.
- **"I want to contribute code"** → `../CONTRIBUTING.md` + `../nrl-architecture.md` §5 (engineering gates: ruff, mypy --strict, parity locks).

## Phase status at a glance

| Phase | What it ships | Status |
|---|---|---|
| P1 | Dense GGUF pass-through via `libllama`; four-metric TPS; muscle-memory; evidence log; manifest v1 | **shipped** |
| P2-Shadow | Background `omega-hybrid` lattice probe reported as advisory; never inflates TPS | **shipped** |
| P2-Active (simulation hinge) | `manifest.gate_skip_ratio_override` / `NRL_GATE_SKIP_RATIO_OVERRIDE` flips the math end-to-end behind an explicit simulation label; enforced by `benchmarks/gguf_golden.py --mode p2active-sim` | **shipped (dev / CI only — not a performance claim)** |
| P2-Active (prefill cache — first structural gate) | `nrlpy.gates.PrefillGate` + manifest `prefill_cache = session` drives `gate_skip_ratio` from real shared-prefix reuse; maps to libllama's free `n_past` carry-over on native backend; enforced by `benchmarks/gguf_golden.py --mode p2active-prefill` (9-assertion two-turn proof) | **shipped** |
| P2-Active (layer-skip callback) | Patched libllama layer-skip hook behind a feature flag; the `gate_source = "layer_skip"` lane already reserved in the accounting layer | planned |
| P3 | KV gate + expert gate for MoE models; muscle-memory write path on by default | planned |
| P4 | Native C runner linking `libllama` directly from `engine/src/` | planned, guarded on P1–P3 showing the Python orchestrator is the bottleneck |
