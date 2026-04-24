<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL GGUF Runner — `llama.nrl` Architecture

**Status:** P1 design + implementation landed. P2 (layer gate) and P3 (KV + expert gates) are scoped below.
**Supersedes:** the twenty-odd `examples/nrl_gguf_runner_v*.py`, `examples/romanai_nrl_*.py`, and `examples/*_phi3*.py` experiments, which are kept only as history.
**Tone contract:** this document is bound by [`nrl-architecture.md`](../nrl-architecture.md) §0.1–§0.5 (honest accounting), §1.5 (sandboxed control), §5.3 (publication policy), and §15 (virtual throughput note).

---

## 0. Identity

> **NRL is the execution supervisor and routing lattice. `libllama` is the numerics library it supervises.**

There is no transformer GEMM inside any `nrl_v1_*` kernel, and there will not be one. That is not a limitation — it is the only framing under which:

- `executed_gops` / `virtual_gops` / `skip_ratio` retain their [§15](../nrl-architecture.md) semantics for text generation.
- `benchmark_class = A` (deterministic, replay-locked) can be published with a straight face.
- The existing `nrl_v1_*` ABI stays frozen — the GGUF runner is **additive only**.

Three planes, bottom-up:

| Plane | Owns | Code |
|---|---|---|
| **L0 — Numerics** | GGUF parse, dequant, GEMM, attention, sampler | `libllama` via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) (P1), later a C bridge (P4) |
| **L1 — Gating lattice** | Per-token layer / expert / KV-keep masks, deterministic | `nrl_v1_gate_*` kernels reusing `braincore_int4` + `zpm_omega_router` primitives (P2, P3) |
| **L2 — Orchestrator** | `.nrl` manifest, control hints, muscle memory, telemetry, evidence JSONL | [`nrlpy.gguf`](../nrlpy/src/nrlpy/gguf.py), extensions to `engine/src/main.c` (P4) |

P1 ships **L0 + L2** end-to-end with the L1 slot zeroed but structurally present, so the telemetry format is final.

---

## 1. Four-metric TPS contract

llama.cpp and every other GGUF runner publishes one number: `tokens/sec`. NRL publishes four, because NRL's own contract ([§15](../nrl-architecture.md)) already splits materialized from accounted work. Applied to decoding:

| Metric | Definition | What it answers |
|---|---|---|
| `executed_tps` | Fresh tokens materialized via `libllama` during wall window | "How many tokens hit the screen per second?" |
| `virtual_tps` | `executed_tps / (1 − gate_skip_ratio)`. `gate_skip_ratio` is the **applied** skip ratio — the fraction of layer work the active gate caused libllama to elide. `0.0` in P1 and P2-Shadow, so `virtual_tps == executed_tps` on real runs until P2-Active lands. | "If the model were not layer-gated, how many baseline-equivalent tokens/sec would this wall-clock rate imply?" |
| `cache_tps` | Muscle-memory replay tokens / cache-read wall seconds | "On repeat workloads, how fast do we ship cached sequences?" |
| `effective_tps` | `executed_tps + cache_tps` across a session window | "Aggregate tokens delivered per second, materialized + replayed, honestly labeled" |

**Banner policy (CLI).** `nrl run model.gguf` prints **three independent blocks** — decode TPS, NRL attestation, NRL lattice observation — that are never cross-cited in the math. The headline is `max(executed_tps, effective_tps, virtual_tps, cache_tps)` with the metric name *always* printed next to the number. Example P1/P2-Shadow run:

```text
decode TPS
  headline      executed_tps      62.40   (gate_skip_ratio=0.000 (P1/P2-Shadow: virtual_tps == executed_tps))
  executed_tps       62.40   (materialized, fresh tokens)
  virtual_tps        62.40   (executed / (1 - gate_skip_ratio))
  cache_tps           0.00   (muscle-memory replays)
  effective_tps      62.40   (executed + cache)

NRL attestation (engine-sanity probe, not decode TPS)
  available     yes
  skip_ratio    0.999756   (lattice, not libllama)
  virtual_gops     6953.161   (lattice §15)

NRL lattice observation (advisory; NOT applied to decode TPS until P2-Active)
  available     yes
  profile       omega-hybrid
  skip_ratio    0.250000   (gate preview, lattice work)
  note          advisory; not applied to token flow in P1 / P2-Shadow
```

No line ever appears without its metric name. Same discipline the demo already uses for `virtual_gops`; extended to TPS.

### 1.0 The honesty hinge (how the three blocks relate)

Two separate concepts, never conflated:

- **`TpsReport.gate_skip_ratio`** — applied. The fraction of libllama forward-pass work the gate *caused to be skipped*. `0.0` in P1 and P2-Shadow because no gate is wired into libllama yet. `virtual_tps = executed_tps / (1 - gate_skip_ratio)`, so `virtual_tps == executed_tps` until P2-Active.
- **`NrlLatticeObservation.skip_ratio`** — advisory. What the NRL lattice would skip under `omega-hybrid` on its *own* workload, measured in a background thread during decode. Never multiplied into decode TPS. Exists so operators can see the gate signal NRL produces, before we wire it into libllama in P2-Active.

The rule: **until a gate actually elides libllama work, `virtual_tps == executed_tps`**. The banner says so explicitly. No exception.

#### 1.0.1 Gate sources and resolution order

Four named sources can set `TpsReport.gate_skip_ratio`. Exactly one fires per run. Resolution order (first non-zero wins):

1. **`prefill_cache`** — structural shared-prefix policy (`nrlpy.gates.PrefillGate`). On native libllama this is real KV-cache reuse via `n_past` carry-over; on stub / cli backends the accounting is structural but the physical work is not elided (banner says so).
2. **`override`** — numeric fixture from `manifest.gate_skip_ratio_override` or env `NRL_GATE_SKIP_RATIO_OVERRIDE`. Development / CI only.
3. *(future)* **`layer_skip`** — callback from a patched libllama layer gate. Not yet wired.
4. *(none)* — P1 / P2-Shadow: `gate_skip_ratio = 0`, `virtual_tps == executed_tps`.

The resolved source is recorded in `GgufRunResult.gate_source`, emitted in every `nrl.gguf_run.v1` evidence-log event, and shown as a tag in the banner headline (e.g., `[P2-Active (prefill cache)]` vs `[P2-Active simulation (override)]`).

#### 1.0.2 The prefill-cache gate (first structural source)

`PrefillGate` is session-scoped state: on each turn it diffs the new prompt's token sequence against the previous one, reports `shared_prefix_len / prompt_token_count` as the skip ratio (clamped below 1.0 so the `virtual_tps` formula never divides by zero), and then observes the new tokens for the next turn. Turn 1 of a session always reports `0.0` (no history). Turn 2 onward reports any real shared prefix as an honest `gate_skip_ratio`.

On the native backend this gate reading corresponds to free libllama elision: `llama.cpp` automatically skips re-prefilling shared-prefix tokens when `n_past` is carried across `decode()` calls on the same context, so the structural skip ratio maps one-to-one to real wall-clock savings. On stub / cli backends the gate fires the same code path for CI — but the banner appends `note: on native libllama this is real KV-cache reuse; on stub/cli backends the accounting is structural`, and the evidence log records which backend was used.

The gate is enforced in CI by [`benchmarks/gguf_golden.py --mode p2active-prefill`](../benchmarks/gguf_golden.py), which runs two sequential stub-backend turns with a shared prefix and asserts nine invariants spanning turn-1 hinge preservation, turn-2 flipped hinge from the structural source, formula match, banner label correctness, and the non-claim invariant (a prefill-gate run must not wear the simulation label).

#### 1.0.3 The P2-Active simulation override (dev / CI only)

An explicit, named, opted-in escape hatch: `manifest.gate_skip_ratio_override` (or env `NRL_GATE_SKIP_RATIO_OVERRIDE`). When set to a value in `[0.0, 1.0)` and **no structural gate is firing**, it writes directly into `TpsReport.gate_skip_ratio` *after* decode, flipping `virtual_tps` above `executed_tps` by exactly `1 / (1 - override)`. This exists so the accounting math, banner layout, and evidence-log schema could be locked in CI before the prefill gate existed. Now that the prefill gate ships, the override remains useful only as (a) a fixed-point fixture for testing the math in isolation and (b) a smoke-test input for harnesses that can't easily construct multi-turn scenarios.

The override is **not** a performance claim. Every surface treats it as simulation:

- Banner prints `[P2-Active simulation (override)]` next to the headline and adds a trailing block showing `gate_skip_ratio_override` and the self-check `virtual_tps_formula_ok    yes`.
- Evidence log emits `gate_simulation_active: true` and the numeric override alongside the TPS block.
- Cache-hit replays do **not** apply the override.
- A live prefill-cache gate **always wins** over the override (structural > numeric fixture).

### 1.1 What `virtual_tps` is **not**

- It is not a claim that the model produced 1248 coherent tokens this second.
- It is not comparable to another runner's `tokens/sec` unless that runner also ships `skip_ratio` per decode step.
- It collapses to `executed_tps` under `sovereign`, exactly as `virtual_gops` collapses to `executed_gops` today.

These warnings are printed inline with the banner on every run. Non-negotiable.

### 1.2 What `virtual_tps` **is**

The token-level analog of `virtual_gops`. Exactly. If a layer gate elides 95% of per-token weight work, the model-equivalent throughput at the observed wall rate is 20× the materialized rate. That number has a name in your architecture doc, and it is now extended to tokens.

---

## 2. `.nrl` manifest v1 (schema-gated extension of v0)

Backward-compatible with [`language/spec/minimal_nrl_v0.md`](../language/spec/minimal_nrl_v0.md). v0 files with no `schema` line keep parsing under v0 rules. v1 activates when `schema = nrl.manifest.v1` is the first non-comment line.

New keys (all optional unless marked required):

```ini
schema = nrl.manifest.v1
mode   = gguf_run
profile = omega-hybrid

# --- L0 (numerics) ---
model        = models/phi-3-mini-4k.Q4_K_M.gguf       # required
model_sha256 =                                        # optional; if set, mismatch aborts
prompt       = "Hello. Tell me a short fact about space."
prompt_file  =                                        # mutually exclusive with prompt
max_tokens   = 128
temperature  = 0.7
top_p        = 0.9
top_k        = 40
repeat_penalty = 1.1
seed         = 42                                     # required for benchmark_class=A
n_ctx        = 2048
n_threads    = 0                                      # 0 = auto (os.cpu_count())
n_batch      = 512
chat_format  = none                                   # none|chatml|phi3|llama2

# --- L1 (gating lattice; P2+) ---
gate_layer_policy  = none                             # none|omega|omega-hybrid
gate_expert_policy = none
gate_kv_policy     = none
gate_min_active    = 4
gate_wake_rate     = 0.25

# --- L2 (orchestrator) ---
respect_control_hints = true
muscle_memory         = on                            # on|off|replay-only
evidence_log          =                               # defaults to build/immune/events.jsonl
telemetry_granularity = summary                       # summary|per_token
benchmark_class       = B                             # A=deterministic (requires seed+replay)
                                                      # B=adaptive (default; libllama may use clock)
```

The full grammar and parser contract live at [`language/spec/nrl_manifest_v1.md`](../language/spec/nrl_manifest_v1.md). The native parser extension for `cmd_file()` in `engine/src/main.c` is P4 work; in P1 the manifest is parsed in Python under [`nrlpy.gguf.load_manifest`](../nrlpy/src/nrlpy/gguf.py).

---

## 3. P1 — shipped in this commit

| Component | File | Status |
|---|---|---|
| Manifest v1 parser (Python) | [`nrlpy/src/nrlpy/gguf.py`](../nrlpy/src/nrlpy/gguf.py) | done |
| Muscle-memory (on-disk, FNV-1a64 keyed) | same | done |
| Dense runner via `llama-cpp-python` | same | done |
| Four-metric TPS reporter | same | done |
| Evidence-log emission (`nrl.gguf_run.v1` events) | same | done |
| Control-hints honored (`respect_control_hints`) | same | done |
| `nrlpy run model.gguf` and `nrlpy <path.gguf>` sugar | [`nrlpy/src/nrlpy/cli.py`](../nrlpy/src/nrlpy/cli.py) | done |
| `nrlpy gguf <manifest.nrl>` | same | done |
| Reference example (replaces v1–v5) | [`examples/nrl_run_gguf.py`](../examples/nrl_run_gguf.py) | done |
| Unit tests (mocked `libllama`) | [`nrlpy/tests/test_gguf.py`](../nrlpy/tests/test_gguf.py) | done |
| Reference manifests | [`language/examples/phi3_dense.nrl`](../language/examples/phi3_dense.nrl), [`phi3_omega_flex.nrl`](../language/examples/phi3_omega_flex.nrl) | done |

**Explicitly not in P1**, and reported as `0.0` / `none` in the telemetry so the format is final:

- `gate_skip_ratio` — always `0.0` (no active layer gate wired into libllama yet; `virtual_tps` == `executed_tps`).
- `gate_kv_skip_ratio` — always `0.0`.
- `experts_executed` / `experts_total` — always `0` (dense path).

**P2-Shadow additions (also shipped now):** advisory `NrlLatticeObservation` produced by a background `omega-hybrid` bench probe during decode. Never applied to decode TPS, never multiplied in. Exists so the operator sees the gate signal NRL would produce under a balanced policy, and so the telemetry format is final *before* P2-Active lands.

This is the honest P1 / P2-Shadow story: **the runner is a clean, high-TPS reference path with the full NRL telemetry contract bolted on**. The flex numbers unlock in P2-Active.

---

## 4. P2-Active — layer gate (next code-producing phase)

- New C module `engine/src/gate_int4.c` exposing `nrl_v1_gate_layers()` that takes a per-block residual-norm proxy and emits a packed-bit `layer_mask`.
- Reuses `zpm_omega_router.c` primitives: one sub-lattice per transformer block, wake/prune with `gate_min_active` as the `omega-hybrid` active floor.
- Determinism: `(model_sha256, seed, token_index, prev_residual_digest) → layer_mask` is a pure function. `benchmark_class = A` becomes legal.
- Python side: `nrlpy.gguf.LayerGate` wraps the kernel; `llama-cpp-python`'s `eval_tokens`/low-level decode path is patched to honor the mask. If patching is not feasible without a llama.cpp fork, P2 falls back to **early-exit gating** via `logits_all=True` + layer-cache clipping, which gives ~30–50% of the win honestly.

Realistic envelope on Phi-3 Mini 4K Q4_K_M, 8-core laptop:

- `executed_tps`: 40–90 (vs 30–60 dense) — real compute saved by actually skipping blocks
- `gate_skip_ratio`: 0.2 – 0.5 (coherence floor; more and quality drops)
- `virtual_tps`: 50 – 180 honestly
- With a 1.1B draft model for speculative decode on top: 100 – 300 `executed_tps`, `virtual_tps` correspondingly higher

## 5. P3 — KV + expert gates

- ZPM static-prefix collapse of KV-keep mask for long fixed system prompts (the single most legitimate "infinite TPS" story on repeat workloads with stable prefixes).
- Expert gate for MoE models (`gate_expert_policy = omega-hybrid`), routing on the model's own router logits bridged into packed INT4.
- Muscle-memory write path enabled by default.

## 6. P4 — native C runner

Only if P1–P3 show `llama-cpp-python` is the bottleneck. Moves the manifest parser into `cmd_file()` and links `libllama.so`/`llama.dll` directly from `engine/src/llama_bridge.c`. Until then, Python orchestration is not the bottleneck — `libllama` decode is, and that's in C already.

---

## 7. Honest accounting checklist (before publishing numbers)

Every time a TPS number leaves this repo, it must carry:

1. Command line (including the `.nrl` manifest path or inline flags).
2. `nrl --version` and `nrlpy --version`.
3. Model file name **and SHA-256**.
4. CPU feature string from `nrl --features`.
5. The **full four-metric block** (never just the headline).
6. For `virtual_tps`: the `gate_skip_ratio` that produced the multiplication (NOT the lattice-observation `skip_ratio`), and the `(§15)` tag. In P1 / P2-Shadow these are always equal to `executed_tps`; any published non-identity requires P2-Active wiring.
7. Benchmark class (A or B).
8. Artifact file path (JSON under `build/bench/gguf_*.json`) if published officially.

This is the same bar as [§5.3](../nrl-architecture.md) in the main architecture doc. Applied verbatim.

**Automated enforcement.** Run [`benchmarks/gguf_golden.py`](../benchmarks/gguf_golden.py) (or [`scripts/live_readiness_gguf.*`](../scripts/live_readiness_gguf.ps1)) before any publication. The harness produces `build/gguf_golden/gguf_golden.{json,md}` and asserts the honesty-hinge invariants (1)–(6) as test assertions; the stub mode is a required CI gate on Ubuntu and Windows, and the real-mode path (gated on `NRL_GGUF_GOLDEN_MODEL`) records the completion SHA + four-metric block in a diffable artifact. Do not ship a TPS figure that wasn't produced by this harness.

---

## 8. Why this beats Grok's earlier sketch

| Dimension | Grok's sketch | This design |
|---|---|---|
| What NRL owns in attention/FFN | "sparse matmul" (doesn't match the ABI) | Gating lattice emitting layer/expert/KV masks; libllama keeps the math |
| Entry point | Ad-hoc `nrl run` flags | `.nrl` manifest v1 (schema-gated extension of v0); CLI flags are sugar |
| Headline number | "real TPS" (undefined) | `virtual_tps` with `(§15)` footnote + materialized `executed_tps` printed alongside |
| KV cache | "NRL lattice blocks" (vague) | ZPM-collapse KV-keep mask over fixed groups; numerics untouched, parity-lockable |
| Muscle memory | Python `OrderedDict` | On-disk FNV-1a64-keyed cache under `$NRL_ROOT/cache/mm/`, cap-aware |
| Determinism | Not specified | Class-A replay-locked per-token when seed is set (P2+) |
| Control plane | Not mentioned | Honored via existing `resolve_*_with_control_hints` |
| Rust FFI phase | Phase 3 | Optional, post-v1.0 |
| LM consent | Not mentioned | GGUF runs require `nrl -ai on` (this is an LM surface, §0.1) |

The one-line summary: **NRL owns the decisions, libllama owns the arithmetic.**
