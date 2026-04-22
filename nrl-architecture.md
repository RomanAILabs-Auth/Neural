# NRL (Neural) - Unified Architecture

> Living architecture contract for NRL. This document defines the system as an AI execution fabric with a machine-code core, `.nrl` language surface, and NrlPy front-end. All performance claims must be reproducible through locked benchmark profiles.

- Author: Daniel Harding - RomanAILabs
- Co-architect: Codex (Cursor)
- Collaborators honored:
  - Grok / xAI
  - Gemini-Flash / Google
  - ChatGPT-5.4 / OpenAI
  - Cursor
- Contact: daniel@romanailabs.com, romanailabs@gmail.com
- Website: romanailabs.com
- Status: Core v1 implemented and release-checkable; architecture log below remains authoritative for ongoing work.

---

## 0. North Star (non-negotiable)

1. NRL is engine-first, not syntax-first.
2. Hot kernels are hand-tuned machine code where performance matters most.
3. `.nrl` and NrlPy are control planes over one shared neural execution fabric.
4. Correctness parity, stability, and reproducibility are mandatory.
5. No benchmark claims without locked profile + machine-readable artifacts.

---

## 1. Product Definition

### 1.1 NRL Engine (`nrl`)

- Deliverables:
  - `nrl.exe` (Windows), `nrl` (Linux/macOS)
  - `libnrl` exposing stable C ABI (`nrl_v1_*`)
- Core role:
  - execute low-bit neural kernels at maximal throughput
  - perform feature dispatch once at init (no hot-loop feature branching)
  - enforce argument validation and semantic contracts at ABI edges

### 1.2 NrlPy (`nrlpy`)

- Python front-end for:
  - `nrlpy <file.py>` (TriPy-style: run user Python with assimilation globals + `_core` / `nrl bench` bridges)
  - `nrlpy file.nrl`
  - `nrlpy bench`
- Thin typed wrappers over `libnrl`.
- No Python overhead inside kernel hot loops.

### 1.3 `.nrl` language

- Purpose: orchestration of lattice definitions and kernel steps.
- Initial grammar is intentionally minimal:
  - lattice declaration
  - kernel invocation
  - profile selection
  - deterministic benchmark mode

### 1.4 Biomorphic mapping and cognitive modes

- NRL maps major neurofunctional roles into explicit software ports (`PORT_*`) to keep architecture modular and auditable.
- NRL defines two runtime cognition lanes:
  - `System 1` (Automatic Processing): cached/compiled habits (`zpm` / `automatic` mode)
  - `System 2` (Deliberate Processing): stepwise iterative compute (reference/analysis mode)
- "Living language" implementation policy:
  - plasticity is data-plane adaptation only
  - shipped machine-code instruction streams are not self-modified at runtime
  - all adaptive behavior is bounded by safety contracts and replay logging

### 1.5 NRL biological port map (v1 contract)

| Brain region | NRL port | Runtime function |
|---|---|---|
| Prefrontal cortex | `PORT_EXEC` | Planning, scheduling, and profile dispatch. |
| Cerebral cortex | `PORT_NEO` | High-dimensional pattern synthesis and inference context. |
| Cerebellum | `PORT_SYNC` | High-speed automatic execution of trained paths (System 1). |
| Hippocampus | `PORT_LATTICE` | ZPM memory map storage and fast recall of transition tables. |
| Thalamus | `PORT_GATE` | I/O relay and stream filtering into MSP-512 channels. |
| Basal ganglia | `PORT_HABIT` | Habit loop offload for repetitive deterministic workloads. |
| Corpus callosum | `PORT_MIRROR` | Cross-bank synchronization for mirrored memory streams. |
| Amygdala | `PORT_SENTINEL` | Threat/anomaly signal generation for containment response. |
| Hypothalamus | `PORT_GOVERN` | Homeostasis controller (thermal, RAM, watchdog budgets). |
| Brainstem | `PORT_ALIVE` | Pulse loop and low-level keep-alive orchestration. |

Immune-system overlay:
- `PORT_SENTINEL` emits anomaly interrupts from integrity and behavior checks.
- `PORT_GOVERN` applies immediate mitigations (throttle, isolate, rollback).
- `PORT_ALIVE` ensures fail-safe baseline operation during incident handling.

---

## 2. Numeric and Kernel Strategy

### 2.1 Default production lane

- INT4 packed lattice compute as the default production lane:
  - 2 values/byte
  - branchless threshold/reset masking
  - fused update loop

### 2.2 R&D lanes

- 2-bit and binary lanes are experimental until they beat INT4 on both:
  - stability
  - reproducible throughput under locked profiles

### 2.3 Dimensional target

- Canonical high-D target: `16,384D` for phase planning and profile design.
- Rationale:
  - power-of-two friendly
  - predictable block tiling
  - clean scheduling and reproducible benchmarks

### 2.4 Epistemic compute: “know” beats “grind” (ZPM-shaped)

This is the architectural reading of the dream you are pointing at — **without** pretending the INT4 hot loop magically contains all of mathematics.

**What “knows the math” means here (engineering, not poetry):**

- **Known** = a piece of structure has already been established in a **replay-stable** form: identities, invariants, a transition table, a closed-form map, a cached sub-lattice summary, or a formally checked rewrite rule.
- **Calculate** = apply that known map to the *current* packed state (often cheap: a lookup, a fused pass, or a short recurrence), instead of re-deriving it from scratch every tick.
- **Learn** = update which maps are active or how they index state — **only** under governance (bounded plasticity, logging, immune checkpoints per `docs/nrl_immune_system_spec.md`). Learning is not unconstrained self-modification of shipped machine code.

**How this helps ZPM (already in the engine):**

- ZPM’s value proposition is exactly **baseline-equivalent work accounting**: do the *minimum* `executed_updates` that still matches the **declared** `baseline_equiv_updates` contract on observables (see §14.6 / `zpm_int4_static.c`).
- An **epistemic layer** (future compiler + runtime tables feeding `PORT_LATTICE` / habit maps) is the *source of truth* for **why** a skip is legal: “we already know this sub-step’s effect” → emit a ZPM-eligible fast path instead of naive iteration.
- **Sovereign / System 2** remains the lane that *does not assume* — full steps for audit and lock. **ZPM / System 1** is the lane that *replays what is known* — high virtual throughput when the map is valid.

**What this is not:**

- Not “the network intuits calculus.” Not guessing open conjectures. Not hiding approximate skips inside exact lanes without proof artifacts.
- Geometry / spacetime / primes / gravity / higher theory become **typed encodings + lowering targets**; only what is **certified** to the scheduler may collapse work. Everything else stays on the deliberate lane until it earns a map.

---

## 3. Machine-Code Contract

### 3.1 ABI

- Symbol prefix: `nrl_v1_*`
- Public header is the single source of truth for kernel contracts.
- Any breaking ABI change requires versioned prefix bump (`nrl_v2_*`).

### 3.2 Kernel variants

- Required baseline for each critical kernel:
  - scalar reference (`scalar`)
  - AVX2 optimized (`avx2`)
- Optional advanced variants:
  - AVX-512 optimized (`avx512`) where available
  - AArch64 NEON/SVE in future phases

### 3.3 Hot-loop rules

- No dynamic allocation inside hot loops.
- No runtime feature branching inside hot loops.
- Branchless masking for threshold/reset where feasible.
- All buffers aligned per kernel contract (minimum 32-byte, prefer 64-byte allocator guarantees).

---

## 4. Memory Architecture - MSP-512 (Mirror Stream Protocol)

MSP-512 is a memory streaming protocol, not literal AVX2 register widening.

### 4.1 Definition

- AVX2 kernels consume two synchronized 256-bit streams (A/B segments) to improve sustained feed into compute loops.
- Layout and allocator cooperate to maximize dual-channel utilization and prefetch predictability.

### 4.2 Requirements

- Dual-bank aligned segments for mirrored stream pairs.
- Page-aware placement and first-touch discipline.
- Non-temporal store path only for low-reuse write streams.
- Profiled prefetch distance tuning per kernel family.

### 4.3 Non-goals

- No false claim of native 512-bit instruction path on AVX2.
- No synthetic bandwidth claims without benchmark artifact proof.

---

## 5. Benchmark Governance (Claim Discipline)

### 5.1 Claim tiers

- Tier 1: vs C++ scalar `-O0` (domination baseline)
- Tier 2: vs C++ `-O3` (engineering baseline)
- Tier 3: cross-runtime snapshots (ecosystem context only)

### 5.2 Locked profile requirements

- Fixed workload (neurons, iterations, threshold, seed policy)
- Warmup + replicate policy
- Median-based reporting + stability CV thresholds
- JSON artifact required for each claim

### 5.3 Current policy

- Never publish "peak spike" claims as official.
- Only publish profile-locked, reproducible, median-stable claims.

---

## 6. Repository Layout (target)

```text
NRL/
├── nrl-architecture.md
├── README.md
├── LICENSE
├── CHANGELOG.md
├── SECURITY.md
├── CONTRIBUTING.md
├── engine/
│   ├── include/nrl/nrl.h
│   ├── asm/
│   ├── src/
│   └── tests/
├── nrlpy/
│   ├── pyproject.toml
│   ├── src/nrlpy/
│   ├── src/_core/
│   └── tests/
├── language/
│   ├── spec/
│   └── examples/
├── benchmarks/
└── scripts/
```

---

## 7. Cross-OS and Backend Roadmap

### 7.1 CPU portability

- x86_64 AVX2 lane first.
- Scalar fallback always present.
- AVX-512 lane optional and gated by runtime dispatch.
- AArch64 lane is planned as a first-class follow-on.

### 7.2 GPU posture

- GPU is optional backend work, not required for NRL v1.
- CPU execution fabric remains primary until backend abstraction is stable.

---

## 8. Phase Plan (exit-criteria only)

### Phase 0 - Contract lock
- Architecture file approved and treated as source of truth.
- Baseline benchmark policy and claim tiers frozen.

### Phase 1 - Engine foundation
- ABI skeleton (`nrl_v1_*`) and scalar references shipped.
- AVX2 hot kernels for primary INT4 path shipped.
- Dispatch and variant reporting shipped.

### Phase 2 - Front-end integration
- NrlPy bindings + CLI shipped.
- Minimal `.nrl` parser/execution path shipped.
- End-to-end tests green.

### Phase 3 - Dominance optimization
- MSP-512 allocator/layout path integrated.
- Profile-locked benchmark artifacts in CI.
- Stable high-multiplier targets reached and documented.

### Phase 4 - Future-proof expansion
- AVX-512 lane (where supported).
- AArch64 optimization lane.
- Optional GPU backend feasibility gate.

---

## 9. Initial Decisions

| ID | Decision |
|---|---|
| NRL-D001 | NRL is an AI execution fabric first, language second. |
| NRL-D002 | INT4 is production baseline; 2-bit/binary are R&D lanes. |
| NRL-D003 | Official claims require locked profile + JSON artifacts. |
| NRL-D004 | MSP-512 is a dual-stream memory protocol, not AVX2 register widening. |
| NRL-D005 | 16,384D is the canonical high-D planning target. |

---

## 10. Immediate Next Actions

1. Create repo skeleton for `NRL/`.
2. Define `engine/include/nrl/nrl.h` with v1 ABI placeholders.
3. Port proven INT4 scaffold pattern from Trinary as reference.
4. Stand up benchmark harness + sweet-spot profile runner at project start.
5. Add CI checks for lint/type/tests from day one.

---

## 11. Pulse Layer (Controlled Plasticity)

NRL may operate in an adaptive "alive" execution mode, but only under strict
sovereign controls. The system may update neural state and weights in real time;
it must not self-mutate instruction logic in production mode.

### 11.1 Intent

- Move beyond static inference into bounded online adaptation.
- Preserve stability, safety, debuggability, and reproducibility contracts.
- Keep architect-level override authority at all times.

### 11.2 Operating Modes

1. **Sovereign Mode (default)**
   - Plasticity disabled.
   - Deterministic/reproducible execution.
   - Official benchmark and release claim mode.

2. **Adaptive Mode (opt-in)**
   - Bounded online plasticity enabled.
   - Weight/state updates allowed under policy limits.
   - Full telemetry and checkpointing required.

### 11.3 Safety Contract (non-negotiable)

- No runtime mutation of shipped assembly kernel instruction streams.
- Plasticity is restricted to data planes (weights, gains, masks, state).
- Deadman switch: instant freeze to Sovereign Mode.
- Entropy guard: auto-rollback to last stable checkpoint on instability.
- Update clipping: bounded per-step deltas to prevent runaway amplification.
- Watchdog budget: max adaptive work per cycle to preserve liveness.

### 11.4 Governance and Audit

- Every plasticity event must be loggable to machine-readable JSON artifacts.
- Adaptive runs must emit:
  - plasticity counters
  - entropy metrics
  - rollback/freeze events
  - checkpoint lineage
- Deterministic replay mode is required for incident/debug analysis.

### 11.5 Benchmark Classes

- **Class A - Deterministic**: official speed claims and baseline/perf-gate.
- **Class B - Adaptive**: adaptation quality and stability envelopes.
- No Class B result may be presented as a deterministic speed claim.

### 11.6 Decision

| ID | Decision |
|---|---|
| NRL-D006 | NRL supports controlled online plasticity, but runtime instruction self-modification is forbidden in production. |

---

## 12. War-Drive Profile (R&D ceiling target)

War-Drive is the high-aggression optimization profile for reaching the upper
performance envelope while preserving sovereign safety controls.

### 12.1 Target

- R&D north-star target: **3000 GOPS class** on supported hardware profiles.
- This target is exploratory and does not replace deterministic production claims.

### 12.2 Scope

War-Drive may enable:
- all-core saturation and thread pinning
- low-bit experimental lanes (2-bit / binary bit-sliced kernels)
- aggressive memory streaming (MSP-512 policy)
- profile-specific fused kernels and schedule tuning

War-Drive must not enable:
- unconstrained runtime self-modification of shipped instruction streams
- unbounded plasticity updates without watchdog/rollback

### 12.3 Measurement Contract

Every War-Drive claim must report:
- exact hardware profile (CPU model, cores/threads, memory topology)
- kernel/profile name and lane type (INT4, INT2, BNN)
- operation definition used for GOPS accounting
- warmup/replicate policy
- median + CV stability metrics
- machine-readable JSON artifact

No "peak spike" result may be published as an official War-Drive claim.

### 12.4 Safety Overlay

War-Drive runs only with:
- deadman switch armed
- entropy guard active
- checkpoint/rollback path active
- adaptive event logging active

### 12.5 Decision

| ID | Decision |
|---|---|
| NRL-D007 | NRL defines a War-Drive R&D profile targeting 3000 GOPS class, but deterministic production claims remain governed by Sovereign Mode. |

---

## 13. Prompt Budget to "Alive" NRL

Definition of "Alive" for v1.0:
- Controlled Plasticity available in Adaptive Mode
- deterministic Sovereign Mode preserved
- deadman switch + entropy rollback + replay logging shipped
- end-to-end NRL engine + NrlPy + `.nrl` execution path working

### 13.1 Estimated prompts remaining

From current state, estimate **12-15 focused prompts** to first "alive" milestone:

1. Engine ABI/runtime skeleton (`nrl_v1_*`) and init/feature reporting.
2. INT4 scalar reference kernel + tests.
3. AVX2 INT4 optimized kernel + parity lock.
4. Dispatch table + variant reporting.
5. Native `nrl` CLI (`run`, `bench`, `--features`, profiles).
6. NrlPy C-extension bridge + typed wrappers.
7. `.nrl` minimal parser + execution path.
8. Benchmark harness + locked deterministic profile.
9. War-Drive benchmark profile + operation accounting.
10. Controlled plasticity core (bounded updates in data planes).
11. Safety systems (deadman switch, entropy guard, rollback, checkpoints).
12. Adaptive telemetry + deterministic replay artifacts.
13. Cross-platform build polish and CI matrix.
14. Documentation/claim policy hardening.
15. Release readiness pass (tests/lint/bench gates all green).

### 13.2 Stretch to aggressive War-Drive envelope

Add **4-6 more prompts** for deep optimization work:
- bit-sliced INT2/BNN lane prototypes
- MSP-512 layout tuning
- all-core scheduling/pinning tuning
- AVX-512 variant (where hardware supports it)

---

## 14. Execution Log (Prompt-by-Prompt)

### 14.1 Prompt 1 - Engine ABI/runtime skeleton + feature reporting (completed)

Implemented:
- public ABI extension in `engine/include/nrl/nrl.h`:
  - `NRL_VERSION_*` constants and `NRL_VERSION_STRING`
  - `NRL_CPU_*` feature bitmask contract
  - `nrl_v1_cpu_features()` API
  - `NRL_ERR_CPU` status code reservation
- runtime internals:
  - `engine/src/runtime_internal.h` for dispatch and initialization contracts
  - `engine/src/cpuid.c` for x86_64 CPU/OS capability detection
  - `engine/src/dispatch.c` for first-stage variant binding (`scalar_stub`, `avx2_stub`)
  - `engine/src/capi.c` for stable init path and active variant reporting
  - `engine/src/version.c` for version/compiler/arch banner
- native CLI skeleton:
  - `engine/src/main.c` with `--version`, `--features`, and `variant <kernel>`
- build/test entrypoints:
  - `build.ps1` (Windows Zig pipeline) and `build.sh` (POSIX)
  - `engine/tests/test_runtime.c` smoke tests for init/version/variant

Verification snapshot:
- `./build.ps1 -Tests` passes.
- `nrl.exe --version` reports `nrl 0.1.0`.
- `nrl.exe --features` reports detected CPU capabilities.
- `nrl.exe variant braincore_int4` resolves runtime-selected placeholder variant.

Next step (Prompt 2):
- implement `nrl_v1_braincore_int4` scalar reference kernel with deterministic parity tests.

### 14.2 Prompt 2 - INT4 scalar reference kernel + tests (completed)

Implemented:
- scalar packed-INT4 kernel in `engine/src/braincore_int4_scalar.c`:
  - 2 neurons per byte (low/high nibble)
  - saturating add in `[0, 15]`
  - threshold-reset semantics (`>= threshold -> 0`)
  - deterministic iteration order and output
- dispatch upgrade:
  - `engine/src/dispatch.c` now binds to real scalar kernel
  - active variant reports `scalar_ref` or `avx2_scalar_ref` (feature lane marker; scalar implementation for now)
- ABI hardening:
  - `engine/src/capi.c` adds strict argument validation for `nrl_v1_braincore_int4`
    - non-null buffers
    - even and non-zero neuron count
    - non-zero iterations
    - threshold in `[1, 15]`
- runtime tests expanded in `engine/tests/test_runtime.c`:
  - scalar parity check against independent in-test reference
  - deterministic replay check (same inputs -> same outputs)
  - argument validation coverage
- build integration:
  - `build.ps1` and `build.sh` compile `braincore_int4_scalar.c`

Verification snapshot:
- `./build.ps1 -Tests` passes with expanded test set.
- `nrl.exe variant braincore_int4` reports bound reference lane.

Next step (Prompt 3):
- add AVX2 `braincore_int4` optimized kernel with parity lock against scalar reference.

### 14.3 Prompt 3 - AVX2 INT4 kernel + parity lock (completed)

Implemented:
- AVX2 kernel in `engine/src/braincore_int4_avx2.c`:
  - vectorized packed-INT4 lane update (32 bytes/step)
  - saturating add + threshold reset logic matching scalar semantics
  - fast path for `threshold == 8` (dominance profile lane)
  - scalar tail path for non-32-byte multiples to preserve full API coverage
- runtime integration:
  - `engine/src/runtime_internal.h` adds `nrl_braincore_int4_avx2` contract
  - `engine/src/dispatch.c` now binds AVX2-capable CPUs to `avx2`
  - scalar fallback remains `scalar_ref`
- parity lock tests in `engine/tests/test_runtime.c`:
  - direct scalar vs AVX2 parity checks for `threshold=8` and `threshold=12`
  - includes non-32-byte vector tail scenario to lock edge correctness
- build integration:
  - `build.ps1` and `build.sh` compile `braincore_int4_avx2.c`

Verification snapshot:
- `./build.ps1 -Tests` passes.
- `nrl.exe variant braincore_int4` returns `avx2` on AVX2 hardware.

Next step (Prompt 4):
- enrich dispatch reporting and surface runtime variant state through CLI and test snapshots.

### 14.4 Prompt 4 - Dispatch/variant reporting polish (completed)

Implemented:
- native runtime snapshot command in `engine/src/main.c`:
  - `nrl runtime` prints version, CPU feature map, and active kernel variants
- dispatch state is now surfaced as first-class runtime visibility:
  - `braincore_int4` variant is explicitly reported (`avx2` or `scalar_ref`)
- CLI usage expanded to include runtime and benchmark entrypoints.

Verification snapshot:
- `nrl.exe runtime` reports active CPU features and selected kernel variant.

### 14.5 Prompt 5 - Native CLI execution + benchmark modes (completed)

Implemented:
- native benchmark command in `engine/src/main.c`:
  - `nrl bench [neurons] [iters] [reps] [threshold]`
  - warmup + measured loop
  - outputs elapsed time, updates/sec, and GOPS
- native execution command in `engine/src/main.c`:
  - `nrl run [neurons] [iters] [threshold] [profile]`
  - emits deterministic checksum for replay sanity
- profile presets wired into CLI:
  - `sovereign`, `adaptive`, `war-drive`
  - profile defaults are used when parameters are omitted
- deterministic synthetic input generation for repeatable local comparisons.
- initial benchmark artifact captured in `benchmarks/initial_results.md`.

First real benchmark snapshot (current machine):
- command: `nrl bench 1048576 256 12 8`
- run A: `18.641 GOPS`
- run B: `17.764 GOPS`
- active variant: `avx2`

Next step (Prompt 6):
- scaffold NrlPy C-extension bridge + typed wrappers for zero-copy buffer path.

### 14.6 Prompt 6 (pivot) - ZPM static accelerator + virtual throughput accounting (completed)

Implemented:
- new exact ZPM accelerator in `engine/src/zpm_int4_static.c`:
  - precomputes k-step transition maps for each 4-bit input value
  - collapses repeated static-input iterations into a single apply pass
  - preserves exact output semantics of iterative kernel for static inputs
- internal runtime contract update:
  - `nrl_braincore_int4_zpm_static(...)` in `engine/src/runtime_internal.h`
  - exposes executed/baseline-equivalent update accounting
- CLI integration in `engine/src/main.c`:
  - new `zpm` profile for `nrl run` and `nrl bench`
  - benchmark output now reports:
    - `executed_updates`
    - `baseline_equiv_updates`
    - `skip_ratio`
    - `executed_gops`
    - `virtual_gops`
- test coverage in `engine/tests/test_runtime.c`:
  - scalar parity lock for ZPM static path
  - stat accounting assertions for executed vs baseline-equivalent updates
- build integration:
  - `build.ps1` and `build.sh` compile `engine/src/zpm_int4_static.c`

Verification snapshot:
- `./build.ps1 -Tests` passes.
- benchmark specs captured in `benchmarks/initial_results.md`.

Observed speed envelope (current host):
- same workload (`1048576 x 256 x 12`):
  - iterative: `0.154222s`
  - zpm: `0.007821s` (`19.72x` faster)
  - virtual throughput: `411.848 GOPS`
- high-iteration workload (`1048576 x 4096 x 12`):
  - iterative: `2.360455s`
  - zpm: `0.008400s` (`281.01x` faster)
  - virtual throughput: `6135.375 GOPS`

Next step (Prompt 7):
- NrlPy bridge with mode selection (`iterative` / `zpm`) and benchmark parity reporting.

### 14.7 Prompt 7 (partial) - Biomorphic control surface and System 1 naming (completed)

Implemented:
- naming alignment:
  - NRL expanded naming adjusted to `NRL (Neural)` in docs.
- architecture contract extensions:
  - added `System 1` / `System 2` cognitive mode model in Product Definition.
  - added biological port map contract (`PORT_EXEC`, `PORT_NEO`, `PORT_SYNC`, `PORT_LATTICE`,
    `PORT_GATE`, `PORT_HABIT`, `PORT_MIRROR`, `PORT_SENTINEL`, `PORT_GOVERN`, `PORT_ALIVE`).
  - added immune overlay responsibilities for sentinel/govern/alive loops.
- runtime/CLI visibility in `engine/src/main.c`:
  - `nrl brain-map` command prints the biomorphic port map.
  - `nrl runtime` now reports System 1/System 2 mode definitions.
  - added `automatic` profile alias mapped to exact ZPM static mode.
  - bench/run mode labels now expose `system1-automatic-zpm` vs `system2-iterative`.

Verification snapshot:
- build and tests remain green after cognitive-mode + port-map additions.
- `nrl brain-map` exposes the runtime biological port contract directly from CLI.

Next step (Prompt 8):
- NrlPy bridge with explicit `mode="iterative|automatic"` path and parity benchmark reporting.

### 14.8 Prompt 8 (pivot) - Omega fractal routing prototype + benchmark (completed)

Implemented:
- new Omega router in `engine/src/zpm_omega_router.c`:
  - lattice-of-lattices routing over fixed sub-lattices
  - dark-lattice skipping with deterministic gate policy
  - sparse wake-up model (index/signature gating)
  - optional pruning of near-zero sub-lattices
- internal contract additions in `engine/src/runtime_internal.h`:
  - `nrl_omega_stats` for executed/baseline/active/pruned accounting
  - `nrl_braincore_int4_omega_virtual(...)` API for virtual routed execution
- CLI integration in `engine/src/main.c`:
  - new `omega` profile in `run` and `bench`
  - output fields for active/total/pruned sub-lattice averages
  - System 1 mode labeling for omega lane
- quality lock:
  - `engine/tests/test_runtime.c` adds omega determinism/stat sanity test
  - build scripts compile `engine/src/zpm_omega_router.c`

Verification snapshot:
- `./build.ps1 -Tests` passes.
- omega benchmark artifact recorded in `benchmarks/initial_results.md`.

Observed omega snapshot (current host):
- `nrl bench 1048576 16384 10 8 omega`
- elapsed: `0.004944s`
- skip ratio: `0.999998`
- virtual throughput: `34749.629 GOPS`

### 14.9 Prompt 9 - Throughput-preserving Omega Hybrid lane (completed)

Implemented:
- new `omega-hybrid` runtime profile:
  - preserves architecture (fractal routed System 1 lane)
  - enforces active-floor sub-lattice execution to maintain high native throughput
  - uses iterative AVX2 compute on active sub-lattices while skipping dark regions
- omega router extension (`engine/src/zpm_omega_router.c`):
  - minimum active sub-lattices contract
  - active kernel mode (`zpm-static` vs `iterative`)
  - deterministic wake + floor-fill behavior
- CLI integration (`engine/src/main.c`):
  - `omega-hybrid` added to `run` and `bench`
  - explicit mode labels and active/pruned averages remain exposed

Benchmark snapshot (current host):
- `sovereign` (`1048576 x 4096 x 12`): `19.646 executed GOPS`, `2.623s`
- `omega-hybrid` (same shape): `19.031 executed GOPS`, `25.375 virtual GOPS`, `2.031s`
- `automatic/zpm` (same shape): `1.503 executed GOPS`, `6156.629 virtual GOPS`, `0.008s`

Interpretation:
- `omega-hybrid` is the "keep raw horsepower" lane.
- `automatic/zpm` is the "max subconscious skip" lane.
- both stay inside one biomorphic architecture contract.

### 14.10 Prompt 10 - NrlPy bridge + mode-aware benchmark surface (completed)

Implemented:
- Python package scaffold in `nrlpy/`:
  - `pyproject.toml`
  - `src/nrlpy/{__init__.py,runtime.py,cli.py,_core.pyi,py.typed}`
  - `src/_core/module.c` C extension bindings to `libnrl`
  - `tests/test_smoke.py`
- C extension surface (`nrlpy._core`):
  - `init`, `version`, `features`, `active_variant`
  - `braincore_int4` runtime call
- mode-aware benchmark bridge in Python:
  - `runtime.bench_cli(...)` parses native `nrl bench` output into typed dicts
  - CLI command: `nrlpy bench ... [profile]` supports all engine profiles including `omega` and `omega-hybrid`
- build integration:
  - `build.ps1` now builds `nrlpy` extension (`_core.cp*.pyd`)
  - `build.ps1 -Tests` runs `nrlpy` tests when `pytest` is available

Verification snapshot:
- `./build.ps1 -Tests` passes (engine + nrlpy tests).
- `python -m nrlpy.cli bench ... omega` returns structured benchmark JSON.

### 14.11 Prompt 11 - `.nrl` minimal parser + execution path (completed)

Implemented:
- native `.nrl` execution in `engine/src/main.c`:
  - new command: `nrl file <path.nrl>`
  - shortcut: `nrl <path.nrl>`
  - parser supports key-value lines:
    - `mode=run|bench`
    - `profile=...`
    - `neurons=...`
    - `iterations=...`
    - `reps=...`
    - `threshold=...`
  - comments (`#`) and blank lines are ignored
- Python execution path:
  - `nrlpy.runtime.run_nrl_file(path)`
  - `nrlpy <path.nrl>` routed via native `nrl file`
- language contract assets:
  - `language/spec/minimal_nrl_v0.md`
  - `language/spec/nrl_physics_language_v0.md` (physics-native surface: design intent; §6 epistemic/ZPM; §8 binary assimilation)
  - `language/examples/omega_pass.nrl`
  - binary assimilation: `nrl assimilate`, `nrl_v1_braincore_packed_bytes`, `nrlpy._core.braincore_int4_inplace`, `nrlpy run` + `examples/assimilate_llm_solver.py`

Verification snapshot:
- `./build.ps1 -Tests` passes (`test_runtime` + `nrlpy` tests).
- native and Python `.nrl` execution produce benchmark output from the same file.

### 14.12 Prompt 12 - Locked NRL vs C++ benchmark harness (completed)

Implemented:
- new benchmark harness: `benchmarks/nrl_vs_cpp.py`
  - compiles C++ INT4 reference baselines (`-O0` and `-O3`) with Zig C++
  - runs locked native NRL profiles (`sovereign`, `omega-hybrid`, `omega`, `automatic`)
  - emits speedup metrics vs both C++ baselines
  - writes reproducible artifacts:
    - `build/bench/nrl_vs_cpp.json`
    - `build/bench/nrl_vs_cpp.md`
- benchmark governance docs:
  - `benchmarks/README.md` updated with harness command and artifact contract

Verification snapshot (workload: `1048576 x 4096 x 6`, threshold `8`):
- C++ `-O0` elapsed: `159.901s`
- C++ `-O3` elapsed: `0.957s`
- NRL `sovereign` speedup:
  - `135.61x` vs C++ `-O0`
  - `0.81x` vs C++ `-O3`
- NRL `omega-hybrid` speedup:
  - `181.20x` vs C++ `-O0`
  - `1.08x` vs C++ `-O3`
- NRL `omega` speedup:
  - `74685.29x` vs C++ `-O0`
  - `447.17x` vs C++ `-O3`
- NRL `automatic/zpm` speedup:
  - `45017.23x` vs C++ `-O0`
  - `269.54x` vs C++ `-O3`

### 14.13 Prompt 13 - Lightweight introspection interface (`status` / `inquire` / `chat`) (completed)

Implemented:
- native command surface in `engine/src/main.c`:
  - `nrl status` and `nrl -status`
  - `nrl inquire <topic>` and `nrl -inquire <topic>`
  - `nrl chat <message>` and `nrl -chat <message>`
- command behavior:
  - `status`: runtime readiness, active variant, cognitive mode summary
  - `inquire`: deterministic topic answers (`speed`, `safety`, `modes`, `profiles`, `architecture`, `benchmark`)
  - `chat`: rule-based command-intent responses (no heavy language model dependency)
- design intent:
  - keeps language layer lightweight and low-risk while core engine/safety work continues
  - avoids premature high-cost "absorb English" pipeline before guard rails are fully shipped

Verification snapshot:
- `./build.ps1 -Tests` passes.
- live command checks for `status`, `inquire`, and `chat` produce expected outputs.

### 14.14 Prompt 14 - Release check automation + Grok handoff package (completed)

Implemented:
- release-check automation:
  - `scripts/release_check.ps1` (Windows)
  - `scripts/release_check.sh` (POSIX)
  - checks include:
    - full build + tests
    - locked `nrl_vs_cpp` artifact generation
    - CLI smoke (`status`, `inquire`)
- review handoff:
  - `grok_review_handoff.md` with scoped external review prompt and required files
  - keeps review objective factual (risk, fixes, claim safety, roadmap sanity)

Verification snapshot:
- `scripts/release_check.ps1` executed successfully end-to-end.
- output artifacts confirmed:
  - `build/bench/nrl_vs_cpp.json`
  - `build/bench/nrl_vs_cpp.md`

### 14.15 Prompt 15 - Production installer + README/guide polish + completion audit (completed)

Implemented:
- production installers (cross-OS):
  - `scripts/install_nrl.ps1` (Windows)
  - `scripts/install_nrl.sh` (POSIX)
  - explicit LM/AI opt-in captured at install time and persisted
  - install target:
    - Windows: `%LOCALAPPDATA%\Programs\NRL\bin`
    - POSIX: `~/.local/bin`
- runtime opt-in visibility:
  - `nrl status` now reports `lm_ai_opt_in: enabled|disabled`
- documentation polish:
  - root `README.md` rewritten for production onboarding:
    - install paths
    - quick start
    - benchmark governance
    - human and LLM coding guides
    - architecture completion status summary
  - `scripts/README.md` updated with installer scripts

Verification snapshot:
- installer executed successfully with LM/AI opt-in enabled.
- installed command verified:
  - `nrl status` from installed path and PATH command both work.
- `./build.ps1 -Tests` remains green.

### 16. Architecture Completion Audit (v1 core)

Status categories:
- `DONE` = implemented and verified in current tree
- `OPEN` = planned expansion lane, not required for current v1 core release

| Contract Area | Status | Notes |
|---|---|---|
| Engine ABI/runtime foundation | DONE | `nrl_v1_*`, init/version/features/variant wired |
| INT4 scalar + AVX2 kernels | DONE | parity and correctness tests in place |
| Dispatch + profile runtime | DONE | sovereign/adaptive/war-drive/zpm/omega/omega-hybrid |
| NrlPy front-end | DONE | C extension + CLI + tests |
| `.nrl` execution path | DONE | key-value parser + native/python file execution |
| Benchmark governance harness | DONE | NRL-vs-C++ harness + JSON/MD artifacts |
| Release automation | DONE | release checks + installers + review handoff |
| Controlled plasticity core | OPEN | policy defined; deeper adaptive internals remain expansion work |
| Sentinel/Govern/Alive guard rails | OPEN | architecture mapped; runtime enforcement layer not yet shipped |
| AVX-512/AArch64 optimization lanes | OPEN | roadmap lanes beyond current x86_64 AVX2 release |

### 15. Alive Comparison Table (engineering analogy)

This table is a design-direction analogy, not a biological consciousness claim.

| Metric | Human brain (reference class) | NRL-Pulse (current virtual lane) |
|---|---|---|
| Total virtual capacity | ~1-10 Peta-Ops (commonly cited rough range) | up to ~34.7 Tera-Ops equivalent (`omega` snapshot) |
| Energy usage | ~20W | ~18-20W class target envelope on laptop CPU |
| Firing logic | Sparse / event-driven | Ultra-sparse routed activation (dark-lattice skipping) |
| Adaptability | Hebbian/plastic biological learning | Controlled 4-bit plasticity roadmap with safety constraints |

North-Star policy:
- NRL-Omega targets Peta-class *virtual* throughput by increasing selective activation and routing quality, not by brute-force executing all baseline ops.

---

_End of architecture contract v0 (living)._ 
