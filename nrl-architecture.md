# NRL (Neural) — Architecture Specification

Author: Daniel Harding, RomanAILabs · Co-architect: Codex (Cursor)  
Collaborators: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor  
Contact: `daniel@romanailabs.com`, `romanailabs@gmail.com` · `romanailabs.com`  
**Status:** Core v1 implemented, release-checkable. This document is the engineering contract for behavior, benchmarks, and extensions.

---

## 0. Design principles

1. **Engine-first:** hot paths live in hand-tuned C/assembly; control languages orchestrate, they do not define numerics inside loops.
2. **Stable ABI:** public surface is `nrl_v1_*` in `engine/include/nrl/nrl.h`; breaking changes require a versioned prefix (`nrl_v2_*`).
3. **Dispatch once:** CPU capability selection and kernel variant binding occur at initialization, not per inner iteration.
4. **Reproducibility:** official throughput claims require locked `(neurons, iterations, reps, threshold, profile)`, stated warmup policy, and machine-readable artifacts.
5. **Honest accounting:** `virtual_gops` and `skip_ratio` encode **baseline-equivalent work per wall second** and **fraction of baseline updates not executed** on instrumented paths—not a claim that every skipped operation was executed at that rate.

### 0.1 Frontier AI / LM positioning (inside this contract)

NRL and Nrlpy are intentionally a **frontier AI/LM-oriented stack**: natural-language control (`nrl chat`, `nrl control`), assimilated script surfaces (`nrlpy run`, `llm_globals()`), explicit **LM/AI opt-in** (`nrl -ai`, installer consent), and a **roadmapped** bounded adaptation plane (specialization cache, shadow promotion, plasticity — see [`docs/nrl_alive_language_evolution_architecture.md`](./docs/nrl_alive_language_evolution_architecture.md)). That is **not** in tension with principles §0.1–0.5: **lattice semantics stay deterministic and replay-locked**, while **language, policy, evidence, and optional upstream models** live in the orchestration layer.

External narratives that dismiss the whole system as “not AI” because the INT4 inner loop has no transformer are **misaligned with this product**: the correct statement is **NAL-002** — no **uncontracted** ML inference *inside* the neuron hot path by default; control-plane and operator-consented LM surfaces are **first-class**.

### 0.2 External automation prompts (LLM / Grok-style refactors)

Third-party prompts that demand **global double-buffering**, **“no in-place mutation ever”** or **graph-style WAR fixes** must be **checked against this repository** before execution:

- The shipped **packed INT4** update (`braincore_int4_*`) applies **independent per-cell** dynamics each sub-step: each packed byte reads **only** its own prior potential and the **fixed** input drive for that index. There is **no neighbor coupling** in that reference model, so **intra-iteration scan order does not change semantics** (still test-locked for regression).
- **Omega / ZPM** add **declared** skipping and accounting; determinism is by **locked seeds, buffers, and workload descriptors** — not by banning all in-place writes in production kernels.
- A **double-buffer formalism** is a **valid correctness reference** (see `engine/tests/test_runtime.c`) and may guide future research paths; wholesale hot-path refactors require **ABI, perf, and parity review** — not “output only code” blind drops.

---

## 1. Components

### 1.1 Native engine (`nrl` / `libnrl`)

- Ships as `nrl.exe` (Windows) or `nrl` (POSIX) linked with `libnrl`.
- Responsibilities: INT4 packed lattice updates, optional **static transition collapse** for fixed drives, optional **hierarchical sparse routing** with pruning, CLI and bench harness, `.nrl` v0 dispatch.

### 1.2 NrlPy (`nrlpy`)

- Python package with `nrlpy._core` C extension calling `libnrl` for in-process primitives (`braincore_int4_inplace`, packed-byte helpers, etc.).
- `nrlpy.runtime` shells out to `nrl bench`, `nrl assimilate`, `nrl file` for locked CLI semantics and stable stdout parsing.
- `nrlpy.cli` exposes `bench`, `run`, `demo`, and TriPy-style `nrlpy <script.py>`.
- **Seamless assimilated scripts:** `nrlpy.compat.llm_globals()` injects `nrl` / `NRL` plus `nrlpy.seamless` helpers (`next_prime`, `is_prime`, `fabric_pulse`) so `nrlpy script.py` can run **without** `import nrlpy`. This is **not** automatic translation of arbitrary Python into lattice kernels; it is a curated, zero-boilerplate surface (see `nrlpy/README.md`, `examples/prime.py`).
- **Adaptive runtime (design):** bounded plasticity, verified specialization cache, and shadow promotion are specified in [`docs/nrl_alive_language_evolution_architecture.md`](./docs/nrl_alive_language_evolution_architecture.md). JSON schemas: [`docs/schemas/`](./docs/schemas/).

### 1.3 `.nrl` v0

- Minimal key-value files: `mode`, `profile`, `neurons`, `iterations`, `reps`, `threshold`, optional `expected_fnv1a64` (digest lock for synthetic mix profile only).
- Grammar: [`language/spec/minimal_nrl_v0.md`](./language/spec/minimal_nrl_v0.md). Physics-oriented language design notes (non-normative for v0 execution): [`language/spec/nrl_physics_language_v0.md`](./language/spec/nrl_physics_language_v0.md).

### 1.4 `PORT_*` telemetry identifiers

The CLI **`nrl brain-map`** prints a fixed table of identifiers. They label **software subsystems and telemetry hooks** for modular documentation—not neuroscientific claims.

| Identifier | Intended subsystem / reading |
|------------|-------------------------------|
| `PORT_EXEC` | Scheduler / profile dispatch and last bench probe parameters. |
| `PORT_NEO` | ISA capability summary (`nrl_v1_cpu_features`). |
| `PORT_SYNC` | Sparse-router skip statistics (`skip_ratio`). |
| `PORT_LATTICE` | Sub-lattice counts from hierarchical router (`active` / `total` / `pruned`). |
| `PORT_GATE` | Packed INT4 lane width / vector path notes. |
| `PORT_HABIT` | Input sparsity pattern used by the probe. |
| `PORT_MIRROR` | Buffer lifetime (potentials/inputs released after probe). |
| `PORT_SENTINEL` | Last kernel return code. |
| `PORT_GOVERN` | Process RSS vs peak ratio (coarse resource signal). |
| `PORT_ALIVE` | Probe wall-clock duration. |

Runtime hardening (sentinel / governor / watchdog) is specified under [`docs/nrl_immune_system_spec.md`](./docs/nrl_immune_system_spec.md); enforcement depth beyond CLI labels remains roadmap work.

### 1.5 Sandboxed CLI control layer (`nrl chat`, `nrl control`)

High-level natural-language strings are accepted only as **hints** mapped to a small closed set of intents. This layer **is** an LM-facing operator surface; there is **no LLM inside the INT4 hot path** — classification here is deterministic substring rules in the CLI (`engine/src/main.c`), outside lattice kernels, with full audit JSONL (§1.5 rules).

**Sandbox rules (normative for this layer):**

1. **Write scope:** The only automatic persistence is under `$NRL_ROOT/build/control` when `NRL_ROOT` is set, otherwise `./build/control`: `preferences.json` (structured hints: recommended profile, optional `power_until_unix`, `throttle_hint`; JSON Schema: [`docs/schemas/control_preferences_v1.schema.json`](./docs/schemas/control_preferences_v1.schema.json)) and `control_audit.jsonl` (one JSON object per line; line shape: [`docs/schemas/control_audit_line_v1.schema.json`](./docs/schemas/control_audit_line_v1.schema.json)). No other paths are opened for writes by this layer. **`throttle_hint`** is `none` by default; CPU-throttle intents set `conservative`; volatile-market “extra gate” intents set **`gated`** so consumers can treat the file as a sticky conservative bias without a clock. **Nrlpy:** `nrlpy.runtime.bench_cli(..., respect_control_hints=True)` (and `native.run_bench`) resolves the bench profile from this file when an active power window applies or when `throttle_hint` is `conservative` or `gated` — subprocess only; kernels unchanged.
2. **Blocked by default:** Any message that looks like OS manipulation, raw URLs, shell one-liners (`taskkill`, `shutdown`, `registry`, `powershell -`, `curl`, `http://`, `https://`) is classified as **deny** and only audited — no preference write. Layman phrases that imply **external** servers (“optimize the server”) produce **advisory** stdout only; the engine does not touch the host fleet.
3. **Trading boundary:** Messages about **real** trading (`buy`, `sell`, `stock`, or standalone `trading` not tied to “algorithm” / “simulate”) are denied. Phrases such as “trading algorithm … extra gated” are treated as **internal** policy hints and may map to `sovereign` after confirmation.
4. **Confirmation for writes:** Intents with `govern_verdict = allow_write` (profile changes, power window, throttle hint) require **`nrl control --yes`** (or `-y`) or environment **`NRL_CONTROL_CONFIRM=1`**. Without that, the CLI prints `DEFER`, appends `defer_confirm` to the audit log, and does not overwrite `preferences.json`. **`nrl chat` never enables writes** (`allow_write` is always false).
5. **Immune mapping (documentation):** Each handled message prints a fixed `control_immune:` block: **`PORT_SENTINEL`** — bounded scan / token policy for the control path; **`PORT_GOVERN`** — classifier verdict and intent id. This mirrors the vocabulary in §1.4 for traceability; full immune enforcement is still per [`docs/nrl_immune_system_spec.md`](./docs/nrl_immune_system_spec.md).
6. **Status surface:** `nrl status` prints `control_preferences_path`, a one-line parse of `recommended_profile` / `throttle_hint` / `power_until_unix` when the v1 file exists, and **`control_hints_active_for_nrlpy`** (`yes` when the same conditions as `nrlpy.runtime.resolve_bench_profile_with_control_hints` apply). Read-only; does not create directories. **Nrlpy:** `python -m nrlpy.cli control status` emits JSON (`preferences`, `hints_active`, audit path); `python -m nrlpy.cli control audit tail [N]` tails `control_audit.jsonl`.

**Example interactions (safe behavior):**

```text
$ nrl chat "how do I go faster"
NRL chat
  user: how do I go faster
control_immune:
  PORT_SENTINEL: input_scanned ok (length and token bounds)
  PORT_GOVERN: verdict=allow_advisory intent=profile_advisory_speed
control_advisory:
  virtual: `nrl bench … omega` — executed: sovereign / omega-hybrid.
  …
```

```text
$ nrl control "buy AAPL now"
NRL control (sandboxed)
  user: buy AAPL now
control_immune:
  PORT_SENTINEL: input_scanned ok (length and token bounds)
  PORT_GOVERN: verdict=deny_trading_sandbox intent=trading_external_denied
control_outcome: BLOCKED (sandbox policy — no OS / trading / network side effects)
```

```text
$ nrl control "the market is volatile, keep the trading algorithm extra gated"
NRL control (sandboxed)
  …
  PORT_GOVERN: verdict=allow_write intent=volatile_market_extra_gate
control_outcome: DEFER — persisted writes need `nrl control --yes ...` or NRL_CONTROL_CONFIRM=1
```

```text
$ nrl control --yes "the market is volatile, keep the trading algorithm extra gated"
…
control_outcome: OK (preferences.json written under build/control)
```

---

## 2. Numerics and kernels

### 2.1 Production lane: packed INT4

- Two 4-bit potentials per byte; saturating add in `[0,15]`; threshold reset to zero when `potential >= threshold`.
- Reference implementation: `engine/src/braincore_int4_scalar.c`. Optimized AVX2: `engine/src/braincore_int4_avx2.c`. Parity-locked in `engine/tests/test_runtime.c`.

### 2.2 Experimental lower-bit lanes

- 2-bit and binary representations remain **R&D** until they beat INT4 on stability and locked-bench throughput.

### 2.3 Scheduling target

- Canonical high-dimensional planning reference: **16,384** units (power-of-two tiling, reproducible block decompositions in hierarchical kernels).

### 2.4 Certified collapse vs full iteration

**Definitions (engineering):**

- **Known structure** — transition function, invariant, lookup table, or checked rewrite rule stored in replay-stable form and applied to the current packed state without re-deriving it each timestep.
- **Full iteration** — each logical timestep applies the packed INT4 update across the whole lattice count (`baseline_equiv_updates`).

**Static transition collapse (`zpm` / `automatic`):**

- Implementation: `engine/src/zpm_int4_static.c`.
- For **static** input fields, precomputes per-nibble k-step maps and applies them in a fused pass. Preserves output semantics of the iterative kernel on supported inputs while reporting large `skip_ratio` and high `virtual_gops`.
- Legitimacy of skips is tied to **exact parity** with the iterative reference on the declared static-input contract (tests lock this).

**Hierarchical sparse routing (`omega` / `omega-hybrid`):**

- Implementation: `engine/src/zpm_omega_router.c`.
- Partitions the lattice into fixed-size sub-lattices, applies wake / signature gating, optional pruning of near-zero regions, and statistics (`nrl_omega_stats`).
- **`omega-hybrid`** enforces a **minimum active sub-lattice count** and mixes **collapsed** and **dense AVX2** execution so `executed_gops` remains substantial while retaining partial skips.

**Epistemic / compiler direction (future):**

- A compile-time or runtime **eligibility layer** may mark sub-steps as safe to collapse when invariants are certified; until then, collapse paths shipped today are those **explicitly implemented and tested** in `zpm_int4_static.c` and `zpm_omega_router.c`.

---

## 3. ABI and hot-loop contract

### 3.1 Surface

- Prefix `nrl_v1_*`; header `engine/include/nrl/nrl.h` is authoritative.

### 3.2 Variants

- Each critical kernel ships **scalar** and **AVX2** variants; optional AVX-512 / AArch64 are roadmap extensions.

### 3.3 Hot-loop rules

- No heap allocation inside hot loops.
- No dynamic CPU feature branches inside hot loops.
- Prefer branchless threshold/reset masks.
- Buffers meet alignment contracts (minimum 32-byte; 64-byte preferred).

---

## 4. Memory streaming (dual 256-bit path)

**MSP-512** denotes a **dual aligned 256-bit stream** feeding AVX2 kernels—an allocator / layout discipline to improve sustained bandwidth and prefetch predictability. It is **not** a claim of a native 512-bit SIMD execution width on AVX2-only hardware.

Requirements: dual-bank segments, page-aware placement, optional non-temporal stores for low-reuse writes, prefetch distance tuned per kernel family. Claims require bench artifacts.

---

## 5. Benchmark governance

### 5.1 Tiers

- Tier 1: vs C++ scalar `-O0` (reference domination baseline).
- Tier 2: vs C++ `-O3` (engineering baseline).
- Tier 3: cross-runtime snapshots (context only).

### 5.2 Locked profile

- Fixed `(neurons, iterations, reps, threshold, profile)` and seed policy.
- Warmup and replication policy documented with the run.
- Report medians and stability (CV) where multi-replicate harnesses apply.
- JSON artifact alongside narrative for external citation.

### 5.3 Policy

- Do not publish single-shot “peak” numbers as official.
- Always ship command line, `nrl --version`, and artifact path with claims.

---

## 6. Repository layout (target)

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
│   ├── src/
│   └── tests/
├── nrlpy/
├── language/
│   ├── spec/
│   └── examples/
├── benchmarks/
└── scripts/
```

---

## 7. Portability roadmap

- **CPU:** x86_64 AVX2 primary; scalar fallback always; optional AVX-512; AArch64 lane planned.
- **GPU:** optional future backend; CPU fabric remains primary until abstractions stabilize.

---

## 8. Phase checklist (exit criteria)

| Phase | Gate |
|-------|------|
| 0 — Contract | Architecture doc + benchmark policy adopted. |
| 1 — Foundation | `nrl_v1_*`, scalar + AVX2 INT4, dispatch, tests green. |
| 2 — Control plane | NrlPy, `.nrl` v0 path, end-to-end tests. |
| 3 — Integration | MSP-512 layout hooks, CI bench artifacts on locked profiles. |
| 4 — Extension | AVX-512 / AArch64 / optional GPU feasibility. |

---

## 9. Architecture decisions (selected)

| ID | Decision |
|----|----------|
| NRL-D001 | Engine and ABI precede surface syntax. |
| NRL-D002 | INT4 is the production numeric lane; lower-bit lanes are R&D. |
| NRL-D003 | Public throughput claims require locked profile + JSON artifacts. |
| NRL-D004 | MSP-512 is a dual-stream memory discipline, not AVX2 register width. |
| NRL-D005 | 16,384 is the canonical high-D planning reference. |
| NRL-D006 | Bounded online adaptation may update **data planes** only; shipped instruction bytes are not self-modified in production. |
| NRL-D007 | The `war-drive` profile is an **R&D ceiling** configuration; deterministic production claims remain tied to full-iteration profiles unless a separate policy explicitly allows otherwise. |

---

## 10. Execution modes (runtime policy)

### 10.1 Deterministic default (`Sovereign` in legacy docs → **frozen mode**)

- Plasticity disabled for official benches and releases unless explicitly overridden by policy.
- Reproducible execution and logging.

### 10.2 Bounded adaptation (opt-in)

- Data-plane updates only (weights, masks, gains, state tables).
- Requires telemetry, checkpoints, rollback, and replay tooling per `docs/nrl_immune_system_spec.md`.

### 10.3 Benchmark classes

- **Class A — Deterministic:** official throughput and regression gates.
- **Class B — Adaptive:** quality/stability envelopes; **must not** be reported as Class A throughput.

---

## 11. Aggressive R&D profile (`war-drive`)

- **Purpose:** explore upper performance envelope (thread pinning, aggressive streaming, experimental low-bit kernels) under the same **non-self-modifying-code** constraint.
- **Measurement:** hardware model, kernel/profile name, op definition for GOPS, warmup/replicates, median + CV, JSON artifact.
- **Safety:** watchdog, rollback path, and logging overlays required for R&D runs that exceed frozen-mode assumptions.

---

## 12. Release maturity checklist (v1.0 target)

- Frozen deterministic mode preserved.
- Optional bounded adaptation path behind explicit opt-in.
- Watchdog / rollback / replay artifacts defined for adaptive class.
- `nrl` + NrlPy + `.nrl` path operational end-to-end.
- Automated release scripts exercise build, tests, and locked harness generation.

---

## 13. Milestone log (implementation history)

Facts below reference the tree at the time of each milestone; re-run `build.ps1 -Tests` and harnesses on current hardware before citing numbers externally.

### M13.1 — ABI skeleton and reporting

- `nrl_v1_*` in `engine/include/nrl/nrl.h`, CPU bitmask, `nrl_v1_cpu_features()`, init and variant reporting (`cpuid.c`, `dispatch.c`, `capi.c`, `version.c`).
- CLI: `--version`, `--features`, `variant <kernel>`.
- Build: `build.ps1`, `build.sh`; tests: `engine/tests/test_runtime.c`.

### M13.2 — Scalar INT4 reference

- `braincore_int4_scalar.c`: packed nibbles, saturating add, threshold reset, deterministic order.
- Strict `nrl_v1_braincore_int4` argument validation in `capi.c`.
- Tests: scalar parity, replay, validation cases.

### M13.3 — AVX2 INT4 + parity lock

- `braincore_int4_avx2.c`: 32-byte step vector path, threshold-8 fast path, scalar tail for remainder.
- Dispatch binds AVX2 when available.
- Tests: scalar vs AVX2 for thresholds 8 and 12 including non-multiple-of-32 tail.

### M13.4 — Runtime introspection

- CLI `nrl runtime`: version, CPU map, active variants.

### M13.5 — `nrl bench` / `nrl run`

- Warmup + timed loops; checksum on `nrl run` for replay sanity.
- Profiles `sovereign`, `adaptive`, `war-drive` with preset defaults.
- Initial numbers captured in `benchmarks/initial_results.md`.

### M13.6 — Static transition collapse (`zpm` / `automatic`)

- `zpm_int4_static.c`: per-nibble k-step maps; single fused pass for static inputs.
- Bench output adds `executed_updates`, `baseline_equiv_updates`, `skip_ratio`, `executed_gops`, `virtual_gops`.
- Tests lock parity vs iterative path and accounting invariants.
- Example speedups and `virtual_gops` recorded in `benchmarks/initial_results.md` (host-specific).

### M13.7 — Hierarchical sparse router (`omega`)

- `zpm_omega_router.c`: sub-lattice partition, wake/prune, `nrl_omega_stats`.
- CLI profiles `omega`; bench exposes sub-lattice averages.
- Example: `nrl bench 1048576 16384 10 8 omega` — sub-second class wall times with very high `virtual_gops` on recorded host.

### M13.8 — Hybrid router profile (`omega-hybrid`)

- Enforces minimum active sub-lattices; combines collapsed and dense passes.
- Documented comparison vs `sovereign` and `zpm` in `benchmarks/initial_results.md`.

### M13.9 — NrlPy

- Package layout under `nrlpy/`; `_core` extension; `runtime.bench_cli` parses native stdout; `pytest` in `build.ps1 -Tests` when available.

### M13.10 — `.nrl` v0

- `nrl file` / `nrl <path.nrl>` in `main.c`; `nrlpy.runtime.run_nrl_file`.
- Specs under `language/spec/`; example `language/examples/omega_pass.nrl`.
- Assimilation: `nrl assimilate`, `braincore_int4_inplace`, `examples/assimilate_llm_solver.py`.

### M13.11 — NRL vs C++ harness

- `benchmarks/nrl_vs_cpp.py` emits `build/bench/nrl_vs_cpp.{json,md}`; documents speedups vs `-O0` and `-O3` baselines for locked shapes.

### M13.12 — Operator CLI

- `nrl status`, `nrl inquire`, `nrl chat` — operator-facing strings; **no LLM inside lattice kernels**; upstream LMs may sit above Nrlpy/scripts per consent and product policy.

### M13.13 — Release automation and install

- `scripts/release_check.{ps1,sh}`; `grok_review_handoff.md`.
- `scripts/install_nrl.{ps1,sh}`; `nrl status` surfaces LM/AI opt-in (env `NRL_LM_AI_OPT_IN`, else `~/.nrl/consent.json`).
- Toggle after install: **`nrl -ai on|off`** (or **`nrlpy -ai on|off`**) — updates consent and, on Windows, **`setx NRL_LM_AI_OPT_IN`** for new shells (`nrl inquire consent`).

### M13.14 — Synthetic mix benchmark (`aes256-synth`)

- Deterministic XOR/rotate ladder over 64-bit words; FNV-1a64 over 32-byte state; **not** AES-256.
- Invoked via `nrl bench … aes256-synth` or `.nrl` profile `aes256-synth`; optional `expected_fnv1a64` in `.nrl` on builds that parse it.

---

## 14. Contract audit (v1 core)

| Area | Status | Note |
|------|--------|------|
| ABI / init / features | Done | `nrl_v1_*` |
| INT4 scalar + AVX2 | Done | parity tests |
| Dispatch + profiles | Done | `sovereign`, `adaptive`, `war-drive`, `zpm`, `automatic`, `omega`, `omega-hybrid`, `aes256-synth` |
| NrlPy | Done | extension + CLI + tests |
| `.nrl` v0 | Done | parser + file dispatch |
| Bench harness | Done | `nrl_vs_cpp` artifacts |
| Release / install | Done | scripts + handoff doc |
| Deep adaptive internals | Open | policy present; full runtime deferred |
| Sentinel / governor automation | Open | labels in CLI; enforcement layer deferred |
| AVX-512 / AArch64 | Open | roadmap |

---

## 15. Virtual throughput note

On **`omega`** and **`zpm`** profiles, **`virtual_gops`** can reach **10¹²–10¹⁶ scale** on laptop-class CPUs because `elapsed_s` becomes small while `baseline_equiv_updates` remains the full nominal count. That metric answers: *“If baseline work were billed at the observed wall-clock rate, what throughput would that imply?”* It does **not** assert that every baseline update was physically executed at that rate. Use **`executed_gops`** when reporting materialized work, and disclose both numbers with the accounting definition when publishing.

---

_End of architecture specification._
