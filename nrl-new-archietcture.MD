<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL New Architecture (Reset Plan)

This document replaces the current "dense-pass-through first" trajectory with the intended design:

- NRL is the decision engine.
- GGUF numerics are execution substrate.
- Decode compute is actively elided by NRL gate policy.
- Throughput is reported honestly as executed vs virtual vs effective.

The goal is not "llama-like chat with extra telemetry."  
The goal is a fast, coherent, stable NRL-governed chat runner.

---

## 1) Problem Statement (Current Gap)

Current runner behavior:

- Multi-turn chat works.
- Coherence is improved with Phi-3 prompt format.
- Path handling and REPL stability are mostly fixed.

But current performance profile is still unacceptable for target outcomes because:

1. Decode remains mostly dense in libllama.
2. NRL skip signals are not yet fully applied to decode work in the hot loop.
3. NRL benchmark burn speed and GGUF chat decode speed are not yet coupled by a strict gate execution contract.

Result: chat feels similar to or slower than standard dense decode.

---

## 2) Clarification of Throughput Claims

"45k words/sec" style values come from extreme virtual accounting envelopes in NRL lattice benchmarks, not current dense chat decode.

Going forward, all claims must be split and published as:

- `executed_tps` / `executed_words_per_sec` (materialized decode)
- `virtual_tps` / `virtual_words_per_sec` (baseline-equivalent under applied skip)
- `effective_tps` / `effective_words_per_sec` (materialized + replay)

No single number without label.

---

## 3) Architecture Target (Authoritative)

### Plane A: NRL Control + Policy

- Source of truth is `.nrl` manifest (every GGUF run materializes one).
- NRL defines gate policy per token:
  - layer mask
  - expert mask (future MoE path)
  - KV keep/reuse mask
- NRL owns deterministic replay lock for policy decisions.

### Plane A.5: ZPM Zero-Point Identity Resolver (Pre-libllama)

NRL-native port of ROMA-ZPM v2.0 (`docs/ROMA-ZPM-Grok.txt`). Lives in
`nrlpy/src/nrlpy/zpm.py`. Opt-in via `manifest.zpm_nullspace = true` or
`NRL_ZPM=on`. Near-match tolerance via `zpm_threshold_bits` / `NRL_ZPM_THRESHOLD`
(0 = exact-only).

Pipeline per turn, before `muscle_memory_lookup` falls through to libllama:

1. **Anchor (Stage I)** — materialize a 256-bit state (4 × uint64) from
   `model_sha256 | rendered_prompt | sampler_fp | seed | max_tokens` via four
   independent FNV-1a64 projections (identity, reverse, half-swap, quarter-shift).
   Deterministic, replay-lockable.
2. **Inversion (Stage II)** — GF(2) row parity: `rows[i] = t[i] ^ (t[i] << 1)
   ^ 0x9E3779B97F4A7C15`. Row XOR determinant = UNITY signature.
3. **Rotor (Stage III)** — Cl(4,0) scalar/bivector with seed-derived angle
   `(t[0] ^ t[1]) * 1e-9`. Phase-locked when `sqrt(s² + b_xy²) > 0.999`
   (true for every 64-bit seed by `sin² + cos² = 1`).
4. **Nullspace (Stage IV)** — XOR + popcount against every stored unity state
   in `$NRL_ROOT/cache/zpm/<model_sha>/index.bin`. Reports Hamming distance
   in bits. `dist == 0` → exact. `0 < dist ≤ threshold` → near-match.
5. **Verify (Stage VI)** — bit-symmetry audit: residual must be zero for
   an exact match before the stored reply is served. Never serve a
   near-match under the exact label.

**Contract**: ZPM only persists what muscle-memory already accepted
(`muscle_memory == "on"` gate for writes). This keeps the index a strict
refinement of MM, never looser. Every ZPM hit records
`gate_source="zpm_nullspace"` + `distance_bits` in the evidence log, so
near-match serves are auditable and the honesty hinge (§7) is preserved.

**Why this hits the 1000 WPS target on *any* hardware**: a ZPM hit emits
tokens at memory-I/O speed (disk read + byte copy to stream_to), not at
decode speed. With the `wps_chat_bench` replay phase we already measure
>7,000 WPS on CPU-only hardware for exact-hash muscle-memory hits. ZPM
generalises those hits across cosmetic prompt variations via the Hamming
tolerance knob, broadening the addressable hit surface.

**What ZPM is NOT**: it is not a model. It is not a learner. It is not
sparse matmul. It never synthesizes novel tokens. It is a strict
replay-identity resolver with a declared confidence bound.

**Phase-1 status (shipped)**: exact-match collapse via FNV-1a64 anchor.
Every turn that renders byte-for-byte identical to a stored entry serves
at memory-I/O speed (`distance_bits = 0`, `gate_source =
"zpm_nullspace"`). Measured on stub backend: ~10 ms serve including
dataclass allocation; on muscle-memory replay elsewhere in this repo
we've measured >7,000 WPS effective throughput for the same class.

**Phase-2 follow-up (planned)**: locality-sensitive anchor. The current
FNV-1a64 projection is cryptographically-scrambled — a single byte
flip in the rendered prompt changes ~half the output bits, so the
`zpm_threshold_bits > 0` knob is architecturally wired but not yet
useful in practice (near-matches look like full misses). The next
commit slot is `anchor_simhash()`: SimHash over char-trigram tokens so
small edit-distance changes (trailing whitespace, case, simple
punctuation variants) yield small Hamming distances. The existing
`ZpmIndex.lookup(..., threshold_bits=K)` path, the evidence log's
`distance_bits` field, and the CLI already support this — only the
anchor function swaps in.

Demo: `nrlpy zpm [--target <hex>] [--text "..."]` prints the full
Stage I–VI banner for any input.

### Plane B: Applied Decode Gate

- Gate decisions are applied before/inside decode execution path.
- Shared-prefix / prefill cache, then layer skip, then KV policy.
- Skip policy changes real compute, not just metadata.

### Plane C: Numerics

- GGUF execution backend performs numerics for active work only.
- NRL does not pretend to be GEMM; it decides what is computed.

---

## 4) Performance Objective

Primary product target:

- Minimum `1000 words/sec` in production throughput mode.
- No artificial upper limit on throughput; the system should scale with model, hardware, and applied skip policy.
- Coherence and stability remain mandatory (speed is not allowed to degrade output quality beyond declared mode contracts).

Universal throughput model:

```text
updates_per_sec = gops * 1e9
tokens_per_sec  = updates_per_sec / updates_per_token
words_per_sec   = tokens_per_sec * words_per_token
```

Equivalent single expression:

```text
words_per_sec = (gops * 1e9 / updates_per_token) * words_per_token
```

Success definition:

1. Coherent chat output under stable preset.
2. Reproducible accounting logs.
3. Demonstrable uplift from NRL-applied skip in real decode path.
4. Measured and published min/p50/p95 words/sec per mode and per model family.

---

## 5) Hard Delivery Phases

## Phase 1 - Manifest-First Execution (Immediate)

- `nrl run <model.gguf>` always generates a concrete run manifest.
- Every run logs manifest path + key policy values.
- No hidden runtime state for critical decode policy.

Acceptance:

- CLI prints generated manifest path.
- Re-running same manifest reproduces run policy.

## Phase 2 - Applied Prefill/KV Gate (Real Compute Elision)

- Promote shared-prefix gate from structural accounting to enforced decode reuse path.
- Ensure chat turns reuse context instead of rebuilding avoidable work.

Acceptance:

- Reduced decode wall-time across repeated-prefix turns.
- `gate_source` reflects active applied source in evidence log.

## Phase 3 - Layer Gate in Decode Hot Loop

- Bind per-token NRL layer mask to decode execution path.
- Preserve deterministic policy under fixed seed and model hash.

Acceptance:

- Non-zero applied skip ratio from layer gate.
- Measured executed decode uplift vs dense baseline on identical prompt set.

## Phase 4 - Coherence and Stability Guardrails

- Add fast/coherent policy lanes:
  - `fast-stable`
  - `fast-balanced`
  - `max-throughput`
- Enforce safe generation constraints per lane.

Acceptance:

- Regression suite validates response quality floor + no runaway degenerate output.

## Phase 5 - Words/sec KPI Harness

- Add benchmark harness that reports:
  - words/sec and tokens/sec
  - executed/virtual/effective split
  - coherence indicators (basic policy checks)
  - min/p50/p95 and tail latency

Acceptance:

- Automated artifact generation for publishable runs.

## Phase 6 - Minimum 1000 WPS Gate

- Add a release gate asserting `words_per_sec >= 1000` for designated throughput mode(s) on certified hardware/model lanes.
- Gate is profile-aware and model-family-aware; no universal cap is applied.

Acceptance:

- CI/live readiness artifact includes pass/fail for the 1000 WPS minimum.
- Report includes exact mode, model SHA, and calibration constants used.

---

## 6) Coherence-First Throughput Modes

Default chat mode should be `fast-stable`:

- lower randomness
- bounded reply length
- context reuse on by default
- minimal non-essential probes

`max-throughput` mode can be more aggressive, but must still declare:

- reduced quality guardrails
- benchmark-only suitability when applicable

Proof-of-concept operator surface:

- `nrl run -specs <model.gguf>`
- This is a simplified high-speed coherent lane intended to show immediate
  interactive behavior without requiring full manifest tuning.

---

## 7) Instrumentation Contract

Per turn, record:

- model hash
- applied gate source
- applied skip ratio
- executed/virtual/effective tps
- words/sec estimate from emitted text
- cache hit/miss
- wall time

This data must be emitted consistently to evidence logs and optional summary view.

Calibration constants required per model lane:

- `updates_per_token` (measured from controlled dense reference runs)
- `words_per_token` band (e.g., 0.6 to 0.8) and observed session value

Projected and measured throughput must both be recorded:

- `projected_words_per_sec_{min,max}`
- `measured_words_per_sec_{executed,virtual,effective}`

---

## 8) Non-Negotiable Rules

1. No throughput claim without metric label.
2. No "virtual" claim unless applied skip source is explicit.
3. No "fast mode" that silently destroys coherence.
4. No runner path that bypasses `.nrl` policy provenance.

---

## 9) Immediate Next Build Actions

1. Wire manifest materialization and print path for every `nrl run <gguf>`.
2. Enable strict run provenance logging.
3. Implement applied prefill/KV reuse metrics with hard tests.
4. Add words/sec reporting helper in session and bench summaries.

---

## 10) Definition of Done (This Reset)

This reset is complete when:

- Users can run one command and get coherent chat plus high stable throughput,
- NRL policy is visibly in control of decode compute,
- Reported numbers are reproducible and separated by executed vs virtual vs effective,
- Performance deltas are attributable to applied NRL gates, not accidental prompt/cache artifacts.

