<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Immune System Specification

**Version:** 1.0  
**Status:** Normative for v1.0 implementation scope  
**Scope:** Observability, policy enforcement, and safety hardening — not an attempt at sentience or consciousness.

---

## 1. Purpose

The NRL Immune System is a **control-plane** subsystem responsible for:

- Detecting anomalies in workload shape, routing behavior, resource use, and runtime integrity.
- Enforcing **Sovereign** vs **Adaptive** policy boundaries.
- Emitting **auditable, machine-readable** evidence for every response action.
- Preserving **reproducibility** of official benchmarks and deterministic execution paths.

Names such as `PORT_SENTINEL`, `PORT_GOVERN`, and `PORT_ALIVE` are **high-level design analogies and naming conventions only**. Implementations MUST use ordinary software modules (e.g. `runtime_guard`, `policy_engine`, `telemetry_sink`) mapped to those ports in documentation — not literal biological simulation.

---

## 2. Philosophy and Non-Goals

| Principle | Requirement |
|-------------|-------------|
| Hot-path isolation | No immune logic inside AVX2 / ZPM / Omega inner loops. |
| Control-plane only | Detection, policy, and logging run outside per-neuron hot paths. |
| Explicit modes | Sovereign (default) vs Adaptive (opt-in) behavior is always explicit. |
| Evidence | Every escalated action produces JSON suitable for CI and incident review. |
| No anthropomorphism | No product claims implying consciousness, sentience, or autonomous will. |

---

## 3. v1.0 Scope (Normative)

**Implemented in v1.0:**

- **Level 1:** Warning + structured logging (JSON event).
- **Level 2:** Light throttling (e.g. cap concurrent benchmark reps, reduce adaptive step budget, or defer non-critical work) + logging.

**Designed but deferred to Phase 2 (documented only, not required for v1.0 ship):**

- **Level 3:** Isolate affected sub-lattices or workloads.
- **Level 4:** Rollback to last checkpoint + forced Sovereign Mode.
- **Level 5:** Full freeze + diagnostic dump.

---

## 4. Guardrails Checklist

### 4.1 Input validation and sanitization

- [ ] Reject malformed CLI arguments and `.nrl` keys (unknown keys, out-of-range integers, odd `neurons` where forbidden).
- [ ] Enforce maximum `iterations`, `reps`, and allocation sizes at the ABI/CLI boundary before kernel invocation.
- [ ] Reject paths to `.nrl` outside allowed roots when running in CI or hardened profiles (optional profile flag).
- [ ] Validate `profile` names against an allow-list for official benchmark harnesses.

### 4.2 Runtime state integrity

- [ ] Verify `nrl_v1_init` completed successfully before any kernel dispatch.
- [ ] Detect impossible combinations (e.g. `omega` profile with workload dimensions that violate router contracts).
- [ ] Optional checksum sampling on output buffers for harness-only regression modes (not default hot path).

### 4.3 Plasticity and Adaptive Mode safety

- [ ] Adaptive Mode MUST be opt-in via explicit flag or environment contract; Sovereign is default.
- [ ] When Adaptive is off, any plasticity mutation API MUST return `disabled` without side effects.
- [ ] Bounded update magnitude and per-tick budget MUST be enforced in the plasticity controller (when present), not in kernels.

### 4.4 Memory and resource abuse prevention

- [ ] Hard caps on total bytes requested per `bench` / `run` invocation at CLI layer.
- [ ] Detect runaway subprocess fan-out from benchmark harnesses (single-flight guard).
- [ ] Time wall-clock ceilings for harness runs in CI profiles.

### 4.5 Skip ratio and Omega routing anomalies

- [ ] Flag sudden drop in `skip_ratio` vs locked baseline for the same profile and hardware class (regression signal).
- [ ] Flag `skip_ratio == 1.0` with **zero** executed updates when profile semantics require non-zero active work (integrity signal).
- [ ] Flag inconsistent `avg_active_sublattices` vs `total_sublattices` for `omega` family profiles.

### 4.6 Entropy and instability monitoring

- [ ] Track rolling variance of wall-clock per locked workload; spike beyond threshold → Level 1.
- [ ] Optional: hash of benchmark JSON artifact mismatch vs committed baseline in CI → Level 1.

### 4.7 Self-modification prevention

- [ ] Immune layer MUST NOT rewrite executable code pages or shipped kernel images.
- [ ] All mutation limited to data planes explicitly defined by architecture (weights, masks, state buffers).

---

## 5. Detection Signals and Metrics

| Signal ID | Metric | Source | Typical Level 1 trigger |
|-----------|--------|--------|-------------------------|
| `SIG_WALL_SPIKE` | Median wall time / CV across harness repeats | harness | CV or delta vs baseline > policy |
| `SIG_SKIP_ANOMALY` | `skip_ratio` vs expected band for profile | `nrl bench` parse | Outside `[low, high]` for locked profile |
| `SIG_ZERO_EXEC` | `executed_updates == 0` with non-degenerate workload | bench output | Always investigate |
| `SIG_ARG_VIOLATION` | ABI error rate | CLI / C ABI | Any `NRL_ERR_ARGS` in official harness |
| `SIG_MEM_CAP` | Requested bytes > cap | CLI pre-check | Block + Level 1 |
| `SIG_ADAPTIVE_LEAK` | Plasticity API invoked while Sovereign | controller | Level 2 throttle + log |

Metrics MUST be collected in the **control plane** (Python harness, CLI wrapper, or post-run analyzer), not inside per-byte kernel loops.

---

## 6. Escalation Ladder

### Level 1 — Warning + log

**Actions:**

- Emit JSON event: `{"level":1,"signal_id":"...","action":"warn",...}`  
- Continue execution unless policy forbids.

**v1.0:** REQUIRED.

### Level 2 — Light throttle + log

**Actions:**

- Reduce allowed `reps`, shrink watchdog budget for Adaptive sandbox, or insert cooling delay between back-to-back harness invocations.
- Emit JSON event with `action:"throttle"` and parameters applied.

**v1.0:** REQUIRED.

### Level 3 — Isolate (Phase 2)

**Actions:**

- Mark affected workload slice as quarantined; route remaining work on clean path.
- Emit JSON with isolation scope.

**v1.0:** SPEC ONLY.

### Level 4 — Rollback + Sovereign (Phase 2)

**Actions:**

- Restore last checkpointed data-plane state; force Sovereign Mode.
- Emit JSON with checkpoint id and reason.

**v1.0:** SPEC ONLY.

### Level 5 — Freeze + diagnostic dump (Phase 2)

**Actions:**

- Halt adaptive controllers; write full diagnostic bundle path in JSON.

**v1.0:** SPEC ONLY.

---

## 7. Implementation Constraints

1. **Zero hot-path overhead:** No immune branches in `braincore_int4` AVX2 inner loops, ZPM static maps, or Omega router inner loops beyond existing validation already required by ABI.
2. **Single writer for policy:** One module owns escalation state machine; kernels remain stateless aside from documented outputs.
3. **JSON schema stability:** Event field names are stable across patch releases within v1.x; additive fields allowed.
4. **Testability:** Level 1 and Level 2 behaviors MUST be unit-testable without GPU and without network.

---

## 8. Integration Points

| Component | Integration |
|-----------|-------------|
| CLI (`nrl` entry) | Pre-flight validation; optional `--immune-strict` for CI. |
| Dispatch / runtime init | Publish feature and mode snapshot consumed by immune checks. |
| Benchmark harness (`nrl_vs_cpp.py`, future gates) | Post-parse validation of JSON; compare to baselines. |
| Plasticity controller (future) | Sole mutation path; immune hooks on entry/exit of adaptive ticks. |
| `PORT_*` (documentation) | Maps to `sentinel_metrics`, `govern_policy`, `alive_watchdog` modules in implementation docs only. |
| Adaptive runtime architecture | [`nrl_alive_language_evolution_architecture.md`](./nrl_alive_language_evolution_architecture.md) — workload identity, specialization store, shadow executor, plasticity; MUST stay consistent with §4.3 and §4.7. Schemas in [`schemas/`](./schemas/). |

---

## 9. Logging and Audit Requirements

- Every Level 1 and Level 2 event MUST append or write one JSON object per line (JSONL) to a configurable path, default: `build/immune/events.jsonl` (development) or user-specified directory (production).
- Required fields: `ts_utc`, `level`, `signal_id`, `action`, `profile` (if any), `workload_hash` (optional), `message`, `build_id` or `nrl_version`.
- Official CI MUST archive `events.jsonl` as an artifact when `--immune-strict` is enabled.

---

## 10. Sovereign vs Adaptive Behavior

| Mode | Plasticity | Immune default | Benchmark official claims |
|------|------------|----------------|---------------------------|
| Sovereign | Disabled | Level 1–2 only; block adaptive APIs | Allowed |
| Adaptive | Enabled per contract | Level 1–2 active; Phase 2 escalation enabled when implemented | Class B only; never substitute for Sovereign claims |

---

## 11. Acceptance Criteria for v1.0 Immune Ship

- [ ] Level 1 and Level 2 behaviors implemented behind a single feature flag or CLI flag.
- [ ] At least three unit tests: arg cap, skip anomaly detection, wall spike warning.
- [ ] Documentation link from root `README.md` to this spec.
- [ ] No measurable regression in median `sovereign` bench time on reference hardware when immune is enabled vs disabled (threshold to be defined in harness config, e.g. `< 2%`).

---

## 12. Document Control

| Version | Date | Author | Notes |
|---------|------|--------|------|
| 1.0 | 2026-04-22 | RomanAILabs | Initial v1.0 scope: L1–L2 only; Phase 2 ladder reserved |

_End of specification._
