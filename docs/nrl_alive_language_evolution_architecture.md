# NRL Adaptive Runtime and Specialization Architecture

**Version:** 1.0  
**Status:** Normative intent for control-plane evolution; native hot paths remain frozen unless separately specified.  
**Normative peers:** [`nrl-architecture.md`](../nrl-architecture.md), [`nrl_immune_system_spec.md`](./nrl_immune_system_spec.md).  
**JSON schemas:** [`schemas/`](./schemas/).

---

## 1. Scope

This document specifies how NRL may **accumulate verified performance artifacts** and **bounded data-plane adaptations** without:

- modifying shipped executable code in production,
- moving policy or ML inference into INT4 inner loops,
- or implying non-measurable “intelligence” or agency.

Execution remains **`nrl` / `libnrl`**; orchestration and evidence remain **Python (`nrlpy`) and harnesses**.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Workload descriptor** | A JSON-serializable record of **inputs** that determine kernel selection and accounting semantics: harness id, profile(s), `neurons`, `iterations`, `reps`, `threshold`, static-drive fingerprint (when applicable), engine ABI version string, and CPU feature class. **Excludes** wall times and GOPS (those are *observations*, not identity). |
| **`structural_hash`** | `SHA-256` (hex) of the **canonical** JSON representation of the workload descriptor: UTF-8, `sort_keys=True`, minimal separators (`","`, `":"`). Same descriptor → same hash. |
| **`workload_id`** | Opaque string `"{harness_id}|{structural_hash}"` used as a primary key in logs and specialization indices. |
| **Verified specialization cache** | A store of **pre-validated** auxiliary data (e.g. static transition maps, router tables, fused schedules, PGO-style tuning hints) **keyed** by `workload_id` / `structural_hash`, with a **compatibility manifest** (engine version, ABI, policy id). |
| **Plasticity controller** | The sole **adaptive-write** path: applies **budgeted** updates to **allowed data planes** (weights, masks, gains, state tables) when **Adaptive** mode is enabled; returns `disabled` in **Sovereign** mode. Per immune spec §4.3. |
| **Shadow executor** | CI or batch facility that runs a **reference** configuration and a **candidate** configuration on identical seeds and buffers, then compares **contracted observables** (checksums, tolerance envelopes, `executed_updates`, `skip_ratio` bands). **Promotion** of a candidate requires passing policy. |
| **Evidence store** | Append-only (or WORM-style) log of immune and specialization events (`specialization_built`, `specialization_promoted`, `specialization_rejected`, `rollback`, `freeze`, `drift_detected`) with JSON lines suitable for audit. |
| **Workload similarity index** | Optional offline component producing **ranked suggestions** for near-matching descriptors (e.g. embedding of IR or tensor-shape features). Outputs **candidates only**; never bypasses shadow promotion. |

**Execution-profile vocabulary** (replace informal names in prose):

| Informal / legacy UI name | Technical reading |
|---------------------------|-------------------|
| Sovereign / frozen mode | **Full iterative execution**: baseline-equivalent updates materialized per policy. |
| ZPM / automatic | **High-skip static-collapse mode**: precomputed transition application for fixed drives; large `skip_ratio` when valid. |
| Omega | **Hierarchical sparse routing**: sub-lattice wake/prune; interpret `virtual_gops` with accounting. |
| Omega-hybrid | **Sparse routing with enforced dense floor**: lower bound on active sub-lattices. |
| Adaptive / war-drive (policy) | **Aggressive optimization / R&D profile**: bounded adaptation or ceiling tuning **only** under explicit policy; never substitutes for locked Class-A throughput claims. |

---

## 3. Layered architecture

Four layers; responsibilities do not merge.

| Layer | Components | May mutate |
|-------|------------|------------|
| **Human / CI** | Review, policy compile, merge gates | Policy repos, harness configs |
| **Control plane** | `nrlpy`, harnesses, evidence exporters | Evidence files, reports |
| **Adaptation services** | Plasticity controller, specialization store, shadow executor, similarity index (optional) | Data-plane blobs + manifests only |
| **Native engine** | `nrl`, `libnrl`, AVX2/scalar kernels | Nothing at runtime in production ship mode |

**Contract NAL-001:** No adaptation or cache promotion **inside** per-neuron hot loops. Specializations enter only after **equivalence** checks (shadow + manifest) keyed by `structural_hash` and policy id.

**Contract NAL-002:** No ML inference on the neuron update critical path unless a future **signed coprocessor contract** exists; default remains control-plane only.

---

## 4. Modules (functional names)

### 4.1 `workload_signature`

**Inputs:** Workload descriptor fields (§2).  
**Outputs:** `workload_id`, `structural_hash`, canonical JSON bytes (for replay).  
**Role:** Identity for specialization lookup and deduplication.

### 4.2 `specialization_store`

**Stores:** Opaque blobs + metadata (format version, engine hash, policy id, promotion timestamp).  
**Retrieval:** Query by `structural_hash`; rank by observed wall time and checksum success under shadow history.  
**Invalidation:** Mismatch on engine version, ABI, policy root, or CPU class → entry **inactive** until re-promoted.

### 4.3 `plasticity_controller`

Sovereign: read-only / `disabled`. Adaptive: bounded writes with per-tick budget and magnitude caps; each mutation emits **JSON evidence** (`immune_event_v1` family; see schemas).

### 4.4 `shadow_executor`

Reference run vs candidate run → compare contracted observables → **promote** | **reject** | **defer**. No promotion without passing policy.

### 4.5 `evidence_log`

Append-only JSONL; supports replay and incident analysis. Required fields per immune spec §9 (extend with `workload_id`, `structural_hash` when present).

### 4.6 `telemetry_tuning` (optional)

Selects variant / tile / thread count within **policy caps** from historical telemetry. Does not skip shadow validation for new specializations.

### 4.7 `workload_similarity` (optional, R&D)

Offline or sidecar: embeddings → ranked candidate `workload_id`s. **Suggestions only**; shadow path mandatory.

---

## 5. `nrlpy` responsibilities

| Area | Responsibility |
|------|----------------|
| Descriptor + hash | Implement `workload_signature` (see `nrlpy.workload`). |
| Harnesses | Emit `workload_id` / `structural_hash` in benchmark JSON artifacts. |
| Evidence | Append immune and promotion events; future CLI: `nrlpy evidence export`. |
| Operator chat (`nrlpy chat`) | English phrases map to **deterministic** telemetry (version, features, INT4 probe, evidence JSONL tail, optional `psutil` thermals). Session turns append to `build/nrlpy_chat/session.jsonl` for recall only — not model training. |
| Bounded vocab store (`nrlpy learn`) | On-disk **frequency table** under `NRL_LEARN_DIR` (default `build/nrlpy_learn`): `vocab.json` + `growth.jsonl`. Default cap **4 GiB** (`max_bytes`); lowest-count tokens pruned when serialized size would exceed `0.85 × max_bytes`. Chat ingests tokens from every line unless `NRL_LEARN_DISABLE=1`. |
| ML / embeddings | Out-of-process or optional import; never block `braincore` inner loops. |

---

## 6. Phased delivery

| Phase | Deliverable | Acceptance |
|-------|-------------|------------|
| **P0** | JSON Schemas: `workload_descriptor_v1`, `immune_event_v1` (minimal), `specialization_manifest_v1` (stub) | Validated examples in `docs/schemas/` |
| **P1** | Official harness emits `workload_id` + `structural_hash` | `nrl_vs_cpp` artifact contains `workload_identity` block |
| **P2** | Read-only specialization index (file or DB) | Replay timing within declared tolerance vs baseline |
| **P3** | Shadow executor in CI for one profile family | Zero false promotions on locked suite |
| **P4** | Plasticity controller stub (Sovereign default; Adaptive = log-only writes) | Immune §4.3 satisfied |
| **P5** | Bounded data-plane updates + rollback checkpoint | Level-4 rollback test |
| **P6** | Optional `workload_similarity` suggestions | Human or CI gate on promotion |

Order is mandatory: **P0 → P1 → evidence/shadow before plasticity writes.**

---

## 7. Instant reuse (correct statement)

| Allowed | Condition |
|---------|-----------|
| O(1) reuse of **verified** specialization | Exact `structural_hash` match (or declared equivalence class with proof artifact). |
| O(1) retrieval of **cached numeric result** | Same workload descriptor and same numerical policy. |

| Disallowed without proof | |
|-----------------------------|---|
| Correctness for **novel** workload classes | Requires proof, exhaustive bounded check, or explicit statistical risk budget (separate product lane). |

Default path: **hash match** for instant reuse; similarity feeds **shadow** only.

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| Performance regression | Locked harness + artifact diff in CI |
| Poisoned specialization | Signed manifests; path allow-lists |
| Latency blow-up | Keep adaptation off hot path |
| Speculative claims | Separate research publications from ship artifacts |

---

## 9. Summary

The production architecture is **strict at the kernel**, **adaptive only in the data plane**, and **evidence-backed everywhere learning is claimed**. `nrlpy` and harnesses provide identity, hashes, and logs; the engine executes; **specialization and plasticity are optional services under immune policy**, not properties of the inner loop itself.

---

## 10. Operator control plane (CLI, separate from lattice evolution)

Natural-language **operator bias** (`nrl chat`, `nrl control`) is intentionally **not** part of the alive-language / plasticity spec above: it writes only sandboxed JSON under `build/control` and never reaches the INT4 hot loop. See [`nrl-architecture.md`](../nrl-architecture.md) §1.5. Nrlpy may optionally apply those hints when shelling out to `nrl bench` (`respect_control_hints`). A future immune bridge could reconcile operator hints with plasticity policy; until then, treat them as **orthogonal control channels**.
