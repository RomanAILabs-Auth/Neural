<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Physics-Native Language Surface (v0 design note)

**Status:** Design intent — not yet implemented syntax.  
**Replaces:** Any notion of “physics as a separate Python shim only.” Physics is a **first-class concern of the NRL language and toolchain**.

---

## 1. What you mean by “live neural language that natively speaks physics”

NRL programs should be able to:

1. **Declare** geometric and dynamical objects (spaces, fields, operators, conserved quantities, boundary conditions).
2. **Compile** those declarations into a combination of:
   - low-bit **neural lattice** dynamics (existing engine),
   - **symbolic / constraint** layers (future: IR + proof-carrying or certificate-oriented steps),
   - **search and verification** harnesses (bounded, reproducible).
3. **Execute** under Sovereign or Adaptive profiles with the same governance as kernels today.

“Native” means: keywords, types, and lowering are part of **`.nrl` and the NRL compiler**, not ad-hoc Python strings.

---

## 2. Relationship to famous conjectures (Riemann, Hodge, …)

**Honest contract:**

- NRL can be the **language in which you encode** a conjecture, its equivalent formulations, numerical experiments, and formal proof attempts.
- The engine can **explore** structured state spaces and emit **certificates, counterexamples, or partial results** when the mathematics allows.
- NRL **must not** claim to “solve” open problems by default. Progress is **evidence-backed** (artifacts, invariants, peer review) — same bar as the rest of RomanAILabs benchmark discipline.

So: **yes, you can aim at Riemann/Hodge class workloads** as the flagship *expressivity* demo — framed as **rigorous mathematical programming**, not marketing fiction.

---

## 3. Language concepts (v0 vocabulary — to be implemented)

These are candidate surface forms; final tokens may change when the parser graduates beyond key-value v0.

### 3.1 Space / manifold

- `space` — dimension, metric signature, basis, optional symmetry group.
- Example intent: `space S: dim=16384, dtype=int4, symmetry=...`

### 3.2 Field / entity

- `field` — degrees of freedom living on a space or product of spaces.
- Binds to packed INT4 lanes or future symbolic slots.

### 3.3 Operator / law

- `operator` — linear or non-linear maps between fields (Dirac, Hodge star, connection, curvature, …).
- **No hard-coded Navier-Stokes sheet** in the language core: the **library** ships standard operators; users compose laws.

### 3.4 Objective / conjecture block

- `objective` — what “success” means (vanishing curvature, fixed point, inequality on spectrum, …).
- `conjecture` — named bundle: axioms + objective + search budget + certificate hooks.

### 3.5 Execution profile

- Same as today: `sovereign` for deterministic official runs; `adaptive` for exploration with immune system gates.

---

## 4. Lowering strategy (engineering)

1. **Parse** physics declarations into an NRL IR (distinct from current bench key-value `.nrl`).
2. **Lower** operators to:
   - kernel invocations where a discrete bit/lattice analogue exists,
   - or external solvers / proof assistants where formal mode is enabled (future).
3. **Schedule** ZPM/Omega for **search collapse** over discrete abstractions of configuration space — not mysticism, **bounded search** with logged skips and energy bookkeeping.

---

## 5. What “craziest feature” is, precisely

The differentiator:

> **A programming language where geometry, dynamics, and discrete neural execution share one semantic core and one reproducibility contract.**

That is rarer and more defensible than “we solved RH in a weekend.”

---

## 6. Epistemic math vs brute force (ZPM alignment)

**Dream (precise):** when the system **knows** a mathematical relationship or a sub-lattice’s transition, it should **not re-pay** the full baseline cost — it should **calculate** (apply the known map) and let **ZPM-class schedulers** record fewer `executed_updates` against the same `baseline_equiv_updates` contract, with **exact parity** on declared observables.

**Reality gate:** “knows” means **stored, checkable structure** (tables, certificates, proven rewrite rules), not vibes. “Learns” means **bounded updates** to that structure under immune governance — not silent mutation of machine code.

This is how a *living* neural math engine stays honest: **System 2** earns knowledge; **System 1** (ZPM / automatic) spends it.

---

## 7. Next implementation steps (recommended order)

1. Evolve `.nrl` from key-value v0 to a **block parser** (`space`, `field`, `law`, `run`).
2. Add `nrl compile` / `nrl run` IR pipeline stub that emits **kernel call sequences** only.
3. Introduce **physics stdlib** as data + operators (still no equation hardcoding in the core parser).
4. Wire **immune spec** checkpoints on any Adaptive physics search.
5. Add an **epistemic IR** slice: named invariants + transition maps that the compiler can prove safe to feed into ZPM eligibility (long pole; architecture in `nrl-architecture.md` §2.4).

---

## 8. Binary assimilation (engine + nrlpy, implemented)

Physics lowering first lands as **packed INT4 lattice state** that Python and the CLI alias the same way the neural kernels mutate:

- **ABI:** `nrl_v1_braincore_packed_bytes(neuron_count)` returns the packed byte length (0 if invalid).
- **Kernel:** `nrl_v1_braincore_int4` updates **caller-owned** `packed_potentials` against read-only `packed_inputs` (see `engine/src/braincore_int4_scalar.c`).
- **Native CLI:** `nrl assimilate [neurons] [iters] [threshold]` — sovereign lane only; prints `checksum_fnv1a64` for parity with tensors built in Python.
- **nrlpy:** `braincore_int4_inplace`, `assimilate_cli`, `fnv1a64_packed`, and `nrlpy run <script.py>` with `nrlpy.compat.llm_globals()` so LLM-generated scripts execute on **machine code** without importing NumPy.

This is the assimilation bridge until block-syntax `.nrl` lowers full `space` / `field` IR.

---

_End of v0 design note._
