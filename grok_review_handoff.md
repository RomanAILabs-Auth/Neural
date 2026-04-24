<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Grok Review Handoff

Use this prompt with Grok for an external architecture and implementation review.

## Prompt

Review the NRL project at:
- `C:\Users\Asus\Desktop\Documents\NRL`

Scope:
1. Validate machine-code-first architecture integrity.
2. Audit benchmark claim discipline (`virtual_gops` vs executed throughput).
3. Review safety posture and identify missing guard rails.
4. Identify highest-impact performance and reliability improvements.

Constraints:
- Avoid hype language.
- Separate confirmed facts from conjecture.
- Provide findings ordered by severity.

Must review these files:
- `nrl-architecture.md` (include §2.4 epistemic compute and language contract list)
- `language/spec/nrl_physics_language_v0.md` (§6 epistemic / §8 binary assimilation)
- `docs/nrl_immune_system_spec.md` (if evaluating plasticity / learning claims)
- `engine/src/main.c`
- `engine/include/nrl/nrl.h` and `engine/src/capi.c` (packed-byte ABI + `nrl_v1_braincore_int4`)
- `engine/src/zpm_int4_static.c`
- `engine/src/zpm_omega_router.c`
- `nrlpy/src/_core/module.c` and `nrlpy/src/nrlpy/runtime.py` (inplace assimilation path)
- `benchmarks/nrl_vs_cpp.py`
- `build/bench/nrl_vs_cpp.json` (if present after harness run)

Output format:
1. Critical risks
2. High-value fixes (next 3 prompts)
3. Claim safety notes (what can/cannot be publicly claimed)
4. Performance roadmap sanity check

## Current known context

- NRL supports System 1 lanes (`zpm`, `omega`, `omega-hybrid`) and System 2 lanes (`sovereign`, `adaptive`, `war-drive`).
- Native and Python (`nrlpy`) execution paths are functional; `nrlpy run` preloads assimilation globals; `braincore_int4_inplace` mutates caller-owned packed buffers.
- `.nrl` minimal parser path is functional (`nrl file <path.nrl>`).
- `nrl assimilate` exercises sovereign INT4 with checksum parity vs Python-owned tensors.
- Architecture documents **epistemic compute** (known math → fewer executed updates vs baseline under ZPM rules) as design north star, not as “all math in the hot loop.”
- Locked NRL-vs-C++ harness artifacts are generated in `build/bench/` when the harness is run.
