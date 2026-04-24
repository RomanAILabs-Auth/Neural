<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Engine

The NRL engine is the machine-code-first core: INT4 neural lattice kernels, CPU feature dispatch, ZPM / Omega routing, and the native `nrl` CLI bound to a stable C ABI (`nrl_v1_*`).

## Scope

- Kernel execution (`braincore_int4` scalar + AVX2; ZPM static; Omega virtual / hybrid)
- One-shot feature detection and dispatch table init (`engine/src/cpuid.c`, `dispatch.c`)
- Public ABI surface (`engine/include/nrl/nrl.h`, `engine/src/capi.c`)
- Native CLI: version, features, profiles, bench/run, `.nrl` file execution, assimilate, operator introspection

## How to work in engine (human + LLM)

- Treat `engine/include/nrl/nrl.h` as the contract; breaking changes require version discipline and changelog entries.
- Keep SIMD paths parity-locked to the scalar reference for `braincore_int4`.
- Keep ZPM and Omega paths **exact** on declared observables where the architecture promises exactness; document any intentional statistical / virtual lanes separately.
- Prefer measurable changes: extend `engine/tests/test_runtime.c`, re-run `build.ps1 -Tests`, then update `benchmarks/initial_results.md` only with locked harness context when publishing.

## Entry points

| Concern | Location |
|---------|----------|
| CLI and commands | `engine/src/main.c` |
| Public ABI | `engine/include/nrl/nrl.h`, `engine/src/capi.c` |
| INT4 scalar / AVX2 | `engine/src/braincore_int4_scalar.c`, `braincore_int4_avx2.c` |
| ZPM static accelerator | `engine/src/zpm_int4_static.c` |
| Omega router | `engine/src/zpm_omega_router.c` |
| Runtime tests | `engine/tests/test_runtime.c` |

## Collaborators and attribution

RomanAILabs honors collaboration with Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, and Cursor.

Primary author: Daniel Harding (RomanAILabs) — `daniel@romanailabs.com` | `romanailabs@gmail.com` | `romanailabs.com`
