# `engine/src`

Implementation of the NRL native engine: dispatch, CPU id, public C ABI, INT4 kernels (scalar and AVX2), ZPM static transition collapse, Omega fractal routing, CLI, and versioning.

## Layout intent

| Kind | Examples |
|------|-----------|
| ABI and lifecycle | `capi.c`, `version.c` |
| Feature routing | `cpuid.c`, `dispatch.c` |
| Hot kernels | `braincore_int4_scalar.c`, `braincore_int4_avx2.c`, `zpm_int4_static.c`, `zpm_omega_router.c` |
| Operator surface | `main.c` |

Hot-path rule: **machine-code-first** where latency and throughput dominate; control and policy stay outside inner loops.

## See also

- [`../README.md`](../README.md) — engine scope and entry-point table  
- [`../../nrl-architecture.md`](../../nrl-architecture.md) — full system contract  
