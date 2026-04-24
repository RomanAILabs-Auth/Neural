<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL language

The `.nrl` surface is the smallest deterministic control plane over the same engine profiles and kernels that `nrl` and `nrlpy` drive.

## Shipped today (v0)

| Asset | Description |
|-------|-------------|
| [`spec/minimal_nrl_v0.md`](./spec/minimal_nrl_v0.md) | Key-value grammar: `mode`, `profile`, `neurons`, `iterations`, `reps`, `threshold`; optional `expected_fnv1a64` on current `nrl` only |
| [`examples/omega_pass.nrl`](./examples/omega_pass.nrl) | Example Omega bench program |
| [`examples/aes256.nrl`](./examples/aes256.nrl) | Synthetic mix benchmark (portable keys) |
| [`examples/aes256_locked.nrl`](./examples/aes256_locked.nrl) | Same + digest lock (`expected_fnv1a64`; requires matching `nrl` build) |

Parser scope is intentionally minimal: one orchestration line style, no general-purpose expression language in v0.

## Design (not yet full syntax)

| Document | Topic |
|----------|--------|
| [`spec/nrl_physics_language_v0.md`](./spec/nrl_physics_language_v0.md) | Physics-native language intent: spaces, fields, operators, objectives; **§6** epistemic math vs ZPM; **§8** binary assimilation bridge |

Lowering goal: physics declarations compile to kernel sequences and packed state — not a separate “physics Python shim” as the source of truth.

## See also

- [`nrl-architecture.md`](../nrl-architecture.md) — language contract listing and execution fabric  
- Root [README](../README.md) — how to run `nrl file …`  
