<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# `engine/tests`

C-level tests for the NRL engine: ABI sanity, INT4 reference parity, AVX2 lockstep checks, ZPM static accounting, and Omega statistics consistency.

## Running

From repo root (Windows):

```powershell
.\build.ps1 -Tests
```

This compiles `nrl-tests.exe` and runs the suite. POSIX: use `build.sh Release 1`.

## Expectations

- Optimized lanes must match scalar semantics on the same packed buffers where the contract demands exactness.
- Argument validation failures must return defined `nrl_v1_status` values, not crash.

## See also

- [`../README.md`](../README.md) — engine overview  
- [`../../nrlpy/tests/`](../nrlpy/tests/) — Python smoke and assimilation parity tests  
