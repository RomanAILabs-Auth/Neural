<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Bio-Digital Brain MVP v3.0 Release Notes

Release status: **GitHub Release Ready**

## Summary

Bio-Digital Brain MVP v3.0 turns NRL into a local, disk-native lifecycle for
GGUF models: absorb once into an LMO, chat through a resolution ladder, map weak
ZPM regions while idle, recover through WAL/snapshots, and keep disk use bounded
with quota pruning.

## Highlights

- **P1 LMO Absorption:** `nrlpy absorb` creates disk-native model objects with
  headers, packed INT4 tiles, retained source bytes, router graph, manifest, and
  attestation.
- **P2 Learn Daemon:** idle-gated background learning with CPU cap and safe
  status snapshots.
- **P3 Chat Integration:** `nrlpy chat <model.gguf> --rewired` routes through
  muscle memory, ZPM nullspace, Omega resolve, and decode fallback.
- **P4 Persistence:** ZPM WAL, snapshots, recovery, and `nrlpy lmo info`.
- **P5 Drift Conqueror:** coverage metrics, weak-bucket targeting, bounded
  conquest prompts, and `nrlpy lmo coverage`.
- **P6 Quota Management:** `NRL_LMO_MAX_GB`, dry-run prune, LRU ZPM eviction,
  WAL compaction, and `nrlpy lmo prune`.
- **P7 Final Integration:** end-to-end lifecycle test, `NRL_SAFE_MODE=1`,
  `nrlpy doctor`, and operator documentation.
- **P8 Release Prep:** GitHub templates, contribution/security policy, examples,
  release packaging, and future roadmap.
- **P9 IP-Protective License:** RomanAILabs Proprietary Source-Available
  Evaluation License 1.0, copyright retention, patent reservation, commercial licensing
  path, attribution requirements, and anti-competitive restrictions.

## Quick Verification

```powershell
cd nrlpy
python -m pytest tests/test_bio_digital_brain_e2e.py -v
python -m nrlpy doctor
```

## Important Environment Variables

| Variable | Purpose |
|----------|---------|
| `NRL_ROOT` | Runtime cache root for LMO/ZPM/MM data. |
| `NRL_SAFE_MODE=1` | Disable background Learn Mode, WAL writes, and auto-prune hooks. |
| `NRL_LEARN_MODE=1` | Enable idle Learn Daemon cycles. |
| `NRL_LEARN_MAX_GROWTH_PCT=5` | Cap ZPM index growth in a 24-hour idle window. |
| `NRL_LMO_MAX_GB=100` | Per-model footprint quota. |
| `NRL_LMO_AUTO_PRUNE=1` | Optional post-persist quota check; off by default. |

## Known Boundaries

- NRL does not ship model weights.
- GGUF/model licenses remain the responsibility of the operator.
- Quota pruning evicts ZPM/snapshot/WAL data; it does not delete immutable LMO
  retained source bytes or packed tiles.
- The E2E soak in CI uses fake time plus memory assertions; run a real soak for
  release hardware validation.
