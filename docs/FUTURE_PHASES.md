<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Future Phases: Bio-Digital Brain P7+

The Bio-Digital Brain MVP v3.0 is complete. The next phases should remain
local-first, opt-in, reversible, and measurable. Each phase below is intentionally
scoped as a roadmap item, not a shipped claim.

## P7: Multi-Model Fusion

**Goal:** absorb and merge multiple GGUFs into one coordinated brain.

Scope:

- Absorb multiple LMOs under one fusion manifest.
- Build a shared router graph across model anchors.
- Track per-model provenance for every ZPM entry and muscle-memory replay.
- Allow operator-selected policies: ensemble, specialist routing, or fallback.

Safety:

- Never erase original LMOs.
- Fusion outputs must cite source model SHA prefixes in telemetry.
- Merging must be dry-run previewable before writes.

Acceptance direction:

- Two tiny fixture LMOs can be fused, queried, unfused, and recovered without
  corrupting either source.

## P8: Distributed Learn Mode

**Goal:** run Learn Daemon across multiple machines and sync ZPM deltas safely.

Scope:

- Append-only ZPM delta bundles with signatures and model SHA binding.
- Conflict detection for duplicate anchors and divergent replies.
- Pull-only and push-pull sync modes.
- Bandwidth and disk quotas per peer.

Safety:

- Default off.
- No remote code execution.
- Deltas are data-only and replay through the same WAL validation path.

Acceptance direction:

- Two local roots exchange ZPM deltas and converge to the same entry count after
  recovery.

## P9: Vision + Audio Absorption

**Goal:** extend LMO absorption beyond text-only GGUF into multimodal towers.

Scope:

- CLIP-style vision tower metadata and tile layout.
- Whisper-style audio encoder metadata and tile layout.
- Shared manifest schema for modality-specific anchors.
- Coverage metrics split by text, vision, and audio buckets.

Safety:

- Keep modality loaders explicit; never infer file type from extension alone.
- Preserve source hashes and licensing metadata for every tower.

Acceptance direction:

- Fixture vision/audio towers can be absorbed, inspected, and unloaded without
  changing text-only behavior.

## P10: Bounded Self-Modifying Code

**Goal:** allow opt-in code patch proposals with rollback, tests, and operator approval.

Scope:

- Generate patch candidates only in a sandbox worktree.
- Require tests, diff summary, and operator approval before applying.
- Keep rollback checkpoints for every accepted patch.
- Restrict editable paths by policy.

Safety:

- Default off and gated behind explicit consent.
- No secret access, network access, or destructive git operations.
- Every accepted change must be reproducible from a patch artifact.

Acceptance direction:

- The system can propose a small documentation patch, run tests, and roll it
  back cleanly in a sandbox.

## Stretch Goals

- LMO federation browser for local models.
- Visual coverage heatmap for Drift Conqueror buckets.
- Signed release artifacts for native binaries and wheels.
- A deterministic benchmark pack that separates decode, replay, ZPM, and prune costs.
- Policy profiles for research, offline production, and safe demo mode.
