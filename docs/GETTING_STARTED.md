<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Getting Started: Bio-Digital Brain MVP v3.0

This guide walks through the final MVP path: absorb a GGUF into an LMO, chat,
let Learn Mode map weak ZPM regions while idle, inspect coverage, and prune if
the disk quota is getting tight.

## 1. Install And Check Health

```powershell
cd C:\Users\Asus\Desktop\Documents\NRL\nrlpy
python -m pip install -e .
nrlpy doctor
```

`nrlpy doctor` checks Python, `NRL_ROOT`, disk space, native core import, and
safety flags. A healthy install prints `status: healthy`.

## 2. Absorb A Model

```powershell
nrlpy absorb C:\models\Phi-3-mini-4k-instruct.Q4_K_M.gguf
```

Absorption writes the disk-native LMO under:

```text
$NRL_ROOT/cache/lmo/<model_sha256>/
```

Record the first 8 or more hex characters of the printed `model_sha256`; the
commands below accept that SHA prefix.

## 3. Chat Through The Resolution Ladder

```powershell
nrlpy chat C:\models\Phi-3-mini-4k-instruct.Q4_K_M.gguf --rewired
```

The rewired preset keeps the normal decode path available while allowing R0
muscle memory, R1 ZPM nullspace, and R2 Omega resolve to serve repeatable turns
when they are eligible.

## 4. Let It Dream Overnight

Enable Learn Mode before leaving the machine idle:

```powershell
$env:NRL_LEARN_MODE = "1"
$env:NRL_LEARN_CONQUEST_IDLE_SEC = "300"
$env:NRL_LEARN_MAX_GROWTH_PCT = "5"
```

After 5 minutes of user idle time, the Drift Conqueror targets weak ZPM anchor
buckets with bounded micro-queries. Growth is capped by
`NRL_LEARN_MAX_GROWTH_PCT` per 24-hour window.

Use Safe Mode when you want a frozen, read-mostly session:

```powershell
$env:NRL_SAFE_MODE = "1"
```

Safe Mode disables background Learn Mode, WAL writes, and auto-prune hooks.

## 5. Check Coverage And Disk Health

```powershell
nrlpy lmo coverage <sha-prefix>
nrlpy lmo info <sha-prefix>
```

`lmo coverage` shows coverage percentage, weak buckets, growth headroom, and the
last conquest run. `lmo info` shows footprint, quota, quota percentage, ZPM entry
count, WAL size, snapshots, growth rate, and last prune.

## 6. Preview And Apply Pruning

Preview first:

```powershell
nrlpy lmo prune <sha-prefix> --dry-run
```

Apply the default prune target (keeps at least 10% quota headroom when ZPM data
is the part that can be evicted):

```powershell
nrlpy lmo prune <sha-prefix>
```

Use a stronger target when needed:

```powershell
nrlpy lmo prune <sha-prefix> --aggressive
```

## 7. Final Verification Commands

```powershell
python -m pytest tests/test_bio_digital_brain_e2e.py -v
nrlpy doctor
nrlpy lmo coverage <sha-prefix>
nrlpy lmo info <sha-prefix>
nrlpy lmo prune <sha-prefix> --dry-run
```

If the E2E test and doctor pass, the local MVP lifecycle is healthy.
