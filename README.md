<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Bio-Digital Brain

**Disk-native lattice memory for local GGUF models: absorb once, chat, recover, dream, and prune safely.**

[![License: Proprietary Source-Available](https://img.shields.io/badge/license-Proprietary%20Source--Available-111827.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](./nrlpy/pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0A0A0A.svg)](./README.md#quick-start)
[![Bio-Digital Brain](https://img.shields.io/badge/Bio--Digital%20Brain%20v3.0-MVP%20Complete-16A34A.svg)](./AUDIT_REPORT.md)
[![engine-ci](https://github.com/RomanAILabs-Auth/NRL/actions/workflows/engine-ci.yml/badge.svg)](./.github/workflows/engine-ci.yml)
[![python-ci](https://github.com/RomanAILabs-Auth/NRL/actions/workflows/python-ci.yml/badge.svg)](./.github/workflows/python-ci.yml)

**NRL** is a CPU-first stack for **packed INT4 lattice dynamics** plus a
disk-native **Bio-Digital Brain MVP v3.0** lifecycle: GGUF absorption into LMO,
chat through a resolution ladder, idle-time Drift Conqueror mapping, WAL-backed
ZPM recovery, quota pruning, and final health diagnostics.

Copyright (c) RomanAILabs — Daniel Harding (GitHub: RomanAILabs-Auth)  
Collaborators: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor  
Contact: `daniel@romanailabs.com` · `romanailabs@gmail.com` · `romanailabs.com`

**License notice:** This software is **NOT open source**. It is released under
the **RomanAILabs Proprietary Source-Available Evaluation License 1.0**.
Evaluation is limited to reading, inspecting, and running one local copy.
Commercial, redistribution, derivative, hosted, AI-training, and competing use
requires a separate written license from RomanAILabs. Full intellectual
property rights are retained by Daniel Harding / RomanAILabs.

---

## Bio-Digital Brain MVP v3.0

The **Bio-Digital Brain** is the disk-native GGUF/LMO lifecycle in this repo:
absorb a model once, chat through the resolution ladder, let Learn Mode map weak
ZPM regions while idle, recover through WAL/snapshots after interruption, and
keep the on-disk footprint bounded with quota pruning.

Phases **P1-P10 are complete and release-ready**:

| Phase | Status | Operator Surface |
|-------|--------|------------------|
| P1 | LMO absorption | `nrlpy absorb <model.gguf>` |
| P2 | Learn Daemon | `NRL_LEARN_MODE=1` |
| P3 | Chat + learning UX | `nrlpy chat <model.gguf> --rewired` |
| P4 | ZPM WAL + recovery | `nrlpy lmo info <model-or-sha>` |
| P5 | Drift Conqueror coverage | `nrlpy lmo coverage <model-or-sha>` |
| P6 | HD quota + prune | `nrlpy lmo prune <model-or-sha> --dry-run` |
| P7 | Final integration + doctor | `nrlpy doctor` |
| P8 | GitHub release prep | `docs/FUTURE_PHASES.md`, examples, templates |
| P9 | Proprietary license lock | `LICENSE`, `LICENSES/NOTICE` |
| P10 | Final architecture audit | `AUDIT_REPORT.md`, `RELEASE_CHECKLIST.md` |

## Quick Start

```powershell
git clone https://github.com/RomanAILabs-Auth/NRL.git
cd NRL\nrlpy
python -m pip install -e .; python -m nrlpy doctor
```

Then absorb and chat with a local GGUF:

```powershell
nrlpy absorb C:\models\Phi-3-mini-4k-instruct.Q4_K_M.gguf
nrlpy chat C:\models\Phi-3-mini-4k-instruct.Q4_K_M.gguf --rewired
```

If `nrlpy` is not on PATH yet, run `python -m pip install -e .` from `NRL\nrlpy`
or use `python -m nrlpy doctor`.

## Architecture

The v3.0 architecture is a disk-native lifecycle around a local GGUF: absorb
once into an LMO, serve conversation through the Resolution Ladder, grow ZPM
coverage during idle Learn Mode, recover state through WAL/snapshots, and keep
storage bounded through quota-aware pruning. P10 confirms these surfaces are
integrated, documented, and ready for GitHub release under the proprietary
source-available evaluation license.

```text
GGUF model
   |
   v
P1 absorb -> LMO on disk (header, tiles, retained blobs, router)
   |
   +--> P3 chat resolution ladder
   |       R0 muscle memory -> R1 ZPM -> R2 Omega -> R5 decode
   |
   +--> P2/P5 Learn Daemon + Drift Conqueror
   |       idle weak-bucket micro-queries -> bounded ZPM growth
   |
   +--> P4 WAL + snapshots
   |       crash recovery for ZPM index
   |
   +--> P6 Disk Manager
           quota, dry-run prune, LRU ZPM eviction, WAL compaction
```

Let the brain map weak ZPM regions while the machine is idle:

```powershell
$env:NRL_LEARN_MODE = "1"
$env:NRL_LEARN_CONQUEST_IDLE_SEC = "300"
$env:NRL_LMO_MAX_GB = "100"
nrlpy lmo coverage <sha-prefix>
nrlpy lmo info <sha-prefix>
nrlpy lmo prune <sha-prefix> --dry-run
```

Safety switches:

| Variable | Effect |
|----------|--------|
| `NRL_SAFE_MODE=1` | Disables background Learn Mode, WAL writes, and auto-prune hooks. |
| `NRL_LEARN_MODE=0` | Keeps the learn supervisor idle. |
| `NRL_LMO_MAX_GB=100` | Per-model LMO+ZPM+muscle-memory footprint cap. |
| `NRL_LMO_AUTO_PRUNE=1` | Optional post-persist quota check; off by default. |

See [`docs/GETTING_STARTED.md`](./docs/GETTING_STARTED.md) for a step-by-step
absorb -> chat -> overnight dream -> coverage workflow.

## What's Next

The MVP is complete. Future work starts from the P7-P10 roadmap rather than
changing the v3.0 release surface: tighten lifecycle diagnostics, add release
artifact signing, expand operator UX around coverage/pruning, and plan post-MVP
multi-model or distributed Learn Mode work in [`docs/FUTURE_PHASES.md`](./docs/FUTURE_PHASES.md).

Release operators should review [`AUDIT_REPORT.md`](./AUDIT_REPORT.md),
[`RELEASE_CHECKLIST.md`](./RELEASE_CHECKLIST.md), and
[`RELEASE_NOTES_v3.0.md`](./RELEASE_NOTES_v3.0.md) before tagging.

## Safety Disclaimer

NRL is local-first research software. It does not replace model safety,
red-team review, sandboxing, backups, or human approval for sensitive use. Keep
`NRL_SAFE_MODE=1` when you want background learning, WAL writes, and auto-prune
hooks disabled. Do not run untrusted GGUFs or scripts outside a sandbox.

## License

This software is **NOT open source**. It is released under the
**RomanAILabs Proprietary Source-Available Evaluation License 1.0**. You may
read, inspect, and run one local copy for personal evaluation only. Commercial,
redistribution, derivative, hosted, AI-training, enterprise, and competing use
requires a separate written agreement from RomanAILabs. Full intellectual
property rights are retained by Daniel Harding / RomanAILabs. Model files,
datasets, and third-party dependencies remain governed by their own licenses.

---

## What this repository is

| Layer | Role |
|-------|------|
| **`libnrl` / `nrl_v1_*`** | Stable C ABI: INT4 packed layout, `braincore_int4`, feature detection, variant reporting. |
| **Native `nrl`** | CLI: build, bench, assimilate, `.nrl` execution, status, diagnostics. |
| **`nrlpy`** | Typed Python API, `nrlpy._core` extension, subprocess bridges to `nrl bench` / `nrl assimilate` / `nrl file`. |
| **Profiles** | Named presets (`sovereign`, `adaptive`, `war-drive`, `zpm`, `automatic`, `omega`, `omega-hybrid`) selecting iterative kernels, **static transition collapse** for fixed inputs, or **hierarchical sparse routing** with optional pruning; see [Architecture](./nrl-architecture.md). |

Design principles (see [`nrl-architecture.md`](./nrl-architecture.md) for contracts):

- **Machine-code hot path**: no allocation and no CPU feature branching inside inner update loops; dispatch fixed at init.
- **One ABI, multiple surfaces**: Python never executes kernel inner loops; it marshals buffers and CLI subprocesses.
- **Reproducible reporting**: benchmarks emit stable `key: value` lines suitable for parsers and locked harnesses under [`benchmarks/`](./benchmarks/).
- **Governance**: bounded adaptation and runtime safety directions are specified in [`docs/nrl_immune_system_spec.md`](./docs/nrl_immune_system_spec.md).

---

## Execution modes (conceptual)

NRL distinguishes two **families** of profile (CLI names unchanged):

| Family | Profiles | Mechanism (summary) |
|--------|----------|------------------------|
| **Full iterative** | `sovereign`, `adaptive`, `war-drive` | Every timestep applies the packed INT4 update across the lattice (AVX2 or scalar); `executed_updates` equals `baseline_equiv_updates` on the bench path. |
| **High-avoidance** | `zpm`, `automatic` | For **static** drive fields, a **precomputed transition map** applies k logical iterations in one pass; `skip_ratio` is large; `virtual_gops` reflects baseline-equivalent work per wall second. |
| **Sparse hierarchical** | `omega` | Fixed-size **sub-lattices** with wake/prune policy; most baseline work is skipped by routing; very high `virtual_gops`, low `executed_updates`. |
| **Sparse + throughput floor** | `omega-hybrid` | Same router, but a **minimum active sub-lattice count** forces dense AVX2 work on a subset so `executed_gops` stays high while retaining partial skip gains. |

**`aes256-synth`**: deterministic XOR/rotate micro-benchmark over a 32-byte state; **not** AES-256. See [`language/examples/aes256.nrl`](./language/examples/aes256.nrl).

---

## Benchmarks

Official numbers require **locked commands**, **`nrl --version`**, and **generated artifacts** (e.g. from [`benchmarks/nrl_vs_cpp.py`](./benchmarks/nrl_vs_cpp.py)). Figures in this README are illustrative.

### `nrl bench`

```text
nrl bench <neurons> <iterations> <reps> <threshold> <profile>
```

| Profile | Read first | Notes |
|---------|------------|--------|
| `sovereign` | `executed_gops` | Reference full-iteration INT4 throughput. |
| `adaptive`, `war-drive` | `executed_gops` | Same kernel class as `sovereign`; different default scale in `nrl run`. |
| `zpm`, `automatic` | `skip_ratio`, `virtual_gops` | Static-input collapse; parity-checked vs iterative semantics on supported inputs. |
| `omega` | `virtual_gops`, `skip_ratio`, sub-lattice stats | Hierarchical sparse execution; interpret `virtual_gops` with accounting in mind. |
| `omega-hybrid` | `executed_gops` and `virtual_gops` | Hybrid: enforced active sub-lattice floor + skips elsewhere. |
| `aes256-synth` | `state_fnv1a64`, `mix_throughput` | Synthetic mix only; optional `expected_fnv1a64` in `.nrl` on current `nrl`. |

**Output fields** (representative):

| Field | Meaning |
|-------|---------|
| `elapsed_s` | Timed interval after warmup. |
| `executed_updates` | Hardware-applied packed updates in the window. |
| `baseline_equiv_updates` | Nominal full-lattice work units for the same `(neurons, iterations, reps)`. |
| `skip_ratio` | `1 − executed/baseline` (bench window). |
| `executed_gops` | Executed updates per second (10⁹). |
| `virtual_gops` | `baseline_equiv_updates / elapsed_s` (10⁹); rises when wall time drops via collapse or pruning. |

**Smoke commands** (resize for slower hosts):

```powershell
.\build\bin\nrl.exe bench 1048576 4096 6 8 sovereign
.\build\bin\nrl.exe bench 1048576 4096 6 8 zpm
.\build\bin\nrl.exe bench 1048576 16384 4 8 omega
```

```bash
./build/bin/nrl bench 1048576 4096 6 8 sovereign
./build/bin/nrl bench 1048576 4096 6 8 zpm
./build/bin/nrl bench 1048576 16384 4 8 omega
```

### `.nrl` v0

| File | Purpose |
|------|---------|
| [`language/examples/omega_pass.nrl`](./language/examples/omega_pass.nrl) | Regression-oriented bench parameters for the sparse hierarchical profile. |
| [`language/examples/aes256.nrl`](./language/examples/aes256.nrl) | Portable synthetic mix (no digest key). |
| [`language/examples/aes256_locked.nrl`](./language/examples/aes256_locked.nrl) | Same + `expected_fnv1a64` (requires parser support in your `nrl` build). |

```bash
./build/bin/nrl file language/examples/omega_pass.nrl
./build/bin/nrl file language/examples/aes256.nrl
./build/bin/nrl file language/examples/aes256_locked.nrl
```

### Runtime snapshot

`nrl brain-map` runs one short INT4 bench probe, prints RSS, and a fixed **`PORT_*`** status table for quick inspection (not a multi-configuration sweep).

---

## NRL-AI: pure-lattice inference (≥1000 wps, no GPU, no libllama)

`NRL-AI` is the **native** path: a retrieval + associative-composition engine that runs entirely on NRL primitives (SimHash anchors, ZPM transition collapse, Omega fragment routing). No GPU. No CUDA. No libllama decode. Every reply is served by a replay-locked lattice walk against a corpus you ingest once.

**Throughput contract.** Bench gate = mean **words per second** ≥ `NRL_AI_WPS_TARGET` (default **1000**). On a reference laptop (AMD64, 8 cores, Windows 11) the bench lands at **~5,900 wps mean / ~10,000 wps peak**, gate `PASS` — see `nrl-ai bench` below.

### Out-of-box demo

```powershell
python -m nrlpy nrl-ai demo
```

Compiles the packaged seed corpus (conversational Q&A about NRL-AI itself) into a demo index and launches the polished chat REPL. No user corpus required.

### The seven commands

| Command | Purpose |
|---------|---------|
| `nrl-ai ingest <corpus.txt>` | Build an on-disk index: fragments + 256-bit SimHash anchors + Omega transitions + manifest. |
| `nrl-ai resolve "<query>"`   | Anchor a query and find the closest fragment by Hamming distance. JSON on stdout. |
| `nrl-ai compose "<query>"`   | Resolve + Omega-routed fragment walk → streamed reply. JSON on stdout. |
| `nrl-ai chat`                | Polished REPL. `[NRL-AI]` fast lane only. Streaming output. Slash commands: `/help /stats /reset /status /quit`. |
| `nrl-ai bench`               | Replay-locked WPS gate. Emits `schema=nrl_ai.bench.v1` JSON. Exit code 0 on PASS, 4 on FAIL, 2 on missing index. |
| `nrl-ai status`              | Roadmap + index readiness JSON. |
| `nrl-ai demo`                | One-shot: ingest seed corpus → chat REPL. |

### Typical flow

```powershell
python -m nrlpy nrl-ai ingest my_corpus.txt --out .\idx
python -m nrlpy nrl-ai bench  --index .\idx --turns 32 --warmup 2
python -m nrlpy nrl-ai chat   --index .\idx
```

### Contracts at a glance

- **Deterministic.** Same corpus + same query → bit-identical reply, word count, and stop reason. No RNG, no wall-clock in the hot path.
- **Honest miss.** When a query lands outside the Hamming threshold (default 96 bits), NRL-AI tells you the closest fragment distance and **never** synthesizes. Pure retrieval, pure composition.
- **Corpus-bounded.** Every word in a reply came from a fragment you ingested. The lattice cannot hallucinate outside its corpus.
- **Replay-locked bench.** The query stream is a pure function of `corpus_sha256` — two bench runs on the same index produce the same queries, so the only source of variation is real silicon wall-time.

### Architecture reference

See [`nrl-new-archietcture.MD`](./nrl-new-archietcture.MD) for the seven-prompt plan, the NRL-AI throughput math, and why this path hits ≥1000 wps on CPU while the libllama decode path physically cannot.

---

## Run a GGUF model (legacy libllama path)

NRL also includes a full **GGUF runner (`llama.nrl`, P1)** where NRL is the execution supervisor and `libllama` (via `llama-cpp-python` or `llama-cli.exe`) owns the numerics. It reports a **four-metric TPS contract** — `executed_tps`, `virtual_tps`, `cache_tps`, `effective_tps` — with honest accounting banners.

**Full architecture:** [`docs/nrl_gguf_runner_architecture.md`](./docs/nrl_gguf_runner_architecture.md). **Manifest grammar:** [`language/spec/nrl_manifest_v1.md`](./language/spec/nrl_manifest_v1.md). **Docs index:** [`docs/README.md`](./docs/README.md).

### One-shot inference

```powershell
python -m nrlpy run models\phi-3-mini-4k-instruct.Q4_K_M.gguf `
  --prompt "Tell me one short, surprising fact about space." `
  --max-tokens 128 --seed 42 --chat-format phi3
```

```bash
python -m nrlpy run models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --prompt "Tell me one short, surprising fact about space." \
  --max-tokens 128 --seed 42 --chat-format phi3
```

Or via a `.nrl` v1 manifest (reference: [`language/examples/phi3_dense.nrl`](./language/examples/phi3_dense.nrl)):

```bash
python -m nrlpy gguf language/examples/phi3_dense.nrl
```

### Interactive chat REPL

```bash
python -m nrlpy chat models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --system "You are concise." --seed 7 --chat-format phi3
```

Inside the REPL, `/help` lists every slash command:

| Command | Effect |
|---|---|
| `/clear` | reset conversation history (system prompt preserved) |
| `/system <text>` | set system prompt (resets history for safety) |
| `/tps` | session-aggregate four-metric TPS banner |
| `/save <path>` / `/load <path>` | JSON snapshot; `/load` refuses snapshots recorded against a different `model_sha256` |
| `/seed <n>` | change sampler seed for subsequent turns |
| `/history` | compact `(role, length)` list |
| `/quit` / `/exit` | leave |

**Per-turn muscle memory.** The rendered full-history prompt is the FNV-1a64 cache key, so replaying the same `(system, history, user_text, sampler)` tuple hits a cached reply on any subsequent run. Cache lives under `$NRL_ROOT/cache/mm/<model_sha256>/`.

### Backend selector

Set `NRL_INFERENCE` to choose how inference actually happens:

| Value | Behavior |
|---|---|
| `native` (default) | In-process `llama-cpp-python` (`pip install llama-cpp-python`). |
| `cli` | Spawn `llama-cli.exe` once per request and stream stdout. Configure via `NRL_LLAMA_CLI=/path/to/llama-cli`. |
| `stub` | Deterministic fake. No weights required. Used by CI and unit tests. |

### The honesty hinge

Until P2-Active (layer gate actually wired into `libllama`), **`virtual_tps == executed_tps`** on real runs. The banner prints this identity explicitly. The `omega-hybrid skip_ratio` you see in the "NRL lattice observation" block is **advisory only** — it's what NRL's lattice would skip under a balanced gate, measured in parallel with decode, never multiplied into the decode TPS math. See [`docs/nrl_gguf_runner_architecture.md`](./docs/nrl_gguf_runner_architecture.md) §1.0 for the full contract.

### Native `nrl run <model>.gguf`

The native `nrl.exe` currently **routes** GGUF paths to the Python orchestrator with an actionable message rather than silently shelling out — native C GGUF execution is P4 (see the runner architecture doc §6). The Python path is the supported surface today.

### Python (`nrlpy`)

`python -m nrlpy.cli bench …` invokes the same native `nrl bench` (binary resolved via `NRL_BIN`, `NRL_ROOT`, or repo `build/bin`). `nrlpy run` / `nrlpy <script.py>` inject assimilation globals for control-plane scripts plus **seamless builtins** (`next_prime`, `is_prime`, `fabric_pulse`) so normal Python files can run without `import nrlpy` (see [`nrlpy/README.md`](./nrlpy/README.md) and [`examples/prime.py`](./examples/prime.py)). Primality stays deterministic Python; the INT4 extension is still touched for attestation.

### Artifacts

| Path | Role |
|------|------|
| [`benchmarks/initial_results.md`](./benchmarks/initial_results.md) | Narrative snapshot; regenerate before external citation. |
| [`benchmarks/nrl_vs_cpp.py`](./benchmarks/nrl_vs_cpp.py) | Locked harness → `build/bench/nrl_vs_cpp.{json,md}`. |
| [`benchmarks/README.md`](./benchmarks/README.md) | Harness conventions. |

**Publication policy:** cite throughput only with command line, engine version, and artifact or log excerpt from the same run.

---

## Quick start

### Windows (PowerShell)

```powershell
.\build.ps1 -Config Release
.\build\bin\nrl.exe --version
.\build\bin\nrl.exe status
.\build\bin\nrl.exe bench 1048576 4096 6 8 sovereign
.\build\bin\nrl.exe assimilate 4096 256 10
.\build\bin\nrl.exe demo
```

```powershell
$env:PYTHONPATH = "nrlpy\src"
python -m nrlpy.cli --version
python -m nrlpy.cli bench 1048576 16384 4 8 omega-hybrid
python -m nrlpy.cli run examples\assimilate_llm_solver.py
python -m nrlpy.cli run examples\ultimate_power_demo.py
python -m nrlpy.cli run examples\global_lightning_lattice.py --max-cycles 1
python -m nrlpy.cli demo
```

### Linux / macOS

```bash
./build.sh Release 1
./build/bin/nrl --version
./build/bin/nrl status
./build/bin/nrl bench 1048576 4096 6 8 sovereign
./build/bin/nrl assimilate 4096 256 10
./build/bin/nrl demo
```

```bash
export PYTHONPATH=nrlpy/src
python3 -m nrlpy.cli --version
python3 -m nrlpy.cli bench 1048576 16384 4 8 omega-hybrid
python3 -m nrlpy.cli run examples/assimilate_llm_solver.py
python3 -m nrlpy.cli run examples/ultimate_power_demo.py
python3 -m nrlpy.cli run examples/global_lightning_lattice.py --max-cycles 1
python3 -m nrlpy.cli demo
```

```bash
./build/bin/nrl file language/examples/omega_pass.nrl
./build/bin/nrl file language/examples/aes256.nrl
./build/bin/nrl language/examples/omega_pass.nrl
```

---

## Production install

### Windows

```powershell
cd C:\path\to\NRL
.\scripts\install_nrl.ps1
```

Installs `nrl.exe`, `nrlpy.cmd`, mirrored **`build\bin\nrl.exe`** for legacy path expectations, copies `examples/` and `py/nrlpy/`, sets user **`NRL_ROOT`** and PATH under `%LOCALAPPDATA%\Programs\NRL`. Open a **new** shell after install. Use **`-OptInLMAI`** to record LM/AI opt-in without the prompt.

### POSIX

```bash
cd /path/to/NRL
./scripts/install_nrl.sh
```

Installs `nrl` and `nrlpy` under `~/.local/bin`, copies assets to `~/.local/share/nrl`, appends `NRL_ROOT` to shell rc when needed. Reload the shell.

**Pre-live checklist:** [`docs/PRODUCTION_READINESS.md`](./docs/PRODUCTION_READINESS.md) and `scripts/live_readiness.ps1` / `live_readiness.sh`.

---

## Selected CLI commands

| Command | Purpose |
|---------|---------|
| `nrl status` | Version, active kernel variant, LM/AI opt-in (env `NRL_LM_AI_OPT_IN`, else `~/.nrl/consent.json`). |
| `nrl -ai on` / `off` | Toggle LM/AI consent + Windows `setx` (new shells); `nrlpy -ai on` / `off` same. |
| `nrl runtime` | CPU feature bitmask and bound variants. |
| `nrl bench` / `nrl run` | Benchmark and timed run with profile selection. |
| `nrl assimilate` | Packed INT4 pass + FNV-1a64 over potentials buffer. |
| `nrl file <path.nrl>` / `nrl <path.nrl>` | Parse v0 control file and dispatch bench or run. |
| `nrl brain-map` | Short probe + RSS + `PORT_*` table. |
| `nrl inquire` / `nrl chat` | Deterministic help and intent stubs (no model dependency). |
| `nrl run <model>.gguf` / `nrl <model>.gguf` | Prints actionable guidance pointing at the Python GGUF runner (see [Run a GGUF model](#run-a-gguf-model)). Direct C-side linkage is P4. |
| `python -m nrlpy run <model>.gguf` | One-shot GGUF inference with four-metric TPS report (P1). |
| `python -m nrlpy chat <model>.gguf` | Multi-turn REPL with per-turn muscle memory (`/help` inside). |
| `python -m nrlpy gguf <manifest.nrl>` | Run a `.nrl` v1 manifest (`mode = gguf_run`). |
| `nrlpy …` | See [`nrlpy/README.md`](./nrlpy/README.md); `pip install -e ./nrlpy` adds `nrlpy` to PATH on dev machines. |

Full behavior: [`nrl-architecture.md`](./nrl-architecture.md), `engine/src/main.c`.

---

## Repository layout

| Path | Contents |
|------|----------|
| `engine/` | Kernels, dispatch, CLI, C ABI — [`engine/README.md`](./engine/README.md) |
| `nrlpy/` | Python package + `_core` — [`nrlpy/README.md`](./nrlpy/README.md) |
| `language/` | `.nrl` spec and examples — [`language/README.md`](./language/README.md) |
| `benchmarks/` | Harnesses — [`benchmarks/README.md`](./benchmarks/README.md) |
| `scripts/` | Install and release — [`scripts/README.md`](./scripts/README.md) |
| `docs/` | Architecture, manifest specs, governance, prior-art notes — index at [`docs/README.md`](./docs/README.md) |
| `examples/` | Cross-surface demos |
| Root | `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, [`grok_review_handoff.md`](./grok_review_handoff.md) (external review brief) |

---

## Engineering gates

- **C:** `-Wall -Wextra -Wpedantic` via `build.ps1` / `build.sh` (Zig `cc` on Windows).
- **Python:** `ruff`, `mypy --strict` per [`nrlpy/pyproject.toml`](./nrlpy/pyproject.toml).
- **Tests:** `engine/tests/test_runtime.c`, `nrlpy/tests/test_smoke.py` (`build.ps1 -Tests`).

Release checks: [`scripts/release_check.ps1`](./scripts/release_check.ps1), [`scripts/release_check.sh`](./scripts/release_check.sh).

---

## Maintainer notes

1. Use profile **`sovereign`** for baseline throughput and regression locks.
2. Use **`omega-hybrid`** when reporting both high `executed_gops` and non-trivial skip statistics.
3. Treat **`virtual_gops`** on **`omega`** / **`zpm`** as **accounting throughput**, not naive “every op executed at that rate.”
4. Do not change `nrl bench` / `nrl assimilate` line formats without a versioned parser bump.
5. ABI changes require a new `nrl_v2_*` prefix policy (see architecture doc).

---

## License

**RomanAILabs Proprietary Source-Available Evaluation License 1.0.** This software is **NOT open source**. Reading, inspection, and one local evaluation copy are permitted only under [`LICENSE`](./LICENSE). Commercial, redistribution, derivative, hosted, AI-training, enterprise, and competing use requires a separate written agreement. Full intellectual property rights are retained by Daniel Harding / RomanAILabs.

Enterprise: `daniel@romanailabs.com` · `romanailabs@gmail.com` · `romanailabs.com`

---

## Acknowledgments

Grok/xAI · Gemini-Flash/Google · ChatGPT-5.4/OpenAI · Cursor
