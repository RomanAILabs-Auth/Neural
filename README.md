# NRL (Neural)

[![License: RBSL 1.1](https://img.shields.io/badge/license-RBSL%201.1-111827.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](./nrlpy/pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0A0A0A.svg)](./README.md#quick-start)

**Engine-first neural execution on CPU.**  
NRL is a low-bit machine-code stack: hand-tuned INT4 lattice kernels, load-time dispatch, and strict separation between **System 2** (deliberate, full-step) and **System 1** (ZPM / Omega compute-avoidance) lanes. Control planes are the native `nrl` CLI, the `nrlpy` Python front-end, and a minimal `.nrl` orchestration surface.

Copyright (c) RomanAILabs - Daniel Harding (GitHub: RomanAILabs-Auth)  
Honored collaborators: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor  
Contact: `daniel@romanailabs.com` | `romanailabs@gmail.com` | `romanailabs.com`

---

## Core contract

- **Machine-code-first hot path**: scalar reference plus AVX2 INT4 execution; feature selection at init, not inside inner loops.
- **One ABI, multiple surfaces**: `libnrl` (`nrl_v1_*`) backs both `nrl` and `nrlpy._core` without Python inside kernels.
- **Reproducible claims**: published throughput comparisons use locked profiles and machine-readable artifacts (see [Benchmarks](#benchmarks) and [`benchmarks/`](./benchmarks/)).
- **Governed plasticity**: adaptive and “fast” lanes are bounded, logged, and documented; long-term guard rails are specified in [`docs/nrl_immune_system_spec.md`](./docs/nrl_immune_system_spec.md).
- **Living architecture**: scope, phase, and engineering decisions are tracked in [`nrl-architecture.md`](./nrl-architecture.md).

---

## Kernel and control surface

| Area | Responsibility |
|------|----------------|
| `braincore_int4` | Packed INT4 lattice update (2 neurons / byte); primary production kernel. |
| ZPM static | Exact transition-collapse for static-input muscle-memory (`zpm` / `automatic`). |
| Omega / Omega-hybrid | Fractal-routed virtual lane + hybrid executed-throughput preservation. |
| `.nrl` v0 | Key-value run/bench control files (`mode`, `profile`, `neurons`, …). |
| Assimilation | `nrl assimilate` and `nrlpy` inplace buffers share the same packed layout and checksum contract. |

Profiles (runtime):

| Lane | Profiles |
|------|----------|
| System 2 (deliberate) | `sovereign`, `adaptive`, `war-drive` |
| System 1 (automatic) | `zpm`, `automatic`, `omega`, `omega-hybrid` |

---

## Benchmarks

NRL publishes benchmarks as **repeatable CLI runs** with **stable, parseable stdout** (`nrl bench`, `nrl assimilate`, `.nrl` control files). Treat any numbers in this README as **illustrative** unless you attach fresh artifacts from your own machine.

### Native CLI (`nrl bench`)

```text
nrl bench <neurons> <iterations> <reps> <threshold> <profile>
```

| Profile | Lane | What to read first |
|---------|------|---------------------|
| `sovereign` | System 2 | `executed_gops` — full-step INT4 baseline. |
| `adaptive`, `war-drive` | System 2 | Same shape as sovereign; different default scale in `nrl run`. |
| `zpm`, `automatic` | System 1 | `skip_ratio`, `virtual_gops` vs `executed_gops`; semantics match sovereign accounting. |
| `omega` | System 1 | Very high `virtual_gops`; `skip_ratio` near 1; `executed_updates` stays small vs `baseline_equiv_updates`. |
| `omega-hybrid` | System 1 | Balance of skip gains and executed throughput (`executed_gops` remains meaningful). |
| `aes256-synth` | Synthetic | Deterministic XOR/rotate mix + `state_fnv1a64` (see `.nrl` examples); **not** real AES. |

**Key fields** (machine-readable `key: value` lines):

| Field | Meaning |
|-------|---------|
| `elapsed_s` | Wall time for the timed `reps` loop (after warmup). |
| `executed_updates` / `baseline_equiv_updates` | Hardware work vs nominal full-lattice work. |
| `skip_ratio` | `1 − executed/baseline` on the bench window. |
| `executed_gops` | Executed neuron-updates per second, in GOPS. |
| `virtual_gops` | Baseline-equivalent rate from the same wall clock (Omega / ZPM “virtual” accounting). |

**Smoke commands** (after `./build.ps1` or `./build.sh`; adjust sizes for slower CPUs):

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

### `.nrl` orchestration

Version-0 files drive the same bench/run paths as the CLI. Shipped examples:

| File | Role |
|------|------|
| [`language/examples/omega_pass.nrl`](./language/examples/omega_pass.nrl) | Locked Omega bench parameters (good regression shape). |
| [`language/examples/aes256.nrl`](./language/examples/aes256.nrl) | **Synthetic** XOR/rotate throughput (not real AES). Runs on any `.nrl`-capable `nrl`. |
| [`language/examples/aes256_locked.nrl`](./language/examples/aes256_locked.nrl) | Same workload + `expected_fnv1a64` digest check; **requires a current `nrl` build** from this tree (older binaries report `unknown key`). |
| [`examples/aes256.nrl`](./examples/aes256.nrl) | Same as `language/examples/aes256.nrl`; copied beside demos on install. |

```bash
./build/bin/nrl file language/examples/omega_pass.nrl
./build/bin/nrl file language/examples/aes256.nrl
./build/bin/nrl file language/examples/aes256_locked.nrl   # after ./build.ps1 / ./build.sh
```

### Runtime snapshot

`nrl brain-map` runs one short INT4 bench probe, prints process RSS, and a fixed `PORT_*` table (seconds, not a full benchmark sweep). Use beside `nrl bench` for quick sanity checks.

### Python parity (`nrlpy`)

`python -m nrlpy.cli bench …` shells out to the same native `nrl bench` binary (resolved via `NRL_BIN`, `NRL_ROOT`, or repo `build/bin`). Use it when benchmarks are driven from notebooks or CI.

### Artifacts and governance

| Artifact | Purpose |
|----------|---------|
| [`benchmarks/initial_results.md`](./benchmarks/initial_results.md) | Narrative snapshot (regenerate before citing externally). |
| [`benchmarks/nrl_vs_cpp.py`](./benchmarks/nrl_vs_cpp.py) | Locked harness → `build/bench/nrl_vs_cpp.json` + `.md` (gitignored until generated). |
| [`benchmarks/README.md`](./benchmarks/README.md) | Harness layout and conventions. |

**Policy:** do not cite throughput in papers, posts, or decks without attaching the exact command line, engine version (`nrl --version`), and generated artifact or log excerpt from the same run.

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

Python (from repo root, with `PYTHONPATH=nrlpy\src` or after `pip install -e nrlpy`):

```powershell
$env:PYTHONPATH = "nrlpy\src"
python -m nrlpy.cli --version
python -m nrlpy.cli bench 1048576 16384 4 8 omega-hybrid
python -m nrlpy.cli run examples\assimilate_llm_solver.py
python -m nrlpy.cli run examples\ultimate_power_demo.py
python -m nrlpy.cli demo
```

### POSIX (Linux / macOS)

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
python3 -m nrlpy.cli demo
```

`.nrl` examples:

```bash
./build/bin/nrl file language/examples/omega_pass.nrl
./build/bin/nrl file language/examples/aes256.nrl
./build/bin/nrl file language/examples/aes256_locked.nrl   # digest verify; needs freshly built nrl
# or pass the path directly:
./build/bin/nrl language/examples/omega_pass.nrl
```

---

## Production install

### Windows

```powershell
cd C:\path\to\NRL
.\scripts\install_nrl.ps1 -OptInLMAI
```

Installs `nrl.exe` and **`nrlpy.cmd`** under `%LOCALAPPDATA%\Programs\NRL\bin`, updates PATH, writes `%USERPROFILE%\.nrl\consent.json`, and sets `NRL_LM_AI_OPT_IN` when opted in. Also sets user **`NRL_ROOT`** to `%LOCALAPPDATA%\Programs\NRL` and copies **`examples/`** + **`py/nrlpy/`** so **`nrl demo`** / **`nrlpy script.py`** work without a checkout. **Open a new PowerShell** after install so `nrlpy` is recognized.

### Linux / macOS

```bash
cd /path/to/NRL
NRL_INSTALL_OPT_IN_LM_AI=1 ./scripts/install_nrl.sh
```

Installs **`nrl`** and **`nrlpy`** launchers under `~/.local/bin`, updates shell `PATH` when needed, writes `~/.nrl/consent.json`, and exports `NRL_LM_AI_OPT_IN` when opted in. Copies demo assets to **`~/.local/share/nrl`** and appends **`export NRL_ROOT=...`** to `~/.bashrc` or `~/.zshrc`. Reload the shell so **`nrlpy myfile.py`** works.

---

## Runtime commands (selection)

| Command | Purpose |
|---------|---------|
| `nrl status` / `-status` | Engine version, variant, cognitive lanes, LM/AI opt-in, health. |
| `nrl inquire <topic>` | Deterministic operator help (`speed`, `epistemic`, `assimilate`, …). |
| `nrl chat <message>` | Lightweight rule-based intent helper. |
| `nrl brain-map` | Runtime snapshot: one-shot INT4 bench probe, RSS, `PORT_*` table. |
| `nrl assimilate [N] [I] [T]` | Sovereign INT4 pass + `checksum_fnv1a64` (binary assimilation contract). |
| `nrl demo` | Runs `examples/ultimate_power_demo.py` via Python (`PYTHONPATH` set to `nrlpy/src` or install `py/`). |
| `nrlpy demo` | Same demo from Python entrypoint (`python -m nrlpy.cli demo`); uses `run_path` with injected `nrl` globals. |
| `nrlpy <file.py>` | TriPy-style: run Python with assimilation globals (same as `nrlpy run <file.py>`). |

**If `nrlpy` is not recognized:** you are not on the install PATH yet (new terminal), or you have not run `pip install -e ./nrlpy` from a dev checkout. Install puts `nrlpy.cmd` next to `nrl.exe`; editable install puts `nrlpy.exe` in Python `Scripts`.

Full CLI contract: [`nrl-architecture.md`](./nrl-architecture.md) and `engine/src/main.c`.

---

## Current phase

- **Core v1**: implemented — engine, profiles, `.nrl`, `nrlpy`, assimilation path, benchmark harness, release check.
- **Next (architecture)**: epistemic IR feeding ZPM eligibility; immune runtime gates; optional AArch64 / AVX-512 lanes.

Roadmap and execution log:

- [`nrl-architecture.md`](./nrl-architecture.md)
- [`CHANGELOG.md`](./CHANGELOG.md)

External review prompt:

- [`grok_review_handoff.md`](./grok_review_handoff.md)

---

## Engineering quality gates

- C: `-Wall -Wextra -Wpedantic` via `build.ps1` / `build.sh` (Zig `cc` toolchain on Windows).
- Python: `ruff` + `mypy --strict` per [`nrlpy/pyproject.toml`](./nrlpy/pyproject.toml).
- Tests: `engine/tests/test_runtime.c`, `nrlpy/tests/test_smoke.py` (wired into `build.ps1 -Tests`).

Release verification:

- Windows: [`scripts/release_check.ps1`](./scripts/release_check.ps1)
- POSIX: [`scripts/release_check.sh`](./scripts/release_check.sh)

---

## Repository layout

| Path | Role |
|------|------|
| `engine/` | Kernels, dispatch, CLI, C ABI — see [`engine/README.md`](./engine/README.md) |
| `nrlpy/` | Python package + `_core` extension — see [`nrlpy/README.md`](./nrlpy/README.md) |
| `language/` | `.nrl` spec and examples — see [`language/README.md`](./language/README.md) |
| `benchmarks/` | Harnesses and snapshot docs — see [`benchmarks/README.md`](./benchmarks/README.md) |
| `scripts/` | Install and release automation — see [`scripts/README.md`](./scripts/README.md) |
| `docs/` | Immune system and long-horizon safety contracts |
| `examples/` | Cross-surface examples (e.g. assimilation) |

---

## Guides

**Humans**

1. Use `sovereign` for official deterministic baselines.
2. Use `omega-hybrid` when you need strong executed GOPS with partial skip gains.
3. Use `omega` when studying virtual throughput / time-to-answer (read `virtual_gops` with care).
4. Never publish numbers without the JSON/MD artifacts from the locked harness.
5. Treat `nrl-architecture.md` as the contract of record before changing behavior.

**LLMs**

1. Preserve ABI semantics in `engine/include/nrl/nrl.h`.
2. Do not break scalar ↔ optimized parity without tests and documented rationale.
3. Keep `nrl bench` / `nrl assimilate` output stable for parsers.
4. Keep `.nrl` v0 deterministic (key-value); bump spec for grammar changes.
5. Prefer measurable edits: benchmark, then document.

---

## License

This repository uses the **RomanAILabs Business Source License 1.1 (RBSL 1.1)**.

- Non-commercial research, education, and evaluation are permitted under the license terms.
- Commercial use requires a separate written commercial license unless covered by the Additional Use Grant in [`LICENSE`](./LICENSE).
- Each version converts to **Apache License 2.0** after the defined Change Date (see license text).

Commercial and enterprise contact: `daniel@romanailabs.com` | `romanailabs@gmail.com` | `romanailabs.com`

---

## Honored collaborators

- Grok / xAI  
- Gemini-Flash / Google  
- ChatGPT-5.4 / OpenAI  
- Cursor  
