# NRL (Neural)

[![License: RBSL 1.1](https://img.shields.io/badge/license-RBSL%201.1-111827.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](./nrlpy/pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0A0A0A.svg)](./README.md#quick-start)

**NRL** is a CPU-first stack for **packed INT4 lattice dynamics**: scalar reference and AVX2 hot paths, load-time dispatch, and multiple **execution profiles** that trade **fully materialized neuron updates** against **pruned or collapsed schedules** while preserving defined accounting invariants (`executed_updates`, `baseline_equiv_updates`, `skip_ratio`, `virtual_gops`). **NrlPy** exposes the same ABI and native CLI bridges from Python. **`.nrl` v0** is a minimal key-value control format for reproducible bench and run invocations.

Copyright (c) RomanAILabs — Daniel Harding (GitHub: RomanAILabs-Auth)  
Collaborators: Grok/xAI, Gemini-Flash/Google, ChatGPT-5.4/OpenAI, Cursor  
Contact: `daniel@romanailabs.com` · `romanailabs@gmail.com` · `romanailabs.com`

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
| `docs/` | Long-horizon safety / governance notes |
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

**RomanAILabs Business Source License 1.1 (RBSL 1.1).** Non-commercial research, education, and evaluation are permitted per [`LICENSE`](./LICENSE). Commercial use requires a separate agreement unless covered by the Additional Use Grant. Versions convert to **Apache 2.0** after the Change Date in the license text.

Enterprise: `daniel@romanailabs.com` · `romanailabs@gmail.com` · `romanailabs.com`

---

## Acknowledgments

Grok/xAI · Gemini-Flash/Google · ChatGPT-5.4/OpenAI · Cursor
