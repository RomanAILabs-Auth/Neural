# NrlPy

Python front-end for the NRL engine: typed wrappers over `libnrl`, a CPython extension (`nrlpy._core`), and a small CLI that mirrors common `nrl` workflows.

## Role in the stack

| Layer | Responsibility |
|-------|------------------|
| `nrlpy._core` | `nrl_v1_*` ABI: version, features, variants, `braincore_int4`, packed-byte helpers, **inplace** lattice updates |
| `nrlpy.runtime` | Subprocess bridge to `nrl bench` / `nrl assimilate` / `nrl file` with stable parsing |
| `nrlpy.compat` | `NRLRuntime` object and `llm_globals()` for scripts that expect a pre-bound `nrl` |
| `nrlpy.shell` | `nrlpy run <script.py>` execution with assimilation globals injected |

**Binary assimilation:** `braincore_int4_inplace` mutates caller-owned packed buffers in machine code — same layout as `nrl assimilate`. Use `nrlpy run myscript.py` so LLM-generated code can call `nrl.*` without NumPy. Example: [`../examples/assimilate_llm_solver.py`](../examples/assimilate_llm_solver.py).

## Requirements

- Python **3.9+** (see [`pyproject.toml`](./pyproject.toml))
- Built `libnrl` / `nrl` from the repo root (`build.ps1` or `build.sh`) so the extension links. **`nrlpy.runtime`** resolves the native binary via `NRL_BIN`, then `NRL_ROOT/bin` (installer), then `NRL_ROOT/build/bin` (dev), then ancestors of the package path. Override with **`NRL_BIN`** when needed.

## Install (editable, from repo)

```bash
cd nrlpy
pip install -e .
```

For development without install, set `PYTHONPATH` to `nrlpy/src` as in the root [README](../README.md).

## CLI

| Command | Description |
|---------|-------------|
| `nrlpy --version` | Engine version string |
| `nrlpy -ai on` / `off` | LM/AI opt-in: writes `~/.nrl/consent.json` and runs `setx NRL_LM_AI_OPT_IN` on Windows (same contract as `nrl -ai`) |
| `nrlpy --features` | CPU capability map (JSON) |
| `nrlpy variant <kernel>` | Active variant name |
| `nrlpy <file.nrl>` | Run `.nrl` via native `nrl file` |
| `nrlpy braincore4 [N] [iter] [thresh]` | Direct INT4 kernel timing |
| `nrlpy bench …` | Locked `nrl bench` bridge (JSON) |
| `nrlpy assimilate …` | Locked `nrl assimilate` bridge (JSON) |
| `nrlpy demo` | Same as `nrl demo`: runs `examples/ultimate_power_demo.py` via `run_path` (needs repo layout + built `nrl` for bench phases) |
| `nrlpy run <script.py> [-- args…]` | Run script with `nrl` / `NRL` preloaded |
| `nrlpy <script.py> [-- args…]` | Same as `run` (TriPy-style shorthand) |
| `nrlpy chat` / `nrlpy talk` | English-friendly **status** loop: temperature (optional `psutil`), INT4 throughput probe, `nrl` path, immune evidence tail, session recall. Not an LLM unless you add one upstream. |
| `nrlpy chat --one "…"` | Single non-interactive reply (for scripts / tests). |
| `nrlpy evidence tail [N]` | Print last **N** JSONL lines from `NRL_EVIDENCE_LOG` or `build/immune/events.jsonl`. |

Optional: `pip install "nrlpy[chat]"` for thermal sensors via `psutil`.

### Bounded learn store (`nrlpy learn`)

| Command | Purpose |
|---------|---------|
| `nrlpy learn status` | Disk used vs **byte cap** (default **4 GiB**), unique word count, observation total. |
| `nrlpy learn cap BYTES` | Set cap (minimum **4096** bytes; use **≥ 1 MiB** in production). |

- **Directory:** `build/nrlpy_learn/` or override with **`NRL_LEARN_DIR`**.
- **Files:** `config.json`, `vocab.json` (word → count), `growth.jsonl` (observe/prune audit).
- **Chat:** every user line tokenizes alphanumerics (length ≥ 3) into the store; ask **“how much have you grown”** for stats. Disable with **`NRL_LEARN_DISABLE=1`**.
- **Schema:** [`../docs/schemas/learn_config_v1.schema.json`](../docs/schemas/learn_config_v1.schema.json).

### Seamless mode (no `import nrlpy` in your script)

`nrlpy compat.llm_globals()` merges **plain names** into the script namespace:

| Name | Role |
|------|------|
| `nrl` / `NRL` | `NRLRuntime` (lattice, assimilate, bench, …) |
| `next_prime(n)` | Smallest prime `> n` (Python + deterministic Miller–Rabin for `n < 2**64`) |
| `is_prime(n)` | Primality test (same bound) |
| `fabric_pulse(...)` | One-shot `braincore_int4` timing dict (explicit INT4 “pulse”) |

On first use, a **micro** `braincore_int4` run ties the process to the extension (no-op if the extension fails to load). **This is not** automatic compilation of arbitrary Python into lattice code; it is **zero-boilerplate** access plus honest math in CPython. Example: [`../examples/prime.py`](../examples/prime.py) — run `nrlpy examples/prime.py` from the repo root.

**Getting `nrlpy` on your PATH:** `pip install -e .` from this directory creates `nrlpy.exe` in Python `Scripts`. The repo installer (`scripts/install_nrl.ps1` / `install_nrl.sh`) also drops **`nrlpy.cmd`** or **`nrlpy`** next to `nrl` and sets `NRL_ROOT`; open a **new** shell so PATH updates apply.

## Python API (selected)

```python
import nrlpy

nrlpy.version()
nrlpy.braincore_packed_bytes(16)  # -> 8
nrlpy.braincore_int4_inplace(potentials, inputs, neurons, iterations, threshold)
nrlpy.assimilate_cli(neurons=4096, iterations=256, threshold=10)
from nrlpy.compat import nrl
from nrlpy import llm_globals, next_prime, is_prime  # explicit import when not using ``nrlpy run``
```

Benchmark profiles accepted by `bench_cli` match native: `sovereign`, `adaptive`, `war-drive`, `automatic` / `zpm`, `omega`, `omega-hybrid`.

## Quality

- Lint / format: `ruff` (config in `pyproject.toml`)
- Types: `mypy --strict`

## See also

- Root [README](../README.md) — full stack quick start and governance  
- [`nrl-architecture.md`](../nrl-architecture.md) — ABI and profile contracts  
