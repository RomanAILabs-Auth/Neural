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
| `nrlpy --features` | CPU capability map (JSON) |
| `nrlpy variant <kernel>` | Active variant name |
| `nrlpy <file.nrl>` | Run `.nrl` via native `nrl file` |
| `nrlpy braincore4 [N] [iter] [thresh]` | Direct INT4 kernel timing |
| `nrlpy bench …` | Locked `nrl bench` bridge (JSON) |
| `nrlpy assimilate …` | Locked `nrl assimilate` bridge (JSON) |
| `nrlpy demo` | Same as `nrl demo`: runs `examples/ultimate_power_demo.py` via `run_path` (needs repo layout + built `nrl` for bench phases) |
| `nrlpy run <script.py> [-- args…]` | Run script with `nrl` / `NRL` preloaded |
| `nrlpy <script.py> [-- args…]` | Same as `run` (TriPy-style shorthand) |

**Getting `nrlpy` on your PATH:** `pip install -e .` from this directory creates `nrlpy.exe` in Python `Scripts`. The repo installer (`scripts/install_nrl.ps1` / `install_nrl.sh`) also drops **`nrlpy.cmd`** or **`nrlpy`** next to `nrl` and sets `NRL_ROOT`; open a **new** shell so PATH updates apply.

## Python API (selected)

```python
import nrlpy

nrlpy.version()
nrlpy.braincore_packed_bytes(16)  # -> 8
nrlpy.braincore_int4_inplace(potentials, inputs, neurons, iterations, threshold)
nrlpy.assimilate_cli(neurons=4096, iterations=256, threshold=10)
from nrlpy.compat import nrl
from nrlpy import llm_globals
```

Benchmark profiles accepted by `bench_cli` match native: `sovereign`, `adaptive`, `war-drive`, `automatic` / `zpm`, `omega`, `omega-hybrid`.

## Quality

- Lint / format: `ruff` (config in `pyproject.toml`)
- Types: `mypy --strict`

## See also

- Root [README](../README.md) — full stack quick start and governance  
- [`nrl-architecture.md`](../nrl-architecture.md) — ABI and profile contracts  
