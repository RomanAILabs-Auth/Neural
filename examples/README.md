# NRL examples

Small, self-contained programs that demonstrate cross-surface workflows (native CLI, `nrlpy`, packed tensors).

| Example | Demonstrates |
|---------|----------------|
| [`assimilate_llm_solver.py`](./assimilate_llm_solver.py) | In-place packed INT4 assimilation vs `nrl assimilate` checksum parity; works with `nrlpy run` or plain `python` after `PYTHONPATH=nrlpy/src` |
| [`ultimate_power_demo.py`](./ultimate_power_demo.py) | Full-stack spectacle: assimilation waves, sovereign vs ZPM vs Omega vs hybrid, live stats, FNV parity, interactive `nrl chat` — run with **`nrl demo`** (recommended) or `python -m nrlpy.cli run examples/ultimate_power_demo.py` |
| [`real_prime_flex.py`](./real_prime_flex.py) | Digest verification: two `nrl assimilate` runs + bench; lattice spec from argv/env (no giant literals) |
| [`nrl_apocalypse_benchmark.py`](./nrl_apocalypse_benchmark.py) | Multi-size stress sweep with **correct** `nrl bench` argument order; explains what flat omega skip does (and does not) mean |

Run:

```bash
nrl demo
# or:
export PYTHONPATH=nrlpy/src   # POSIX
python3 -m nrlpy.cli run examples/assimilate_llm_solver.py
```

See root [README](../README.md) for Windows equivalents.
