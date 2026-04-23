# NRL scripts

Automation entry points for build, install, benchmark-related release checks, and contributor sanity. Scripts preserve **deterministic defaults** and **explicit opt-in** for LM/AI and adaptive behavior.

## Scripts

| Script | Platform | Purpose |
|--------|----------|---------|
| [`install_nrl.ps1`](./install_nrl.ps1) | Windows | Build, install `nrl.exe` to user tree, PATH, consent JSON; `-OptInLMAI` |
| [`install_nrl.sh`](./install_nrl.sh) | POSIX | Same for `~/.local/bin`; `NRL_INSTALL_OPT_IN_LM_AI=1` |
| [`release_check.ps1`](./release_check.ps1) | Windows | Build, tests, locked `nrl_vs_cpp` harness, `workload_identity` JSON check, `nrlpy learn`/`chat`/`control status`, smoke `nrl status` / `inquire` / `nrl control` (sandbox) |
| [`release_check.sh`](./release_check.sh) | POSIX | Equivalent release pipeline + JSON check when Python available |
| [`live_readiness.ps1`](./live_readiness.ps1) | Windows | `release_check` + optional full `nrlpy` pytest when `_core*.pyd` exists |
| [`live_readiness.sh`](./live_readiness.sh) | POSIX | Same pattern for `_core*.so` / `.pyd` |

Installers also set **`NRL_ROOT`**, copy `examples/` + `py/nrlpy/`, and add **`nrlpy.cmd`** (Windows) or **`nrlpy`** (POSIX) next to **`nrl`** so **`nrlpy script.py`** works like TriPy after a new shell / PATH reload.

## Prerequisites

- **Zig** (`zig cc`) — used by root `build.ps1` on Windows; `build.sh` may use system `cc` on POSIX (see script headers).
- **Python 3.9+** — for `nrlpy` tests and harnesses when release check runs them.

## See also

- Root [README](../README.md) — Quick start and production install  
- [`nrl-architecture.md`](../nrl-architecture.md) — operator and consent contracts  
