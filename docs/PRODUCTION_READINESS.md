# Production readiness (NRL + nrlpy)

Use this checklist before **live** or **customer** runs. Nothing here replaces your own security and compliance review.

## 1. Build and native smoke

| Step | Command (Windows) | Pass |
|------|---------------------|------|
| Engine + optional pyd + tests | `.\build.ps1 -Config Release -Tests` | `nrl-tests.exe` + `pytest` green |
| Locked harness + workload identity | `.\scripts\release_check.ps1` | `nrl_vs_cpp.json` contains `workload_identity` |
| Full gate | `.\scripts\live_readiness.ps1` | release + pytest when `_core` exists |

POSIX: `./build.sh Release 1` then `./scripts/release_check.sh`; optional `./scripts/live_readiness.sh`.

## 2. nrlpy control plane

| Check | Command |
|-------|---------|
| Version | `python -m nrlpy.cli --version` |
| Learn store cap | `python -m nrlpy.cli learn status` (default 4 GiB cap) |
| Chat one-shot | `python -m nrlpy.cli chat --one "status"` |
| Evidence tail | `python -m nrlpy.cli evidence tail 10` (requires `NRL_EVIDENCE_LOG` or `build/immune/events.jsonl`) |
| Lightning lattice PoC | `pip install -e nrlpy/[lightning]` then `python -m nrlpy.cli run examples/global_lightning_lattice.py --max-cycles 1` (see script docstring for `NRL_LIGHTNING_HTTP` / `NRL_LIGHTNING_JSONL`) |

## 3. Configuration

| Variable | Purpose |
|----------|---------|
| `NRL_ROOT` / `NRL_BIN` | Resolve `nrl` for benches and assimilate |
| `NRL_EVIDENCE_LOG` | Immune / audit JSONL path |
| `NRL_LEARN_DIR` | Bounded vocabulary store root |
| `NRL_LEARN_DISABLE=1` | Stop chat from ingesting tokens (air-gapped runs) |
| `NRL_LIGHTNING_HTTP` / `NRL_LIGHTNING_JSONL` | NDJSON strike feed for `examples/global_lightning_lattice.py` when WebSocket is blocked |
| `NRL_BLITZ_WS_URIS` | Comma-separated `wss://…` override list for the same example |

## 4. Governance

- Official throughput: locked profile + harness artifact + engine version in the same log ([`nrl-architecture.md`](../nrl-architecture.md)).
- Adaptive / aggressive profiles: Class B only unless policy explicitly allows Class A claims.
- Trading or money-moving automation: **out of scope** for this repo; never connect unreviewed LLM output to execution.

## 5. CI

- GitHub: [`.github/workflows/engine-ci.yml`](../.github/workflows/engine-ci.yml) builds the C engine and runs `nrl-tests` on Ubuntu.
- Windows + `nrlpy._core`: run locally via `live_readiness.ps1` (hosted runners vary).
