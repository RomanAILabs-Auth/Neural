<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Production readiness (NRL + nrlpy)

Use this checklist before **live** or **customer** runs. Nothing here replaces your own security and compliance review.

## 1. Build and native smoke

| Step | Command (Windows) | Pass |
|------|---------------------|------|
| Engine + optional pyd + tests | `.\build.ps1 -Config Release -Tests` | `nrl-tests.exe` + `pytest` green |
| Locked harness + workload identity | `.\scripts\release_check.ps1` | `nrl_vs_cpp.json` contains `workload_identity` |
| Full gate | `.\scripts\live_readiness.ps1` | release + pytest when `_core` exists |
| GGUF runner gate (stub-only) | `.\scripts\live_readiness_gguf.ps1 -Mode stub` | golden harness `PASS`, writes `build\gguf_golden\gguf_golden.{json,md}` |
| GGUF runner gate (real model) | `.\scripts\live_readiness_gguf.ps1 -Mode real -Model C:\path\to\model.gguf` | golden harness `PASS` against real `libllama`; set `$env:NRL_GGUF_GOLDEN_MODEL` once to persist |

POSIX: `./build.sh Release 1` then `./scripts/release_check.sh`; optional `./scripts/live_readiness.sh` and `./scripts/live_readiness_gguf.sh [stub|real [model.gguf]]`.

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
- GitHub: [`.github/workflows/python-ci.yml`](../.github/workflows/python-ci.yml) runs `ruff` + `mypy --strict` + `pytest` on Ubuntu and Windows with `NRL_INFERENCE=stub`. Required gates are scoped to the GGUF runner files (`nrlpy/src/nrlpy/gguf.py`, `gguf_chat.py`, `__init__.py`, and their tests) plus the [`benchmarks/gguf_golden.py`](../benchmarks/gguf_golden.py) stub harness. Full-tree ruff and the full test suite run as advisory (`continue-on-error: true`). Golden artifacts (`build/gguf_golden/*`) are uploaded per-matrix-leg for cross-OS diffing.
- Windows + `nrlpy._core`: run locally via `live_readiness.ps1` (hosted runners vary).

### Known lint / test debt (not gating)

Tracked here because the CI workflow intentionally runs it as **advisory**, not a required gate, to avoid red-flagging every push on pre-existing issues unrelated to the GGUF runner work.

| File(s) | Kind | Notes |
|---|---|---|
| `nrlpy/src/nrlpy/chat.py`, `consent_ai.py`, `evidence.py`, `learn_store.py`, `native.py`, `paths.py`, `seamless.py`, `shadow.py`, `shell.py`, `workload.py` | ruff `I001` / `E501` / `RUF100` / `S603` / `S607` / `UP035` | Pre-existing; mostly legacy `noqa` directives for rules that were since disabled plus lines > 100 chars. Fix in a separate PR scoped "repo-wide lint cleanup". |
| `nrlpy/tests/test_cli_control.py`, `test_cli_evidence.py`, `test_control_hints.py`, `test_smoke.py` | ruff `E501`, pytest host-isolation | Tests read from real `$NRL_ROOT` / `%USERPROFILE%`; fail locally when those paths have state, pass on clean CI runners. Fix by teaching each test to `monkeypatch.setenv("NRL_ROOT", tmp_path)` consistently. |
| `nrlpy/src/nrlpy/cli.py` | ruff `E501` | Long USAGE strings and error prints; widen ruff `line-length` or refactor the usage block. |
