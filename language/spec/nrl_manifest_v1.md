<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# .nrl Manifest v1 Spec

This extends [`minimal_nrl_v0.md`](./minimal_nrl_v0.md) for GGUF inference runs.

**Backward compatibility:** files without a `schema` line continue to parse under v0 rules. v1 activates when the first non-comment, non-blank line is `schema = nrl.manifest.v1`.

## Syntax (same as v0)

- UTF-8 text
- one `key = value` per line (whitespace around `=` optional)
- comments start with `#`
- blank lines ignored
- string values may be quoted with `"..."` to preserve leading/trailing whitespace

## Modes

- `mode = run` — v0 lattice run
- `mode = bench` — v0 lattice bench
- `mode = gguf_run` — **v1 only**, GGUF inference (requires `schema = nrl.manifest.v1`)

## Keys (v1 additions)

### L0 — numerics

| Key | Type | Default | Notes |
|---|---|---|---|
| `model` | path | — | **required** for `gguf_run`. Resolved relative to the manifest file. |
| `model_sha256` | hex string | none | If set, mismatch aborts the run before tokenization. |
| `prompt` | string | `""` | Mutually exclusive with `prompt_file`. |
| `prompt_file` | path | none | UTF-8 prompt read from disk. |
| `max_tokens` | int | `128` | Hard ceiling on generated tokens (not counting the prompt). |
| `temperature` | float | `0.7` | Passed to libllama sampler. |
| `top_p` | float | `0.9` | Nucleus sampling. |
| `top_k` | int | `40` | 0 disables top-k. |
| `repeat_penalty` | float | `1.1` | libllama repetition penalty. |
| `seed` | int | `0` | `0` = non-deterministic (wall-clock). Any non-zero seed pins the run. Required for `benchmark_class = A`. |
| `n_ctx` | int | `2048` | libllama context size. |
| `n_threads` | int | `0` | `0` = auto (`os.cpu_count()`). |
| `n_batch` | int | `512` | libllama eval batch. |
| `chat_format` | string | `none` | `none` / `chatml` / `phi3` / `llama2`. |

### L1 — gating lattice (P2+)

| Key | Type | Default | Notes |
|---|---|---|---|
| `gate_layer_policy` | string | `none` | `none` / `omega` / `omega-hybrid`. |
| `gate_expert_policy` | string | `none` | MoE routing policy. |
| `gate_kv_policy` | string | `none` | `none` / `zpm` (static-prefix collapse). |
| `gate_min_active` | int | `4` | omega-hybrid active-block floor. |
| `gate_wake_rate` | float | `0.25` | Target active sub-lattice fraction. |

In P1 these keys parse and are stored in the manifest, but the runner ignores them (dense execution only). This keeps manifests forward-compatible.

#### L1 — P2-Active structural gate (prefill cache)

| Key | Type | Default | Notes |
|---|---|---|---|
| `prefill_cache` | string | `"off"` | `"off"` / `"session"`. When `"session"` and the caller passes a `nrlpy.gates.PrefillGate` into `run_gguf(prefill_gate=...)`, the gate's shared-prefix report drives `TpsReport.gate_skip_ratio` with `gate_source = "prefill_cache"`. On the native libllama backend this is free via `n_past` carry-over (real KV-cache reuse); on stub / cli backends the accounting is structural (clearly labeled in the banner). A live prefill gate always wins over `gate_skip_ratio_override`. |

#### L1 — P2-Active simulation hinge

| Key | Type | Default | Notes |
|---|---|---|---|
| `gate_skip_ratio_override` | float | `0.0` | **Development / harness only.** Must be in `[0.0, 1.0)`. When > 0 and no structural gate fires, sets `TpsReport.gate_skip_ratio` directly. Flips the `virtual_tps == executed_tps` hinge and banner / evidence-log labels the run as `P2-Active simulation (override)`. Ignored on muscle-memory cache hits. Overridable per-host via `NRL_GATE_SKIP_RATIO_OVERRIDE`; explicit manifest value wins over env. Not a performance claim — use [`benchmarks/gguf_golden.py --mode p2active-sim`](../../benchmarks/gguf_golden.py) to enforce the math. |

### L2 — orchestrator

| Key | Type | Default | Notes |
|---|---|---|---|
| `respect_control_hints` | bool | `true` | Reads `build/control/preferences.json`. Same policy as `nrlpy.runtime.resolve_bench_profile_with_control_hints`. |
| `muscle_memory` | string | `on` | `on` / `off` / `replay-only`. |
| `muscle_memory_key_fields` | string | `model_sha256,prompt,sampler,seed,n_ctx` | Closed set; any subset. |
| `evidence_log` | path | `build/immune/events.jsonl` | JSONL, schema `nrl.gguf_run.v1`. |
| `telemetry_granularity` | string | `summary` | `summary` / `per_token`. |
| `benchmark_class` | string | `B` | `A` (deterministic; requires non-zero seed) or `B` (adaptive). |

### L0 — host-tuning knobs (tight-RAM operator pain)

| Key | Type | Default | Notes |
|---|---|---|---|
| `kv_cache_dtype` | string | `""` (engine default, f16) | One of `f32 / f16 / bf16 / q8_0 / q4_0 / q4_1 / iq4_nl / q5_0 / q5_1`. Maps to `-ctk/-ctv` in CLI backend, `type_k/type_v` in native backend. Lets 8 GB hosts run 7B models. |
| `no_repack` | bool | `false` | Pass `--no-repack` to llama.cpp. Set when you hit `CPU_REPACK` / allocation errors on load. |

These can be overridden per-host without editing the manifest via `NRL_KV_CACHE`, `NRL_NO_REPACK`, and `NRL_CTX` env vars (env wins over manifest default, not over explicit manifest value).

### Environment variables (runtime knobs, not manifest keys)

| Env | Meaning |
|---|---|
| `NRL_INFERENCE` | `native` (default, llama-cpp-python), `cli` (spawn `llama-cli.exe`), `stub` (deterministic fake for CI). |
| `NRL_LLAMA_CLI` | Full path to `llama-cli.exe` for `NRL_INFERENCE=cli` (falls back to `LLAMA_CLI` / `PATH`). |
| `NRL_KV_CACHE` | Fills `kv_cache_dtype` if unset in manifest. |
| `NRL_NO_REPACK` | Sets `no_repack=true` if unset in manifest. |
| `NRL_CTX` | Overrides `n_ctx` (useful for tight-RAM hosts, default 2048). |
| `NRL_STREAM_CHUNK_MS` | Per-token sleep in ms for visible-stream demos. When >0, the TPS banner adds `paced=Xms` (demo pacing, **not** a native throughput claim). |
| `NRL_ROOT` | Root for `cache/mm/` muscle-memory files and `build/immune/events.jsonl` (defaults to CWD). |
| `NRL_GATE_SKIP_RATIO_OVERRIDE` | P2-Active simulation hinge (see L1 table). Accepts `[0.0, 1.0)`; out-of-range silently ignored. Loses to an explicit manifest `gate_skip_ratio_override`. |

## Resolution order for CLI vs manifest

When both are present, CLI flags override manifest values — except `model`, which must match (a CLI `--model` differing from the manifest aborts).

## Example (minimal)

```ini
schema = nrl.manifest.v1
mode = gguf_run
model = models/phi-3-mini-4k.Q4_K_M.gguf
prompt = "Hello. Tell me a short fact about space."
max_tokens = 128
seed = 42
```

## Example (max flex)

See [`language/examples/phi3_omega_flex.nrl`](../examples/phi3_omega_flex.nrl).

## Errors

- Unknown key with `schema = nrl.manifest.v1` → parse error (strict, same discipline as v0).
- `model` missing when `mode = gguf_run` → parse error.
- Both `prompt` and `prompt_file` set → parse error.
- `benchmark_class = A` with `seed = 0` → parse error.
- `mode = gguf_run` without `schema = nrl.manifest.v1` → parse error ("mode=gguf_run requires schema v1").
