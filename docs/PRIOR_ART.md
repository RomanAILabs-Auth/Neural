<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

  # Prior art — what NRL's GGUF runner borrows from sibling RomanAILabs projects

This doc tracks which patterns in `nrlpy.gguf` were cross-pollinated from adjacent RomanAILabs work and — equally important — **which patterns we refused to pull in** because they would violate [`nrl-architecture.md`](../nrl-architecture.md) §0.5 (honest accounting) or §15 (no phantom throughput).

## `RomanAI-4D-GGUF-Engine` (sibling: 4D-geometric GGUF runner)

Reviewed: `C:\Users\Asus\Downloads\RomanAI-4D-GGUF-Engine-main`.

### Patterns taken

- **`NRL_STREAM_CHUNK_MS`** ← their `FOURD_STREAM_CHUNK_MS`. Per-token visible-stream pacing for video demos. The TPS banner flags paced runs with `paced=Xms (demo pacing, not native throughput)` so the pacing can never be quoted as raw TPS.
- **`NRL_INFERENCE` backend selector** ← their `FOURD_INFERENCE`. Supports `native` (llama-cpp-python, default) / `cli` (spawn `llama-cli.exe`) / `stub` (CI).
- **Honest-positioning doc tone.** Their `ARCHITECTURE.md` uses the same "*measured results before adjectives*" stance as our `nrl-architecture.md`. Convergent, not copied.

### Patterns refused

- **`four_d_engine` Rust crate** — quaternion RoPE / Cl(4,0) / w-axis tensor ops. Different architectural bet than NRL's packed-INT4 threshold-reset lattice. Merging them dilutes both stories. Keep as sibling, do not integrate.
- **Ollama forwarder mode** (`FOURD_INFERENCE=ollama`). Different product (Ollama drop-in). NRL is a runner with skip accounting; scope-guarded.
- **`gguf_inference.py`.** Non-streaming `create_chat_completion` with per-request model load. Superseded by our P1 contract.

## `Ghost_Compressor` (sibling: `nrlllama` launcher + "ghost seed" theatre)

Reviewed: `C:\Users\Asus\Desktop\RomanAILabs\Ghost_Compressor`.

### Patterns taken (from the honest half — `runner.py` / `ARCHITECTURE.md`)

- **`NRL_INFERENCE=cli` backend.** Resolves `llama-cli.exe` via `NRL_LLAMA_CLI` / `LLAMA_CLI` / `PATH`, spawns once, streams stdout. Zero per-token subprocess overhead. This is the right fallback for Windows hosts that cannot build `llama-cpp-python` against their Python.
- **Memory-pressure knobs.** `kv_cache_dtype` manifest key + `NRL_KV_CACHE` env var (`q8_0`, `q4_0`, …) maps to `-ctk/-ctv` in CLI mode and `type_k/type_v` in native mode. `no_repack` + `NRL_NO_REPACK` for `CPU_REPACK` allocation failures. `NRL_CTX` env override with `2048` default for 8 GB hosts.
- **`_diagnose_bad_model` helper.** When `model = X.gguf` doesn't resolve, we print which `.gguf` files live in the parent folder. Saves real user time.
- **Architecture-alignment quote.** Their `ARCHITECTURE.md` (`NrlLama`, v0.3): "LM work lives in upstream `llama.cpp`; NRL identity and RomanAILabs seeds attach before spawn, without pretending the INT4 lattice is doing transformer inference." This is exactly our stance. Preserved.

### Patterns refused

- **`ghost_compressor.py` — the "100 MB seed = compressed Llama-3-70B" claim.** A 100 MB `np.random.randn` buffer is not a compressed 70 B model. The `qwen2_5_72b_ghost.nrl` 100 MB file in that repo is random float32 bytes; the entropy of real quantized Llama-3 weights is ~4–5 bits/param, so 350× compression is past the Shannon limit by orders of magnitude. Adopting this would violate `nrl-architecture.md` §0.5 on day one.
- **`Roman_NRL_Engine_v7.py` — decorated Ollama stream.** Calls Ollama's streaming API, runs a no-op `_ = ghost_seed[:4096] * 0.1` in the consumer loop, then reports Ollama's TPS under the label `MANIFOLD METRICS`. This is re-labeling someone else's throughput. Hard refused.
- **`Nrlllama_Rogue_Direct_v2.py` — `Popen.stdout.read(1)` + heuristic break on `>`.** Slow (char-per-char Python I/O), desyncs on any prompt containing `>`. Superseded by the clean `runner.py` in the same folder; our `_CliLlm` uses line-buffered `readline` instead.
- **The "virtual throughput = params / elapsed" formula.** NRL already has the honest version from §15: `virtual_tps = executed_tps / (1 - skip_ratio)`, tied to real engine accounting. Do not stack a second, fabricated "virtual throughput" on top.

## Upstream (external)

- **`llama.cpp` / `llama-cpp-python`** — numerics library we supervise. L0 in `docs/nrl_gguf_runner_architecture.md`. Unchanged.
- **GGUF format** — Georgi Gerganov et al. Parsed via libllama.

---

*The rule: take ergonomic patterns, refuse accounting theatre. When a RomanAILabs sibling's story says the same thing as NRL's (honest "integration with AI"), port the pattern. When it invents throughput numbers, leave them in that sibling project and cross-reference here.*
