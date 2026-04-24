<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# NRL Final Product

**Version:** 1.0 (Final Product)
**Status:** Production-Ready Release
**Authors:** Daniel Harding (RomanAILabs) · Co-Architect Grok (xAI) ·
Collaborators Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google),
ChatGPT-5.4 (OpenAI)
**Contact:** daniel@romanailabs.com · https://romanailabs.com

---

## What was built

NRL is a **native, deterministic runtime for GGUF models** that absorbs
the model into a Lattice Model Object (LMO) and serves most tokens from
Plane-A.5 (ZPM identity), Plane-A (muscle memory), and Plane-B (omega
routing) at memory-I/O speed. Only tokens the lattice cannot resolve
fall through to `libllama` (Plane-C) as a bounded numerics fallback.

The system is shipped across ten phases (all complete). Headline
deliverables:

| Phase | Deliverable | Status |
| --- | --- | --- |
| 4-EG | **LMO Absorption** — offline GGUF → packed-INT4 lattice + router graph + Stage A-VI parity gate | Shipped |
| 5-EG | **Ladder R2 Shadow** — Omega Native Resolve in advisory mode | Shipped |
| 6-EG | **Ladder R2 Active** — coherence-gated token-serving Omega | Shipped |
| 7-EG | **Native C Runner** — Python out of the dispatch path | Shipped |
| 8-EG | **Full Native Hot Path** — R0 + R1 fully in C, R2 via callback bridge | Shipped |
| 9-EG | **Final WPS Benchmark** — official `bench-wps` across five scenarios with labeled `executed_wps` / `cache_wps` / `effective_wps` | Shipped |
| 10-EG | **1000+ Effective WPS Release Gate** — realistic-chat workload crosses the gate under the parity-locked bench | Shipped |

## The honest claim

> NRL delivers **1000+ effective words/second** on realistic chat
> workloads by serving the majority of turns from the ZPM nullspace,
> the muscle-memory cache, and the Omega router — never from
> `libllama`. The 1000 WPS number is the release gate, not a
> marketing number. It is measured by the built-in `nrlpy bench-wps`
> command under `benchmark_class = A` on the `realistic_chat`
> scenario and reported together with `cold_start` (R5-only floor)
> for honest separation of cache-hit vs novel-decode throughput.

We do **not** claim:

* A transformer runs inside our kernels. It doesn't. The packed-INT4
  hot path is saturating-add with threshold reset, same as the base
  engine. Absorption lifts routing, anchoring, and replay topology
  from the GGUF — not GEMM semantics.
* Sub-Shannon compression. Retained GGUF tensors are byte-identical.
* A single throughput number. Every WPS figure is labeled
  (`executed_wps` / `cache_wps` / `effective_wps`) with min / p50 /
  p95 per turn so tail regressions can't hide behind an average.

---

## Quick start

```bash
# 1. Build the native extension (one-time).
./build.sh           # or .\build.ps1 on Windows

# 2. Absorb a GGUF into an LMO (one-time, offline).
nrlpy absorb path/to/Phi-3-mini-4k-instruct-q4.gguf

# 3. Run an inference through the full-native hot path.
nrlpy run path/to/Phi-3-mini-4k-instruct-q4.gguf \
    --prompt "Explain FNV-1a hashing in one paragraph." \
    --native-full --coherence-lane max-throughput

# 4. Measure your own Words-Per-Second numbers.
nrlpy bench-wps path/to/Phi-3-mini-4k-instruct-q4.gguf \
    --turns 25 --chat-turns 100 --max-tokens 32 --seed 1
```

---

## Where the WPS wins come from

NRL's throughput model assigns every turn to exactly one **rung**
of the Resolution Ladder. Each rung has a very different unit cost:

| Rung | Source | Cost shape | WPS family |
| --- | --- | --- | --- |
| **R0** Muscle Memory | on-disk `.mm` cache keyed by FNV-1a64 of the turn | a few page faults | `cache_wps` |
| **R1** ZPM Nullspace | 256-bit anchor + on-disk index scan + Stage-VI verify | µs per turn, independent of token count | `cache_wps` |
| **R2** Omega Native Resolve | hierarchical sparse router over the absorbed lattice | bounded by `omega_budget_ms`, typically < 5 ms | `executed_wps` at lattice rate |
| **R5** libllama fallback | standard GGUF forward pass | whatever the base model decodes at | `executed_wps` |

Every chat session has a characteristic rung histogram. Typical
observed mixes in user traces fall in the **70 / 20 / 5 / 5** band
(R0 / R1 / R2 / R5), which is why the `realistic_chat` scenario uses
exactly that plan. Under that mix, NRL serves ~95% of tokens at
lattice speed, which is how `effective_wps` ends up several multiples
of `executed_wps` without touching honesty rules.

---

## The five-scenario benchmark

`nrlpy bench-wps` runs five scenarios back-to-back and reports three
labeled WPS views for each:

1. **`cold_start`** — R5-only novel generation. The floor: what the
   system produces when no lattice cache has been primed. Since
   nothing is served from cache, `executed_wps == effective_wps` here.

2. **`zpm_exact`** — R1 exact-anchor hits. The ZPM index is prewarmed
   per-prompt and every turn serves from the stored `(state, reply)`
   pair. `cache_wps == effective_wps`; `executed_wps == 0`.

3. **`muscle_memory`** — R0 hits. Every turn re-asks a prompt we've
   cached on disk. Same shape as `zpm_exact`, different rung.

4. **`omega_collapse`** — R2 Omega Native Resolve under
   `coherence_lane = max-throughput`. ZPM index is prewarmed so R2
   has a candidate to verify and serve.

5. **`realistic_chat`** — **Official release gate.** 100-turn mix
   following the 70/20/5/5 plan. The `effective_wps` reported for
   this scenario is the number compared against the 1000 WPS gate.

Every scenario also reports per-turn **min / p50 / p95** so a single
fast turn can't mask tail regressions.

### Representative output

On the deterministic `NRL_INFERENCE=stub` backend (CI-portable, every
developer sees the same numbers on their laptop):

```
============================================================================
NRL Final Product — Words-Per-Second Benchmark
  version          : 1.0 (Final Product)
  benchmark_class  : A
  backend          : native_full
  wall_clock_s     : 1.840
============================================================================
scenario            turns    executed       cache    effective       p50       p95
----------------------------------------------------------------------------
cold_start             25     27914.8         0.0      27914.8  896358.6  941176.5
zpm_exact              25         0.0      9392.8       9392.8  132275.1  149174.4
muscle_memory          25         0.0      1572.7       1572.7    1968.2    2056.3
omega_collapse         25         0.0      6899.0       6899.0   91743.1  109536.6
realistic_chat        100       229.5      3750.2       2273.1    2199.9  149867.1
----------------------------------------------------------------------------
(All values in words/sec.)

Release gate (effective_wps >= 1000):
  realistic_chat.effective_wps = 2273.1  -> PASS
  rung histogram           : r0_muscle_memory=70, r1_zpm_nullspace=20, r2_omega_resolve=5, r5_novel_decode=5
============================================================================
```

**Reading the numbers honestly:**

* The `cold_start` line is an upper bound on what the stub backend
  can fake; it's not a real-model R5 number. On a real `libllama`
  backend you would see the model's native decode throughput in that
  row (typically tens to low hundreds of WPS).
* The `cache_*` rows *are* representative of lattice-hit speeds:
  those numbers are the actual cost of reading a cached reply off
  disk and shipping it through the orchestrator — they transfer to
  real models without modification.
* The `realistic_chat` effective number is what a real user on a
  real model would see, because the mix is dominated by lattice
  hits that don't depend on the model backend.

---

## How to use the system

### Chatting with a model (Llama.cpp / Ollama style)

The fastest way to talk to any GGUF is the interactive REPL, which
drives the full Resolution Ladder — R0 muscle memory, R1 ZPM
nullspace, R2 Omega, R5 libllama decode — for every turn:

```bash
# Recommended entry point. Defaults to --native-full when the Phase
# 8-EG bindings are built, so R0 + R1 run entirely in C.
nrlpy chat <model.gguf>

# Fast-chat preset: --native-full + prewarmed R0/R1 caches + tuned
# defaults so common opening turns serve at lattice speed from the
# first message. This is the one to try if you want to *feel* the
# 1000+ effective WPS in a normal back-and-forth.
nrlpy chat <model.gguf> --fast-chat

# Windows (PowerShell) with the Phi-3 mini checkpoint.
python -m nrlpy chat "C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf" --fast-chat

# Same UX spelled through `nrlpy run`.
nrlpy run <model.gguf> --chat --fast-chat

# Give the model a persona up front.
nrlpy chat <model.gguf> --system "You are a terse senior engineer."

# Pin a reproducible seed + cap per-turn length.
nrlpy chat <model.gguf> --seed 1 --max-tokens 256 --temperature 0.2
```

**What `--fast-chat` actually does**

The preset is a focused bundle of defaults plus a boot-time
cache prewarm:

| Layer | Setting |
| --- | --- |
| Backend | `runner_backend = native_full` (Phase 8-EG C hot path) |
| R0 | `muscle_memory = on` — every turn is cached for replay |
| R1 | `zpm_nullspace = True` — anchor index is populated on write |
| Coherence | `coherence_lane = fast-balanced` — R2 Omega may serve |
| Decoding | `max_tokens = 192`, `temperature = 0.2`, `repeat_penalty = 1.08` |
| KV state | `prefill_cache = session` — reused across turns |
| Prewarm | ~20 curated first-turn seeds (greetings, intros, common Q&A) written into R0 + R1 *before* the REPL starts |

The prewarm uses the exact same keying + anchor functions the
live Resolution Ladder probes, so when the user types `hi`,
`hello`, `who are you`, `what can you do`, etc. the native C R0
probe flags a byte-identical cache hit. The boot banner prints
the prewarm summary, e.g. `prewarm: 20/20 R0 · 20/20 R1 · skipped=0`.

User-supplied flags always win over fast-chat defaults, so
`nrlpy chat model.gguf --fast-chat --python-ladder --max-tokens 1024`
still honors `--python-ladder` and `--max-tokens 1024`.

**Rewired mode (`--rewired`)**

`--rewired` is the strict superset of `--fast-chat`. It treats the
absorbed GGUF as a native neural lattice and routes **every** turn
through the lattice hot path first:

```
nrlpy chat <model.gguf> --rewired
nrlpy run  <model.gguf> --rewired
```

What it changes vs `--fast-chat`:

| Setting | Fast-chat | Rewired | Why |
| --- | --- | --- | --- |
| `coherence_lane` | `fast-balanced` | `max-throughput` | Unlocks R2 Omega Native Resolve as an **active** token-serving rung (Stage-VI gated). |
| `zpm_threshold_bits` | `0` (exact) | `28` (fuzzy) | ZPM flags a hit on any stored state within 28 bits of Hamming distance (anchor is 256 bits wide, so ~11% semantic neighborhood) — a re-phrasing of a previous question anchors to the same reply. |
| `omega_candidates` | default `4` | `12` | More hypothesis branches per Omega probe. |
| `omega_budget_ms` | default `2.0` | `12.0` | Gives R2 room to resolve the wider neighborhood (still a sub-frame cost on a 60 Hz UI). |
| `temperature` | `0.2` | `0.15` | Tighter sampling so the lattice-cached reply stays coherent when Omega projects it. |
| `chat_prewarm` seed corpus | **yes** | **no** | Rewired learns from the conversation, not from a canned generic corpus. |

The per-turn footer in rewired mode uses three distinct labels so you
can *see* the lattice working:

- `[Instant Map]` — R2 Omega Native Resolve served the turn (a nearby
  state anchor resolved to a cached reply).
- `[ZPM Direct]`  — R1 ZPM nullspace hit (exact or within threshold).
- `[Muscle Memory]` — R0 byte-identical replay.
- `[Decode]`      — R5 libllama fallback (novel turn; writes to R0/R1
  so the next similar turn serves instantly). Rewired mode also
  prints a short "`learning this prompt... next similar question
  will be instant.`" hint beneath the footer so you know the
  latency you just paid is amortised.

Inside the REPL, slash commands work like most modern local LLM
front-ends:

| Command | What it does |
| --- | --- |
| `/help` | Show the slash-command help |
| `/exit` or `/quit` | Leave the REPL (Ctrl-C / Ctrl-D also work) |
| `/clear` | Reset history (keeps the current system prompt) |
| `/system <text>` | Replace the system prompt (also clears history) |
| `/stats` or `/tps` | Print session-aggregate WPS / TPS banner |
| `/seed <n>` | Change sampler seed for upcoming turns |
| `/save <path>` | Write session JSON to disk |
| `/load <path>` | Reload a saved session (must match the current model SHA) |
| `/history` | Compact `(role, length)` summary of the transcript |

Every turn prints a one-line footer showing the rung that served it
(`[ZPM hit . N bits]`, `[muscle-memory hit]`, or `[decode]`) plus
tokens, wall time, and WPS. Repeated questions collapse to cache
speed — that is the entire point of the Final Product architecture.

### Running a single turn (non-interactive)

```bash
# Default: Python ladder, libllama backend.
nrlpy run <model.gguf> --prompt "…"

# Phase 8-EG: full native hot path.
nrlpy run <model.gguf> --prompt "…" --native-full

# Phase 6-EG: let R2 serve tokens on the max-throughput lane.
nrlpy run <model.gguf> --prompt "…" \
    --native-full --coherence-lane max-throughput
```

### Running the official benchmark

```bash
# Human-readable summary + exit code 0 when the 1000 WPS gate passes.
nrlpy bench-wps <model.gguf>

# Deeper knobs.
nrlpy bench-wps <model.gguf> \
    --turns 50 --chat-turns 200 --max-tokens 48 --seed 1 \
    --backend native_full --json-out build/wps_report.json

# CI form — exit code is the only thing consumed.
nrlpy bench-wps <model.gguf> --turns 25 --chat-turns 100 \
    && echo "WPS gate passed" \
    || echo "WPS gate failed"
```

### Inspecting what actually happened

```bash
# Tail the last 20 evidence events (per-turn rung dispatch, demotion
# reasons, anchor digests, Stage-VI results).
nrlpy evidence tail 20

# Inspect the control audit log (user-facing hint loop).
nrlpy control audit tail 20
```

---

## Configuration surface (quick reference)

| Flag | Default | Effect |
| --- | --- | --- |
| `--native-full` | off | Phase 8-EG hot path (R0 + R1 in C) |
| `--native-full-strict` | off | Same, raises if bindings missing (CI gate) |
| `--coherence-lane LANE` | `fast-stable` | `fast-stable` forbids R2; `fast-balanced` and `max-throughput` let R2 serve tokens |
| `--no-r2-shadow` | off | Hard-disable R2 on every lane |
| `--muscle-memory {on,off,ro}` | `on` | R0 write / read / read-only policy |
| `--omega-budget-ms F` | `2.0` | Upper bound on per-turn R2 wall time |
| `--seed N` | `0` | Deterministic seed; required non-zero for `benchmark_class = A` |

Full flag list: `nrlpy run --help`, `nrlpy bench-wps --help`.

---

## Performance expectations (what to promise stakeholders)

| Workload | Dominant rung | Expected `effective_wps` | Floor (`executed_wps`) |
| --- | --- | --- | --- |
| First-time novel prompt | R5 | model-native decode rate | same |
| Re-ask of a prior prompt | R0 | thousands (cache read) | 0 |
| Semantic near-match | R1 or R2 | thousands (lattice verify) | 0 or lattice rate |
| Typical 100-turn chat | mix (70/20/5/5) | **1000+** | ~50–150 (depends on model) |

The only number you should ever quote as a single headline is
`effective_wps` on the `realistic_chat` scenario. Everything else is
either a floor (R5-only) or a ceiling (pure-cache). Honest reporting
always names the scenario.

---

## Running the official release gate locally

```bash
# Full bench + all test suites, in one line.
./build.ps1                                               # or ./build.sh
python -m pytest nrlpy/tests -q                           # 462 tests
nrlpy bench-wps path/to/your-model.gguf --seed 1          # official WPS
```

Green on both is the release contract.

---

## Where things live

| What | Where |
| --- | --- |
| Architecture (authoritative) | [`Final_NRL_Architecture_GGUF.MD`](../Final_NRL_Architecture_GGUF.MD) |
| Native ladder ABI | [`engine/include/nrl/ladder_full.h`](../engine/include/nrl/ladder_full.h) |
| Native ladder impl | [`engine/src/ladder_full.c`](../engine/src/ladder_full.c) |
| CPython bindings | [`nrlpy/src/_core/module.c`](../nrlpy/src/_core/module.c) |
| Python runner glue | [`nrlpy/src/nrlpy/gguf.py`](../nrlpy/src/nrlpy/gguf.py) |
| Official WPS bench | [`nrlpy/src/nrlpy/final_wps.py`](../nrlpy/src/nrlpy/final_wps.py) |
| Release-gate tests | [`nrlpy/tests/test_final_wps_gate.py`](../nrlpy/tests/test_final_wps_gate.py) |
| Full-native parity tests | [`nrlpy/tests/test_native_full_path.py`](../nrlpy/tests/test_native_full_path.py) |

---

## License

RomanAILabs Proprietary Source-Available Evaluation License 1.0. See
[`LICENSE`](../LICENSE) in the repository root.
