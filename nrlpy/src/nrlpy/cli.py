# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""NRLPy CLI.

Contract (aligned with TriPy-style ergonomics):
  nrlpy <file.py>     Run a Python file with NRL assimilation globals injected (same as ``nrlpy run``).
  nrlpy <file.nrl>    Run a control file via native ``nrl file``.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import runtime
from .shell import run_path

USAGE = """\
nrlpy - Python front-end for the NRL machine-code engine

Usage:
  nrlpy <file.py>              Run Python with ``nrl`` / ``NRL`` + seamless builtins
                               (``next_prime``, ``is_prime``, ``fabric_pulse``) pre-injected.
  nrlpy run <script.py> [-- extra args...]   Same as above; explicit form.
  nrlpy run <model.gguf> [--prompt "..."] [--max-tokens N] [--seed N]
                               Bare filename is resolved from CWD, NRL_MODELS_DIR,
                               NRL_ROOT/models, ~/Desktop/Documents/RomaPy Engine.
  nrlpy run -specs <model.gguf>  Proof-of-concept speed lane (aggressive fast/coherent defaults)
                               Without --prompt/--prompt-file: start interactive GGUF chat.
                               With --prompt or --prompt-file: one-shot inference + TPS banner.
  nrlpy run <model.gguf> --chat [--native-full]
                               Force the interactive chat REPL (same UX as
                               ``nrlpy chat``). Honors all Resolution Ladder
                               flags (--native-full, --coherence-lane, etc.).
  nrlpy run <model.gguf> --fast-chat
                               Fast-chat preset: --chat + native_full hot
                               path + prewarmed R0 muscle-memory and R1
                               ZPM nullspace caches + tuned defaults
  nrlpy run <model.gguf> --rewired
                               Rewired preset (strict superset of --fast-
                               chat): R2 Omega Native Resolve is the
                               primary fast path, ZPM runs with a fuzzy
                               32-bit Hamming threshold, every R5 turn
                               feeds R0 + R1. No prewarmed seed corpus;
                               the lattice learns from the conversation.
                               (max_tokens=192, temperature=0.2,
                               coherence_lane=fast-balanced). The first
                               few turns of common conversation hit the
                               lattice cache so the 1000+ effective WPS
                               gate is felt in live chat, not just in
                               the benchmark. Same flag is accepted on
                               ``nrlpy chat``.
  nrlpy <model.gguf>           Same behavior, shorthand form.
  nrlpy gguf <manifest.nrl>    Run a ``.nrl`` v1 manifest (mode=gguf_run).
  nrlpy --version
  nrlpy doctor                 Check NRL_ROOT, disk space, Python, native core, and safety flags
  nrlpy --features
  nrlpy variant <kernel>
  nrlpy <file.nrl>
  nrlpy braincore4 [N] [ITER] [THRESHOLD]
  nrlpy bench [N] [ITER] [REPS] [THRESHOLD] [PROFILE] [--respect-control-hints]
  nrlpy control status                     JSON: prefs path, parsed file, hints_active
  nrlpy control audit tail [N]            Last N lines of control_audit.jsonl
  nrlpy assimilate [N] [ITER] [THRESHOLD]
  nrlpy demo
  nrlpy chat | talk            English-friendly status loop (telemetry + evidence tail)
  nrlpy chat --one "…"         Single non-interactive reply
  nrlpy chat <model.gguf> [--system "…"] [--seed N] [--max-tokens N] [--temperature F] [--chat-format FMT]
                               [--fast-chat | --rewired] [--native-full | --native | --python-ladder]
                               Multi-turn GGUF chat REPL — Llama.cpp / Ollama
                               style back-and-forth on the Final Product
                               Resolution Ladder. Defaults to --native-full
                               when the Phase 8-EG bindings are built.
                               --fast-chat additionally prewarms the R0
                               muscle-memory and R1 ZPM nullspace caches
                               with a curated seed corpus so common first
                               turns serve from cache instantly. Type
                               /help inside for slash commands
                               (/help, /exit, /clear, /stats, /system, /seed,
                               /save, /load, /history).
  nrlpy chat <manifest.nrl>    Same, parameters from a .nrl v1 manifest
  nrlpy adaptive-chat <model.gguf> [--system "…"] [--fast-chat|--rewired]
                               Phase 15-EG R&D REPL: native_full + phi3 only;
                               sidecar primes ZPM RAM after >3 consecutive R5 turns.
  nrlpy learn status           Bounded vocab store: disk use vs cap (default 4 GiB)
  nrlpy learn cap BYTES        Set learn-store byte cap (min 1 MiB)
  nrlpy evidence tail [N]     Last N lines of immune JSONL (NRL_EVIDENCE_LOG or build/immune/events.jsonl)
  nrlpy -ai|--ai on|off       LM/AI opt-in: ~/.nrl/consent.json + setx on Windows
  nrlpy wps-gate [--ci]      Full WPS/gguf gate (pytest slice + golden + autopilot).
                               Works from any cwd if NRL_REPO points at the git clone
                               (set automatically by install_nrl.ps1).
  nrlpy wps-chat-bench <model.gguf> [--auto-tune]
                               Real-model chat WPS bench: preload once, scripted
                               transcript (cold + warm + replay), JSON + MD artifacts.
                               Bare filenames resolve via NRL_MODELS_DIR.
  nrlpy bench-wps <model.gguf> [--turns N] [--chat-turns N] [--max-tokens N]
                               [--seed N] [--backend native_full|native|python]
                               [--json] [--json-out PATH]
                               Final Product (Phase 9-EG + 10-EG) official WPS
                               benchmark: measures executed_wps / cache_wps /
                               effective_wps across five scenarios (cold_start,
                               zpm_exact, muscle_memory, omega_collapse,
                               realistic_chat). The realistic_chat
                               effective_wps is the 1000+ WPS release gate.
                               benchmark_class = A (seeded, reproducible).
  nrlpy zpm [--target HEX] [--text "…"] [--no-color]
                               ROMA-ZPM v2.0 pipeline demo: anchor → inversion
                               → rotor → nullspace → verify → singularity. Used
                               to inspect Plane-A.5 state for a given input.
  nrlpy lmo info <model.gguf|lmo_dir|sha-prefix> [--json]
                               P4 persistence: ZPM index size, entry count,
                               WAL + snapshot metadata, muscle-memory stats.
  nrlpy lmo coverage <model.gguf|lmo_dir|sha-prefix> [--json]
                               P5 Drift Conqueror: lattice coverage %, weak ZPM
                               buckets, 24h growth headroom, last conquest run.
  nrlpy lmo prune <model.gguf|lmo_dir|sha-prefix> [--dry-run] [--aggressive] [--force] [--json]
                               P6 HD quota: LRU ZPM eviction + WAL compaction;
                               preview with --dry-run (no writes).
  nrlpy absorb <model.gguf> [--out-root DIR] [--force] [--no-libllama] [--json]
                               Phase 4-EG: absorb a GGUF into a Lattice Model
                               Object (LMO). Writes packed INT4 tiles,
                               retained/ byte slices, router.graph, lmo.header,
                               attest.json and manifest.nrl under
                               $NRL_ROOT/cache/lmo/<sha256>/. Stage A-VI is a
                               hard gate (retained byte-identity + determinism
                               + header round-trip); any failure refuses to
                               write attest.json.
  nrlpy lmo-inspect <lmo_dir> <token_id> [--row-units N] [--origin NAME] [--preview K]
                               Phase 13 R0' foundation: extract and print one
                               token's INT4 packed row from an absorbed LMO.
                               Returns the NRL packed projection for that
                               vocab index (deterministic, addressable). This
                               is NOT the Q4_K-dequantized F16 embedding; that
                               is Phase 13.5 (requires a Q4_K decoder).
  nrlpy nrl-ai <subcommand> [...]
                               NRL-AI native inference path (no libllama, no GPU).
                               Subcommands: status | ingest | chat | bench.
                               Target: >=1000 words/sec on CPU. Landing across
                               prompts #2-#7; run `nrl-ai status` for roadmap.
  nrlpy run <model.gguf> ... [--legacy-llama]
                               Forward-compat flag: pins the libllama decode
                               path. No-op today (still the default); becomes
                               meaningful when NRL-AI ships as the default.
  nrlpy run <model.gguf> ... [--coherence-lane LANE] [--no-r2-shadow]
                               [--omega-budget-ms F] [--omega-candidates N] [-v]
                               Phase 6-EG Rung-R2 (Omega Native Resolve) controls.
                               LANE is one of fast-stable (default; R2 off),
                               fast-balanced, or max-throughput. On the two
                               non-stable lanes R2 runs in ACTIVE token-serving
                               mode: on a ZPM hit + Stage-VI verify it emits the
                               stored reply and skips libllama. --no-r2-shadow
                               forces R2 off entirely (all lanes; name is kept
                               for flag stability — see Phase 5-EG docs).
                               -v/--verbose prints a one-line R2 activity
                               summary to stderr after the TPS banner.
  nrlpy run <model.gguf> ... [--native | --python-ladder | --native-strict]
                               Phase 7-EG dispatcher selection.
                               --python-ladder (default) uses the pure-Python
                               §4.2 Resolution Ladder (Phase 6-EG behavior).
                               --native routes the rung-selection decision and
                               libllama bridge call through the C dispatcher
                               (engine/src/ladder_native.c); silently falls
                               back to Python if nrlpy._core lacks the Phase
                               7-EG bindings. --native-strict raises an error
                               instead of falling back (CI-gate use).
  nrlpy run <model.gguf> ... [--native-full | --native-full-strict]
                               Phase 8-EG full-native hot path selection.
                               --native-full drives R0 (muscle memory) and
                               R1 (ZPM nullspace) entirely in C with R2/R5
                               served through registered bridge callbacks;
                               on the common cache-hit path Python never
                               sits in the decode loop. Silently falls back
                               to --native and then --python-ladder if the
                               Phase 8-EG bindings are missing. --native-
                               full-strict raises instead of falling back.
"""


def _parse_gguf_cli_args(args: list[str]) -> tuple[str, dict[str, object]]:
    """Parse ``run <model.gguf>`` / ``<model.gguf>`` trailing flags into kwargs."""
    model_path = args[0]
    kwargs: dict[str, object] = {}
    i = 1
    while i < len(args):
        flag = args[i]
        if flag in {"--prompt", "-p"} and i + 1 < len(args):
            kwargs["prompt"] = args[i + 1]
            i += 2
        elif flag in {"--prompt-file"} and i + 1 < len(args):
            kwargs["prompt_file"] = args[i + 1]
            i += 2
        elif flag in {"--max-tokens", "-n"} and i + 1 < len(args):
            kwargs["max_tokens"] = int(args[i + 1])
            i += 2
        elif flag in {"--temperature", "--temp"} and i + 1 < len(args):
            kwargs["temperature"] = float(args[i + 1])
            i += 2
        elif flag == "--top-p" and i + 1 < len(args):
            kwargs["top_p"] = float(args[i + 1])
            i += 2
        elif flag == "--top-k" and i + 1 < len(args):
            kwargs["top_k"] = int(args[i + 1])
            i += 2
        elif flag == "--repeat-penalty" and i + 1 < len(args):
            kwargs["repeat_penalty"] = float(args[i + 1])
            i += 2
        elif flag == "--seed" and i + 1 < len(args):
            kwargs["seed"] = int(args[i + 1])
            i += 2
        elif flag == "--n-ctx" and i + 1 < len(args):
            kwargs["n_ctx"] = int(args[i + 1])
            i += 2
        elif flag == "--n-threads" and i + 1 < len(args):
            kwargs["n_threads"] = int(args[i + 1])
            i += 2
        elif flag == "--n-batch" and i + 1 < len(args):
            kwargs["n_batch"] = int(args[i + 1])
            i += 2
        elif flag == "--chat-format" and i + 1 < len(args):
            kwargs["chat_format"] = args[i + 1]
            i += 2
        elif flag == "--system" and i + 1 < len(args):
            kwargs["_system"] = args[i + 1]
            i += 2
        elif flag == "--profile" and i + 1 < len(args):
            kwargs["profile"] = args[i + 1]
            i += 2
        elif flag == "--no-muscle-memory":
            kwargs["muscle_memory"] = "off"
            i += 1
        elif flag == "--no-control-hints":
            kwargs["respect_control_hints"] = False
            i += 1
        elif flag == "--bench-class" and i + 1 < len(args):
            kwargs["benchmark_class"] = args[i + 1]
            i += 2
        elif flag == "--no-stream":
            kwargs["_no_stream"] = True
            i += 1
        elif flag == "--chat":
            # Force the interactive GGUF chat REPL even if --prompt or
            # --prompt-file is present. Without this flag ``nrlpy run
            # model.gguf`` already starts the chat when no prompt is
            # supplied; --chat makes the intent explicit and is the
            # documented entry point alongside ``nrlpy chat``.
            kwargs["_chat"] = True
            i += 1
        elif flag == "--fast-chat":
            # Short-cut: ``--fast-chat`` implies ``--chat`` + prewarmed
            # R0/R1 caches + tuned defaults (native_full backend,
            # zpm_nullspace on, bounded reply length). See
            # :func:`nrlpy.gguf_chat._apply_fast_chat_defaults`.
            kwargs["_chat"] = True
            kwargs["_fast_chat"] = True
            i += 1
        elif flag == "--rewired":
            # Strict superset of ``--fast-chat``: the absorbed GGUF is
            # treated as a native neural lattice. R2 Omega Native
            # Resolve is unlocked as the primary fast path, ZPM runs
            # with a fuzzy Hamming threshold, and every R5 turn
            # feeds R0 + R1 so the system learns the conversation
            # live. No prewarmed seed corpus.
            # See :func:`nrlpy.gguf_chat._apply_rewired_defaults`.
            kwargs["_chat"] = True
            kwargs["_rewired"] = True
            i += 1
        elif flag in {"--specs", "-specs"}:
            kwargs["_specs"] = True
            i += 1
        elif flag == "--legacy-llama":
            # Forward-compatible switch: today the libllama decode path is
            # still the default for ``nrl run <gguf>``. When NRL-AI lands as
            # the default in prompt #5, this flag opts back into the old
            # path without a rebuild. Accepted + recorded now.
            kwargs["_legacy_llama"] = True
            i += 1
        elif flag == "--coherence-lane" and i + 1 < len(args):
            # Phase 5-EG §4.5 lane whitelist. ``fast-stable`` (default)
            # forbids R2; ``fast-balanced`` + ``max-throughput`` enable
            # the R2 shadow probe. See Final_NRL_Architecture_GGUF.MD.
            kwargs["coherence_lane"] = args[i + 1]
            i += 2
        elif flag == "--no-r2-shadow":
            kwargs["r2_shadow_enabled"] = False
            i += 1
        elif flag == "--omega-budget-ms" and i + 1 < len(args):
            kwargs["omega_budget_ms"] = float(args[i + 1])
            i += 2
        elif flag == "--omega-candidates" and i + 1 < len(args):
            kwargs["omega_candidates"] = int(args[i + 1])
            i += 2
        elif flag in {"--verbose", "-v"}:
            kwargs["_verbose"] = True
            i += 1
        elif flag == "--native":
            # Phase 7-EG: route §4.2 dispatch through the C ladder.
            # Falls back to the Python ladder if nrlpy._core is missing
            # the Phase 7-EG bindings (e.g. host hasn't built it).
            kwargs["runner_backend"] = "native"
            i += 1
        elif flag == "--native-strict":
            # CI-gate variant: raises RuntimeError instead of falling
            # back. Useful for proving the native runner is engaged in
            # benchmark gates.
            kwargs["runner_backend"] = "native_strict"
            i += 1
        elif flag == "--python-ladder":
            # Forward-compat: pin the Python ladder explicitly. Today
            # this is the default; once Phase 8-EG passes its release
            # gate the default flips to ``native_full`` and this flag
            # opts back into the Python path for debugging.
            kwargs["runner_backend"] = "python"
            i += 1
        elif flag == "--native-full":
            # Phase 8-EG: drive R0 + R1 natively in C, with R2 and R5
            # on bridge callbacks. Falls back to --native and then the
            # Python ladder if the Phase 8-EG bindings are missing.
            kwargs["runner_backend"] = "native_full"
            i += 1
        elif flag == "--native-full-strict":
            # CI-gate variant of --native-full: raises RuntimeError if
            # the full-native bindings are unavailable.
            kwargs["runner_backend"] = "native_full_strict"
            i += 1
        else:
            raise ValueError(f"unknown flag for gguf run: {flag!r}")
    return model_path, kwargs


def _detect_split_gguf_slot(args: list[str], start: int) -> int:
    """Return index of the first arg ending with ``.gguf`` from ``start``."""
    for i in range(start, len(args)):
        if args[i].lower().endswith(".gguf"):
            return i
    return -1


def _infer_chat_format_from_model(model_path: str) -> str:
    """Best-effort chat template guess from model filename."""
    low = Path(model_path).name.lower()
    if "phi-3" in low or "phi3" in low:
        return "phi3"
    if "llama-2" in low or "llama2" in low:
        return "llama2"
    return "none"


def _apply_chat_fast_stable_defaults(
    manifest: Any,
    *,
    user_overrides: set[str],
) -> None:
    # Defaults for interactive chat:
    #   - coherent answers at bounded length (max_tokens=256)
    #   - cache-accelerated throughput via prefill_cache=session
    #   - muscle memory ON so exact-repeat turns collapse to I/O speed
    #   - ZPM identity resolver ON (Plane A.5) so every served turn is
    #     indexed for the next identical request
    if "n_batch" not in user_overrides:
        manifest.n_batch = max(int(manifest.n_batch), 1024)
    if "max_tokens" not in user_overrides:
        manifest.max_tokens = 256
    if "temperature" not in user_overrides:
        manifest.temperature = 0.2
    if "repeat_penalty" not in user_overrides:
        manifest.repeat_penalty = 1.08
    manifest.prefill_cache = "session"
    if manifest.muscle_memory not in {"off", "replay-only"}:
        manifest.muscle_memory = "on"
    if "zpm_nullspace" not in user_overrides:
        manifest.zpm_nullspace = True


def _apply_specs_defaults(
    manifest: Any,
    *,
    user_overrides: set[str],
) -> None:
    # PoC specs lane: aggressive sampling + throughput-friendly batch,
    # coherent answer length (max_tokens big enough for a full sentence),
    # no artificial WPS cap. Same fast-lane caches as fast-stable.
    if "n_batch" not in user_overrides:
        manifest.n_batch = max(int(manifest.n_batch), 2048)
    if "max_tokens" not in user_overrides:
        manifest.max_tokens = 256
    if "temperature" not in user_overrides:
        manifest.temperature = 0.15
    if "repeat_penalty" not in user_overrides:
        manifest.repeat_penalty = 1.1
    if "n_ctx" not in user_overrides:
        manifest.n_ctx = max(int(manifest.n_ctx), 4096)
    manifest.prefill_cache = "session"
    if manifest.muscle_memory not in {"off", "replay-only"}:
        manifest.muscle_memory = "on"
    if "zpm_nullspace" not in user_overrides:
        manifest.zpm_nullspace = True


def _candidate_model_dirs() -> list[Path]:
    """Ordered list of directories to search for a bare ``<name>.gguf``."""
    dirs: list[Path] = [Path.cwd()]
    env_dir = os.environ.get("NRL_MODELS_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir))
    nrl_root = os.environ.get("NRL_ROOT", "").strip()
    if nrl_root:
        dirs.append(Path(nrl_root) / "models")
    home = Path.home()
    dirs.extend([
        home / "Desktop" / "Documents" / "RomaPy Engine",
        home / "Desktop" / "Documents" / "NRL" / "models",
        home / "models",
        home / ".nrl" / "models",
    ])
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _resolve_model_path(model_path: str) -> str:
    """Resolve a ``<name>.gguf`` argument to a real file path.

    If the literal path exists, return it untouched. Otherwise search the
    canonical list of candidate model directories for a matching basename.
    Returns the original string unchanged when no match is found (the caller
    surfaces a helpful "not found + list of candidates" error downstream).
    """
    if not model_path:
        return model_path
    p = Path(model_path)
    if p.is_file():
        return model_path
    basename = p.name
    if not basename.lower().endswith(".gguf"):
        return model_path
    # Only search when the given path was a bare filename; we don't want to
    # silently remap a user-typed absolute path to a different location.
    if p.parent != Path("."):
        return model_path
    for d in _candidate_model_dirs():
        candidate = d / basename
        if candidate.is_file():
            return str(candidate)
    return model_path


def _normalize_specs_prefix(
    model_path: str,
    kwargs: dict[str, object],
) -> tuple[str, bool]:
    specs_mode = bool(kwargs.pop("_specs", False))
    stripped = model_path.strip()
    low = stripped.lower()
    for prefix in ("-specs ", "--specs "):
        if low.startswith(prefix):
            specs_mode = True
            stripped = stripped[len(prefix) :].strip()
            break
    return stripped, specs_mode


def _run_gguf_from_cli(model_path: str, kwargs: dict[str, object]) -> int:
    from . import gguf

    user_overrides = set(kwargs.keys())
    stream_flag = bool(kwargs.pop("_no_stream", False))
    system_prompt = str(kwargs.pop("_system", ""))
    verbose = bool(kwargs.pop("_verbose", False))
    force_chat = bool(kwargs.pop("_chat", False))
    fast_chat = bool(kwargs.pop("_fast_chat", False))
    rewired = bool(kwargs.pop("_rewired", False))
    user_overrides.discard("_verbose")
    user_overrides.discard("_chat")
    user_overrides.discard("_fast_chat")
    user_overrides.discard("_rewired")
    if rewired:
        # Rewired mode is the strict superset of fast-chat and wins
        # over fast-chat if both are set. It unlocks R2 Omega Native
        # Resolve as the primary fast path via coherence_lane=
        # max-throughput and a fuzzy ZPM Hamming threshold.
        from .gguf_chat import _apply_rewired_defaults  # noqa: PLC0415

        _apply_rewired_defaults(kwargs, user_overrides)
        user_overrides |= {
            "runner_backend",
            "muscle_memory",
            "max_tokens",
            "temperature",
            "repeat_penalty",
            "coherence_lane",
            "prefill_cache",
            "omega_budget_ms",
            "omega_candidates",
        }
    elif fast_chat:
        # Fast chat mode is a curated preset — equivalent to what
        # ``nrlpy chat --fast-chat`` applies. Fill in only the slots
        # the user didn't set explicitly.
        from .gguf_chat import _apply_fast_chat_defaults  # noqa: PLC0415

        _apply_fast_chat_defaults(kwargs, user_overrides)
        user_overrides |= {
            "runner_backend",
            "muscle_memory",
            "coherence_lane",
            "prefill_cache",
            "max_tokens",
            "temperature",
            "repeat_penalty",
        } & set(kwargs.keys())
    # Scaffold flag for the NRL-AI pivot (prompt #1 of the rebuild). Today
    # libllama is still the default decode path; ``--legacy-llama`` is a
    # no-op guard so the switch to NRL-AI-by-default in prompt #5 does not
    # silently break anyone's scripts -- they just add ``--legacy-llama``.
    user_overrides.discard("_legacy_llama")
    kwargs.pop("_legacy_llama", None)
    model_path, specs_mode = _normalize_specs_prefix(model_path, kwargs)
    if not model_path or model_path.startswith("-"):
        print("error: specs mode requires a model path ending in .gguf", file=sys.stderr)
        return 2
    model_path = _resolve_model_path(model_path)
    try:
        manifest = gguf.manifest_from_args(model_path, **kwargs)  # type: ignore[arg-type]
    except gguf.ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if rewired:
        from .gguf_chat import apply_rewired_post_build  # noqa: PLC0415

        apply_rewired_post_build(manifest, user_overrides)
    elif fast_chat:
        from .gguf_chat import apply_fast_chat_post_build  # noqa: PLC0415

        apply_fast_chat_post_build(manifest, user_overrides)
    if "chat_format" not in kwargs and manifest.chat_format == "none":
        manifest.chat_format = _infer_chat_format_from_model(model_path)
    if force_chat or (not manifest.prompt and not manifest.prompt_file):
        from .gguf_chat import run_gguf_chat_repl

        if force_chat:
            # --chat was explicit: ignore any --prompt / --prompt-file and
            # start the interactive REPL. No surprises: the user asked for
            # chat, they get chat.
            manifest.prompt = ""
            manifest.prompt_file = ""
            user_overrides.discard("prompt")
            user_overrides.discard("prompt_file")
        if specs_mode:
            _apply_specs_defaults(manifest, user_overrides=user_overrides)
            if not system_prompt:
                system_prompt = (
                    "You are NRL specs chat. Respond clearly and coherently. "
                    "Keep answers concise unless user asks for detail."
                )
            print(
                "NRL specs mode: fast coherent PoC defaults "
                "(max_tokens=256, temperature=0.15, n_batch>=2048, n_ctx>=4096)"
            )
        else:
            _apply_chat_fast_stable_defaults(manifest, user_overrides=user_overrides)
        manifest.prompt = ""
        manifest.prompt_file = ""
        try:
            run_gguf_chat_repl(
                manifest,
                system=system_prompt,
                fast_chat=fast_chat and not rewired,
                rewired=rewired,
            )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0
    try:
        if specs_mode:
            _apply_specs_defaults(manifest, user_overrides=user_overrides)
        result = gguf.run_gguf(manifest, stream_to=None if stream_flag else sys.stdout)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(gguf.format_banner(result))
    if verbose:
        shadow = result.omega_shadow
        print(
            "[nrl.r2] lane={lane} mode={mode} status={status} "
            "served={served} served_tokens={stoks} hits={hits} "
            "demotions={dem} wall_ms={wall:.2f}".format(
                lane=shadow.coherence_lane,
                mode=shadow.mode,
                status=shadow.status,
                served="yes" if shadow.served else "no",
                stoks=shadow.served_tokens,
                hits=shadow.hits,
                dem=",".join(shadow.demotion_reasons) or "(none)",
                wall=shadow.wall_ms,
            ),
            file=sys.stderr,
        )
        print(
            "[nrl.runner] backend={rb} gate_source={gs}".format(
                rb=result.manifest.runner_backend,
                gs=result.gate_source or "(none)",
            ),
            file=sys.stderr,
        )
    return 0


def _run_gguf_manifest(path: str) -> int:
    from . import gguf

    try:
        manifest = gguf.load_manifest(path)
    except gguf.ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if manifest.mode != "gguf_run":
        print(
            f"error: manifest mode is {manifest.mode!r}; use `nrlpy <path>` for non-gguf modes",
            file=sys.stderr,
        )
        return 2
    try:
        result = gguf.run_gguf(manifest, stream_to=sys.stdout)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(gguf.format_banner(result))
    return 0


def _zpm_cli(args: list[str]) -> int:
    """``nrlpy zpm`` — ROMA-ZPM v2.0 pipeline demo.

    Mirrors the output of the ``zpm`` C++ CLI in docs/ROMA-ZPM-Grok.txt so a
    user can visually verify Plane-A.5 state for any given input.

    Flags:
      ``--target <hex>``   32-hex-nibble 64-bit target word 0 (word[1..3] keep
                           their ROMA defaults).
      ``--text "…"``       Compute the state from UTF-8 text (overrides --target).
      ``--no-color``       Disable ANSI color escapes.
    """
    from . import zpm as _zpm

    target_word0: int | None = None
    text_in: str | None = None
    ansi = True
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--target" and i + 1 < len(args):
            try:
                target_word0 = int(args[i + 1], 16)
            except ValueError:
                print("error: --target expects a 64-bit hex value", file=sys.stderr)
                return 2
            i += 2
        elif a == "--text" and i + 1 < len(args):
            text_in = args[i + 1]
            i += 2
        elif a in {"--no-color", "--no-ansi"}:
            ansi = False
            i += 1
        elif a in {"-h", "--help"}:
            print(
                "nrlpy zpm [--target <hex>] [--text \"…\"] [--no-color]\n"
                "  Runs the ROMA-ZPM Stage I–VII pipeline against the given "
                "state/text\n  and prints the anchor/inversion/rotor/nullspace"
                "/verify banner.\n"
            )
            return 0
        else:
            print(f"error: unknown zpm flag {a!r}\n", file=sys.stderr)
            return 2

    if text_in is not None:
        state = _zpm.anchor(text_in.encode("utf-8"))
    elif target_word0 is not None:
        state = (
            target_word0 & 0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
        )
    else:
        # ROMA-ZPM v2.0 DEFAULT
        state = (
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
        )
    print(_zpm.format_stage_banner(state, solution=state, ansi=ansi))
    return 0


def _absorb_cli(args: list[str]) -> int:
    """``nrlpy absorb <model.gguf>`` — Phase 4-EG LMO absorption.

    Flags:
      ``--out-root DIR``   Root directory for the LMO cache (defaults to
                           ``$NRL_ROOT/cache/lmo`` or ``$CWD/cache/lmo``).
      ``--force``          Re-run absorption even if the LMO already exists.
      ``--no-libllama``    Skip the optional libllama forward smoke test.
      ``--json``           Emit a JSON status object instead of a text banner.
    """
    from . import lmo as _lmo

    gguf_path: str | None = None
    out_root: str | None = None
    force = False
    attempt_libllama = True
    as_json = False

    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-h", "--help"}:
            print(
                "nrlpy absorb <model.gguf> "
                "[--out-root DIR] [--force] [--no-libllama] [--json]\n"
                "  One-time, offline absorption of a GGUF into a Lattice Model Object.\n"
                "  Stage A-VI (byte-identity + determinism + header round-trip) is a hard gate.\n"
            )
            return 0
        if a == "--out-root" and i + 1 < len(args):
            out_root = args[i + 1]
            i += 2
            continue
        if a == "--force":
            force = True
            i += 1
            continue
        if a == "--no-libllama":
            attempt_libllama = False
            i += 1
            continue
        if a == "--json":
            as_json = True
            i += 1
            continue
        if a.startswith("-"):
            print(f"error: unknown absorb flag {a!r}\n\n" + USAGE, file=sys.stderr)
            return 2
        if gguf_path is None:
            gguf_path = a
            i += 1
            continue
        print(f"error: unexpected positional arg {a!r}\n\n" + USAGE, file=sys.stderr)
        return 2

    if not gguf_path:
        print("error: nrlpy absorb requires a <model.gguf> path\n\n" + USAGE, file=sys.stderr)
        return 2

    resolved = _resolve_model_path(gguf_path)
    if not Path(resolved).is_file():
        print(f"error: GGUF not found: {gguf_path}", file=sys.stderr)
        return 2

    try:
        handle = _lmo.absorb_gguf(
            resolved,
            out_root=out_root,
            force=force,
            attempt_libllama=attempt_libllama,
        )
    except _lmo.GgufParseError as e:
        print(f"error: GGUF parse failed: {e}", file=sys.stderr)
        return 2
    except _lmo.LmoError as e:
        print(f"error: LMO absorption failed (Stage A-VI gate): {e}", file=sys.stderr)
        return 2

    if as_json:
        r_ok = bool(handle.attest.get("retained_byte_identity_ok", False))
        d_ok = bool(handle.attest.get("determinism_self_parity_ok", False))
        h_ok = bool(handle.attest.get("header_roundtrip_ok", False))
        payload = {
            "lmo_dir": str(handle.lmo_dir),
            "model_sha256": handle.model_sha256,
            "lmo_anchor": [f"{w:016x}" for w in handle.header.lmo_anchor],
            "tile_count": handle.header.tile_count,
            "tile_plan_digest": handle.header.tile_plan_digest,
            "router_graph_digest": handle.header.router_graph_digest,
            "nrl_version": handle.header.nrl_version,
            "absorption_partial": bool(handle.attest.get("absorption_partial", False)),
            "retained_byte_identity_ok": r_ok,
            "determinism_self_parity_ok": d_ok,
            "header_roundtrip_ok": h_ok,
            "stage_a_vi_ok": bool(r_ok and d_ok and h_ok),
            "libllama_forward": handle.attest.get("libllama_forward", {}),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"LMO ready: {handle.lmo_dir}")
        print(f"  model_sha256         : {handle.model_sha256}")
        print(f"  tile_count           : {handle.header.tile_count}")
        print(f"  tile_plan_digest     : {handle.header.tile_plan_digest}")
        print(f"  router_graph_digest  : {handle.header.router_graph_digest}")
        print(
            "  lmo_anchor           : "
            + " ".join(f"{w:016x}" for w in handle.header.lmo_anchor)
        )
        print(f"  stage A-VI           : byte-identity + determinism + round-trip OK")
        llm = handle.attest.get("libllama_forward", {}) or {}
        print(f"  libllama_forward     : {llm.get('status', 'skipped')}")
    return 0


def _lmo_resolve_model_and_lmo_dir(raw: str) -> tuple[str, Path]:
    """Resolve GGUF path, LMO directory, or sha prefix to ``(model_sha256, lmo_dir)``."""
    from . import gguf
    from . import lmo as _lmo

    p = Path(raw)
    root = _lmo._default_lmo_root()
    if p.is_file() and str(p).lower().endswith(".gguf"):
        model_sha = gguf.sha256_file(p)
        lmo_dir = root / model_sha
        if not (lmo_dir / "lmo.header").is_file():
            raise FileNotFoundError(
                f"No LMO for this GGUF (expected {lmo_dir}); run: nrlpy absorb {raw}"
            )
        return model_sha, lmo_dir
    if p.is_dir() and (p / "lmo.header").is_file():
        h = _lmo.LmoHandle.open(p)
        return h.model_sha256, h.lmo_dir
    s = raw.strip().lower()
    if len(s) >= 8 and len(s) <= 64 and all(c in "0123456789abcdef" for c in s):
        hit: Path | None = None
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                name = child.name.lower()
                if name == s or name.startswith(s):
                    hit = child
                    break
        if hit is None:
            raise FileNotFoundError(f"no LMO under {root} matches prefix {raw!r}")
        h = _lmo.LmoHandle.open(hit)
        return h.model_sha256, h.lmo_dir
    raise FileNotFoundError(
        f"expected .gguf file, LMO directory, or hex sha prefix — got {raw!r}"
    )


def _lmo_cli(args: list[str]) -> int:
    """``nrlpy lmo <subcommand>`` — P4 LMO / persistence tooling."""
    if not args or args[0] in {"-h", "--help"}:
        print(
            "nrlpy lmo info <model.gguf|lmo_dir|sha-prefix> [--json]\n"
            "  Show ZPM + muscle-memory on-disk stats (WAL, snapshots).\n"
            "nrlpy lmo coverage <model.gguf|lmo_dir|sha-prefix> [--json]\n"
            "  P5 lattice coverage, weak buckets, growth headroom vs NRL_LEARN_MAX_GROWTH_PCT.\n"
            "nrlpy lmo prune <model.gguf|lmo_dir|sha-prefix> [--dry-run] [--aggressive] [--force] [--json]\n"
            "  P6 quota prune: LRU ZPM rows + snapshot trim; WAL truncated after snapshot.\n"
        )
        return 0
    if args[0] == "info":
        return _lmo_info_cli(args[1:])
    if args[0] == "coverage":
        return _lmo_coverage_cli(args[1:])
    if args[0] == "prune":
        return _lmo_prune_cli(args[1:])
    print(f"error: unknown lmo subcommand {args[0]!r}\n\n" + USAGE, file=sys.stderr)
    return 2


def _lmo_coverage_cli(args: list[str]) -> int:
    """P5 — Drift Conqueror coverage summary (read-only)."""
    from datetime import datetime

    from . import drift_conqueror as _dc
    from . import gguf
    from . import lmo as _lmo
    from . import zpm_persist as _zp

    as_json = False
    pos: list[str] = []
    for a in args:
        if a in {"-h", "--help"}:
            print(
                "nrlpy lmo coverage <model.gguf|lmo_dir|sha-prefix> [--json]\n"
                "  Requires an absorbed LMO when passing a .gguf path.\n"
            )
            return 0
        if a == "--json":
            as_json = True
        elif not a.startswith("-"):
            pos.append(a)
    if not pos:
        print(
            "error: nrlpy lmo coverage requires a target path or sha prefix\n\n" + USAGE,
            file=sys.stderr,
        )
        return 2
    raw = pos[0]
    try:
        model_sha, lmo_dir = _lmo_resolve_model_and_lmo_dir(raw)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except _lmo.LmoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    probe_dir = lmo_dir / "learn_probe"
    zpath = gguf._zpm_index_path(model_sha)
    _zp.recover_zpm_for_model(model_sha, zpath)
    summ = _dc.summarize_for_cli(probe_dir=probe_dir, zpm_index_path=zpath)
    if as_json:
        print(json.dumps(summ, indent=2))
        return 0
    lc = float(summ.get("last_conquest_unix", 0.0) or 0.0)
    if lc > 0.0:
        last_s = datetime.fromtimestamp(lc).strftime("%Y-%m-%d %H:%M:%S (local)")
    else:
        last_s = "never"
    prev = summ.get("weak_buckets_preview") or []
    prev_s = ", ".join(str(int(x)) for x in prev[:16])
    if len(prev) > 16:
        prev_s += ", …"
    cap = summ.get("max_growth_pct", _dc.max_growth_pct())
    print("NRL lattice coverage (Bio-Digital P5 — Drift Conqueror)")
    print(f"  model_sha256        {model_sha}")
    print(f"  lmo_dir               {lmo_dir}")
    print(f"  coverage_percent      {summ['coverage_percent']}% of anchor buckets touched")
    print(f"  weak_bucket_count     {summ['weak_bucket_count']}")
    print(f"  weak_buckets_preview  {prev_s or '(none)'}")
    print(f"  zpm_index_bytes       {summ['zpm_index_bytes']:,}")
    print(
        f"  growth_24h_window     used {summ['growth_used_pct']}% vs baseline "
        f"(headroom {summ['growth_headroom_pct']}%, cap {cap}%)"
    )
    print(f"  projected_growth_note headroom is remaining index.bin growth allowed this window")
    print(f"  last_conquest_run     {last_s}")
    print(f"  conquest_cycles       {summ.get('conquest_cycles', 0)}")
    return 0


def _lmo_prune_cli(args: list[str]) -> int:
    """P6 — bounded footprint prune (ZPM LRU + optional snapshot trim + WAL compact)."""
    from . import gguf
    from . import lmo as _lmo
    from . import lmo_disk_manager as _ldm
    from . import zpm_persist as _zp

    aggressive = False
    dry_run = False
    force = False
    as_json = False
    pos: list[str] = []
    for a in args:
        if a in {"-h", "--help"}:
            print(
                "nrlpy lmo prune <model.gguf|lmo_dir|sha-prefix> "
                "[--dry-run] [--aggressive] [--force] [--json]\n"
                "  --dry-run     Show planned eviction without writing.\n"
                "  --aggressive  Target 80% of quota headroom (default post-prune target 90%).\n"
                "  --force       Prune toward target even when under hard quota cap.\n"
            )
            return 0
        if a == "--aggressive":
            aggressive = True
        elif a == "--dry-run":
            dry_run = True
        elif a == "--force":
            force = True
        elif a == "--json":
            as_json = True
        elif not a.startswith("-"):
            pos.append(a)
    if not pos:
        print(
            "error: nrlpy lmo prune requires a target path or sha prefix\n\n" + USAGE,
            file=sys.stderr,
        )
        return 2
    raw = pos[0]
    try:
        model_sha, lmo_dir = _lmo_resolve_model_and_lmo_dir(raw)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except _lmo.LmoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    zpath = gguf._zpm_index_path(model_sha)
    _zp.recover_zpm_for_model(model_sha, zpath)
    mm_root = gguf._muscle_memory_root()
    mgr = _ldm.LmoDiskManager(
        model_sha256=model_sha,
        lmo_dir=lmo_dir,
        index_path=zpath,
        mm_root=mm_root,
    )
    result = mgr.prune_if_needed(aggressive=aggressive, dry_run=dry_run, force=force)
    if as_json:
        print(json.dumps(result, indent=2))
        return 0
    print("NRL LMO prune result (P6)")
    for k in sorted(result.keys()):
        print(f"  {k:28} {result[k]}")
    return 0


def _lmo_info_cli(args: list[str]) -> int:
    """Resolve a model or LMO directory and print persistence stats."""
    from . import gguf
    from . import lmo as _lmo
    from . import zpm_persist as _zp

    as_json = False
    pos: list[str] = []
    for a in args:
        if a in {"-h", "--help"}:
            print(
                "nrlpy lmo info <model.gguf|lmo_dir|sha-prefix> [--json]\n"
                "  sha-prefix: first 8+ hex chars of model_sha256 matching one\n"
                "  directory under $NRL_ROOT/cache/lmo/<sha>/\n"
            )
            return 0
        if a == "--json":
            as_json = True
        elif not a.startswith("-"):
            pos.append(a)
    if not pos:
        print("error: nrlpy lmo info requires a target path or sha prefix\n\n" + USAGE, file=sys.stderr)
        return 2
    raw = pos[0]
    p = Path(raw)
    model_sha = ""
    lmo_dir: Path | None = None
    root = _lmo._default_lmo_root()

    if p.is_file() and str(p).lower().endswith(".gguf"):
        model_sha = gguf.sha256_file(p)
    elif p.is_dir() and (p / "lmo.header").is_file():
        try:
            h = _lmo.LmoHandle.open(p)
            model_sha = h.model_sha256
            lmo_dir = h.lmo_dir
        except _lmo.LmoError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        s = raw.strip().lower()
        if len(s) >= 8 and len(s) <= 64 and all(c in "0123456789abcdef" for c in s):
            hit: Path | None = None
            if root.is_dir():
                for child in sorted(root.iterdir()):
                    if not child.is_dir():
                        continue
                    name = child.name.lower()
                    if name == s or name.startswith(s):
                        hit = child
                        break
            if hit is None:
                print(f"error: no LMO under {root} matches prefix {raw!r}", file=sys.stderr)
                return 2
            try:
                h = _lmo.LmoHandle.open(hit)
                model_sha = h.model_sha256
                lmo_dir = h.lmo_dir
            except _lmo.LmoError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        else:
            print(
                f"error: expected .gguf file, LMO directory, or hex sha prefix — got {raw!r}",
                file=sys.stderr,
            )
            return 2

    zpath = gguf._zpm_index_path(model_sha)
    _zp.recover_zpm_for_model(model_sha, zpath)
    mm_root = gguf._muscle_memory_root()
    if lmo_dir is None and model_sha:
        cand = root / model_sha
        if cand.is_dir() and (cand / "lmo.header").is_file():
            lmo_dir = cand
    info = _zp.gather_lmo_persistence_info(
        model_sha256=model_sha,
        index_path=zpath,
        mm_root=mm_root,
        lmo_dir=lmo_dir,
    )
    if lmo_dir is not None:
        info["lmo_dir"] = str(lmo_dir)
    if as_json:
        print(json.dumps(info, indent=2))
        return 0
    from datetime import datetime

    age = info.get("last_snapshot_age_s")
    age_s = f"{age:.1f}s ago" if isinstance(age, (int, float)) and age is not None else "never"
    lp = float(info.get("last_prune_unix", 0.0) or 0.0)
    if lp > 0.0:
        prune_s = datetime.fromtimestamp(lp).strftime("%Y-%m-%d %H:%M:%S (local)")
    else:
        prune_s = "never"
    fp_b = int(info.get("lmo_footprint_bytes", 0) or 0)
    qb_b = int(info.get("lmo_quota_bytes", 0) or 0)
    gr = info.get("footprint_growth_bytes_per_sec")
    gr_s = f"{float(gr):.4f} B/s" if isinstance(gr, (int, float)) and gr is not None else "n/a"
    print("NRL LMO persistence (P4 + P6 quota)")
    print(f"  model_sha256     {model_sha}")
    if lmo_dir:
        print(f"  lmo_dir          {lmo_dir}")
    print(f"  lmo_footprint    {fp_b / (1024**3):.3f} GiB (LMO+ZPM+MM)")
    print(f"  lmo_quota        {qb_b / (1024**3):.3f} GiB (NRL_LMO_MAX_GB)")
    print(f"  quota_used_pct   {100.0 * float(info.get('lmo_quota_used_ratio', 0.0) or 0.0):.2f}%")
    print(f"  growth_rate      {gr_s} (vs last footprint sample in learn_state)")
    print(f"  zpm_index        {info['zpm_index_path']}")
    print(f"  index_bytes      {info['zpm_index_bytes']:,}")
    print(f"  zpm_entries      {info['zpm_entry_count']:,}")
    print(f"  wal_bytes        {info['wal_bytes']:,}")
    print(f"  wal_pending_est  {info['wal_pending_lines_estimate']:,} lines")
    print(f"  snapshots        {info['snapshot_count']}")
    print(f"  last_snapshot    {age_s}")
    print(f"  muscle_memory    {info['muscle_memory_files']} files, {info['muscle_memory_bytes']:,} bytes")
    print(f"  learn_state      {info['learn_state_path']}")
    print(f"  last_prune       {prune_s}  (evicted {info.get('last_prune_evicted_entries', 0)}, freed_est {int(info.get('last_prune_freed_bytes', 0) or 0):,} B)")
    return 0


def _lmo_inspect_cli(args: list[str]) -> int:
    """``nrlpy lmo-inspect <lmo_dir> <token_id>`` — Phase 13 R0' data probe.

    Opens an absorbed LMO, resolves the requested vocab row to its
    backing packed INT4 tile(s), extracts the row's nibbles, and prints
    a small preview plus a determinism digest. Informational only:
    this does **not** serve inference, does not run the model, and
    makes no speed claims. It exists to prove the R0' data foundation
    is real and addressable.
    """
    from . import lmo as _lmo
    from .runtime import fnv1a64_packed

    lmo_dir: str | None = None
    token_id: int | None = None
    row_units = 3072  # Phi-3-mini hidden_dim default
    origin = "token_embd"
    preview = 16
    as_json = False

    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-h", "--help"}:
            print(
                "nrlpy lmo-inspect <lmo_dir> <token_id> "
                "[--row-units N] [--origin NAME] [--preview K] [--json]\n"
                "  Extract one vocab row from an absorbed LMO's packed INT4\n"
                "  tiles. Honest scope: returns the NRL packed projection,\n"
                "  not the dequantized Q4_K F16 weight.\n"
            )
            return 0
        if a == "--row-units" and i + 1 < len(args):
            try:
                row_units = int(args[i + 1])
            except ValueError:
                print(f"error: --row-units expects int, got {args[i+1]!r}", file=sys.stderr)
                return 2
            i += 2
            continue
        if a == "--origin" and i + 1 < len(args):
            origin = args[i + 1]
            i += 2
            continue
        if a == "--preview" and i + 1 < len(args):
            try:
                preview = int(args[i + 1])
            except ValueError:
                print(f"error: --preview expects int, got {args[i+1]!r}", file=sys.stderr)
                return 2
            i += 2
            continue
        if a == "--json":
            as_json = True
            i += 1
            continue
        if a.startswith("-"):
            print(f"error: unknown lmo-inspect flag {a!r}\n\n" + USAGE, file=sys.stderr)
            return 2
        if lmo_dir is None:
            lmo_dir = a
            i += 1
            continue
        if token_id is None:
            try:
                token_id = int(a)
            except ValueError:
                print(f"error: <token_id> must be int, got {a!r}", file=sys.stderr)
                return 2
            i += 1
            continue
        print(f"error: unexpected positional arg {a!r}\n\n" + USAGE, file=sys.stderr)
        return 2

    if lmo_dir is None or token_id is None:
        print(
            "error: nrlpy lmo-inspect requires <lmo_dir> <token_id>\n\n" + USAGE,
            file=sys.stderr,
        )
        return 2

    try:
        handle = _lmo.LmoHandle.open(lmo_dir)
    except _lmo.LmoError as e:
        print(f"error: cannot open LMO at {lmo_dir}: {e}", file=sys.stderr)
        return 2

    try:
        row = _lmo.embedding_row_packed(
            handle, token_id, row_units=row_units, origin=origin
        )
    except _lmo.LmoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    digest = fnv1a64_packed(row)
    preview_nibbles = row[: max(0, min(preview, len(row)))]

    if as_json:
        payload = {
            "lmo_dir": str(handle.lmo_dir),
            "model_sha256": handle.model_sha256,
            "origin": origin,
            "token_id": int(token_id),
            "row_units": int(row_units),
            "row_fnv1a64": f"{digest:016x}",
            "preview_nibbles_hex": preview_nibbles.hex(),
            "scope": "Phase 13 R0' packed projection (informational only; "
                     "NOT Q4_K-dequantized F16 embedding)",
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"LMO             : {handle.lmo_dir}")
        print(f"  model_sha256  : {handle.model_sha256}")
        print(f"  origin        : {origin}")
        print(f"  token_id      : {token_id}")
        print(f"  row_units     : {row_units}")
        print(f"  row_fnv1a64   : {digest:016x}")
        print(
            f"  preview ({len(preview_nibbles)}n)  : "
            + " ".join(f"{b:x}" for b in preview_nibbles)
        )
        print(
            "  scope         : Phase 13 R0' packed projection "
            "(informational only; NOT Q4_K F16)"
        )
    return 0


def _bench_wps_cli(args: list[str]) -> int:
    """``nrlpy bench-wps <model.gguf>`` — Phase 9-EG / 10-EG official WPS benchmark.

    Measures ``executed_wps`` / ``cache_wps`` / ``effective_wps`` across
    the five Final-Product scenarios and prints a clean summary table.
    The ``realistic_chat.effective_wps`` value is the Phase 10-EG release
    gate (>= 1000 words/sec).

    Flags:
      ``--turns N``       Turns per simple scenario (default 25).
      ``--chat-turns N``  Turns in the realistic-chat mix (default 100).
      ``--max-tokens N``  Max tokens per turn (default 32).
      ``--seed N``        Benchmark seed. Must be non-zero for
                          ``benchmark_class = A``. Default 1.
      ``--backend NAME``  ``native_full`` (default), ``native``, or
                          ``python``. Identical behavioral output across
                          all three; only hot-path speed differs.
      ``--json``          Print the full report as JSON to stdout instead
                          of the text banner.
      ``--json-out PATH`` Also write the full JSON report to ``PATH``.
    """
    from . import final_wps as _fwps

    model_path: str | None = None
    turns = 25
    chat_turns = _fwps.DEFAULT_CHAT_WORKLOAD_TURNS
    max_tokens = 32
    seed = 1
    backend = "native_full"
    as_json = False
    json_out: str | None = None

    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-h", "--help"}:
            print(
                "nrlpy bench-wps <model.gguf> "
                "[--turns N] [--chat-turns N] [--max-tokens N] "
                "[--seed N] [--backend native_full|native|python] "
                "[--json] [--json-out PATH]\n"
                "  Official Final-Product WPS benchmark (five scenarios).\n"
                "  Release gate: realistic_chat.effective_wps >= 1000.\n"
            )
            return 0
        if a == "--turns" and i + 1 < len(args):
            turns = max(1, int(args[i + 1]))
            i += 2
            continue
        if a == "--chat-turns" and i + 1 < len(args):
            chat_turns = max(1, int(args[i + 1]))
            i += 2
            continue
        if a == "--max-tokens" and i + 1 < len(args):
            max_tokens = max(1, int(args[i + 1]))
            i += 2
            continue
        if a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
            continue
        if a == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
            continue
        if a == "--json":
            as_json = True
            i += 1
            continue
        if a == "--json-out" and i + 1 < len(args):
            json_out = args[i + 1]
            i += 2
            continue
        if a.startswith("-"):
            print(f"error: unknown bench-wps flag {a!r}\n\n" + USAGE, file=sys.stderr)
            return 2
        if model_path is None:
            model_path = a
            i += 1
            continue
        print(f"error: unexpected positional arg {a!r}\n\n" + USAGE, file=sys.stderr)
        return 2

    if not model_path:
        print(
            "error: nrlpy bench-wps requires a <model.gguf> path\n\n" + USAGE,
            file=sys.stderr,
        )
        return 2
    if seed == 0:
        print(
            "error: bench-wps requires a non-zero --seed for benchmark_class=A",
            file=sys.stderr,
        )
        return 2

    resolved = _resolve_model_path(model_path)
    if not Path(resolved).is_file():
        print(f"error: GGUF not found: {model_path}", file=sys.stderr)
        return 2

    # Resolve NRL_ROOT — default to a dedicated bench cache under the
    # user's ``$NRL_ROOT`` so repeated runs reuse the prewarmed caches.
    nrl_root_env = os.environ.get("NRL_ROOT", "").strip()
    nrl_root = Path(nrl_root_env) if nrl_root_env else (Path.home() / ".nrl")
    nrl_root.mkdir(parents=True, exist_ok=True)

    try:
        report = _fwps.run_final_wps_benchmark(
            model_path=resolved,
            nrl_root=nrl_root,
            runner_backend=backend,
            seed=seed,
            max_tokens=max_tokens,
            turns_per_scenario=turns,
            realistic_chat_turns=chat_turns,
            benchmark_class="A",
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(_fwps.format_final_wps_report(report, as_json=as_json), end="")

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        if not as_json:
            print(f"\n[bench-wps] wrote JSON report -> {out_path}")

    # Exit code: 0 when the release gate passes, 1 when it doesn't so
    # CI can fail the run. This is the same contract as ``nrlpy wps-gate``.
    return 0 if report.passes_gate else 1


def _doctor_cli(args: list[str]) -> int:
    """Final MVP health check: root, disk, Python, native core, and safety flags."""
    import platform
    import shutil

    as_json = "--json" in args
    root = Path(os.environ.get("NRL_ROOT") or Path.cwd()).resolve()
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("python", sys.version_info >= (3, 9), platform.python_version())
    try:
        root.mkdir(parents=True, exist_ok=True)
        add("nrl_root", root.is_dir(), str(root))
    except OSError as exc:
        add("nrl_root", False, f"{root} ({exc})")

    try:
        usage = shutil.disk_usage(root)
        free_gb = usage.free / (1024**3)
        add("disk_free", free_gb >= 1.0, f"{free_gb:.2f} GiB free at {root.anchor or root}")
    except OSError as exc:
        add("disk_free", False, str(exc))

    try:
        ver = runtime.version()
        add("native_core", bool(ver), ver)
    except Exception as exc:  # pragma: no cover - depends on local extension availability
        add("native_core", False, f"{type(exc).__name__}: {exc}")

    try:
        from . import gguf, lmo, zpm  # noqa: F401

        add("python_modules", True, "gguf/lmo/zpm import OK")
    except Exception as exc:
        add("python_modules", False, f"{type(exc).__name__}: {exc}")

    safe = os.environ.get("NRL_SAFE_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
    add("safe_mode", True, "enabled: background learn/WAL/auto-prune disabled" if safe else "disabled")
    add("nrl_lmo_max_gb", True, os.environ.get("NRL_LMO_MAX_GB", "100"))

    healthy = all(bool(c["ok"]) for c in checks)
    payload = {
        "schema_id": "nrl.doctor.v1",
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("NRL doctor")
        print(f"  status: {payload['status']}")
        for c in checks:
            mark = "OK" if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
    return 0 if healthy else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    if args[0] == "wps-gate":
        from .wps_gate import main_wps_gate

        return main_wps_gate(args[1:])
    if args[0] == "wps-chat-bench":
        from .wps_chat_bench_cli import main_wps_chat_bench

        return main_wps_chat_bench(args[1:])
    if args[0] == "bench-wps":
        return _bench_wps_cli(args[1:])
    if args[0] == "doctor":
        return _doctor_cli(args[1:])
    if args[0] == "zpm":
        return _zpm_cli(args[1:])
    if args[0] == "absorb":
        return _absorb_cli(args[1:])
    if args[0] == "lmo":
        return _lmo_cli(args[1:])
    if args[0] == "lmo-inspect":
        return _lmo_inspect_cli(args[1:])
    if args[0] in {"nrl-ai", "nrlai"}:
        # NRL-AI pivot entry point -- pure-NRL-lattice inference path.
        # Scaffolded in prompt #1; stages land in prompts #2-#7.
        from . import nrl_ai

        return nrl_ai.dispatch(args[1:])
    if args and args[0] in ("-ai", "--ai"):
        from .consent_ai import lm_ai_cli_toggle

        if len(args) < 2:
            print("error: nrlpy -ai on|off\n\n" + USAGE, file=sys.stderr)
            return 2
        code = lm_ai_cli_toggle(args[1])
        if code != 0:
            print("error: nrlpy -ai expects on|off (or --on|--off|-on|-off)\n\n" + USAGE, file=sys.stderr)
        return code
    if args[0] == "--version":
        print(runtime.version())
        return 0
    if args[0] == "--features":
        print(json.dumps(runtime.features(), indent=2))
        return 0
    if args[0] == "variant" and len(args) >= 2:
        print(runtime.active_variant(args[1]))
        return 0
    if args[0] == "braincore4":
        neurons = int(args[1]) if len(args) >= 2 else 8_000_000
        iters = int(args[2]) if len(args) >= 3 else 1000
        threshold = int(args[3]) if len(args) >= 4 else 12
        print(
            json.dumps(
                runtime.braincore_int4(neurons=neurons, iterations=iters, threshold=threshold),
                indent=2,
            )
        )
        return 0
    if args[0] == "control":
        if len(args) < 2:
            print("error: nrlpy control status | nrlpy control audit tail [N]\n\n" + USAGE, file=sys.stderr)
            return 2
        if args[1] == "status":
            prefs = runtime.load_control_preferences()
            payload: dict[str, object] = {
                "control_preferences_path": str(runtime.control_preferences_path()),
                "control_audit_log_path": str(runtime.control_audit_log_path()),
                "preferences": prefs,
                "hints_active": runtime.control_hints_active(prefs),
            }
            print(json.dumps(payload, indent=2))
            return 0
        if args[1] == "audit":
            from .evidence import read_jsonl_tail

            if len(args) < 3 or args[2] != "tail":
                print("error: nrlpy control audit tail [N]\n\n" + USAGE, file=sys.stderr)
                return 2
            n = int(args[3]) if len(args) >= 4 else 20
            p = runtime.control_audit_log_path()
            if not p.is_file():
                print(
                    f"error: no control audit log at {p} (run `nrl control` first)\n\n" + USAGE,
                    file=sys.stderr,
                )
                return 2
            for line in read_jsonl_tail(p, n):
                print(line)
            return 0
        print(f"error: unknown control subcommand: {args[1]!r}\n\n" + USAGE, file=sys.stderr)
        return 2
    if args[0] == "evidence":
        from .evidence import read_jsonl_tail
        from .paths import first_existing_evidence_log

        if len(args) < 2 or args[1] != "tail":
            print("error: nrlpy evidence tail [N]\n\n" + USAGE, file=sys.stderr)
            return 2
        n = int(args[2]) if len(args) >= 3 else 20
        evidence_log = first_existing_evidence_log()
        if evidence_log is None:
            print(
                "error: no evidence log found (set NRL_EVIDENCE_LOG or "
                "create build/immune/events.jsonl)",
                file=sys.stderr,
            )
            return 2
        for line in read_jsonl_tail(evidence_log, n):
            print(line)
        return 0
    if args[0] == "learn":
        from .learn_store import LearnStore

        if len(args) < 2:
            print("error: nrlpy learn status | nrlpy learn cap BYTES\n\n" + USAGE, file=sys.stderr)
            return 2
        store = LearnStore()
        if args[1] == "status":
            print(store.stats().summary())
            return 0
        if args[1] == "cap":
            if len(args) < 3:
                print("error: nrlpy learn cap BYTES\n\n" + USAGE, file=sys.stderr)
                return 2
            store.set_max_bytes(int(args[2]))
            print(f"max_bytes set to {store.max_bytes:,}")
            return 0
        print(f"error: unknown learn subcommand: {args[1]!r}\n\n" + USAGE, file=sys.stderr)
        return 2
    if args[0] in ("adaptive-chat", "adaptive_chat"):
        from .adaptive_chat import main_adaptive_chat

        return main_adaptive_chat(args[1:])
    if args[0] in ("chat", "talk"):
        from .chat import main_chat

        return main_chat(args[1:])
    if args[0] == "demo":
        repo = Path(__file__).resolve().parents[3]
        demo = repo / "examples" / "ultimate_power_demo.py"
        if not demo.is_file():
            print(f"error: demo script not found at {demo}\n\n" + USAGE, file=sys.stderr)
            return 2
        run_path(str(demo), [])
        return 0
    if args[0] == "run":
        if len(args) < 2:
            print("error: nrlpy run requires a script path\n\n" + USAGE, file=sys.stderr)
            return 2
        script = args[1]
        if script.lower().endswith(".gguf"):
            try:
                model_path, kwargs = _parse_gguf_cli_args(args[1:])
            except ValueError as e:
                print(f"error: {e}\n\n{USAGE}", file=sys.stderr)
                return 2
            return _run_gguf_from_cli(model_path, kwargs)
        split_slot = _detect_split_gguf_slot(args, 2)
        if split_slot >= 0 and not Path(script).is_file():
            # Accept Windows/PowerShell unquoted paths:
            #   nrl run C:\RomaPy Engine\model.gguf
            model_path = " ".join(args[1 : split_slot + 1])
            try:
                _, kwargs = _parse_gguf_cli_args([model_path, *args[split_slot + 1 :]])
            except ValueError as e:
                print(f"error: {e}\n\n{USAGE}", file=sys.stderr)
                return 2
            return _run_gguf_from_cli(model_path, kwargs)
        extra: list[str] = []
        if "--" in args:
            i = args.index("--")
            extra = args[i + 1 :]
        else:
            extra = args[2:]
        run_path(script, extra)
        return 0
    if args[0] == "gguf":
        if len(args) < 2:
            print("error: nrlpy gguf requires a manifest path\n\n" + USAGE, file=sys.stderr)
            return 2
        return _run_gguf_manifest(args[1])
    if args[0] == "assimilate":
        neurons = int(args[1]) if len(args) >= 2 else 4096
        iters = int(args[2]) if len(args) >= 3 else 256
        threshold = int(args[3]) if len(args) >= 4 else 10
        print(json.dumps(runtime.assimilate_cli(neurons, iters, threshold), indent=2))
        return 0
    if args[0] == "bench":
        bargs = [a for a in args if a != "--respect-control-hints"]
        respect = len(bargs) != len(args)
        neurons = int(bargs[1]) if len(bargs) >= 2 else 1_048_576
        iters = int(bargs[2]) if len(bargs) >= 3 else 4096
        reps = int(bargs[3]) if len(bargs) >= 4 else 12
        threshold = int(bargs[4]) if len(bargs) >= 5 else 8
        profile = bargs[5] if len(bargs) >= 6 else "sovereign"
        print(
            json.dumps(
                runtime.bench_cli(
                    neurons=neurons,
                    iterations=iters,
                    reps=reps,
                    threshold=threshold,
                    profile=profile,
                    respect_control_hints=respect,
                ),
                indent=2,
            )
        )
        return 0
    if args[0].endswith(".nrl"):
        print(runtime.run_nrl_file(args[0]), end="")
        return 0

    if args[0].lower().endswith(".gguf"):
        try:
            model_path, kwargs = _parse_gguf_cli_args(args)
        except ValueError as e:
            print(f"error: {e}\n\n{USAGE}", file=sys.stderr)
            return 2
        return _run_gguf_from_cli(model_path, kwargs)

    if args[0].endswith(".py"):
        py_script = Path(args[0])
        if not py_script.is_file():
            print(f"error: Python file not found: {args[0]}\n\n{USAGE}", file=sys.stderr)
            return 2
        py_extra: list[str] = []
        if "--" in args:
            i = args.index("--")
            py_extra = args[i + 1 :]
        else:
            py_extra = args[1:]
        run_path(str(py_script.resolve()), py_extra)
        return 0

    print(f"error: unknown args: {' '.join(args)}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
