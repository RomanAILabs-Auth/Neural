# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Phase 15-EG — Adaptive high-speed R&D chat REPL. Wraps :func:`nrlpy.gguf_chat.chat_turn`
# with a sidecar streak on consecutive R5 (novel decode) turns and optional
# :func:`nrlpy.zpm.prime` before the next decode when the streak exceeds 3.
"""Adaptive GGUF chat REPL (Phase 15-EG drift recovery).

Entry points:

* :func:`run_adaptive_chat_repl` — interactive loop (``native_full`` + ``phi3``).
* :func:`main_adaptive_chat` — CLI argv handler (``nrlpy adaptive-chat ...``).

Does not modify :mod:`nrlpy.gguf` internals; composes :func:`nrlpy.gguf_chat.chat_turn`
and :func:`nrlpy.zpm.prime` with the same per-turn manifest contract as
``gguf_chat._per_turn_manifest``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import IO, Any

from . import gguf
from . import zpm
from .gguf import GgufManifest, GgufRunResult, load_manifest, manifest_from_args
from .gguf_chat import (
    ChatSession,
    _ansi_supported,
    _boot_banner,
    _format_turn_footer,
    _handle_slash,
    _read_line,
    _reply_label,
    _turn_header,
    build_session,
    chat_turn,
    format_session_banner,
)
from .gguf_chat import _per_turn_manifest  # noqa: PLC2701 — stable chat contract

_ADAPTIVE_USAGE = (
    "usage: nrlpy adaptive-chat <model.gguf|manifest.nrl> "
    "[--system \"...\"] [--seed N] [--max-tokens N] [--temperature F] "
    "[--fast-chat|--rewired] [--response-recall] [--lmo-sha PREFIX]\n"
    "  Backend is always native_full; chat_format is always phi3.\n"
    "  (Use ``nrlpy chat`` for other backends or formats.)"
)


def _tail_token_ids(preloaded_llm: Any) -> list[int]:
    ids = getattr(preloaded_llm, "_input_ids", None) or ()
    return [int(x) for x in ids[-4:]]


def _format_adaptive_turn_footer(
    turn_idx: int,
    tokens: int,
    wall_s: float,
    result: GgufRunResult,
    use_color: bool,
) -> str:
    base = _format_turn_footer(turn_idx, tokens, wall_s, result, use_color).rstrip("\n")
    drift = int(getattr(result, "drift_reprime_count", 0))
    return f"{base} . drift_reprime_count={drift}\n"


def _maybe_phase15_sidecar_prime(
    *,
    session: ChatSession,
    user_text: str,
    preloaded_llm: Any,
    consecutive_r5_turns: int,
) -> None:
    """When the sidecar streak exceeds 3, re-hydrate ZPM index RAM before ``chat_turn``."""
    if consecutive_r5_turns <= 3:
        return
    model_sha = session.model_sha256 or getattr(session.base_manifest, "model_sha256", "")
    if not model_sha:
        return
    per_next = _per_turn_manifest(session, user_text)
    intent_anchor = gguf._zpm_anchor_bytes(per_next, per_next.prompt)
    tail = _tail_token_ids(preloaded_llm)
    try:
        zpm.prime(
            model_sha,
            intent_anchor,
            tail,
            gguf._zpm_index_path(model_sha),
        )
    except Exception:
        pass


def run_adaptive_chat_repl(
    manifest: GgufManifest,
    *,
    system: str = "",
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    fast_chat: bool = False,
    rewired: bool = False,
    lmo_sha_banner: str = "",
) -> ChatSession:
    """Like :func:`nrlpy.gguf_chat.run_gguf_chat_repl` with Phase 15-EG sidecar + footer."""
    from . import gguf as _gguf_mod  # noqa: PLC0415

    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    session = build_session(manifest, system=system)
    preloaded_llm: Any = None

    model_path = Path(manifest.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {manifest.model}")
    actual_sha = _gguf_mod.sha256_file(model_path)
    if manifest.model_sha256 and manifest.model_sha256 != actual_sha:
        raise RuntimeError(
            f"model_sha256 mismatch: manifest={manifest.model_sha256} actual={actual_sha}"
        )
    manifest.model_sha256 = actual_sha
    session.model_sha256 = actual_sha
    preloaded_llm = _gguf_mod._load_llm(manifest)

    session.last_attestation = _gguf_mod.NrlAttestation(profile="", available=False)
    session.last_observation = _gguf_mod.NrlLatticeObservation()

    prewarm_summary = ""
    if fast_chat and not rewired:
        from . import chat_prewarm  # noqa: PLC0415

        pw = chat_prewarm.prewarm_chat_cache(manifest, system=system)
        prewarm_summary = pw.summary_line()

    use_color = _ansi_supported(sout)
    sout.write(
        _boot_banner(
            model_name=Path(manifest.model).name,
            chat_format=manifest.chat_format,
            seed=manifest.seed,
            zpm_on=bool(getattr(manifest, "zpm_nullspace", False)),
            mm_on=(manifest.muscle_memory == "on"),
            use_color=use_color,
            runner_backend=str(getattr(manifest, "runner_backend", "native_full")),
            fast_chat=fast_chat,
            prewarm_summary=prewarm_summary,
            rewired=rewired,
            zpm_threshold_bits=int(getattr(manifest, "zpm_threshold_bits", 0)),
            coherence_lane=str(getattr(manifest, "coherence_lane", "fast-stable")),
        )
    )
    dim_on = "\x1b[2m" if use_color else ""
    reset = "\x1b[0m" if use_color else ""
    sout.write(
        f"  {dim_on}Phase 15-EG Adaptive REPL: after >3 consecutive R5 turns, "
        f"zpm.prime() runs before the next decode (memory-only ZPM index refresh).{reset}\n"
    )
    sout.write(
        f"  {dim_on}Bench note: zpm_exact lattice ~10.6k WPS; realistic_chat novel "
        f"decode is often ~2 WPS — lattice hits show high effective_wps.{reset}\n"
    )
    if lmo_sha_banner.strip():
        sout.write(
            f"  {dim_on}LMO compatibility hint (informational): {lmo_sha_banner.strip()}{reset}\n"
        )
    sout.flush()

    consecutive_r5_turns = 0

    try:
        while True:
            turn_idx = session.turn_count()
            line = _read_line(_turn_header(turn_idx, use_color), sin, sout)
            if line is None:
                break
            stripped = line.strip()
            if not stripped:
                continue
            verdict = _handle_slash(stripped, session, sout)
            if verdict == "quit":
                break
            if verdict == "continue":
                low = stripped.lower()
                if low.startswith("/clear") or low.startswith("/system") or low.startswith("/load"):
                    consecutive_r5_turns = 0
                continue

            _maybe_phase15_sidecar_prime(
                session=session,
                user_text=stripped,
                preloaded_llm=preloaded_llm,
                consecutive_r5_turns=consecutive_r5_turns,
            )
            if consecutive_r5_turns > 3:
                consecutive_r5_turns = 0

            sout.write(_reply_label(use_color))
            sout.flush()
            t0 = time.perf_counter()
            try:
                result = chat_turn(
                    session,
                    stripped,
                    stream_to=sout,
                    preloaded_llm=preloaded_llm,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                sout.write(
                    f"\nerror: {type(e).__name__}: {e}\n"
                    f"       (turn skipped; session continues. /exit to leave.)\n"
                )
                sout.flush()
                continue
            wall = time.perf_counter() - t0
            if not result.cache_hit:
                consecutive_r5_turns += 1
            else:
                consecutive_r5_turns = 0

            sout.write(
                _format_adaptive_turn_footer(turn_idx, result.tokens, wall, result, use_color)
            )
            sout.flush()
    except KeyboardInterrupt:
        sout.write("\nSession ended. Goodbye.\n")

    sout.write("\n" + format_session_banner(session))
    sout.flush()
    return session


def main_adaptive_chat(args: list[str]) -> int:
    """``nrlpy adaptive-chat <model> [...]`` — forces ``native_full`` and ``phi3``."""
    from . import gguf_chat as _gc  # noqa: PLC0415
    from .native_ladder import is_full_native_available  # noqa: PLC0415

    if not is_full_native_available():
        print(
            "error: adaptive-chat requires the native_full C hot path "
            "(Phase 8-EG). Build the extension or use ``nrlpy chat``.\n",
            file=sys.stderr,
        )
        return 2

    if not args or args[0] in {"-h", "--help"}:
        print(_ADAPTIVE_USAGE)
        return 0 if args else 2

    target = args[0]
    rest = args[1:]
    system = ""
    extra_kwargs: dict[str, Any] = {}
    fast_chat = False
    rewired = False
    response_recall = False
    user_overrides: set[str] = set()
    lmo_sha_banner = ""

    forbidden = {
        "--chat-format",
        "--native-full",
        "--native-full-strict",
        "--native",
        "--native-strict",
        "--python-ladder",
    }

    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag in forbidden:
            print(
                f"error: {flag} is not supported on adaptive-chat "
                f"(backend and format are fixed). Use ``nrlpy chat`` instead.\n",
                file=sys.stderr,
            )
            print(_ADAPTIVE_USAGE, file=sys.stderr)
            return 2
        if flag == "--system" and i + 1 < len(rest):
            system = rest[i + 1]
            i += 2
        elif flag == "--seed" and i + 1 < len(rest):
            try:
                extra_kwargs["seed"] = int(rest[i + 1])
            except ValueError:
                print(f"error: --seed expects int, got {rest[i + 1]!r}", file=sys.stderr)
                return 2
            user_overrides.add("seed")
            i += 2
        elif flag == "--max-tokens" and i + 1 < len(rest):
            try:
                extra_kwargs["max_tokens"] = int(rest[i + 1])
            except ValueError:
                print(
                    f"error: --max-tokens expects int, got {rest[i + 1]!r}",
                    file=sys.stderr,
                )
                return 2
            user_overrides.add("max_tokens")
            i += 2
        elif flag == "--temperature" and i + 1 < len(rest):
            try:
                extra_kwargs["temperature"] = float(rest[i + 1])
            except ValueError:
                print(
                    f"error: --temperature expects float, got {rest[i + 1]!r}",
                    file=sys.stderr,
                )
                return 2
            user_overrides.add("temperature")
            i += 2
        elif flag == "--fast-chat":
            fast_chat = True
            i += 1
        elif flag == "--rewired":
            rewired = True
            i += 1
        elif flag == "--response-recall":
            response_recall = True
            i += 1
        elif flag == "--lmo-sha" and i + 1 < len(rest):
            lmo_sha_banner = rest[i + 1]
            i += 2
        elif flag in {"-h", "--help"}:
            print(_ADAPTIVE_USAGE)
            return 0
        else:
            print(f"error: unknown adaptive-chat flag {flag!r}", file=sys.stderr)
            print(_ADAPTIVE_USAGE, file=sys.stderr)
            return 2

    if rewired:
        _gc._apply_rewired_defaults(extra_kwargs, user_overrides)
    elif fast_chat:
        _gc._apply_fast_chat_defaults(extra_kwargs, user_overrides)

    extra_kwargs["runner_backend"] = "native_full"
    user_overrides.add("runner_backend")
    extra_kwargs["chat_format"] = "phi3"
    user_overrides.add("chat_format")

    try:
        if target.lower().endswith(".nrl"):
            manifest = load_manifest(target)
            for k, v in extra_kwargs.items():
                setattr(manifest, k, v)
        else:
            manifest = manifest_from_args(model=target, **extra_kwargs)
    except gguf.ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    manifest.runner_backend = "native_full"
    manifest.chat_format = "phi3"

    if rewired:
        _gc.apply_rewired_post_build(manifest, user_overrides)
    elif fast_chat:
        _gc.apply_fast_chat_post_build(manifest, user_overrides)
    if response_recall:
        _gc.apply_response_recall(manifest, user_overrides)

    manifest.prompt = ""
    manifest.prompt_file = ""

    try:
        run_adaptive_chat_repl(
            manifest,
            system=system,
            fast_chat=fast_chat and not rewired,
            rewired=rewired,
            lmo_sha_banner=lmo_sha_banner,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
