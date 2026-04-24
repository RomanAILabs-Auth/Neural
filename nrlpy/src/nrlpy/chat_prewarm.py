# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Fast-chat cache prewarmer.

Given a ``ChatSession`` that is about to run its first turn, this module
writes a small bundle of ``(rendered_prompt, reply)`` entries into the
on-disk Muscle-Memory cache (R0) and the ZPM index (R1). The entries
are computed with the exact same keying functions the live Resolution
Ladder uses, so the first time the user types one of these common
prompts the native hot path flags a byte-identical cache hit and serves
the reply at lattice speed.

This is what makes ``nrlpy chat --fast-chat`` feel like the ``realistic_chat``
benchmark: the first few turns of a normal conversation collapse to R0 /
R1, giving the user a direct, tactile experience of the 1000+
effective_wps release gate rather than making them wait for the cache
to fill up organically.

Strict guarantees preserved:

* No new runtime behavior beyond writing cache files that the existing
  ladder already knows how to read. If ``muscle_memory = "off"`` or the
  NRL cache root is read-only the prewarm silently skips.
* Deterministic: the same ``(manifest, seeds)`` always produces the same
  on-disk layout.
* Honest accounting: MM/ZPM hits on prewarmed entries show up in
  ``TpsReport`` as ``cache_tokens`` (not ``executed_tokens``), which is
  the correct four-metric labelling.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import gguf, zpm
from .gguf import (
    MUSCLE_MEMORY_MAGIC,
    GgufManifest,
)

__all__ = [
    "FAST_CHAT_SEEDS",
    "PrewarmSeed",
    "PrewarmResult",
    "prewarm_chat_cache",
]


# --------------------------------------------------------------------------- #
# Seed corpus — the 20 most common first-turn utterances.
# --------------------------------------------------------------------------- #
# Replies are intentionally short + generic so they're a reasonable answer
# for any instruct-tuned GGUF (Phi-3, Llama-3, Mistral, Qwen). They are a
# cache win, not a ghostwritten persona; if a user prefers a different
# canonical reply they just clear the cache by overwriting a turn.


@dataclass(frozen=True)
class PrewarmSeed:
    """One ``(prompt, reply)`` pair to prewarm into R0 + R1."""

    prompt: str
    reply: str


FAST_CHAT_SEEDS: tuple[PrewarmSeed, ...] = (
    PrewarmSeed(
        "hi",
        "Hi! I'm running on the NRL Final Product lattice runner. "
        "Ask me anything; repeated questions will serve from cache at "
        "lattice speed.",
    ),
    PrewarmSeed(
        "hello",
        "Hello! I'm running on the NRL Final Product lattice runner. "
        "Ask me anything; repeated questions will serve from cache at "
        "lattice speed.",
    ),
    PrewarmSeed(
        "hey",
        "Hey! I'm ready. This chat is powered by the NRL native hot "
        "path -- muscle memory + ZPM nullspace serve cache hits in "
        "microseconds.",
    ),
    PrewarmSeed(
        "who are you",
        "I'm a chat front-end for a GGUF model running on the NRL "
        "Final Product architecture. The Resolution Ladder (R0 muscle "
        "memory, R1 ZPM, R2 Omega, R5 libllama) decides how each turn "
        "is served. Most repeat questions land on R0 or R1.",
    ),
    PrewarmSeed(
        "what are you",
        "I'm a local chat session. The model weights are a standard "
        "GGUF; the runner around them is NRL (Neural Rewriter Layer) "
        "which caches turns so repeats serve at lattice speed.",
    ),
    PrewarmSeed(
        "what is nrl",
        "NRL is a CPU-first runtime for GGUF models. It caches turns "
        "in a Muscle-Memory store (R0) and a ZPM anchor index (R1) so "
        "repeated questions skip the decoder entirely and serve at "
        "memory-I/O speed.",
    ),
    PrewarmSeed(
        "what can you do",
        "I can answer questions, explain concepts, draft short pieces "
        "of text, and reason step-by-step at the level of the "
        "underlying GGUF model. The NRL runner makes repeats fast.",
    ),
    PrewarmSeed(
        "help",
        "Type anything to chat. Slash commands: /help /exit /clear "
        "/stats /system /seed /save /load /history. Repeat any "
        "question and watch the footer say [ZPM hit] or [Muscle "
        "Memory] instead of [Decode].",
    ),
    PrewarmSeed(
        "how are you",
        "I'm fine. I don't have feelings, but the runner is healthy: "
        "native hot path active, muscle memory and ZPM both warm, "
        "ready for your question.",
    ),
    PrewarmSeed(
        "thanks",
        "You're welcome.",
    ),
    PrewarmSeed(
        "thank you",
        "You're welcome!",
    ),
    PrewarmSeed(
        "ok",
        "Okay -- what would you like to do next?",
    ),
    PrewarmSeed(
        "yes",
        "Got it. What would you like next?",
    ),
    PrewarmSeed(
        "no",
        "Understood. What would you prefer instead?",
    ),
    PrewarmSeed(
        "test",
        "Test received. This reply was served from the NRL cache, "
        "so it printed in microseconds rather than seconds.",
    ),
    PrewarmSeed(
        "ping",
        "pong -- served from the NRL muscle-memory cache.",
    ),
    PrewarmSeed(
        "what is your name",
        "I don't have a personal name. I'm a GGUF chat session "
        "running on the NRL Final Product runner. If you'd like to "
        "give me a nickname, just use `/system`.",
    ),
    PrewarmSeed(
        "how fast are you",
        "Cache-hit turns serve in microseconds (thousands of effective "
        "words per second). Novel turns run at the underlying GGUF's "
        "native decode speed. The 1000+ effective WPS gate is "
        "measured on a realistic chat mix via `nrlpy bench-wps`.",
    ),
    PrewarmSeed(
        "what is the time",
        "I don't have access to the system clock from this chat "
        "session. Check your shell (e.g. `Get-Date` in PowerShell).",
    ),
    PrewarmSeed(
        "are you an ai",
        "I'm a chat front-end for an instruct-tuned GGUF language "
        "model running through the NRL lattice runner. The model does "
        "the language work; NRL handles routing and caching.",
    ),
)


# --------------------------------------------------------------------------- #
# Rendering + on-disk writes (must match the live ladder byte-for-byte).
# --------------------------------------------------------------------------- #


def _render_first_turn(
    manifest: GgufManifest, *, system: str, user: str
) -> str:
    """Render a single first-turn prompt under ``manifest.chat_format``.

    Deferred import of :mod:`nrlpy.gguf_chat` avoids a circular import
    (``gguf_chat`` imports this module when ``--fast-chat`` is set).
    """
    from . import gguf_chat  # noqa: PLC0415

    session = gguf_chat.ChatSession(base_manifest=manifest, system=system)
    return gguf_chat.build_history_prompt(session, user)


def _per_turn_manifest(
    manifest: GgufManifest, *, rendered_prompt: str
) -> GgufManifest:
    """Clone ``manifest`` with ``prompt = rendered`` and ``chat_format = "none"``.

    Mirrors ``gguf_chat._per_turn_manifest`` exactly so the MM key and
    ZPM anchor come out byte-identical on prewarm + live dispatch.
    """
    return dataclasses.replace(
        manifest,
        prompt=rendered_prompt,
        prompt_file="",
        chat_format="none",
    )


def _write_mm(manifest: GgufManifest, reply: str, tokens: int) -> Path | None:
    """Write a Muscle-Memory entry for the current manifest, or return
    ``None`` if MM is disabled. Uses the live keying + path functions
    so the native C R0 probe flags a byte-identical hit.
    """
    if manifest.muscle_memory == "off":
        return None
    key = gguf._muscle_memory_key(manifest)
    path = gguf._muscle_memory_path(manifest, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = reply.encode("utf-8")
        header = MUSCLE_MEMORY_MAGIC + struct.pack("<II", tokens, len(body))
        path.write_bytes(header + body)
    except OSError:
        return None
    return path


def _write_zpm(manifest: GgufManifest, reply: str, tokens: int) -> Path | None:
    """Append a ZPM entry for the current manifest, accumulating any
    existing ``index.bin`` rather than clobbering it. Returns ``None``
    if ZPM nullspace is disabled or the write fails.
    """
    if not bool(getattr(manifest, "zpm_nullspace", False)):
        return None
    try:
        from . import zpm_persist as zpm_persist  # noqa: PLC0415

        state = zpm.anchor(gguf._zpm_anchor_bytes(manifest, manifest.prompt))
        path = gguf._zpm_index_path(manifest.model_sha256 or "unknown")
        path.parent.mkdir(parents=True, exist_ok=True)
        zpm_persist.recover_zpm_for_model(manifest.model_sha256 or "unknown", path)
        if path.is_file():
            try:
                idx = zpm.ZpmIndex.load(path)
            except Exception:  # noqa: BLE001 - corrupt index => rewrite
                idx = zpm.ZpmIndex()
        else:
            idx = zpm.ZpmIndex()
        # Skip if we already have an exact anchor for this state (keeps
        # the index small on repeated prewarms).
        existing_hit, existing_entry = idx.lookup(state, threshold_bits=0)
        if existing_entry is None or not existing_hit.exact:
            ent = zpm.ZpmEntry(
                state=state,
                reply_text=reply,
                tokens=tokens,
                metadata={"src": "chat_prewarm"},
            )
            idx.add(ent)
            zpm_persist.persist_zpm_entry(manifest.model_sha256 or "unknown", path, idx, ent)
    except OSError:
        return None
    return path


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class PrewarmResult:
    """Summary of what actually made it onto disk."""

    seeds_attempted: int = 0
    mm_written: int = 0
    zpm_written: int = 0
    skipped: int = 0

    def summary_line(self) -> str:
        return (
            f"prewarm: {self.mm_written}/{self.seeds_attempted} R0 . "
            f"{self.zpm_written}/{self.seeds_attempted} R1 . "
            f"skipped={self.skipped}"
        )


def prewarm_chat_cache(
    manifest: GgufManifest,
    *,
    system: str = "",
    seeds: Sequence[PrewarmSeed] | None = None,
    stream: IO[str] | None = None,
) -> PrewarmResult:
    """Write MM + ZPM entries for every seed in ``seeds``.

    Parameters
    ----------
    manifest:
        The *live* manifest — must have ``model_sha256`` already set
        (via ``gguf.sha256_file`` at REPL boot) so the cache paths
        resolve correctly.
    system:
        The system prompt that will be in effect for the first turn.
        Included in the rendered template so the key matches.
    seeds:
        Optional override. Defaults to :data:`FAST_CHAT_SEEDS`.
    stream:
        Optional output stream for a one-line status message.

    Returns
    -------
    PrewarmResult
        Counts of what was written and what was skipped. Never raises;
        any I/O failures are silently counted as skips because prewarm
        is a best-effort accelerator, not a correctness requirement.
    """
    chosen: Iterable[PrewarmSeed] = seeds if seeds is not None else FAST_CHAT_SEEDS
    result = PrewarmResult()
    if not manifest.model_sha256:
        # Can't key without a model SHA. Caller forgot to preload.
        if stream is not None:
            stream.write(
                "prewarm: skipped (model_sha256 not set; preload the model first)\n"
            )
        return result
    for seed in chosen:
        result.seeds_attempted += 1
        try:
            rendered = _render_first_turn(manifest, system=system, user=seed.prompt)
            per_turn = _per_turn_manifest(manifest, rendered_prompt=rendered)
            reply_tokens = max(1, len(seed.reply.split()))
            mm_path = _write_mm(per_turn, seed.reply, reply_tokens)
            zp_path = _write_zpm(per_turn, seed.reply, reply_tokens)
            if mm_path is not None:
                result.mm_written += 1
            if zp_path is not None:
                result.zpm_written += 1
            if mm_path is None and zp_path is None:
                result.skipped += 1
        except Exception:  # noqa: BLE001 - best-effort accelerator
            result.skipped += 1
            continue
    if stream is not None:
        stream.write(result.summary_line() + "\n")
    return result
