# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI chat REPL — Neural AI surface routed through the NRL-AI fast lane.

Stage 5 of the NRL-AI rebuild. No libllama, no decode path. Every user
turn flows through:

  1. ``simhash256`` anchor of the raw user utterance
  2. Hamming nullspace scan over the on-disk anchor table (resolver)
  3. HIT  -> Omega-routed fragment walk, streamed sentence-by-sentence
     MISS -> honest "not in threshold" surface (no synthesis)

The banner, per-turn header/label, and footer are the RomanAILabs
polish carried over from ``gguf_chat`` but with a ``[NRL-AI]`` badge so
users can see at a glance that this turn was served entirely by the
lattice (no cold CPU decode).

Streaming: each composed fragment is written with an explicit flush so
the reply appears progressively even at high aggregate WPS; the
``chunk_delay_s`` knob lets bench runs disable the pause.

Session telemetry: cumulative words / wall-time / wps / hits / misses
emitted at ``/stats`` and on exit. That number is the honest measure
of the native NRL-AI throughput and is what prompt #6's bench gates
against.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from . import nrl_ai_compose as _compose
from . import nrl_ai_ingest as _ingest
from . import nrl_ai_resolve as _resolve
from .nrl_ai import NRL_AI_VERSION, NRL_AI_WPS_TARGET, NrlAiPaths

_BANNER_INNER = 62

_REPL_HELP = (
    "NRL-AI chat commands:\n"
    "  /help           show this help\n"
    "  /quit /exit     leave the REPL\n"
    "  /stats          cumulative session WPS and hit ratio\n"
    "  /reset          zero out session counters (index stays loaded)\n"
    "  /status         print index path + fragment count + threshold\n"
)


# ---------------------------------------------------------------------------
# ANSI / banner helpers
# ---------------------------------------------------------------------------


def _ansi_supported(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("NRL_NO_COLOR", "").strip():
        return False
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        return False


def _pad_line(
    raw: str, left_ansi: str, right_ansi: str, reset: str, border_ansi: str
) -> str:
    pad = max(0, _BANNER_INNER - len(raw) - 2)
    return (
        f"{border_ansi}|{reset} "
        f"{left_ansi}{raw}{right_ansi}{' ' * pad} "
        f"{border_ansi}|{reset}"
    )


def _short_path(p: Path) -> str:
    s = str(p)
    max_len = _BANNER_INNER - 16
    if len(s) <= max_len:
        return s
    return "..." + s[-(max_len - 3) :]


def _boot_banner(
    paths: NrlAiPaths,
    fragment_count: int,
    threshold_bits: int,
    use_color: bool,
) -> str:
    C = "\x1b[36m" if use_color else ""
    M = "\x1b[35m" if use_color else ""
    G = "\x1b[32m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    R = "\x1b[0m" if use_color else ""

    rule = f"{C}+{'-' * _BANNER_INNER}+{R}"
    rows = [
        rule,
        _pad_line("Neural AI  .  RomanAILabs  .  NRL-AI native runner", M, R, R, C),
        _pad_line(
            "pure NRL lattice . SimHash + ZPM + Omega . no decode", DIM, R, R, C
        ),
        rule,
        _pad_line(
            f"fast_lane    [NRL-AI]  (target >= {NRL_AI_WPS_TARGET} wps)", G, R, R, C
        ),
        _pad_line(f"version      {NRL_AI_VERSION}", "", "", "", C),
        _pad_line(f"index        {_short_path(paths.root)}", "", "", "", C),
        _pad_line(f"fragments    {fragment_count}", "", "", "", C),
        _pad_line(f"threshold    {threshold_bits} bits (Hamming)", "", "", "", C),
        rule,
        f"  {DIM}tip: '/help' for commands, '/stats' for WPS, '/quit' to leave{R}",
        "",
    ]
    return "\n".join(rows) + "\n"


def _turn_header(turn_idx: int, use_color: bool) -> str:
    DIM = "\x1b[2m" if use_color else ""
    R = "\x1b[0m" if use_color else ""
    return f"{DIM}[you, turn {turn_idx}]{R}\n> "


def _reply_label(use_color: bool) -> str:
    M = "\x1b[35m" if use_color else ""
    R = "\x1b[0m" if use_color else ""
    return f"\n{M}[NRL-AI]{R}\n"


def _format_turn_footer(
    turn_idx: int,
    cresult: Optional[_compose.ComposeResult],
    distance_bits: Optional[int],
    words: int,
    wall_s: float,
    use_color: bool,
) -> str:
    DIM = "\x1b[2m" if use_color else ""
    G = "\x1b[32m" if use_color else ""
    Y = "\x1b[33m" if use_color else ""
    R = "\x1b[0m" if use_color else ""

    wps = (words / wall_s) if wall_s > 0 else 0.0
    turn_tag = f"{DIM}turn {turn_idx:3d}{R}"
    if cresult is None:
        badge = f"{Y}[resolve miss]{R}"
        parts = [
            turn_tag,
            badge,
            f"{words} words",
            f"{wall_s * 1000:6.1f} ms",
        ]
    else:
        dist_txt = f"{distance_bits}b" if distance_bits is not None else "?"
        badge = f"{G}[NRL-AI hit . {dist_txt} dist]{R}"
        parts = [
            turn_tag,
            badge,
            f"{words} words",
            f"{wall_s * 1000:6.1f} ms",
            f"{wps:,.0f} wps",
            f"stop={cresult.stop_reason}",
        ]
    return " . ".join(parts)


# ---------------------------------------------------------------------------
# Session state + reply streaming
# ---------------------------------------------------------------------------


@dataclass
class SessionStats:
    """Cumulative per-session counters for the NRL-AI REPL."""

    turns: int = 0
    hits: int = 0
    misses: int = 0
    total_words: int = 0
    total_wall_s: float = 0.0

    @property
    def wps(self) -> float:
        return self.total_words / self.total_wall_s if self.total_wall_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"turns={self.turns} hits={self.hits} misses={self.misses} "
            f"words={self.total_words} wall_s={self.total_wall_s:.3f} "
            f"cumulative_wps={self.wps:,.0f}"
        )


def _stream_reply(
    cresult: _compose.ComposeResult,
    out: IO[str],
    *,
    chunk_delay_s: float,
) -> int:
    """Write the composed fragments to ``out`` with flush. Returns word count."""
    words = 0
    for i, step in enumerate(cresult.steps):
        if i > 0:
            out.write(" ")
        out.write(step.fragment_text)
        out.flush()
        words += len(step.fragment_text.split())
        if chunk_delay_s > 0:
            time.sleep(chunk_delay_s)
    out.write("\n")
    out.flush()
    return words


def _miss_text(rresult: _resolve.ResolveResult) -> str:
    best = rresult.best
    if best is None:
        return "I don't have anything indexed for that yet."
    return (
        f"I'm not confident on that -- closest corpus fragment was "
        f"{best.distance_bits} bits away at threshold "
        f"{rresult.threshold_bits}."
    )


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def run_nrl_ai_chat_repl(
    paths: Optional[NrlAiPaths] = None,
    *,
    threshold_bits: Optional[int] = None,
    min_sentences: int = _compose.DEFAULT_MIN_SENTENCES,
    max_sentences: int = _compose.DEFAULT_MAX_SENTENCES,
    max_chars: int = _compose.DEFAULT_MAX_CHARS,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
    chunk_delay_s: float = 0.0,
    use_color: Optional[bool] = None,
) -> SessionStats:
    """Interactive NRL-AI chat loop.

    Blocks on ``stdin`` and writes banner + replies + footers to
    ``stdout``. Returns the aggregate ``SessionStats`` so callers /
    tests can assert on cumulative WPS and hit counts.

    Raises ``FileNotFoundError`` when the index does not exist so the
    CLI wrapper can surface a clean exit code 2.
    """
    paths = paths or NrlAiPaths.default()
    if not paths.manifest.is_file():
        raise FileNotFoundError(
            f"no NRL-AI index at {paths.root} "
            f"(run `nrl-ai ingest <corpus>` first)"
        )
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    if use_color is None:
        use_color = _ansi_supported(stdout)
    thr = threshold_bits if threshold_bits is not None else _resolve.DEFAULT_THRESHOLD_BITS

    fragments = _ingest.load_fragments(paths)
    stats = SessionStats()

    stdout.write(_boot_banner(paths, len(fragments), thr, use_color))
    stdout.flush()

    turn_idx = 0
    while True:
        stdout.write(_turn_header(turn_idx, use_color))
        stdout.flush()
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            stdout.write("\n")
            break
        if line == "":
            stdout.write("\n")
            break
        text = line.strip()
        if not text:
            continue

        if text == "/help":
            stdout.write(_REPL_HELP)
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/stats":
            stdout.write(stats.summary() + "\n")
            continue
        if text == "/reset":
            stats = SessionStats()
            stdout.write("session stats reset\n")
            continue
        if text == "/status":
            stdout.write(
                f"index={paths.root} fragments={len(fragments)} threshold={thr}\n"
            )
            continue
        if text.startswith("/"):
            stdout.write(f"unknown command: {text}\n")
            continue

        t0 = time.perf_counter()
        try:
            rresult = _resolve.resolve(
                text, paths=paths, threshold_bits=thr, top_k=1
            )
        except (FileNotFoundError, RuntimeError) as exc:
            stdout.write(f"\n[nrl-ai error] {exc}\n")
            break

        stdout.write(_reply_label(use_color))
        if rresult.hit and rresult.best is not None:
            cresult = _compose.compose(
                rresult.best.fragment_id,
                paths=paths,
                hit=True,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                max_chars=max_chars,
            )
            words = _stream_reply(cresult, stdout, chunk_delay_s=chunk_delay_s)
            wall_s = time.perf_counter() - t0
            stats.hits += 1
            stats.total_words += words
            stats.total_wall_s += wall_s
            footer = _format_turn_footer(
                turn_idx,
                cresult,
                rresult.best.distance_bits,
                words,
                wall_s,
                use_color,
            )
        else:
            msg = _miss_text(rresult)
            stdout.write(msg + "\n")
            stdout.flush()
            wall_s = time.perf_counter() - t0
            words = len(msg.split())
            stats.misses += 1
            stats.total_words += words
            stats.total_wall_s += wall_s
            footer = _format_turn_footer(
                turn_idx, None, None, words, wall_s, use_color
            )

        stdout.write(footer + "\n")
        stats.turns += 1
        turn_idx += 1

    stdout.write(f"\n[session] {stats.summary()}\n")
    stdout.flush()
    return stats


__all__ = [
    "SessionStats",
    "run_nrl_ai_chat_repl",
]
