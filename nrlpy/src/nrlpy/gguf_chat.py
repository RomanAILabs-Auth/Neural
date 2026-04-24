# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Co-Architect: Grok (xAI)
# Collaborators: Cursor, Anthropic Claude (Opus 4.7), Gemini-Flash (Google), ChatGPT-5.4 (OpenAI)
#
# Contact: daniel@romanailabs.com | romanailabs@gmail.com
# Website: https://romanailabs.com
"""Multi-turn GGUF chat on top of :func:`nrlpy.gguf.run_gguf`.

Provides:

* :class:`ChatMessage` — one ``(role, text)`` turn.
* :class:`ChatSession` — conversation state, session-aggregate TPS, per-turn
  muscle-memory by virtue of threading the full rendered history into
  :attr:`GgufManifest.prompt` (so ``_muscle_memory_key`` naturally incorporates
  the history digest).
* :func:`chat_turn` — one decode step, returns :class:`GgufRunResult` plus the
  updated session.
* :func:`run_gguf_chat_repl` — interactive REPL with slash commands.

Per-turn muscle-memory contract
-------------------------------
The key is ``FNV-1a64(model_sha256 || rendered_prompt || sampler || seed || max_tokens)``
via :func:`nrlpy.gguf._muscle_memory_key`. Because ``rendered_prompt`` embeds
the **entire** chat history for this turn, identical ``(history, next_user)``
pairs hit the same cache file across sessions. Changing the system prompt,
editing an earlier message, or adjusting sampler params invalidates the key.

Honesty hinge (unchanged from the single-shot runner): ``gate_skip_ratio``
stays at ``0.0``, so per-turn and session ``virtual_tps == executed_tps`` until
P2-Active wires a gate into libllama. The session banner says so.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from . import gguf
from .gguf import (
    GgufManifest,
    GgufRunResult,
    NrlAttestation,
    NrlLatticeObservation,
    PrefillGate,
    TpsReport,
    load_manifest,
    manifest_from_args,
    run_gguf,
)

__all__ = [
    "ChatMessage",
    "ChatSession",
    "SessionTps",
    "build_history_prompt",
    "chat_turn",
    "format_session_banner",
    "ladder_badge_plain",
    "load_session",
    "run_gguf_chat_repl",
    "save_session",
]


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass
class ChatMessage:
    """Single turn. ``role`` is ``"user"`` or ``"assistant"``; ``"system"`` is
    reserved for the prepended system prompt and never stored in the list."""

    role: str
    text: str


@dataclass
class SessionTps:
    """Roll-up across turns. Preserves the four-metric contract."""

    turns: int = 0
    executed_tokens: int = 0
    executed_wall_s: float = 0.0
    cache_tokens: int = 0
    cache_wall_s: float = 0.0
    # Bio-Digital P3 — word-count split for session WPS (cache vs decode lanes).
    cache_words: int = 0
    executed_words: int = 0
    gate_skip_ratio: float = 0.0  # must stay 0.0 in P1/P2-Shadow

    def merge(self, tps: TpsReport) -> None:
        self.executed_tokens += tps.executed_tokens
        self.executed_wall_s += tps.executed_wall_s
        self.cache_tokens += tps.cache_tokens
        self.cache_wall_s += tps.cache_wall_s
        # gate_skip_ratio is session-identity: weighted average of per-turn values.
        # In P1/P2-Shadow this always stays 0 because every turn contributes 0.
        total = self.executed_tokens
        if total > 0:
            self.gate_skip_ratio = (
                self.gate_skip_ratio * (total - tps.executed_tokens)
                + tps.gate_skip_ratio * tps.executed_tokens
            ) / total

    def as_tps_report(self) -> TpsReport:
        r = TpsReport(
            executed_tokens=self.executed_tokens,
            executed_wall_s=self.executed_wall_s,
            cache_tokens=self.cache_tokens,
            cache_wall_s=self.cache_wall_s,
            gate_skip_ratio=self.gate_skip_ratio,
        )
        r.finalize()
        return r


@dataclass
class ChatSession:
    """Chat state + session-scoped aggregates.

    Mutate via :func:`chat_turn`; slash-command helpers treat the session as the
    one source of truth. Serialize with :func:`save_session` / :func:`load_session`.
    """

    base_manifest: GgufManifest
    model_sha256: str = ""
    system: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    prefill_gate: PrefillGate | None = None
    tps: SessionTps = field(default_factory=SessionTps)
    last_attestation: NrlAttestation = field(default_factory=NrlAttestation)
    last_observation: NrlLatticeObservation = field(default_factory=NrlLatticeObservation)

    def history_digest(self) -> str:
        """Stable SHA-256 over ``(system, messages)``. For diagnostics/UX only.

        Muscle-memory keying uses the rendered prompt (which contains the same
        information), so this digest is not itself a cache key — it is surfaced
        in the REPL so the operator can visually confirm history stability.
        """
        h = hashlib.sha256()
        h.update(b"sys\x1f")
        h.update(self.system.encode("utf-8"))
        for m in self.messages:
            h.update(b"\x1e")
            h.update(m.role.encode("utf-8"))
            h.update(b"\x1f")
            h.update(m.text.encode("utf-8"))
        return h.hexdigest()[:16]

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")


def _as_numpy_if_available(x: Any) -> Any:
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "__array__"):
            return np.asarray(x)
    except Exception:
        pass
    return None


def _safe_cache_hit(result: GgufRunResult) -> bool:
    """True only on a real lattice hit — never ``bool(ndarray)`` on ``cache_hit``."""
    ch = getattr(result, "cache_hit", False)
    if ch is True:
        return True
    if ch is False or ch is None:
        return False
    arr = _as_numpy_if_available(ch)
    if arr is not None:
        if arr.size == 0:
            return False
        flat = arr.ravel()
        try:
            return bool(flat.all()) if flat.dtype == bool else bool(flat.any())
        except Exception:
            return False
    if isinstance(ch, (list, tuple)):
        return len(ch) > 0 and any(bool(x) for x in ch)
    try:
        return bool(int(ch)) != 0
    except (TypeError, ValueError):
        return False


def ladder_badge_plain(result: GgufRunResult) -> str:
    """Plain-text Resolution Ladder badge (sidecar + ``/stats`` diagnostics)."""
    if not _safe_cache_hit(result):
        return "[Decode]"
    gs = getattr(result, "gate_source", None) or ""
    arr = _as_numpy_if_available(gs)
    if arr is not None:
        if arr.size == 0:
            gs = ""
        else:
            try:
                gs = str(arr.flat[0])
            except Exception:
                gs = str(arr)
    else:
        gs = str(gs)
    if gs.startswith("omega_resolve"):
        dist = int((result.gate_report or {}).get("distance_bits", 0))
        sr = result.omega_shadow
        if sr.ngram_rescued:
            return (
                f"[Instant Map · Rescue overlap={sr.ngram_best_overlap:.2f} · {dist}b]"
            )
        return "[Instant Map]" if dist == 0 else f"[Instant Map · {dist}b]"
    if gs == "zpm_nullspace":
        dist = int((result.gate_report or {}).get("distance_bits", 0))
        return "[ZPM Direct]" if dist == 0 else f"[ZPM Direct · {dist}b]"
    return "[Muscle Memory]"


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def _render_chatml(system: str, messages: Iterable[ChatMessage], next_user: str) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for m in messages:
        if m.role == "user":
            parts.append(f"<|im_start|>user\n{m.text}<|im_end|>\n")
        elif m.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{m.text}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{next_user}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def _render_phi3(system: str, messages: Iterable[ChatMessage], next_user: str) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|system|>\n{system}<|end|>\n")
    for m in messages:
        if m.role == "user":
            parts.append(f"<|user|>\n{m.text}<|end|>\n")
        elif m.role == "assistant":
            parts.append(f"<|assistant|>\n{m.text}<|end|>\n")
    parts.append(f"<|user|>\n{next_user}<|end|>\n<|assistant|>\n")
    return "".join(parts)


def _render_llama2(system: str, messages: list[ChatMessage], next_user: str) -> str:
    # Llama-2 conventions: only the *first* turn carries <<SYS>>.
    parts: list[str] = []
    first = True
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages:
        if m.role == "user":
            pending_user = m.text
        elif m.role == "assistant" and pending_user is not None:
            pairs.append((pending_user, m.text))
            pending_user = None
    for u, a in pairs:
        if first:
            sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
            parts.append(f"[INST] {sys_block}{u} [/INST] {a} </s><s>")
            first = False
        else:
            parts.append(f"[INST] {u} [/INST] {a} </s><s>")
    if first:
        sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        parts.append(f"[INST] {sys_block}{next_user} [/INST]")
    else:
        parts.append(f"[INST] {next_user} [/INST]")
    return "".join(parts)


def _render_plain(system: str, messages: Iterable[ChatMessage], next_user: str) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"System: {system}\n")
    for m in messages:
        prefix = "User" if m.role == "user" else "Assistant"
        parts.append(f"{prefix}: {m.text}\n")
    parts.append(f"User: {next_user}\nAssistant:")
    return "".join(parts)


def build_history_prompt(
    session: ChatSession, next_user: str, fmt: str | None = None
) -> str:
    """Render ``(system, messages, next_user)`` into a single string under ``fmt``.

    ``fmt`` defaults to ``session.base_manifest.chat_format``. Supported values:
    ``"none"`` (plain User/Assistant), ``"chatml"``, ``"phi3"``, ``"llama2"``.
    """
    chosen = fmt if fmt is not None else session.base_manifest.chat_format
    if chosen == "chatml":
        return _render_chatml(session.system, session.messages, next_user)
    if chosen == "phi3":
        return _render_phi3(session.system, session.messages, next_user)
    if chosen == "llama2":
        return _render_llama2(session.system, list(session.messages), next_user)
    return _render_plain(session.system, session.messages, next_user)


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #


def _per_turn_manifest(session: ChatSession, next_user: str) -> GgufManifest:
    """Clone ``base_manifest`` with per-turn prompt baked in.

    ``chat_format`` is forced to ``"none"`` on the clone because we have already
    rendered the full history into ``prompt`` — letting the runner re-template
    would wrap the whole transcript in a second user/assistant envelope.
    """
    rendered = build_history_prompt(session, next_user)
    # Phase 11 — scope MM/ZPM cache to the *user intent* of this turn,
    # not the rendered history+prompt envelope. Without this every turn
    # has a unique cache key (because the transcript keeps growing) and
    # repeating / rephrasing a question can never hit R0 or R1 in a live
    # chat. The decoder still sees the full rendered prompt for coherent
    # generation; only the cache lookup key is scoped.
    return dataclasses.replace(
        session.base_manifest,
        prompt=rendered,
        prompt_file="",
        chat_format="none",
        chat_intent=next_user.strip(),
        chat_kv_reuse=True,
    )


def chat_turn(
    session: ChatSession,
    user_text: str,
    *,
    stream_to: IO[str] | None = None,
    observation_profile: str = "",  # disabled per-turn by default; 1x/session is plenty
    observation_timeout_s: float = 5.0,
    preloaded_llm: Any = None,
) -> GgufRunResult:
    """Run one chat turn. Mutates ``session`` in place with user + assistant turns.

    ``observation_profile=""`` skips the per-turn lattice probe by default —
    running ``omega-hybrid`` every turn would add ~25 ms of wall overhead without
    adding information. Call :func:`nrlpy.gguf.nrl_attest` or
    :func:`run_gguf_chat_repl` once at session start instead.
    """
    per_turn = _per_turn_manifest(session, user_text)
    result = run_gguf(
        per_turn,
        stream_to=stream_to,
        attest_profile="",
        observation_profile=observation_profile,
        observation_timeout_s=observation_timeout_s,
        prefill_gate=session.prefill_gate,
        preloaded_llm=preloaded_llm,
        trust_model_sha=preloaded_llm is not None,
    )
    session.messages.append(ChatMessage(role="user", text=user_text))
    session.messages.append(ChatMessage(role="assistant", text=result.text))
    session.tps.turns += 1
    session.tps.merge(result.tps)
    wc = int(getattr(getattr(result, "word_rates", None), "word_count", 0) or 0)
    if _safe_cache_hit(result):
        session.tps.cache_words += max(0, wc)
    else:
        session.tps.executed_words += max(0, wc)
    session.last_attestation = result.nrl_attestation
    if result.lattice_observation.available:
        session.last_observation = result.lattice_observation
    if not session.model_sha256:
        session.model_sha256 = result.model_sha256
    return result


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save_session(session: ChatSession, path: str | Path) -> Path:
    """Write a portable JSON snapshot. Does NOT write model/prompt weights."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "nrl.gguf_chat_session.v1",
        "system": session.system,
        "messages": [{"role": m.role, "text": m.text} for m in session.messages],
        "model_sha256": session.model_sha256,
        "tps": dataclasses.asdict(session.tps),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_session(session: ChatSession, path: str | Path) -> ChatSession:
    """Replace ``session.system`` + ``messages`` from a saved JSON snapshot.

    Preserves ``base_manifest`` — the caller already chose the model. If the
    snapshot's ``model_sha256`` disagrees with the manifest, we refuse: loading
    another model's history into this session would corrupt muscle memory and
    produce nonsense from the current weights.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    saved_sha = str(data.get("model_sha256", ""))
    if (
        saved_sha
        and session.model_sha256
        and saved_sha != session.model_sha256
    ):
        raise RuntimeError(
            f"/load refused: session model_sha256={session.model_sha256[:12]}... "
            f"but snapshot was recorded with {saved_sha[:12]}..."
        )
    session.system = str(data.get("system", ""))
    session.messages = [
        ChatMessage(role=str(m["role"]), text=str(m["text"]))
        for m in data.get("messages", [])
    ]
    session.tps = SessionTps()  # re-aggregate from zero; TPS doesn't replay
    return session


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #


def format_session_banner(session: ChatSession) -> str:
    tps = session.tps.as_tps_report()
    att = session.last_attestation
    obs = session.last_observation
    st = session.tps
    tw = max(0, int(st.cache_words) + int(st.executed_words))
    twall = float(st.cache_wall_s) + float(st.executed_wall_s)
    eff_wps = tw / twall if twall > 1e-12 else 0.0
    lane_cache_wps = (
        float(st.cache_words) / float(st.cache_wall_s)
        if st.cache_wall_s > 1e-12
        else 0.0
    )
    lane_exec_wps = (
        float(st.executed_words) / float(st.executed_wall_s)
        if st.executed_wall_s > 1e-12
        else 0.0
    )
    lines = [
        "",
        "NRL gguf_chat (session aggregate)",
        f"  model           {Path(session.base_manifest.model).name}",
        f"  model_sha256    {session.model_sha256[:16] if session.model_sha256 else 'n/a'}...",
        f"  turns           {session.tps.turns}",
        f"  history_digest  {session.history_digest()}",
        "",
        "session words/sec (P3 rollup - cache vs decode lanes)",
        f"  effective_wps   {eff_wps:>10.2f}   (all turns, words / total wall)",
        f"  cache_lane_wps  {lane_cache_wps:>10.2f}   ({st.cache_words} words / {st.cache_wall_s:.3f}s)",
        f"  decode_lane_wps {lane_exec_wps:>10.2f}   ({st.executed_words} words / {st.executed_wall_s:.3f}s)",
        "",
        "decode TPS (session-aggregate)",
        f"  executed_tokens {tps.executed_tokens:>10}",
        f"  executed_wall   {tps.executed_wall_s:>10.3f} s",
        f"  executed_tps    {tps.executed_tps:>10.2f}",
        f"  virtual_tps     {tps.virtual_tps:>10.2f}   "
        "(P1/P2-Shadow: == executed_tps until gate is active)",
        f"  cache_tps       {tps.cache_tps:>10.2f}",
        f"  effective_tps   {tps.effective_tps:>10.2f}",
        "",
        "NRL attestation (last turn, engine-sanity)",
        f"  available       {'yes' if att.available else 'no'}",
        f"  profile         {att.profile}",
        f"  skip_ratio      {att.skip_ratio:>10.6f}   (lattice, not libllama)",
        "",
        "NRL lattice observation (last turn, advisory-only)",
        f"  available       {'yes' if obs.available else 'no'}",
        f"  profile         {obs.profile}",
        f"  skip_ratio      {obs.skip_ratio:>10.6f}   (gate preview, lattice work)",
        f"  note            {obs.note}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #


_REPL_HELP = """\
Slash commands:
  /help              show this help
  /clear             reset conversation history (keeps system prompt)
  /system <text>     set system prompt (resets history for safety)
  /stats | /tps      print session-aggregate four-metric TPS banner
  /save <path>       write session JSON
  /load <path>       load session JSON (must match model_sha256)
  /seed <n>          set sampler seed for the next turn(s)
  /history           print compact (role, len) summary
  /exit | /quit      leave the REPL (Ctrl-D / Ctrl-C also work)

Anything not starting with '/' is sent as a user turn. The Resolution
Ladder (R0 muscle-memory, R1 ZPM, R2 Omega, R5 libllama decode)
dispatches every turn automatically. Repeated questions collapse to
cache speed on R0/R1; novel turns execute on R5.
"""


# --------------------------------------------------------------------------- #
# Branded UI helpers (ASCII-safe for PowerShell cp1252)
# --------------------------------------------------------------------------- #


def _ansi_supported(stream: IO[str]) -> bool:
    """True when it's safe to emit ANSI escapes. Respects NO_COLOR."""
    import os as _os

    if _os.environ.get("NO_COLOR", "").strip():
        return False
    if _os.environ.get("NRL_NO_COLOR", "").strip():
        return False
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        return False


_BANNER_INNER = 62  # characters between the two `|` pipes


def _pad_line(raw: str, left_ansi: str, right_ansi: str, reset: str, border_ansi: str) -> str:
    """Render a boxed line: |  <content padded to inner width>  |."""
    pad = max(0, _BANNER_INNER - len(raw) - 2)
    return f"{border_ansi}|{reset} {left_ansi}{raw}{right_ansi}{' ' * pad} {border_ansi}|{reset}"


_BACKEND_LABELS = {
    "native_full": "native_full (C hot path, R0+R1+R2 bridge)",
    "native_full_strict": "native_full strict (C hot path, CI gate)",
    "native": "native (C ladder, Python candidate gen)",
    "native_strict": "native strict (C ladder, CI gate)",
    "python": "python ladder (fallback / debug)",
}


def _boot_banner(
    model_name: str,
    chat_format: str,
    seed: int,
    zpm_on: bool,
    mm_on: bool,
    use_color: bool,
    runner_backend: str = "python",
    fast_chat: bool = False,
    prewarm_summary: str = "",
    rewired: bool = False,
    zpm_threshold_bits: int = 0,
    coherence_lane: str = "fast-stable",
) -> str:
    """Polished startup banner. RomanAILabs branding, honest capability hints."""
    C = "\x1b[36m" if use_color else ""  # cyan (border)
    M = "\x1b[35m" if use_color else ""  # magenta (brand)
    G = "\x1b[32m" if use_color else ""  # green (status)
    Y = "\x1b[33m" if use_color else ""  # yellow (fast-chat badge)
    DIM = "\x1b[2m" if use_color else ""
    R = "\x1b[0m" if use_color else ""

    fast = []
    if mm_on:
        fast.append("muscle-memory")
    if zpm_on:
        fast.append("zpm-nullspace")
    fast_str = " + ".join(fast) if fast else "disabled"

    backend_label = _BACKEND_LABELS.get(runner_backend, runner_backend or "python")

    rule = f"{C}+{'-' * _BANNER_INNER}+{R}"
    line_brand = _pad_line("Neural AI  .  RomanAILabs  .  NRL lattice runner", M, R, R, C)
    subtitle = (
        "Final Product v1.0  .  REWIRED  .  ZPM + Omega primary path"
        if rewired
        else "Final Product v1.0  .  Expect speed on repeat questions"
    )
    line_sub = _pad_line(subtitle, M if rewired else DIM, R, R, C)
    line_model = _pad_line(f"model        {model_name}", "", "", "", C)
    line_fmt = _pad_line(f"chat_format  {chat_format}", "", "", "", C)
    line_seed = _pad_line(f"seed         {seed}", "", "", "", C)
    line_backend = _pad_line(f"backend      {backend_label}", G, R, R, C)
    line_fast = _pad_line(f"fast_lane    {fast_str}", G, R, R, C)
    rows = [
        rule,
        line_brand,
        line_sub,
        rule,
        line_model,
        line_fmt,
        line_seed,
        line_backend,
        line_fast,
    ]
    if rewired:
        rows.append(
            _pad_line(
                "rewired      ON (R2 Omega primary; fuzzy ZPM; learns live)",
                M,
                R,
                R,
                C,
            )
        )
        rows.append(
            _pad_line(
                f"coherence    {coherence_lane}  .  zpm_thresh={zpm_threshold_bits}b",
                DIM,
                R,
                R,
                C,
            )
        )
    elif fast_chat:
        rows.append(_pad_line("fast_chat    ON (cache prewarmed; repeats hit R0/R1)", Y, R, R, C))
        if prewarm_summary:
            rows.append(_pad_line(prewarm_summary, DIM, R, R, C))
    rows.append(rule)
    rows.append(
        f"  {DIM}tip: '/help' for commands, '/stats' for TPS, '/exit' to leave{R}"
    )
    rows.append("")
    return "\n".join(rows) + "\n"


def _turn_header(turn_idx: int, use_color: bool) -> str:
    DIM = "\x1b[2m" if use_color else ""
    R = "\x1b[0m" if use_color else ""
    return f"{DIM}[you, turn {turn_idx}]{R}\n> "


def _reply_label(use_color: bool) -> str:
    M = "\x1b[35m" if use_color else ""
    R = "\x1b[0m" if use_color else ""
    return f"\n{M}[Neural AI]{R}\n"


def _format_turn_footer(
    turn_idx: int,
    tokens: int,
    wall_s: float,
    result: GgufRunResult,
    use_color: bool,
) -> str:
    """One compact, WPS-first status line per reply.

    ``executed_wps`` is the real silicon throughput on a cache miss.
    ``effective_wps`` is the served-throughput (which dominates on MM/ZPM
    hits). The badge shows which rung of the Resolution Ladder served
    the turn -- ``[ZPM hit]`` for R1 and Omega active, ``[Muscle Memory]``
    for R0, and ``[Decode]`` for novel R5 generation.
    """
    DIM = "\x1b[2m" if use_color else ""
    G = "\x1b[32m" if use_color else ""
    Y = "\x1b[33m" if use_color else ""
    R = "\x1b[0m" if use_color else ""

    M = "\x1b[35m" if use_color else ""
    wr = result.word_rates
    plain = ladder_badge_plain(result)
    if _safe_cache_hit(result):
        if plain.startswith("[Instant Map"):
            badge = f"{M}{plain}{R}"
        elif plain.startswith("[ZPM"):
            badge = f"{G}{plain}{R}"
        else:
            badge = f"{G}{plain}{R}"
        wps_primary = wr.effective_wps if wr is not None else 0.0
        wps_label = "effective_wps"
    else:
        badge = f"{Y}{plain}{R}"
        wps_primary = wr.executed_wps if wr is not None else 0.0
        wps_label = "executed_wps"

    tps = result.tps.cache_tps if _safe_cache_hit(result) else result.tps.executed_tps
    return (
        f"\n{DIM}turn {turn_idx:>3} . {badge}{DIM} . "
        f"{tokens} tok / {wall_s:.2f}s / "
        f"{tps:.1f} tps / {wps_primary:.1f} {wps_label}{R}\n"
    )


def _read_line(prompt: str, stdin: IO[str], stdout: IO[str]) -> str | None:
    stdout.write(prompt)
    stdout.flush()
    line = stdin.readline()
    if not line:
        return None
    return line.rstrip("\r\n")


def _handle_slash(
    cmd: str,
    session: ChatSession,
    stdout: IO[str],
) -> str | None:
    """Dispatch ``/command ...``.

    Returns one of: ``"continue"`` (stay in REPL), ``"quit"`` (exit cleanly),
    or ``None`` if the input was not a slash command.
    """
    if not cmd.startswith("/"):
        return None
    head, _, rest = cmd.partition(" ")
    rest = rest.strip()
    if head in {"/quit", "/exit"}:
        return "quit"
    if head == "/help":
        stdout.write(_REPL_HELP)
        return "continue"
    if head == "/clear":
        session.messages.clear()
        stdout.write("history cleared\n")
        return "continue"
    if head == "/system":
        session.system = rest
        session.messages.clear()
        stdout.write(f"system prompt set ({len(rest)} chars); history cleared\n")
        return "continue"
    if head in {"/tps", "/stats"}:
        stdout.write(format_session_banner(session))
        return "continue"
    if head == "/save":
        if not rest:
            stdout.write("error: /save requires a path\n")
            return "continue"
        p = save_session(session, rest)
        stdout.write(f"saved session to {p}\n")
        return "continue"
    if head == "/load":
        if not rest:
            stdout.write("error: /load requires a path\n")
            return "continue"
        try:
            load_session(session, rest)
        except (OSError, RuntimeError, json.JSONDecodeError) as e:
            stdout.write(f"error: /load failed: {e}\n")
            return "continue"
        stdout.write(
            f"loaded session ({len(session.messages)} messages, "
            f"digest={session.history_digest()})\n"
        )
        return "continue"
    if head == "/seed":
        try:
            session.base_manifest.seed = int(rest)
        except ValueError:
            stdout.write(f"error: /seed requires an integer, got {rest!r}\n")
            return "continue"
        stdout.write(f"seed set to {session.base_manifest.seed}\n")
        return "continue"
    if head == "/history":
        for i, m in enumerate(session.messages):
            stdout.write(f"  [{i:03d}] {m.role:<9} {len(m.text):>6}ch\n")
        return "continue"
    stdout.write(f"error: unknown slash command {head!r} — try /help\n")
    return "continue"


def build_session(
    manifest: GgufManifest,
    *,
    system: str = "",
) -> ChatSession:
    """Construct a :class:`ChatSession` from a parsed manifest."""
    session = ChatSession(
        base_manifest=manifest,
        system=system,
        prefill_gate=PrefillGate() if manifest.prefill_cache == "session" else None,
    )
    return session


def run_gguf_chat_repl(
    manifest: GgufManifest,
    *,
    system: str = "",
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    attest_profile: str = "",
    observation_profile: str = "",
    fast_chat: bool = False,
    rewired: bool = False,
) -> ChatSession:
    """Interactive REPL. One engine-attestation + one lattice-observation at
    session start; per-turn decode via :func:`chat_turn`.

    Returns the final :class:`ChatSession` so callers (or tests) can inspect
    the aggregate TPS and conversation.
    """
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    session = build_session(manifest, system=system)
    preloaded_llm: Any = None

    # Preload model + hash once per REPL. This avoids re-hashing and re-loading
    # weights every turn, which dominates wall time and kills chat TPS.
    model_path = Path(manifest.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {manifest.model}")
    actual_sha = gguf.sha256_file(model_path)
    if manifest.model_sha256 and manifest.model_sha256 != actual_sha:
        raise RuntimeError(
            f"model_sha256 mismatch: manifest={manifest.model_sha256} actual={actual_sha}"
        )
    manifest.model_sha256 = actual_sha
    session.model_sha256 = actual_sha
    preloaded_llm = gguf._load_llm(manifest)

    # One-shot engine attestation + lattice observation at session start.
    # Default is disabled for chat throughput; operators can opt in explicitly.
    session.last_attestation = (
        gguf.nrl_attest(profile=attest_profile)
        if attest_profile
        else gguf.NrlAttestation(profile="", available=False)
    )
    if observation_profile:
        session.last_observation = gguf._collect_lattice_observation(
            observation_profile, neurons=65_536, iterations=64
        )

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
            runner_backend=str(getattr(manifest, "runner_backend", "python")),
            fast_chat=fast_chat,
            prewarm_summary=prewarm_summary,
            rewired=rewired,
            zpm_threshold_bits=int(getattr(manifest, "zpm_threshold_bits", 0)),
            coherence_lane=str(getattr(manifest, "coherence_lane", "fast-stable")),
        )
    )
    sout.flush()

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
                continue

            sout.write(_reply_label(use_color))
            sout.flush()
            t0 = time.perf_counter()
            try:
                result = chat_turn(session, stripped, stream_to=sout, preloaded_llm=preloaded_llm)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — REPL must survive decoder crashes
                # Phase 11 — previously only FileNotFoundError / RuntimeError
                # were caught, so anything else (UnicodeDecodeError on a
                # partial token, OSError on a BrokenPipeError into the
                # terminal, random llama-cpp internals) killed the REPL
                # mid-session. A chat shouldn't die when one turn does.
                sout.write(
                    f"\nerror: {type(e).__name__}: {e}\n"
                    f"       (turn skipped; session continues. /exit to leave.)\n"
                )
                sout.flush()
                continue
            wall = time.perf_counter() - t0
            sout.write(_format_turn_footer(turn_idx, result.tokens, wall, result, use_color))
            # Rewired UX: on an R5 decode fall-back we explicitly tell
            # the user the lattice is learning the prompt. The actual
            # cache write (R0 muscle-memory + R1 ZPM nullspace) has
            # already been performed by the runner — see
            # ``nrlpy.gguf._run_gguf_native_full`` for the native path
            # and ``nrlpy.gguf._run_gguf_python`` for the fallback. We
            # don't invent cache writes here; we just surface the fact
            # that the next semantically similar turn will ride R0/R1
            # / R2 instead of decoding again.
            if rewired and not _safe_cache_hit(result):
                C_DIM = "\x1b[2m" if use_color else ""
                C_M = "\x1b[35m" if use_color else ""
                C_R = "\x1b[0m" if use_color else ""
                sout.write(
                    f"{C_DIM}{C_M}  Learning this prompt... "
                    f"next similar question will be instant.{C_R}\n"
                )
            sout.flush()
    except KeyboardInterrupt:
        sout.write("\nSession ended. Goodbye.\n")

    sout.write("\n" + format_session_banner(session))
    sout.flush()
    return session


# --------------------------------------------------------------------------- #
# CLI entry point (called from nrlpy.cli)
# --------------------------------------------------------------------------- #


_CHAT_USAGE = (
    "usage: nrlpy chat <model.gguf|manifest.nrl> "
    "[--system \"...\"] [--seed N] [--max-tokens N] [--temperature F] "
    "[--chat-format none|chatml|phi3|llama2] [--fast-chat|--rewired] "
    "[--response-recall] [--no-response-recall] "
    "[--native-full|--native|--python-ladder|--native-full-strict|--native-strict]"
    "\n\n"
    "  --rewired: R2 Omega primary, fuzzy ZPM, aggressive R0/R1 writeback "
    "(response recall ON by default). "
    "--no-response-recall disables whole-reply cache while keeping rewired lanes."
)


def _apply_fast_chat_defaults(
    extra_kwargs: dict[str, Any], user_overrides: set[str]
) -> None:
    """Tune ``extra_kwargs`` for the 'fast chat' UX.

    The goal: make R0 + R1 hits as likely as possible during a normal
    back-and-forth so the user feels the lattice-cache wins in the
    first few turns. Concretely:

    * ``runner_backend = native_full`` (Phase 8-EG hot path).
    * ``muscle_memory = on`` so every turn is cached for replay.
    * Bounded reply length + low temperature so repeats are tighter
      and more likely to anchor the same ZPM state twice in a row.
    * ``coherence_lane = fast-balanced`` so R2 Omega Native Resolve
      is allowed to serve tokens on near-matches while staying
      inside the Stage-VI verify gate.
    * ``prefill_cache = session`` so KV-state is reused across turns.

    The ``zpm_nullspace`` toggle is **not** a ``manifest_from_args``
    kwarg — :func:`apply_fast_chat_post_build` handles it as a direct
    attribute assignment after the manifest is built. User-supplied
    flags always win; fast-chat only fills in slots left empty.
    """
    if "runner_backend" not in user_overrides:
        extra_kwargs["runner_backend"] = "native_full"
    # ``--fast-chat`` prewarms R0+R1; MM must be on so seed corpus writes land.
    if "muscle_memory" not in user_overrides:
        extra_kwargs["muscle_memory"] = "on"
    if "max_tokens" not in user_overrides:
        extra_kwargs["max_tokens"] = 192
    if "temperature" not in user_overrides:
        extra_kwargs["temperature"] = 0.2
    if "repeat_penalty" not in user_overrides:
        extra_kwargs["repeat_penalty"] = 1.08
    if "coherence_lane" not in user_overrides:
        extra_kwargs["coherence_lane"] = "fast-balanced"
    if "prefill_cache" not in user_overrides:
        extra_kwargs["prefill_cache"] = "session"


def apply_fast_chat_post_build(
    manifest: GgufManifest, user_overrides: set[str]
) -> None:
    """Post-construction setters for fast-chat (prewarm path).

    Turns on ZPM nullspace + fuzzy threshold so prewarmed seeds populate
    R1 alongside R0 muscle memory. Plain ``nrlpy chat`` without
    ``--fast-chat`` / ``--rewired`` stays conservative; use
    ``--response-recall`` there if you want whole-reply cache without
    the seed prewarmer.
    """
    if "zpm_nullspace" not in user_overrides:
        manifest.zpm_nullspace = True
    if "zpm_threshold_bits" not in user_overrides:
        manifest.zpm_threshold_bits = 28


def _apply_rewired_defaults(
    extra_kwargs: dict[str, Any], user_overrides: set[str]
) -> None:
    """Tune ``extra_kwargs`` for the 'rewired' UX.

    Rewired mode treats the absorbed GGUF as a native neural lattice:
    every turn first tries the lightweight ZPM anchor + Omega router
    before falling back to R5 libllama decode. Once a turn has been
    served by R5 it is written to R0 muscle memory and R1 ZPM index,
    so semantically similar follow-ups instantly map via the lattice.

    This is a strict superset of fast-chat:

    * ``runner_backend = native_full`` (R0 + R1 in C).
    * ``coherence_lane = max-throughput`` — unlocks R2 Omega Native
      Resolve as an active token-serving rung (Stage-VI gated).
    * ``omega_budget_ms = 12.0`` and ``omega_candidates = 12`` — more
      hypothesis branches per probe so Omega can resolve a wider
      neighborhood of states to a cached reply. These are the
      maximum values we'll budget at default without turning a
      single turn into a visible wait (12 ms is a sub-frame latency
      cost even on a 60 Hz UI).
    * ``muscle_memory = on`` + post-build ``zpm_nullspace = True`` +
      post-build ``zpm_threshold_bits = 28`` — ZPM flags a hit on
      any stored state within 28 bits of Hamming distance. The
      anchor is 256 bits wide, so 28 bits ≈ an 11% tolerance — a
      re-phrasing of a previous question still anchors to the same
      reply but genuinely different prompts stay in R5.
    * Low temperature + tight repeat penalty so the lattice-cached
      reply stays coherent when Omega projects it.

    User-supplied flags always win.
    """
    if "runner_backend" not in user_overrides:
        extra_kwargs["runner_backend"] = "native_full"
    # Bio-Digital P3 — rewired defaults include whole-reply recall (R0+R1)
    # plus fuzzy ZPM; ``chat_intent`` scoping keeps rephrases on-lattice.
    if "muscle_memory" not in user_overrides:
        extra_kwargs["muscle_memory"] = "on"
    if "max_tokens" not in user_overrides:
        extra_kwargs["max_tokens"] = 192
    if "temperature" not in user_overrides:
        extra_kwargs["temperature"] = 0.15
    if "repeat_penalty" not in user_overrides:
        extra_kwargs["repeat_penalty"] = 1.08
    if "coherence_lane" not in user_overrides:
        extra_kwargs["coherence_lane"] = "max-throughput"
    if "prefill_cache" not in user_overrides:
        extra_kwargs["prefill_cache"] = "session"
    if "omega_budget_ms" not in user_overrides:
        extra_kwargs["omega_budget_ms"] = 12.0
    if "omega_candidates" not in user_overrides:
        extra_kwargs["omega_candidates"] = 12


def apply_rewired_post_build(
    manifest: GgufManifest, user_overrides: set[str]
) -> None:
    """Post-construction setters for rewired mode (P3 conscious thread).

    Enables fuzzy ZPM nullspace + 28-bit neighborhood threshold unless
    overridden. Use ``--no-response-recall`` on ``nrlpy chat`` to keep
    R2 primary but disable whole-reply R0/R1 writeback (see
    :func:`apply_response_recall` for the same toggles in isolation).
    """
    if "zpm_threshold_bits" not in user_overrides:
        manifest.zpm_threshold_bits = 28
    if "zpm_nullspace" not in user_overrides:
        manifest.zpm_nullspace = True
    if "muscle_memory" not in user_overrides:
        manifest.muscle_memory = "on"


def apply_response_recall(
    manifest: GgufManifest, user_overrides: set[str]
) -> None:
    """Phase 13 opt-in: turn on the whole-reply cache layer.

    Mirrors the pre-Phase-13 defaults: muscle-memory on + ZPM
    nullspace on. This is honest response_recall (byte-replay of a
    stored reply when the new prompt anchor matches a previously-
    served turn within ``zpm_threshold_bits``). It is NOT word-weight
    memoization and makes no claim to be.
    """
    if "muscle_memory" not in user_overrides:
        manifest.muscle_memory = "on"
    if "zpm_nullspace" not in user_overrides:
        manifest.zpm_nullspace = True
    if "zpm_threshold_bits" not in user_overrides:
        manifest.zpm_threshold_bits = 28


def _pick_default_backend() -> str:
    """Prefer ``native_full`` when Phase 8-EG bindings are built; fall back
    to the Python ladder otherwise. This keeps ``nrlpy chat`` fast on
    well-provisioned hosts without forcing users to remember a flag.
    """
    try:
        from . import native_ladder  # noqa: PLC0415

        if native_ladder.is_full_native_available():
            return "native_full"
    except Exception:  # noqa: BLE001
        pass
    return "python"


def main_gguf_chat(args: list[str]) -> int:
    """``nrlpy chat <model.gguf|manifest.nrl> [--system "..."] [--seed N] [...]``.

    Defaults to the ``native_full`` backend (Phase 8-EG full native hot path)
    when it is available, matching the Final Product release gate. Falls back
    to the Python ladder on hosts that haven't built the C extensions.

    Returns the exit code. Errors (bad manifest, missing model, SHA mismatch)
    are printed to stderr with exit code 2.
    """
    if not args:
        print("error: nrlpy chat <model.gguf|manifest.nrl>", file=sys.stderr)
        print(_CHAT_USAGE, file=sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_CHAT_USAGE)
        return 0
    target = args[0]
    rest = args[1:]
    system = ""
    extra_kwargs: dict[str, Any] = {}
    backend_override: str | None = None
    fast_chat = False
    rewired = False
    response_recall = False
    no_response_recall = False
    user_overrides: set[str] = set()

    i = 0
    while i < len(rest):
        flag = rest[i]
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
        elif flag == "--chat-format" and i + 1 < len(rest):
            extra_kwargs["chat_format"] = rest[i + 1]
            user_overrides.add("chat_format")
            i += 2
        elif flag == "--fast-chat":
            fast_chat = True
            i += 1
        elif flag == "--rewired":
            rewired = True
            i += 1
        elif flag == "--native-full":
            backend_override = "native_full"
            i += 1
        elif flag == "--native-full-strict":
            backend_override = "native_full_strict"
            i += 1
        elif flag == "--native":
            backend_override = "native"
            i += 1
        elif flag == "--native-strict":
            backend_override = "native_strict"
            i += 1
        elif flag == "--python-ladder":
            backend_override = "python"
            i += 1
        elif flag == "--response-recall":
            response_recall = True
            i += 1
        elif flag == "--no-response-recall":
            no_response_recall = True
            user_overrides.update({"muscle_memory", "zpm_nullspace"})
            i += 1
        elif flag in {"-h", "--help"}:
            print(_CHAT_USAGE)
            return 0
        else:
            print(f"error: unknown chat flag {flag!r}", file=sys.stderr)
            print(_CHAT_USAGE, file=sys.stderr)
            return 2

    # Record explicit backend choice before any preset fills in defaults,
    # so --python-ladder --rewired still honors the user's backend pick.
    if backend_override is not None:
        extra_kwargs["runner_backend"] = backend_override
        user_overrides.add("runner_backend")

    # --rewired is the strict superset of --fast-chat. When the user
    # asks for rewired, we apply the rewired preset and skip fast-chat
    # prewarm entirely (rewired learns from the conversation, not from
    # a canned seed corpus).
    if rewired:
        if no_response_recall:
            extra_kwargs["muscle_memory"] = "off"
        _apply_rewired_defaults(extra_kwargs, user_overrides)
    elif fast_chat:
        _apply_fast_chat_defaults(extra_kwargs, user_overrides)

    if "runner_backend" not in extra_kwargs:
        extra_kwargs["runner_backend"] = _pick_default_backend()

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

    if rewired:
        apply_rewired_post_build(manifest, user_overrides)
    elif fast_chat:
        apply_fast_chat_post_build(manifest, user_overrides)
    if response_recall and not rewired:
        apply_response_recall(manifest, user_overrides)

    # Chat needs a non-empty history; we don't use manifest.prompt as the first user turn.
    manifest.prompt = ""
    manifest.prompt_file = ""

    try:
        run_gguf_chat_repl(
            manifest,
            system=system,
            fast_chat=fast_chat and not rewired,
            rewired=rewired,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0

