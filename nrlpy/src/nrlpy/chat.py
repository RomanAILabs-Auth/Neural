"""English-friendly control-plane chat (status, telemetry, evidence tail).

This is **not** a claim of agency or consciousness. It maps short natural-language
phrases to **deterministic** queries against ``nrlpy.runtime``, optional ``psutil``
thermal sensors, and local JSONL logs. Session lines append to
``build/nrlpy_chat/session.jsonl`` for a bounded “what we discussed” recall only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from . import evidence, runtime
from .learn_store import default_store
from .paths import immune_evidence_log_paths


def _session_path() -> Path:
    base = Path(os.environ.get("NRL_CHAT_SESSION_DIR", Path.cwd() / "build" / "nrlpy_chat"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "session.jsonl"


def _append_session(role: str, text: str) -> None:
    evidence.append_jsonl(
        _session_path(),
        {
            "schema_id": "nrl.chat_session.v1",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "text": text[:4000],
        },
    )


def _evidence_candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("NRL_EVIDENCE_LOG")
    if env:
        out.append(Path(env))
    here = Path.cwd()
    for anc in [here, *here.parents]:
        p = anc / "build" / "immune" / "events.jsonl"
        if p.is_file():
            out.append(p)
        if anc.parent == anc:
            break
    # de-dupe preserve order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _tail_jsonl(path: Path, max_lines: int) -> list[str]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return raw[-max_lines:]


def _reply_temperature() -> str:
    try:
        import psutil  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        return (
            "I do not have thermal data in this environment. "
            "Install optional dependency: ``pip install psutil``, then ask again."
        )
    try:
        fn = getattr(psutil, "sensors_temperatures", None)
        if not callable(fn):
            return "``psutil`` has no ``sensors_temperatures`` on this platform build."
        sensors = fn()  # type: ignore[misc]
        if not sensors:
            return "No temperature sensors were reported by the OS via psutil."
        parts: list[str] = []
        for name, entries in sensors.items():
            for e in entries[:3]:
                cur = getattr(e, "current", None)
                if cur is not None:
                    parts.append(f"{name}: {float(cur):.1f} °C")
        if not parts:
            return "Thermal sensors list was empty."
        return "Thermal snapshot: " + " | ".join(parts[:8])
    except Exception as exc:  # noqa: BLE001
        return f"Temperature query failed: {exc}"


def _reply_throughput() -> str:
    try:
        r = runtime.braincore_int4(neurons=65_536, iterations=256, threshold=8)
    except Exception as exc:  # noqa: BLE001
        return f"INT4 probe failed: {exc}"
    gn = float(r.get("giga_neurons_per_sec", 0.0))
    var = str(r.get("variant", "?"))
    sec = float(r.get("seconds", 0.0))
    return (
        f"In-process INT4 probe: {gn:.3f} GN/s (variant={var}, wall≈{sec:.4f}s, "
        f"neurons={r.get('neurons')}, iterations={r.get('iterations')}). "
        "This is one local measurement, not a sustained production SLO."
    )


def _reply_activity() -> str:
    cands = [p for p in immune_evidence_log_paths() if p.is_file()]
    if not cands:
        return (
            "No immune evidence file found. Expected ``build/immune/events.jsonl`` "
            "under the repo or set ``NRL_EVIDENCE_LOG`` to a JSONL path."
        )
    path = cands[0]
    lines = _tail_jsonl(path, 6)
    if not lines:
        return f"Evidence file is empty: {path}"
    pretty: list[str] = []
    for line in lines:
        try:
            obj = json.loads(line)
            sid = obj.get("signal_id", "?")
            act = obj.get("action", "?")
            ts = obj.get("ts_utc", "?")
            pretty.append(f"  [{ts}] {sid} → {act}")
        except json.JSONDecodeError:
            pretty.append(f"  (raw) {line[:120]}")
    return f"Last lines from {path}:\n" + "\n".join(pretty)


def _reply_recall() -> str:
    p = _session_path()
    if not p.is_file():
        return "No prior chat session on disk for this working directory."
    lines = _tail_jsonl(p, 12)
    if not lines:
        return "Session file exists but is empty."
    out: list[str] = []
    for line in lines:
        try:
            o = json.loads(line)
            role = o.get("role", "?")
            text = str(o.get("text", ""))[:200]
            out.append(f"  {role}: {text}")
        except json.JSONDecodeError:
            out.append(f"  {line[:120]}")
    return "Recent session turns:\n" + "\n".join(out)


def _reply_help() -> str:
    return "\n".join(
        [
            "Commands (English):",
            "  temperature / temp / hot   — thermal sensors (needs psutil)",
            "  speed / throughput / ops     — one INT4 braincore probe (GN/s)",
            "  version / who are you        — engine version string",
            "  features / cpu               — JSON CPU feature flags",
            "  nrl binary / where nrl       — resolved native ``nrl`` path",
            "  activity / log / events      — tail immune JSONL if present",
            "  recall / history             — tail this chat session",
            "  learn / plasticity           — architecture + bounded vocab store",
            "  growth / how much grown      — disk + word-count vs your 4 GiB cap",
            "  status / overview            — version + nrl path + feature count",
            "  help                         — this list",
            "  quit / exit                  — leave",
            "",
            "This loop is control-plane telemetry only; it does not mutate kernels.",
        ]
    )


def _reply_learn_architecture() -> str:
    return (
        "Bounded learning (architecture): specialization and plasticity live in the "
        "control plane with evidence + shadow promotion. See ``docs/"
        "nrl_alive_language_evolution_architecture.md``. "
        "The local **word store** is frequency-pruned JSON under your byte cap; "
        "it does not train ``nrl`` weights."
    )


def interpret(user_text: str) -> str:
    """Map free-form English to a single reply string."""
    raw = user_text.strip()
    if not raw:
        return "Say something, or type ``help``."
    t = raw.lower()

    if any(k in t for k in ("quit", "exit", "bye")):
        return "__QUIT__"

    if os.environ.get("NRL_LEARN_DISABLE", "").lower() not in ("1", "true", "yes"):
        try:
            default_store().observe_text(raw)
        except OSError:
            pass

    if "help" in t or t == "?":
        return _reply_help()

    if re.search(r"\b(temp|thermal|temperature|hot)\b", t):
        return _reply_temperature()

    if re.search(r"\b(speed|throughput|gops|gn/s|ops|how fast)\b", t):
        return _reply_throughput()

    if re.search(r"\b(version|who are you|what are you)\b", t):
        return f"Engine: {runtime.version()}"

    if re.search(r"\b(features?|cpu|avx|simd)\b", t):
        return "CPU features:\n" + json.dumps(runtime.features(), indent=2)

    if re.search(r"\b(nrl|binary|where).*\b(path|exe)\b", t) or "where nrl" in t:
        return f"Resolved ``nrl`` path: {runtime.nrl_binary_path()}"

    if re.search(r"\b(activity|log|events|immune|been up)\b", t):
        return _reply_activity()

    if re.search(r"\b(recall|session|what did i)\b", t):
        return _reply_recall()

    if re.search(
        r"\b(how much|grown|growth|vocabulary|vocab store|learn budget|disk use|word count)\b",
        t,
    ):
        return default_store().stats().summary()

    if re.search(r"\b(plasticity|shadow|specialization)\b", t):
        return _reply_learn_architecture()

    if re.search(r"\b(learn|learning)\b", t):
        return _reply_learn_architecture() + "\n\n" + default_store().stats().summary()

    if re.search(r"\b(how are you|hows it going|how's it going)\b", t):
        feats = runtime.features()
        n = sum(1 for v in feats.values() if v)
        return (
            f"I am a deterministic status shim, not an agent. Engine `{runtime.version()}`; "
            f"{n} CPU feature flags true. Ask ``throughput`` or ``temperature`` for probes."
        )

    if re.search(r"\b(status|overview)\b", t):
        feats = runtime.features()
        n_true = sum(1 for v in feats.values() if v)
        return (
            f"Version: {runtime.version()}\n"
            f"``nrl`` path: {runtime.nrl_binary_path()}\n"
            f"CPU features on: {n_true} / {len(feats)}"
        )

    return (
        "I only understand a small fixed vocabulary about NRL status. "
        "Try: ``temperature``, ``throughput``, ``version``, ``features``, "
        "``activity``, ``recall``, ``growth``, ``learn``, or ``help``. "
        "Unknown phrases still add tokens to your capped learn store."
    )


def run_chat_loop(stream_in: Any = None, stream_out: Any = None) -> int:
    """Interactive REPL; streams default to stdin/stdout."""
    sin = stream_in or sys.stdin
    sout = stream_out or sys.stdout
    banner = (
        f"nrlpy chat — control-plane status ({runtime.version()})\n"
        "Type ``help`` or ``quit``. Not an autonomous agent; deterministic telemetry.\n"
    )
    sout.write(banner)
    sout.flush()
    while True:
        sout.write("you> ")
        sout.flush()
        line = sin.readline()
        if line == "":
            break
        raw = line.rstrip("\n\r")
        reply = interpret(raw)
        if reply == "__QUIT__":
            sout.write("bye.\n")
            break
        _append_session("user", raw)
        _append_session("assistant", reply)
        sout.write(reply + "\n\n")
        sout.flush()
    return 0


def main_chat(argv: Sequence[str] | None = None) -> int:
    """Interactive REPL, or one-shot: ``nrlpy chat --one \"version\"``."""
    args = list(argv) if argv is not None else sys.argv[2:]
    if args and args[0] == "--one" and len(args) >= 2:
        print(interpret(" ".join(args[1:])))
        return 0
    return run_chat_loop()
