# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI: pure-NRL-lattice inference path (no libllama, no GPU).

Contract
--------
NRL-AI is a retrieval + associative-composition engine that lives entirely
inside NRL primitives:

  * ZPM (Plane A.5) anchors every user turn to a 256-bit topological state.
  * A locality-sensitive anchor (SimHash over char-trigrams) replaces
    FNV-1a64 so that near-matches land in a small Hamming ball of the
    stored keys (Phase-2 of the architecture doc).
  * Omega sub-lattice routing composes the reply by walking a transition
    map compiled once from a corpus. No libllama, no GPU, no transformer.

The three pipeline stages land across the next prompts:

  Prompt #2 -- ingest    : corpus.txt -> fragment table + ZPM transitions
  Prompt #3 -- resolve   : SimHash anchors + nullspace hit at real locality
  Prompt #4 -- compose   : Omega-routed fragment walk -> streamed reply
  Prompt #5 -- REPL      : polished chat surface (NRL-AI becomes default)
  Prompt #6 -- bench     : wps_gate in pure-NRL mode, >=1000 wps contract
  Prompt #7 -- docs      : README + seed corpus + out-of-box demo

This module ships the CLI surface, the on-disk index layout, and the
contract types as first-class code so the later prompts only add
implementations -- no restructuring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

NRL_AI_VERSION = "0.1.0-scaffold"
NRL_AI_INDEX_VERSION = 1
NRL_AI_WPS_TARGET = 1000  # Prompt #6 acceptance threshold.

_INDEX_MANIFEST_NAME = "manifest.json"
_FRAGMENTS_NAME = "fragments.json"
_TRANSITIONS_NAME = "transitions.bin"
_CORPUS_NAME = "corpus.bin"
_ANCHOR_INDEX_NAME = "anchors.bin"


@dataclass(frozen=True)
class NrlAiPaths:
    """On-disk layout for an NRL-AI index.

    Resolution order for the default index root:
      1. ``NRL_AI_INDEX`` environment variable (explicit override).
      2. ``NRL_ROOT/cache/nrl_ai`` when NRL is installed.
      3. ``~/.nrl/nrl_ai`` as the fallback so no install is required.
    """

    root: Path

    @classmethod
    def default(cls) -> NrlAiPaths:
        env = os.environ.get("NRL_AI_INDEX", "").strip()
        if env:
            return cls(Path(env))
        nrl_root = os.environ.get("NRL_ROOT", "").strip()
        if nrl_root:
            return cls(Path(nrl_root) / "cache" / "nrl_ai")
        return cls(Path.home() / ".nrl" / "nrl_ai")

    @property
    def manifest(self) -> Path:
        return self.root / _INDEX_MANIFEST_NAME

    @property
    def fragments(self) -> Path:
        return self.root / _FRAGMENTS_NAME

    @property
    def transitions(self) -> Path:
        return self.root / _TRANSITIONS_NAME

    @property
    def corpus(self) -> Path:
        return self.root / _CORPUS_NAME

    @property
    def anchors(self) -> Path:
        return self.root / _ANCHOR_INDEX_NAME

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.manifest.is_file()


@dataclass
class NrlAiIndexManifest:
    """Schema-gated description of a compiled NRL-AI index."""

    version: int = NRL_AI_INDEX_VERSION
    corpus_sha256: str = ""
    fragment_count: int = 0
    transition_count: int = 0
    simhash_bits: int = 256
    created_utc: str = ""
    source_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "nrl_ai.index.v1",
            "version": self.version,
            "corpus_sha256": self.corpus_sha256,
            "fragment_count": self.fragment_count,
            "transition_count": self.transition_count,
            "simhash_bits": self.simhash_bits,
            "created_utc": self.created_utc,
            "source_path": self.source_path,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NrlAiIndexManifest:
        schema = str(data.get("schema", ""))
        if schema and schema != "nrl_ai.index.v1":
            raise ValueError(f"unsupported nrl-ai index schema: {schema!r}")
        return cls(
            version=int(data.get("version", 0)),
            corpus_sha256=str(data.get("corpus_sha256", "")),
            fragment_count=int(data.get("fragment_count", 0)),
            transition_count=int(data.get("transition_count", 0)),
            simhash_bits=int(data.get("simhash_bits", 256)),
            created_utc=str(data.get("created_utc", "")),
            source_path=str(data.get("source_path", "")),
        )

    @classmethod
    def load(cls, path: Path) -> NrlAiIndexManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_json(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NrlAiUnimplemented(RuntimeError):
    """Raised by scaffolded stages that land in later prompts.

    The message always names the exact prompt index so the user (and any
    automation) can see where a stage is due. This is a typed error so
    tests can assert on it without string matching.
    """

    def __init__(self, stage: str, prompt_no: int, scope: str) -> None:
        self.stage = stage
        self.prompt_no = prompt_no
        self.scope = scope
        super().__init__(
            f"nrl-ai {stage}: lands in prompt #{prompt_no} ({scope}). "
            f"run `nrl-ai status` for the current scaffold state."
        )


def cmd_ingest(args: argparse.Namespace) -> int:
    from . import nrl_ai_ingest as _ingest

    show_progress = sys.stderr.isatty()
    try:
        result = _ingest.ingest(
            args.corpus,
            out_dir=args.out,
            progress=show_progress,
        )
    except FileNotFoundError as exc:
        print(f"nrl-ai ingest: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"nrl-ai ingest: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "schema": "nrl_ai.ingest.v1",
        "status": "ok",
        "index_root": str(result.paths.root),
        "fragment_count": result.fragment_count,
        "transition_count": result.transition_count,
        "corpus_sha256": result.corpus_sha256,
        "simhash_bits": result.manifest.simhash_bits,
        "elapsed_s": round(result.elapsed_s, 6),
        "wps_target": NRL_AI_WPS_TARGET,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from . import nrl_ai_resolve as _resolve

    paths = (
        NrlAiPaths(Path(args.index)) if getattr(args, "index", None) else NrlAiPaths.default()
    )
    threshold = (
        int(args.threshold)
        if getattr(args, "threshold", None) is not None
        else _resolve.DEFAULT_THRESHOLD_BITS
    )
    try:
        result = _resolve.resolve(
            args.query,
            paths=paths,
            threshold_bits=threshold,
            top_k=int(args.top),
        )
    except FileNotFoundError as exc:
        print(f"nrl-ai resolve: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"nrl-ai resolve: {exc}", file=sys.stderr)
        return 2

    def _hit_dict(h: _resolve.ResolveHit) -> dict[str, Any]:
        return {
            "fragment_id": h.fragment_id,
            "distance_bits": h.distance_bits,
            "within_threshold": h.within_threshold,
            "fragment_text": h.fragment_text,
        }

    payload: dict[str, Any] = {
        "schema": "nrl_ai.resolve.v1",
        "query": result.query,
        "query_simhash_hex": [f"{w:016x}" for w in result.query_simhash],
        "threshold_bits": result.threshold_bits,
        "simhash_bits": result.simhash_bits,
        "fragment_count": result.fragment_count,
        "scanned_count": result.scanned_count,
        "elapsed_s": round(result.elapsed_s, 6),
        "hit": result.hit,
        "best": _hit_dict(result.best) if result.best is not None else None,
        "top": [_hit_dict(h) for h in result.top],
    }
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def cmd_compose(args: argparse.Namespace) -> int:
    from . import nrl_ai_compose as _compose
    from . import nrl_ai_resolve as _resolve

    paths = (
        NrlAiPaths(Path(args.index)) if getattr(args, "index", None) else NrlAiPaths.default()
    )
    threshold = (
        int(args.threshold)
        if getattr(args, "threshold", None) is not None
        else _resolve.DEFAULT_THRESHOLD_BITS
    )
    try:
        rresult = _resolve.resolve(
            args.query,
            paths=paths,
            threshold_bits=threshold,
            top_k=1,
        )
    except FileNotFoundError as exc:
        print(f"nrl-ai compose: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"nrl-ai compose: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "schema": "nrl_ai.compose.v1",
        "query": rresult.query,
        "threshold_bits": rresult.threshold_bits,
        "simhash_bits": rresult.simhash_bits,
        "hit": rresult.hit,
        "best_fragment_id": rresult.best.fragment_id if rresult.best else None,
        "best_distance_bits": rresult.best.distance_bits if rresult.best else None,
    }
    if not rresult.hit or rresult.best is None:
        payload["reply"] = None
        payload["stop_reason"] = "resolver_miss"
        payload["sentence_count"] = 0
        payload["char_count"] = 0
        payload["steps"] = []
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    cresult = _compose.compose(
        rresult.best.fragment_id,
        paths=paths,
        hit=True,
        min_sentences=int(args.min_sentences),
        max_sentences=int(args.max_sentences),
        max_chars=int(args.max_chars),
    )
    payload.update(
        {
            "reply": cresult.reply,
            "stop_reason": cresult.stop_reason,
            "sentence_count": cresult.sentence_count,
            "char_count": cresult.char_count,
            "elapsed_s": round(cresult.elapsed_s, 6),
            "steps": [
                {
                    "fragment_id": s.fragment_id,
                    "fragment_text": s.fragment_text,
                    "stop_reason": s.stop_reason,
                }
                for s in cresult.steps
            ],
        }
    )
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from . import nrl_ai_chat as _chat

    paths = (
        NrlAiPaths(Path(args.index)) if getattr(args, "index", None) else NrlAiPaths.default()
    )
    threshold = (
        int(args.threshold) if getattr(args, "threshold", None) is not None else None
    )
    try:
        _chat.run_nrl_ai_chat_repl(
            paths,
            threshold_bits=threshold,
            min_sentences=int(args.min_sentences),
            max_sentences=int(args.max_sentences),
            max_chars=int(args.max_chars),
            chunk_delay_s=float(args.chunk_delay),
        )
    except FileNotFoundError as exc:
        print(f"nrl-ai chat: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import nrl_ai_bench as _bench

    paths = (
        NrlAiPaths(Path(args.index)) if getattr(args, "index", None) else NrlAiPaths.default()
    )
    threshold = (
        int(args.threshold) if getattr(args, "threshold", None) is not None else None
    )
    queries_file = (
        Path(args.queries_file) if getattr(args, "queries_file", None) else None
    )
    out_json = Path(args.out) if getattr(args, "out", None) else None
    return _bench.cli_main(
        paths,
        turns=int(args.turns),
        warmup=int(args.warmup),
        threshold_bits=threshold,
        target_wps=int(args.target),
        queries_file=queries_file,
        min_sentences=int(args.min_sentences),
        max_sentences=int(args.max_sentences),
        max_chars=int(args.max_chars),
        out_json=out_json,
    )


def _status_payload(paths: NrlAiPaths) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "nrl_ai.status.v1",
        "version": NRL_AI_VERSION,
        "phase": "scaffold",
        "index_root": str(paths.root),
        "index_ready": paths.exists(),
        "wps_target": NRL_AI_WPS_TARGET,
        "roadmap": [
            {"prompt": 1, "scope": "strip + scaffold", "status": "shipped"},
            {"prompt": 2, "scope": "ingest: corpus -> SimHash + ZPM transitions", "status": "shipped"},
            {"prompt": 3, "scope": "resolve: SimHash anchors + nullspace", "status": "shipped"},
            {"prompt": 4, "scope": "compose: Omega-routed fragment walk", "status": "shipped"},
            {"prompt": 5, "scope": "REPL: NRL-AI becomes default chat path", "status": "shipped"},
            {"prompt": 6, "scope": f"bench: >={NRL_AI_WPS_TARGET} wps gate", "status": "shipped"},
            {"prompt": 7, "scope": "docs + seed corpus + demo", "status": "shipped"},
        ],
    }
    if paths.exists():
        try:
            payload["manifest"] = NrlAiIndexManifest.load(paths.manifest).to_json()
        except (OSError, ValueError, json.JSONDecodeError) as e:
            payload["manifest_error"] = f"{type(e).__name__}: {e}"
    return payload


def cmd_status(args: argparse.Namespace) -> int:
    paths = NrlAiPaths(Path(args.index)) if getattr(args, "index", None) else NrlAiPaths.default()
    json.dump(_status_payload(paths), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Seed corpus (packaged Q&A for the out-of-box demo)
# ---------------------------------------------------------------------------

SEED_CORPUS_NAME = "seed_corpus.txt"


def seed_corpus_path() -> Path:
    """Return the on-disk path to the packaged seed corpus.

    The file ships under ``nrlpy/data/`` and is registered as package
    data in ``pyproject.toml``. We resolve it via ``__file__`` so it
    works under both editable installs and installed wheels without
    dragging in ``importlib.resources`` semantics.
    """
    p = Path(__file__).resolve().parent / "data" / SEED_CORPUS_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"NRL-AI seed corpus missing from package at {p} "
            "(reinstall nrlpy to restore it)"
        )
    return p


def cmd_demo(args: argparse.Namespace) -> int:
    """Out-of-box demo: ingest the packaged seed corpus, launch chat.

    If ``--index`` is given and already contains a manifest, the
    existing index is reused (no re-ingest). Otherwise we ingest the
    seed corpus into ``--index`` (or a default demo dir) and boot the
    REPL. ``--ingest-only`` prints the ingest result and exits so CI
    can assert the corpus compiles cleanly without opening a TTY loop.
    """
    from . import nrl_ai_chat as _chat
    from . import nrl_ai_ingest as _ingest

    try:
        seed = seed_corpus_path()
    except FileNotFoundError as exc:
        print(f"nrl-ai demo: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "index", None):
        paths = NrlAiPaths(Path(args.index))
    else:
        default = NrlAiPaths.default()
        paths = NrlAiPaths(default.root.parent / "nrl_ai_demo")

    if not paths.manifest.is_file():
        result = _ingest.ingest(seed, out_dir=paths.root)
        print(
            f"nrl-ai demo: ingested seed corpus "
            f"({result.fragment_count} fragments, "
            f"{result.transition_count} transitions) -> {paths.root}",
            file=sys.stderr,
        )
    else:
        print(
            f"nrl-ai demo: reusing existing index at {paths.root}",
            file=sys.stderr,
        )

    if getattr(args, "ingest_only", False):
        return 0

    try:
        _chat.run_nrl_ai_chat_repl(paths)
    except FileNotFoundError as exc:
        print(f"nrl-ai demo: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nrl-ai",
        description=(
            "NRL-AI: pure-NRL-lattice inference path (no libllama, no GPU). "
            f"target: >={NRL_AI_WPS_TARGET} words per second on CPU."
        ),
    )
    parser.add_argument("--version", action="version", version=f"nrl-ai {NRL_AI_VERSION}")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser(
        "ingest",
        help="build a ZPM transition index from a corpus text file (prompt #2).",
    )
    p_ingest.add_argument("corpus", help="path to UTF-8 corpus text file")
    p_ingest.add_argument(
        "--out",
        default=None,
        help="index output directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_resolve = sub.add_parser(
        "resolve",
        help=(
            "anchor a user turn with SimHash256 and return the nearest "
            "corpus fragment via Hamming nullspace search."
        ),
    )
    p_resolve.add_argument("query", help="user turn string to resolve")
    p_resolve.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Hamming threshold in bits (default: DEFAULT_THRESHOLD_BITS=96)",
    )
    p_resolve.add_argument(
        "--top",
        type=int,
        default=3,
        help="number of top-k matches to return (default 3)",
    )
    p_resolve.add_argument(
        "--index",
        default=None,
        help="override index directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_resolve.set_defaults(func=cmd_resolve)

    p_compose = sub.add_parser(
        "compose",
        help=(
            "resolve a query and compose a reply via Omega-routed fragment "
            "walk with deterministic end-of-reply detection."
        ),
    )
    p_compose.add_argument("query", help="user turn string to resolve and compose from")
    p_compose.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="resolver Hamming threshold in bits (default 96)",
    )
    p_compose.add_argument(
        "--min-sentences",
        dest="min_sentences",
        type=int,
        default=2,
        help="minimum sentences before soft-end stop (default 2)",
    )
    p_compose.add_argument(
        "--max-sentences",
        dest="max_sentences",
        type=int,
        default=4,
        help="hard ceiling on emitted sentences (default 4)",
    )
    p_compose.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=400,
        help="hard ceiling on reply characters (default 400)",
    )
    p_compose.add_argument(
        "--index",
        default=None,
        help="override index directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_compose.set_defaults(func=cmd_compose)

    p_chat = sub.add_parser(
        "chat",
        help="run the NRL-AI chat REPL (pure lattice, no libllama decode).",
    )
    p_chat.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="resolver Hamming threshold in bits (default 96)",
    )
    p_chat.add_argument(
        "--min-sentences",
        dest="min_sentences",
        type=int,
        default=2,
        help="minimum sentences before soft-end stop (default 2)",
    )
    p_chat.add_argument(
        "--max-sentences",
        dest="max_sentences",
        type=int,
        default=4,
        help="hard ceiling on emitted sentences (default 4)",
    )
    p_chat.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=400,
        help="hard ceiling on reply characters (default 400)",
    )
    p_chat.add_argument(
        "--chunk-delay",
        dest="chunk_delay",
        type=float,
        default=0.0,
        help="seconds to pause between streamed fragments (default 0)",
    )
    p_chat.add_argument(
        "--index",
        default=None,
        help="override index directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_bench = sub.add_parser(
        "bench",
        help=(
            "replay-locked WPS gate on the ingested corpus. "
            f"PASS iff wps_mean >= target (default {NRL_AI_WPS_TARGET})."
        ),
    )
    p_bench.add_argument(
        "--turns", type=int, default=16, help="measured turns (default 16)"
    )
    p_bench.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup turns excluded from aggregates (default 1)",
    )
    p_bench.add_argument(
        "--target",
        type=int,
        default=NRL_AI_WPS_TARGET,
        help=f"WPS gate threshold (default {NRL_AI_WPS_TARGET})",
    )
    p_bench.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="resolver Hamming threshold in bits (default 96)",
    )
    p_bench.add_argument(
        "--queries-file",
        dest="queries_file",
        default=None,
        help="optional path to a queries file (one per line). "
        "Default: deterministic sampling from the corpus.",
    )
    p_bench.add_argument(
        "--min-sentences",
        dest="min_sentences",
        type=int,
        default=2,
        help="minimum sentences before soft-end stop (default 2)",
    )
    p_bench.add_argument(
        "--max-sentences",
        dest="max_sentences",
        type=int,
        default=4,
        help="hard ceiling on emitted sentences (default 4)",
    )
    p_bench.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=400,
        help="hard ceiling on reply characters (default 400)",
    )
    p_bench.add_argument(
        "--out",
        default=None,
        help="write the JSON bench payload to this path in addition to stdout",
    )
    p_bench.add_argument(
        "--index",
        default=None,
        help="override index directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_bench.set_defaults(func=cmd_bench)

    p_status = sub.add_parser(
        "status",
        help="print scaffold + index status as JSON (always available).",
    )
    p_status.add_argument(
        "--index",
        default=None,
        help="override index directory (default: $NRL_AI_INDEX or ~/.nrl/nrl_ai)",
    )
    p_status.set_defaults(func=cmd_status)

    p_demo = sub.add_parser(
        "demo",
        help=(
            "out-of-box demo: ingest the packaged seed corpus and launch "
            "the NRL-AI chat REPL (no user corpus required)."
        ),
    )
    p_demo.add_argument(
        "--index",
        default=None,
        help="demo index directory (default: sibling of the regular index)",
    )
    p_demo.add_argument(
        "--ingest-only",
        dest="ingest_only",
        action="store_true",
        help="compile the seed corpus into the demo index and exit without chat",
    )
    p_demo.set_defaults(func=cmd_demo)

    return parser


def dispatch(args: list[str]) -> int:
    """Entry point used by ``nrlpy nrl-ai ...``.

    Returns a process exit code. Unimplemented stages return ``3`` so
    callers and CI can distinguish "not built yet" from generic errors.
    """
    parser = build_parser()
    ns = parser.parse_args(args)
    func: Callable[[argparse.Namespace], int] | None = getattr(ns, "func", None)
    if func is None:
        parser.print_help()
        return 0
    try:
        return int(func(ns))
    except NrlAiUnimplemented as exc:
        print(f"nrl-ai: {exc}", file=sys.stderr)
        return 3


__all__ = [
    "NRL_AI_INDEX_VERSION",
    "NRL_AI_VERSION",
    "NRL_AI_WPS_TARGET",
    "SEED_CORPUS_NAME",
    "NrlAiIndexManifest",
    "NrlAiPaths",
    "NrlAiUnimplemented",
    "build_parser",
    "cmd_bench",
    "cmd_chat",
    "cmd_compose",
    "cmd_demo",
    "cmd_ingest",
    "cmd_resolve",
    "cmd_status",
    "dispatch",
    "seed_corpus_path",
]
