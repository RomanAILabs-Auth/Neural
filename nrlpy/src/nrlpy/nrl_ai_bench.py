# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI bench: replay-locked throughput gate (>=1000 wps on CPU).

Stage 6 of the NRL-AI rebuild. This module proves -- with an honest,
host-independent, reproducible contract -- that the pure NRL lattice
path hits or exceeds :data:`~nrlpy.nrl_ai.NRL_AI_WPS_TARGET` words per
second on the user's own CPU.

Contract (``schema = "nrl_ai.bench.v1"``)
-----------------------------------------
1. **Deterministic query stream.** Queries are either supplied by the
   caller (``queries=`` / ``--queries-file``) or auto-sampled from the
   on-disk fragment table using a SHA-256 seed that folds in
   ``corpus_sha256``. Two bench runs on the same index therefore
   produce the same query list, so the *only* source of variation
   between runs is real silicon wall-time.

2. **Replay-locked hot path.** Per turn we call
   :func:`nrl_ai_resolve.resolve` followed by
   :func:`nrl_ai_compose.compose` exactly once. Neither function uses
   RNG or wall-clock state; both read from mmap-friendly binary files
   that are fully described by the corpus SHA. That is what lets us
   call the result "replay-locked on this host" without handwaving.

3. **Warmup exclusion.** The first ``warmup`` turns (default 1) are
   executed but excluded from the aggregate. This removes cold-cache
   / import-time noise from the throughput number.

4. **Gate.** ``wps_mean >= target_wps`` -> ``gate_pass = True``. The
   CLI maps that to exit code 0; failure is exit code 4 (distinct
   from 2=index missing, 3=stub).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import nrl_ai_compose as _compose
from . import nrl_ai_ingest as _ingest
from . import nrl_ai_resolve as _resolve
from .nrl_ai import (
    NRL_AI_INDEX_VERSION,
    NRL_AI_VERSION,
    NRL_AI_WPS_TARGET,
    NrlAiIndexManifest,
    NrlAiPaths,
)

BENCH_SCHEMA = "nrl_ai.bench.v1"
DEFAULT_TURNS = 16
DEFAULT_WARMUP = 1


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BenchTurn:
    """One measured turn of the replay-locked bench."""

    turn: int
    query: str
    hit: bool
    fragment_id: Optional[int]
    distance_bits: Optional[int]
    stop_reason: str
    words: int
    chars: int
    sentence_count: int
    wall_s: float
    wps: float
    warmup: bool

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["wall_s"] = round(self.wall_s, 9)
        d["wps"] = round(self.wps, 3)
        return d


@dataclass
class BenchResult:
    """Full ``nrl_ai.bench.v1`` payload."""

    corpus_sha256: str
    fragment_count: int
    threshold_bits: int
    warmup_turns: int
    measured_turns: int
    total_turns: int
    hits: int
    misses: int
    total_words: int
    total_chars: int
    total_wall_s: float
    wps_mean: float
    wps_p50: float
    wps_p95: float
    wps_min: float
    wps_max: float
    target_wps: int
    gate_pass: bool
    started_utc: str
    elapsed_s: float
    turns: list[BenchTurn] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": BENCH_SCHEMA,
            "version": NRL_AI_VERSION,
            "index_version": NRL_AI_INDEX_VERSION,
            "corpus_sha256": self.corpus_sha256,
            "fragment_count": self.fragment_count,
            "threshold_bits": self.threshold_bits,
            "warmup_turns": self.warmup_turns,
            "measured_turns": self.measured_turns,
            "total_turns": self.total_turns,
            "hits": self.hits,
            "misses": self.misses,
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "total_wall_s": round(self.total_wall_s, 9),
            "wps_mean": round(self.wps_mean, 3),
            "wps_p50": round(self.wps_p50, 3),
            "wps_p95": round(self.wps_p95, 3),
            "wps_min": round(self.wps_min, 3),
            "wps_max": round(self.wps_max, 3),
            "target_wps": self.target_wps,
            "gate_pass": self.gate_pass,
            "started_utc": self.started_utc,
            "elapsed_s": round(self.elapsed_s, 9),
            "env": self.env,
            "turns": [t.to_json() for t in self.turns],
        }


# ---------------------------------------------------------------------------
# Deterministic query selection
# ---------------------------------------------------------------------------


def _seed_for_turn(corpus_sha256: str, turn: int) -> int:
    """SHA-256(corpus_sha||'|'||turn) truncated to 64 bits, little-endian.

    Pure function of (corpus, turn_index). No RNG state, no wall clock.
    """
    src = f"{corpus_sha256}|{turn}".encode("utf-8")
    digest = hashlib.sha256(src).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _mutate_query(fragment: str) -> str:
    """Mutate a fragment so the resolver does near-match work.

    Strategy: lowercase + drop trailing punctuation + drop the last
    word. That keeps the query well inside the default 96-bit Hamming
    threshold on SimHash256 (the overlap is large) but exercises the
    *near-match* path rather than the exact-hash short-circuit. If the
    fragment is too short to drop a word we just lowercase.
    """
    cleaned = fragment.rstrip(" .!?,:;")
    words = cleaned.split()
    if len(words) > 2:
        return " ".join(words[:-1]).lower()
    return cleaned.lower() or fragment.lower()


def select_queries(
    fragments: list[str],
    *,
    total_turns: int,
    corpus_sha256: str,
) -> list[str]:
    """Deterministic, replay-locked query list for ``total_turns`` turns."""
    if not fragments:
        raise ValueError("cannot build bench queries: fragment list is empty")
    out: list[str] = []
    for t in range(total_turns):
        seed = _seed_for_turn(corpus_sha256, t)
        idx = seed % len(fragments)
        out.append(_mutate_query(fragments[idx]))
    return out


# ---------------------------------------------------------------------------
# Environment capture (host-independent schema, host-specific values)
# ---------------------------------------------------------------------------


def _env_snapshot() -> dict[str, Any]:
    try:
        import os

        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "nrl_ai_version": NRL_AI_VERSION,
        "nrl_ai_index_version": NRL_AI_INDEX_VERSION,
    }


# ---------------------------------------------------------------------------
# Main bench loop
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def run_bench(
    paths: Optional[NrlAiPaths] = None,
    *,
    turns: int = DEFAULT_TURNS,
    warmup: int = DEFAULT_WARMUP,
    threshold_bits: Optional[int] = None,
    target_wps: int = NRL_AI_WPS_TARGET,
    queries: Optional[list[str]] = None,
    min_sentences: int = _compose.DEFAULT_MIN_SENTENCES,
    max_sentences: int = _compose.DEFAULT_MAX_SENTENCES,
    max_chars: int = _compose.DEFAULT_MAX_CHARS,
) -> BenchResult:
    """Run the replay-locked WPS bench and return a :class:`BenchResult`.

    Args:
        paths: index paths (defaults to ``NRL_AI_INDEX`` / ``~/.nrl/nrl_ai``).
        turns: number of *measured* turns (warmup is additional).
        warmup: turns executed but excluded from aggregates.
        threshold_bits: resolver Hamming threshold (default 96).
        target_wps: pass/fail gate (default :data:`NRL_AI_WPS_TARGET`).
        queries: optional override for the query stream. When None, the
            bench deterministically samples ``turns+warmup`` queries from
            the corpus fragments using ``corpus_sha256`` as the seed.

    Raises:
        FileNotFoundError: when the index is missing (caller maps to
            exit code 2).
        ValueError: when ``turns <= 0``, when ``warmup < 0``, or when
            the corpus has zero fragments.
    """
    if turns <= 0:
        raise ValueError(f"turns must be positive (got {turns})")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative (got {warmup})")

    paths = paths or NrlAiPaths.default()
    if not paths.manifest.is_file():
        raise FileNotFoundError(
            f"no NRL-AI index at {paths.root} "
            f"(run `nrl-ai ingest <corpus>` first)"
        )

    manifest = NrlAiIndexManifest.load(paths.manifest)
    fragments = _ingest.load_fragments(paths)
    thr = threshold_bits if threshold_bits is not None else _resolve.DEFAULT_THRESHOLD_BITS

    total = turns + warmup
    if queries is None:
        stream = select_queries(
            fragments,
            total_turns=total,
            corpus_sha256=manifest.corpus_sha256,
        )
    else:
        stream = list(queries)
        if len(stream) < total:
            raise ValueError(
                f"provided queries={len(stream)} is less than "
                f"turns+warmup={total}"
            )
        stream = stream[:total]

    started = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_wall_start = time.perf_counter()

    turn_rows: list[BenchTurn] = []
    for t_idx, query in enumerate(stream):
        is_warmup = t_idx < warmup
        t0 = time.perf_counter()
        rresult = _resolve.resolve(
            query, paths=paths, threshold_bits=thr, top_k=1
        )
        if rresult.hit and rresult.best is not None:
            cresult = _compose.compose(
                rresult.best.fragment_id,
                paths=paths,
                hit=True,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                max_chars=max_chars,
            )
            wall_s = time.perf_counter() - t0
            words = len(cresult.reply.split())
            chars = cresult.char_count
            wps = (words / wall_s) if wall_s > 0 else 0.0
            turn_rows.append(
                BenchTurn(
                    turn=t_idx,
                    query=query,
                    hit=True,
                    fragment_id=rresult.best.fragment_id,
                    distance_bits=rresult.best.distance_bits,
                    stop_reason=cresult.stop_reason,
                    words=words,
                    chars=chars,
                    sentence_count=cresult.sentence_count,
                    wall_s=wall_s,
                    wps=wps,
                    warmup=is_warmup,
                )
            )
        else:
            wall_s = time.perf_counter() - t0
            turn_rows.append(
                BenchTurn(
                    turn=t_idx,
                    query=query,
                    hit=False,
                    fragment_id=None,
                    distance_bits=(
                        rresult.best.distance_bits if rresult.best else None
                    ),
                    stop_reason="resolver_miss",
                    words=0,
                    chars=0,
                    sentence_count=0,
                    wall_s=wall_s,
                    wps=0.0,
                    warmup=is_warmup,
                )
            )

    elapsed = time.perf_counter() - t_wall_start

    measured = [r for r in turn_rows if not r.warmup]
    hits = sum(1 for r in measured if r.hit)
    misses = sum(1 for r in measured if not r.hit)
    total_words = sum(r.words for r in measured)
    total_chars = sum(r.chars for r in measured)
    total_wall_s = sum(r.wall_s for r in measured)
    wps_mean = (total_words / total_wall_s) if total_wall_s > 0 else 0.0

    per_turn_wps_hits = sorted(r.wps for r in measured if r.hit)
    wps_p50 = _percentile(per_turn_wps_hits, 0.50)
    wps_p95 = _percentile(per_turn_wps_hits, 0.95)
    wps_min = per_turn_wps_hits[0] if per_turn_wps_hits else 0.0
    wps_max = per_turn_wps_hits[-1] if per_turn_wps_hits else 0.0

    gate_pass = wps_mean >= target_wps

    return BenchResult(
        corpus_sha256=manifest.corpus_sha256,
        fragment_count=manifest.fragment_count or len(fragments),
        threshold_bits=thr,
        warmup_turns=warmup,
        measured_turns=len(measured),
        total_turns=len(turn_rows),
        hits=hits,
        misses=misses,
        total_words=total_words,
        total_chars=total_chars,
        total_wall_s=total_wall_s,
        wps_mean=wps_mean,
        wps_p50=wps_p50,
        wps_p95=wps_p95,
        wps_min=wps_min,
        wps_max=wps_max,
        target_wps=target_wps,
        gate_pass=gate_pass,
        started_utc=started,
        elapsed_s=elapsed,
        turns=turn_rows,
        env=_env_snapshot(),
    )


# ---------------------------------------------------------------------------
# Human-readable summary (stderr-side, for local dev)
# ---------------------------------------------------------------------------


def format_summary(result: BenchResult) -> str:
    verdict = "PASS" if result.gate_pass else "FAIL"
    delta = result.wps_mean - result.target_wps
    sign = "+" if delta >= 0 else ""
    lines = [
        f"NRL-AI bench {verdict}  (target >= {result.target_wps} wps)",
        f"  measured_turns : {result.measured_turns} "
        f"(warmup {result.warmup_turns}, total {result.total_turns})",
        f"  hits / misses  : {result.hits} / {result.misses}",
        f"  total_words    : {result.total_words}",
        f"  total_wall_s   : {result.total_wall_s:.6f}",
        f"  wps_mean       : {result.wps_mean:,.1f}  "
        f"({sign}{delta:,.1f} vs target)",
        f"  wps p50/p95    : {result.wps_p50:,.1f} / {result.wps_p95:,.1f}",
        f"  wps min/max    : {result.wps_min:,.1f} / {result.wps_max:,.1f}",
        f"  corpus_sha256  : {result.corpus_sha256[:16]}...",
        f"  fragments      : {result.fragment_count}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Queries-file helper (JSONL or plain text, one per line)
# ---------------------------------------------------------------------------


def load_queries_file(path: Path) -> list[str]:
    """Parse a queries file. One query per non-blank line.

    ``#``-prefixed lines are treated as comments.
    """
    if not path.is_file():
        raise FileNotFoundError(f"queries file not found: {path}")
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if not out:
        raise ValueError(f"queries file is empty (no usable lines): {path}")
    return out


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------


def cli_main(
    paths: NrlAiPaths,
    *,
    turns: int,
    warmup: int,
    threshold_bits: Optional[int],
    target_wps: int,
    queries_file: Optional[Path],
    min_sentences: int,
    max_sentences: int,
    max_chars: int,
    out_json: Optional[Path] = None,
    stdout=None,
    stderr=None,
) -> int:
    """Entry point used by ``nrl-ai bench``.

    Returns exit code: 0 on PASS, 4 on FAIL, 2 on missing index.
    """
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    queries = load_queries_file(queries_file) if queries_file else None
    try:
        result = run_bench(
            paths,
            turns=turns,
            warmup=warmup,
            threshold_bits=threshold_bits,
            target_wps=target_wps,
            queries=queries,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )
    except FileNotFoundError as exc:
        print(f"nrl-ai bench: {exc}", file=stderr)
        return 2

    payload = result.to_json()
    json.dump(payload, stdout, indent=2, ensure_ascii=False)
    stdout.write("\n")
    stdout.flush()
    stderr.write(format_summary(result) + "\n")
    stderr.flush()

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return 0 if result.gate_pass else 4


__all__ = [
    "BENCH_SCHEMA",
    "BenchResult",
    "BenchTurn",
    "DEFAULT_TURNS",
    "DEFAULT_WARMUP",
    "cli_main",
    "format_summary",
    "load_queries_file",
    "run_bench",
    "select_queries",
]
