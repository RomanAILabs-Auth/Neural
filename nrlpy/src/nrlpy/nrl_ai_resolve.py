# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI resolve: SimHash-anchored query -> nearest corpus fragment.

Stage 3 of the NRL-AI rebuild. Ships the Phase-2 locality path that the
architecture doc calls out but the FNV-1a64 anchor couldn't deliver:

  * The user turn is anchored with the same ``simhash256`` that compiled
    the corpus, so near-matches (word edits, tense flips, typos) sit in
    a small Hamming ball of a stored fragment.
  * A linear Hamming nullspace scan returns the top-k closest fragments
    sorted by distance. At realistic corpus sizes (<1M fragments) this
    is a few MB of packed anchors and scales as micro-to-milliseconds.
  * A configurable ``threshold_bits`` gates what counts as a "hit",
    preserving the ZPM honesty contract: miss = return no fragment.

Determinism contract:
  * Same ``query`` + same index -> same ``ResolveResult`` (byte-stable).
  * Tie-break on equal distance is by ascending fragment id so replay
    locks across hosts.

This module is pure CPU, stdlib-only, O(N) in anchors scanned. Prompt #4
(composer) consumes ``ResolveResult.top`` to walk the ZPM transition
map and stream a reply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .nrl_ai import NrlAiPaths
from .nrl_ai_ingest import (
    SIMHASH_BITS,
    hamming_distance_simhash,
    load_anchors,
    load_fragments,
    simhash256,
)

# Empirically, single-word edits over SAMPLE_CORPUS-sized fragments sit
# under ~60 Hamming bits at SIMHASH_BITS=256. 96 is a loose-but-safe
# default threshold: generous enough to catch "hi" vs "hey" but far from
# the 128-bit orthogonal band. Users can tighten with ``--threshold``.
DEFAULT_THRESHOLD_BITS = 96


@dataclass(frozen=True)
class ResolveHit:
    """A single nullspace-search result.

    ``within_threshold`` is True iff ``distance_bits <= threshold_bits``
    at the time of the resolve call -- callers can still inspect near
    misses when ``top_k > 1`` without the ZPM hit contract applying.
    """

    fragment_id: int
    fragment_text: str
    distance_bits: int
    simhash_words: tuple[int, int, int, int]
    within_threshold: bool


@dataclass
class ResolveResult:
    """Full result of a single resolve call."""

    query: str
    query_simhash: tuple[int, int, int, int]
    threshold_bits: int
    simhash_bits: int
    fragment_count: int
    scanned_count: int
    best: Optional[ResolveHit]
    top: list[ResolveHit]
    elapsed_s: float

    @property
    def hit(self) -> bool:
        """True if the best match is inside the configured threshold."""
        return self.best is not None and self.best.within_threshold


def resolve(
    query: str,
    *,
    paths: Optional[NrlAiPaths] = None,
    threshold_bits: int = DEFAULT_THRESHOLD_BITS,
    top_k: int = 1,
) -> ResolveResult:
    """Anchor ``query`` with SimHash256 and find the nearest stored fragment.

    Raises ``FileNotFoundError`` if no NRL-AI index is present at
    ``paths`` (or the default location). Raises ``RuntimeError`` if the
    on-disk anchors/fragments counts disagree -- an integrity check that
    catches half-written indexes before they propagate a wrong match.
    """
    t0 = time.perf_counter()
    if paths is None:
        paths = NrlAiPaths.default()
    if not paths.manifest.is_file():
        raise FileNotFoundError(
            f"no NRL-AI index at {paths.root} "
            f"(run `nrl-ai ingest <corpus>` first)"
        )

    anchors = load_anchors(paths)
    fragments = load_fragments(paths)
    if len(anchors) != len(fragments):
        raise RuntimeError(
            f"nrl-ai index inconsistency at {paths.root}: "
            f"{len(anchors)} anchors vs {len(fragments)} fragments"
        )

    q = simhash256(query)

    # Linear scan -- O(N) in fragments, 32 bytes per anchor. At N=100k
    # this is ~3.2 MB of bytes and sub-millisecond in practice.
    scored: list[tuple[int, int]] = []
    for i, a in enumerate(anchors):
        scored.append((hamming_distance_simhash(q, a), i))
    # Stable tie-break on fragment id so replay is host-independent.
    scored.sort(key=lambda item: (item[0], item[1]))

    k = max(1, int(top_k))
    top: list[ResolveHit] = []
    for d, fid in scored[:k]:
        top.append(
            ResolveHit(
                fragment_id=fid,
                fragment_text=fragments[fid],
                distance_bits=d,
                simhash_words=anchors[fid],
                within_threshold=d <= threshold_bits,
            )
        )

    best = top[0] if top else None
    return ResolveResult(
        query=query,
        query_simhash=q,
        threshold_bits=threshold_bits,
        simhash_bits=SIMHASH_BITS,
        fragment_count=len(fragments),
        scanned_count=len(anchors),
        best=best,
        top=top,
        elapsed_s=time.perf_counter() - t0,
    )


__all__ = [
    "DEFAULT_THRESHOLD_BITS",
    "ResolveHit",
    "ResolveResult",
    "resolve",
]
