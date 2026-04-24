# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI compose: Omega-routed fragment walk -> streamed reply.

Stage 4 of the NRL-AI rebuild. Takes a starting fragment id (usually
the best hit returned by the resolver) and walks the ZPM transition
map to produce a coherent multi-sentence reply.

Omega routing semantics (honest port of nrl_v1_omega on the lattice):
  * At each source fragment, enumerate outgoing edges in
    ``(desc count, asc dst)`` order. Frequent transitions wake first;
    ties break by ascending destination id for host-independent replay.
  * Destinations already visited this reply are pruned -- same
    wake/prune discipline as the Omega router, producing loop-free
    walks without wall-clock or RNG input.

End-of-reply gates (first to trip wins, always includes the start):

  * ``sentence_cap``   emitted >= max_sentences fragments
  * ``char_cap``       accumulated >= max_chars characters
  * ``soft_end``       current fragment ends with a sentence terminator
                        AND sentence_count >= min_sentences ("natural stop")
  * ``terminal``       no outgoing transitions at all (end of corpus chain)
  * ``cycle_pruned``   all outgoing transitions lead to visited fragments

Determinism contract: given identical (start_id, index, caps), the walk
is byte-stable across hosts. Shared engine with ``compose_stream`` so
the REPL in prompt #5 can emit fragment-by-fragment without duplicating
the state machine.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from .nrl_ai import NrlAiPaths
from .nrl_ai_ingest import load_fragments, load_transitions
from .nrl_ai_resolve import ResolveResult

DEFAULT_MIN_SENTENCES = 2
DEFAULT_MAX_SENTENCES = 4
DEFAULT_MAX_CHARS = 400

# Characters that mark the end of a "natural" sentence for the soft-end
# heuristic. Includes punctuation and closing quotes/brackets so we
# don't stop mid-clause on something like ``"...said he.``.
_TERMINATORS = frozenset('.!?\u2026"\')\u201d\u2019]')


@dataclass(frozen=True)
class ComposeStep:
    """One fragment emitted during the walk.

    ``stop_reason`` is empty for every step except the final one,
    which carries the gate that ended the walk.
    """

    fragment_id: int
    fragment_text: str
    stop_reason: str = ""


@dataclass
class ComposeResult:
    """Full result of an Omega-routed compose pass."""

    reply: str
    steps: list[ComposeStep]
    stop_reason: str
    start_fragment_id: int
    sentence_count: int
    char_count: int
    elapsed_s: float
    hit: bool

    def step_ids(self) -> list[int]:
        return [s.fragment_id for s in self.steps]


def _ends_with_terminator(s: str) -> bool:
    s = s.rstrip()
    return bool(s) and s[-1] in _TERMINATORS


def _build_transition_index(
    transitions: dict[tuple[int, int], int],
) -> dict[int, list[tuple[int, int]]]:
    """src -> [(dst, count), ...] sorted by ``(desc count, asc dst)``."""
    out: dict[int, list[tuple[int, int]]] = {}
    for (src, dst), count in transitions.items():
        out.setdefault(src, []).append((dst, count))
    for lst in out.values():
        lst.sort(key=lambda x: (-x[1], x[0]))
    return out


def _omega_next(
    src: int,
    index: dict[int, list[tuple[int, int]]],
    visited: set[int],
) -> tuple[Optional[int], bool]:
    """Return ``(next_dst, had_candidates)`` under Omega wake/prune rules.

    * ``had_candidates`` is True iff ``src`` has any outgoing edge at
      all -- lets callers distinguish ``terminal`` (no edges) from
      ``cycle_pruned`` (edges existed but all visited).
    * ``next_dst`` is the first unvisited destination in the ranked
      order, or ``None`` if every candidate was already visited.
    """
    candidates = index.get(src)
    if not candidates:
        return None, False
    for dst, _count in candidates:
        if dst not in visited:
            return dst, True
    return None, True


def _walk(
    start_id: int,
    fragments: list[str],
    index: dict[int, list[tuple[int, int]]],
    *,
    min_sentences: int,
    max_sentences: int,
    max_chars: int,
) -> Iterator[ComposeStep]:
    """Core state machine shared by ``compose`` and ``compose_stream``.

    Yields ``ComposeStep`` instances; the last yielded step carries the
    non-empty ``stop_reason``.
    """
    visited: set[int] = set()
    sentence_count = 0
    char_count = 0
    cur = start_id
    while True:
        text = fragments[cur]
        visited.add(cur)
        sentence_count += 1
        char_count += len(text) + 1  # +1 for the join separator

        stop_reason: str
        if sentence_count >= max_sentences:
            stop_reason = "sentence_cap"
        elif char_count >= max_chars:
            stop_reason = "char_cap"
        elif sentence_count >= min_sentences and _ends_with_terminator(text):
            stop_reason = "soft_end"
        else:
            nxt, had = _omega_next(cur, index, visited)
            if nxt is None:
                stop_reason = "terminal" if not had else "cycle_pruned"
            else:
                yield ComposeStep(fragment_id=cur, fragment_text=text, stop_reason="")
                cur = nxt
                continue

        yield ComposeStep(fragment_id=cur, fragment_text=text, stop_reason=stop_reason)
        return


def compose(
    best_fragment_id: int,
    *,
    paths: NrlAiPaths,
    hit: bool = True,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ComposeResult:
    """Compose a reply starting from ``best_fragment_id`` via Omega routing.

    Deterministic and host-independent. Raises ``ValueError`` if
    ``best_fragment_id`` is outside the corpus.
    """
    t0 = time.perf_counter()
    fragments = load_fragments(paths)
    transitions = load_transitions(paths)
    if best_fragment_id < 0 or best_fragment_id >= len(fragments):
        raise ValueError(
            f"fragment id {best_fragment_id} out of range [0, {len(fragments)})"
        )
    index = _build_transition_index(transitions)

    steps = list(
        _walk(
            best_fragment_id,
            fragments,
            index,
            min_sentences=max(1, min_sentences),
            max_sentences=max(1, max_sentences),
            max_chars=max(1, max_chars),
        )
    )
    reply = " ".join(s.fragment_text for s in steps).strip()
    stop_reason = steps[-1].stop_reason if steps else "empty"
    return ComposeResult(
        reply=reply,
        steps=steps,
        stop_reason=stop_reason,
        start_fragment_id=best_fragment_id,
        sentence_count=len(steps),
        char_count=len(reply),
        elapsed_s=time.perf_counter() - t0,
        hit=hit,
    )


def compose_stream(
    best_fragment_id: int,
    *,
    paths: NrlAiPaths,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Iterator[str]:
    """Yield reply fragments one at a time for streamed REPL output.

    Used by prompt #5's chat surface to feel incremental without holding
    the whole reply in memory. Same state machine as ``compose``, so
    byte-stable sentences land in the same order.
    """
    fragments = load_fragments(paths)
    transitions = load_transitions(paths)
    if best_fragment_id < 0 or best_fragment_id >= len(fragments):
        raise ValueError(
            f"fragment id {best_fragment_id} out of range [0, {len(fragments)})"
        )
    index = _build_transition_index(transitions)
    for step in _walk(
        best_fragment_id,
        fragments,
        index,
        min_sentences=max(1, min_sentences),
        max_sentences=max(1, max_sentences),
        max_chars=max(1, max_chars),
    ):
        yield step.fragment_text


def compose_from_resolve(
    resolve_result: ResolveResult,
    *,
    paths: NrlAiPaths,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Optional[ComposeResult]:
    """Glue between resolver output and the composer.

    Returns ``None`` when the resolver reported a miss so the chat glue
    in prompt #5 can fall back to an honest "I don't recognize that"
    surface instead of synthesizing from an out-of-threshold fragment.
    """
    if not resolve_result.hit or resolve_result.best is None:
        return None
    return compose(
        resolve_result.best.fragment_id,
        paths=paths,
        hit=True,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        max_chars=max_chars,
    )


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_SENTENCES",
    "DEFAULT_MIN_SENTENCES",
    "ComposeResult",
    "ComposeStep",
    "compose",
    "compose_from_resolve",
    "compose_stream",
]
