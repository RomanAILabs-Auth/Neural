# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-AI ingest: corpus text -> SimHash-anchored ZPM transition map.

Given a UTF-8 text corpus, produce a deterministic on-disk index:

  * ``fragments.json``  -- ordered fragment table (id, text, char_len)
  * ``anchors.bin``     -- packed 256-bit SimHash anchors (4 * uint64 per frag)
  * ``transitions.bin`` -- packed (src_u32, dst_u32, count_u32) transitions
  * ``manifest.json``   -- schema-gated index metadata (``nrl_ai.index.v1``)

The SimHash anchor is **locality-sensitive** over character trigrams
(Phase-2 of the architecture doc): two strings that share most
trigrams land within a small Hamming distance of each other, which is
what makes ZPM nullspace search (prompt #3) meaningful for user turns
that are *similar* to stored questions rather than byte-identical.

Determinism contract:
  * Fragment order = order of appearance in the corpus.
  * SimHash = fixed trigram window + fixed projection seeds.
  * Transitions = insertion order of consecutive fragment pairs.

No external dependencies -- stdlib only. This ships in prompt #2 of
the NRL-AI rebuild.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .nrl_ai import NrlAiIndexManifest, NrlAiPaths, _now_utc

SIMHASH_BITS = 256
SIMHASH_WORDS = SIMHASH_BITS // 64  # 4

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

# Independent 64-bit projection salts -- one per word of the 256-bit anchor.
# Same golden-ratio constant used in the ZPM inversion stage to keep the
# whole identity resolver family aligned.
_PROJECTION_SEEDS: tuple[int, int, int, int] = (
    0x0000000000000000,
    0x9E3779B97F4A7C15,
    0xCA62C1D1A7A57B43,
    0x5A827999EB5A7B89,
)

_TRIGRAM = 3

# Sentence splitter -- punctuation/newline followed by whitespace + uppercase
# or digit or opening quote/bracket. Deliberately no NLTK; zero deps.
_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+(?=[A-Z0-9\"'(\[])")
_WS_COLLAPSE = re.compile(r"\s+")
_LONG_SPLIT = re.compile(r"(?<=[,;])\s+")

_FRAG_MIN_CHARS = 3
_FRAG_MAX_CHARS = 500

_ANCHOR_RECORD = struct.Struct("<4Q")
_TRANSITION_RECORD = struct.Struct("<III")

_PROGRESS_EVERY = 2000


# ---------------------------------------------------------------------------
# SimHash256 -- locality-sensitive anchor over char trigrams
# ---------------------------------------------------------------------------


def _fnv1a64(data: bytes, seed: int = _FNV_OFFSET) -> int:
    h = seed & _MASK64
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK64
    return h


def _projected_hash(data: bytes) -> tuple[int, int, int, int]:
    """Four 64-bit FNV-1a64 projections of ``data`` with independent salts."""
    return (
        _fnv1a64(_PROJECTION_SEEDS[0].to_bytes(8, "little") + data),
        _fnv1a64(_PROJECTION_SEEDS[1].to_bytes(8, "little") + data),
        _fnv1a64(_PROJECTION_SEEDS[2].to_bytes(8, "little") + data),
        _fnv1a64(_PROJECTION_SEEDS[3].to_bytes(8, "little") + data),
    )


def _iter_trigrams(text: str) -> Iterator[str]:
    """Yield overlapping char-trigrams. Short strings yield the whole string."""
    t = text.lower()
    n = len(t)
    if n < _TRIGRAM:
        if n > 0:
            yield t
        return
    for i in range(n - _TRIGRAM + 1):
        yield t[i : i + _TRIGRAM]


def simhash256(text: str) -> tuple[int, int, int, int]:
    """Compute a 256-bit SimHash anchor over char-trigrams of ``text``.

    Returns four 64-bit words (little-endian bit order) forming a single
    256-bit locality-sensitive hash. Identical inputs -> identical words.
    Small edits typically produce a Hamming distance in the low tens,
    random pairs cluster around 128.

    The empty string anchors to ``(0, 0, 0, 0)``.
    """
    accum: list[list[int]] = [[0] * 64 for _ in range(SIMHASH_WORDS)]
    any_trigram = False
    for tri in _iter_trigrams(text):
        any_trigram = True
        h4 = _projected_hash(tri.encode("utf-8"))
        for w in range(SIMHASH_WORDS):
            word = h4[w]
            row = accum[w]
            for b in range(64):
                row[b] += 1 if (word >> b) & 1 else -1
    if not any_trigram:
        return (0, 0, 0, 0)
    out: list[int] = []
    for w in range(SIMHASH_WORDS):
        bits = 0
        row = accum[w]
        for b in range(64):
            if row[b] > 0:
                bits |= 1 << b
        out.append(bits)
    return (out[0], out[1], out[2], out[3])


def hamming_distance_simhash(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    """Bitwise Hamming distance between two 256-bit SimHash anchors."""
    return (
        int(a[0] ^ b[0]).bit_count()
        + int(a[1] ^ b[1]).bit_count()
        + int(a[2] ^ b[2]).bit_count()
        + int(a[3] ^ b[3]).bit_count()
    )


# ---------------------------------------------------------------------------
# Fragment extraction
# ---------------------------------------------------------------------------


def _normalize_fragment(raw: str) -> str:
    return _WS_COLLAPSE.sub(" ", raw).strip()


def _split_long_fragment(s: str) -> list[str]:
    """Split overly long fragments on commas / semicolons, preserving order."""
    if len(s) <= _FRAG_MAX_CHARS:
        return [s]
    parts = _LONG_SPLIT.split(s)
    out: list[str] = []
    buf = ""
    for p in parts:
        candidate = f"{buf} {p}".strip() if buf else p
        if len(candidate) > _FRAG_MAX_CHARS and buf:
            out.append(buf)
            buf = p
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def iter_fragments(text: str) -> Iterator[str]:
    """Split ``text`` into clean fragment strings.

    Deterministic: sentence-ends via punctuation + whitespace + uppercase,
    whitespace collapsed, minimum/maximum char bounds enforced, over-long
    fragments split on commas.
    """
    if not text:
        return
    for raw in _SENT_SPLIT.split(text):
        frag = _normalize_fragment(raw)
        if len(frag) < _FRAG_MIN_CHARS:
            continue
        for piece in _split_long_fragment(frag):
            if len(piece) >= _FRAG_MIN_CHARS:
                yield piece


# ---------------------------------------------------------------------------
# On-disk layout -- deterministic, mmap-friendly packed binaries + JSON
# ---------------------------------------------------------------------------


def _write_anchors(path: Path, anchors: Iterable[tuple[int, int, int, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("wb") as f:
        for a in anchors:
            f.write(_ANCHOR_RECORD.pack(*a))
            n += 1
    return n


def _read_anchors(path: Path) -> list[tuple[int, int, int, int]]:
    data = path.read_bytes()
    size = _ANCHOR_RECORD.size
    if len(data) % size != 0:
        raise ValueError(
            f"corrupt anchors file {path}: length {len(data)} not multiple of {size}"
        )
    out: list[tuple[int, int, int, int]] = []
    for i in range(0, len(data), size):
        out.append(_ANCHOR_RECORD.unpack(data[i : i + size]))
    return out


def _write_transitions(path: Path, transitions: dict[tuple[int, int], int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("wb") as f:
        for (src, dst), count in transitions.items():
            f.write(_TRANSITION_RECORD.pack(src, dst, count))
            n += 1
    return n


def _read_transitions(path: Path) -> dict[tuple[int, int], int]:
    data = path.read_bytes()
    size = _TRANSITION_RECORD.size
    if len(data) % size != 0:
        raise ValueError(
            f"corrupt transitions file {path}: length {len(data)} not multiple of {size}"
        )
    out: dict[tuple[int, int], int] = {}
    for i in range(0, len(data), size):
        src, dst, count = _TRANSITION_RECORD.unpack(data[i : i + size])
        out[(int(src), int(dst))] = int(count)
    return out


def _write_fragments_json(path: Path, fragments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nrl_ai.fragments.v1",
        "count": len(fragments),
        "fragments": [
            {"id": i, "text": frag, "char_len": len(frag)}
            for i, frag in enumerate(fragments)
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_fragments_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = str(data.get("schema", ""))
    if schema != "nrl_ai.fragments.v1":
        raise ValueError(f"unsupported nrl-ai fragments schema: {schema!r}")
    return [str(f["text"]) for f in data.get("fragments", [])]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Summary payload for a completed ingest run."""

    manifest: NrlAiIndexManifest
    paths: NrlAiPaths
    fragment_count: int
    transition_count: int
    corpus_sha256: str
    elapsed_s: float


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(
    corpus_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    progress: bool = False,
) -> IngestResult:
    """Compile ``corpus_path`` into an on-disk NRL-AI index.

    Writes ``fragments.json``, ``anchors.bin``, ``transitions.bin`` and
    ``manifest.json`` under ``out_dir`` (or the default index root).

    Progress is printed to ``stderr`` when ``progress=True`` and the
    corpus is large enough that per-fragment SimHash work dominates.
    """
    t0 = time.perf_counter()
    src = Path(corpus_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"corpus not found: {src}")

    paths = NrlAiPaths(Path(out_dir).expanduser()) if out_dir else NrlAiPaths.default()
    paths.ensure()

    text = src.read_text(encoding="utf-8", errors="replace")
    fragments = list(iter_fragments(text))

    anchors: list[tuple[int, int, int, int]] = []
    total = len(fragments)
    for i, frag in enumerate(fragments):
        anchors.append(simhash256(frag))
        if progress and total >= _PROGRESS_EVERY and (i + 1) % _PROGRESS_EVERY == 0:
            pct = 100.0 * (i + 1) / total
            print(
                f"[nrl-ai ingest] simhash {i + 1}/{total} ({pct:5.1f}%)",
                file=sys.stderr,
                flush=True,
            )

    transitions: dict[tuple[int, int], int] = {}
    for i in range(total - 1):
        key = (i, i + 1)
        transitions[key] = transitions.get(key, 0) + 1

    _write_fragments_json(paths.fragments, fragments)
    anchor_written = _write_anchors(paths.anchors, anchors)
    trans_written = _write_transitions(paths.transitions, transitions)
    if anchor_written != total:
        raise RuntimeError(
            f"anchor write mismatch: wrote {anchor_written}, expected {total}"
        )

    manifest = NrlAiIndexManifest(
        corpus_sha256=_sha256_of_file(src),
        fragment_count=total,
        transition_count=trans_written,
        simhash_bits=SIMHASH_BITS,
        created_utc=_now_utc(),
        source_path=str(src),
    )
    manifest.save(paths.manifest)

    return IngestResult(
        manifest=manifest,
        paths=paths,
        fragment_count=total,
        transition_count=trans_written,
        corpus_sha256=manifest.corpus_sha256,
        elapsed_s=time.perf_counter() - t0,
    )


def load_anchors(paths: NrlAiPaths) -> list[tuple[int, int, int, int]]:
    return _read_anchors(paths.anchors)


def load_fragments(paths: NrlAiPaths) -> list[str]:
    return _read_fragments_json(paths.fragments)


def load_transitions(paths: NrlAiPaths) -> dict[tuple[int, int], int]:
    return _read_transitions(paths.transitions)


__all__ = [
    "SIMHASH_BITS",
    "IngestResult",
    "hamming_distance_simhash",
    "ingest",
    "iter_fragments",
    "load_anchors",
    "load_fragments",
    "load_transitions",
    "simhash256",
]
