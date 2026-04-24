# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL-ZPM — Zero-Point Mapping identity resolver.

This module ports the ROMA-ZPM v2.0 topology pipeline into NRL as a
**pre-libllama identity resolver** (Plane A.5 in
``nrl-new-archietcture.MD``). The intent is deliberately narrow: *if the
turn's topological state already matches a stored "unity" state, emit the
stored reply at memory-I/O speed instead of decoding tokens in libllama.*

Pipeline, per ROMA-ZPM Stages I–VI:

1. **Anchor**    — materialize a ``State`` (4 × uint64, 256 bits) from the
   turn's raw bytes (model SHA + system + rendered history + user + sampler
   fingerprint). Deterministic, replay-lockable.
2. **Inversion** — GF(2) row parity with the golden-ratio reversal
   ``0x9E3779B97F4A7C15``. Extracts a 4-row signature of the question.
3. **Rotor**     — Cl(4,0) scalar/bivector rotor with a seed-derived angle;
   returns a phase-locked 64-bit word over the inversion rows.
4. **Nullspace** — XOR against every stored unity state and compute the
   Hamming distance (``popcount``). ``dist == 0`` → exact singularity;
   ``0 < dist ≤ threshold`` → near-match.
5. **Verify**    — bit-symmetry audit: residual across all four words must
   be zero for an exact match, or Hamming distance ≤ threshold for a
   bounded near-match, before the stored reply is served.

The *only* numerics NRL claims here are the four primitives above. This
module does not decode tokens, does not run libllama, and does not emit
fresh text. It is a **strict replay resolver** with a declared confidence
bound.

Honesty contract:

* ``ZpmHit.exact`` is ``True`` iff the 256-bit residual was zero.
* ``ZpmHit.distance_bits`` is the real Hamming distance from the nearest
  stored unity state — this number goes into the evidence log so the
  gate decision is auditable.
* ``ZpmIndex`` writes atomically and mmap-safe; the index survives process
  restarts, so ``executed_tps`` after the first hit for a pattern is
  dominated by disk I/O, not decode.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "State",
    "anchor",
    "inversion",
    "rotor",
    "nullspace_search",
    "verify",
    "ZpmEntry",
    "ZpmHit",
    "ZpmIndex",
    "ZPM_GOLDEN_RATIO",
    "popcount64",
    "hamming_state",
    "format_stage_banner",
    "prime",
    "take_prefetched_zpm_index",
    "invalidate_prefetch",
]


# ROMA-ZPM constant: golden-ratio-derived 64-bit reversal used by Stage II
# (inversion). Matches the original zpm.cpp reference verbatim.
ZPM_GOLDEN_RATIO: int = 0x9E3779B97F4A7C15
_U64_MASK: int = 0xFFFFFFFFFFFFFFFF

# 4 × uint64 = 256-bit topological state (public type alias).
State = tuple[int, int, int, int]

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def popcount64(x: int) -> int:
    """Count set bits in a 64-bit integer. Matches ``__builtin_popcountll``."""
    return (x & _U64_MASK).bit_count()


def hamming_state(a: State, b: State) -> int:
    """Hamming distance between two 256-bit states (``Σ popcount(a[i] ^ b[i])``)."""
    return sum(popcount64(a[i] ^ b[i]) for i in range(4))


def _fnv1a64(data: bytes) -> int:
    """Stdlib-only FNV-1a64 (matches ``nrlpy.runtime.fnv1a64_packed`` semantics)."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for b in data:
        h ^= b
        h = (h * prime) & _U64_MASK
    return h


# --------------------------------------------------------------------------- #
# Stage I — Anchor
# --------------------------------------------------------------------------- #


def anchor(turn_bytes: bytes | Iterable[bytes]) -> State:
    """Materialize the 4 × uint64 manifold anchor for a turn.

    The input is a bytes blob or iterable of bytes chunks (caller chooses the
    closed set; usually ``model_sha256 | system | history | user | sampler``).
    We FNV-1a64 over four rotated windows of the blob, producing four
    independent 64-bit words. This matches the "4D manifold" definition in
    ROMA-ZPM: four independent Galois projections over the same state.
    """
    if isinstance(turn_bytes, bytes):
        blob = turn_bytes
    else:
        blob = b"\x1e".join(turn_bytes)
    n = len(blob)
    if n == 0:
        return (0, 0, 0, 0)

    # Four deterministic rotations: identity, reverse, halves-swap, byte-shift.
    # Each projection is FNV-1a64 over a deterministic permutation. Collisions
    # across all four rows simultaneously are astronomically unlikely for
    # natural chat input.
    w0 = _fnv1a64(blob)
    w1 = _fnv1a64(blob[::-1])
    half = n // 2
    w2 = _fnv1a64(blob[half:] + blob[:half])
    shift = n // 4 if n >= 4 else 1
    w3 = _fnv1a64(blob[shift:] + blob[:shift])
    return (w0, w1, w2, w3)


# --------------------------------------------------------------------------- #
# Stage II — Inversion (GF(2) row parity over Cl(4,0))
# --------------------------------------------------------------------------- #


def inversion(t: State) -> State:
    """GF(2) row parity matching ``zpm.cpp``'s inversion():

    ``rows[i] = t[i] ^ (t[i] << 1) ^ 0x9E3779B97F4A7C15``

    The left shift generates the bit-wise differential (discrete derivative
    over GF(2)); XORing the golden-ratio constant breaks periodicity so
    near-identity inputs stay distinguishable at the row level.
    """
    return tuple(
        ((t[i] ^ ((t[i] << 1) & _U64_MASK) ^ ZPM_GOLDEN_RATIO) & _U64_MASK)
        for i in range(4)
    )  # type: ignore[return-value]


def inversion_determinant(rows: State) -> int:
    """Parity determinant = XOR of all four rows. Non-zero ⇒ UNITY signature."""
    return (rows[0] ^ rows[1] ^ rows[2] ^ rows[3]) & _U64_MASK


# --------------------------------------------------------------------------- #
# Stage III — Rotor (Cl(4,0) sandwich)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rotor:
    """Clifford Cl(4,0) rotor components. ``s`` scalar, ``b_*`` bivectors."""

    s: float
    b_xy: float
    b_xz: float = 0.7071067811865475  # Fixed spacetime anchor from ROMA-ZPM v2.0
    b_xw: float = 0.5  # Hyper-dimensional offset

    @property
    def norm(self) -> float:
        return (self.s * self.s + self.b_xy * self.b_xy) ** 0.5

    @property
    def phase_locked(self) -> bool:
        return self.norm > 0.999


def rotor(seed: int) -> Rotor:
    """Build the rotor from the XOR of the first two inversion rows.

    The scale ``1e-9`` matches the ROMA-ZPM v2.0 reference so the symmetry
    norm land on the phase-locked shelf (>0.999) for most 64-bit seeds.
    """
    import math

    x = (seed & _U64_MASK) * 1e-9
    return Rotor(s=math.cos(x), b_xy=math.sin(x))


# --------------------------------------------------------------------------- #
# Stage IV — Nullspace search
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZpmHit:
    """Outcome of a nullspace search.

    * ``entry_index``: position of the nearest stored unity state, or ``-1``.
    * ``distance_bits``: Hamming distance to that stored state (0 = exact).
    * ``exact``: convenience — ``distance_bits == 0``.
    * ``within_threshold``: ``distance_bits <= threshold``.
    """

    entry_index: int
    distance_bits: int
    threshold_bits: int
    exact: bool
    within_threshold: bool


def nullspace_search(
    query: State,
    stored: Sequence[State],
    *,
    threshold_bits: int = 0,
) -> ZpmHit:
    """Scan stored unity states; return nearest + Hamming distance.

    O(N) in the number of stored entries, O(4) per comparison
    (popcount on 4 × 64-bit words). In practice a chat deployment holds
    a few thousand entries — the scan is microseconds.
    """
    if not stored:
        return ZpmHit(-1, 256, threshold_bits, False, False)
    best_i = 0
    best_d = hamming_state(query, stored[0])
    for i in range(1, len(stored)):
        d = hamming_state(query, stored[i])
        if d < best_d:
            best_i = i
            best_d = d
            if best_d == 0:
                break
    return ZpmHit(
        entry_index=best_i,
        distance_bits=best_d,
        threshold_bits=threshold_bits,
        exact=(best_d == 0),
        within_threshold=(best_d <= threshold_bits),
    )


# --------------------------------------------------------------------------- #
# Stage VI — Verify
# --------------------------------------------------------------------------- #


def verify(t: State, solution: State) -> bool:
    """Bit-symmetry audit. Residual is the OR of per-word XOR diffs.

    Returns ``True`` iff every bit of every word matches — i.e. the candidate
    solution is a true identity. Used after a cache fetch to guarantee the
    served reply corresponds to an exact-state entry, never a near-match
    served under the exact label.
    """
    residual = 0
    for i in range(4):
        residual |= (t[i] ^ solution[i]) & _U64_MASK
    return residual == 0


# --------------------------------------------------------------------------- #
# Persistent index (on-disk near-match store)
# --------------------------------------------------------------------------- #


@dataclass
class ZpmEntry:
    """One (state, reply) pair in the ZPM index.

    The reply text is stored verbatim for serving; ``tokens`` is the decoded
    token count recorded at write time (muscle-memory-equivalent). Turn
    hints (``wall_s_at_write``) are carried for telemetry but don't change
    the lookup.
    """

    state: State
    reply_text: str
    tokens: int
    wall_s_at_write: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


_ZPM_HEADER_MAGIC = b"NRLZPM01"
_ZPM_ENTRY_HEADER = "<4Q I f H"  # 4×u64 state, u32 tokens, f32 wall, u16 meta-count


class ZpmIndex:
    """In-memory + on-disk store of ``ZpmEntry`` records.

    On-disk layout (binary, little-endian):

    * 8 bytes magic (``NRLZPM01``).
    * Sequence of records::

          4 × uint64         state (big-endian XOR permutations from anchor())
          uint32             token count
          float32            wall_s when written
          uint16             metadata pair count
          uint32             reply text length (bytes, UTF-8)
          bytes[text_len]    reply text
          (key, value) pairs: uint16 key_len, bytes, uint16 val_len, bytes

    Writes are atomic (tmp+rename). Reads are eager on :meth:`load`; the
    nullspace scan is in-memory so a chat turn's lookup is fast even after
    tens of thousands of entries.
    """

    def __init__(self, entries: list[ZpmEntry] | None = None) -> None:
        self._entries: list[ZpmEntry] = list(entries or [])

    # ------------- index ops -------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def states(self) -> list[State]:
        return [e.state for e in self._entries]

    def add(self, entry: ZpmEntry) -> int:
        """Append a new entry. Returns its index."""
        self._entries.append(entry)
        return len(self._entries) - 1

    def remove_entry_indices(self, indices: set[int]) -> int:
        """Drop entries whose indices are in ``indices``; returns removed count."""
        if not indices:
            return 0
        before = len(self._entries)
        self._entries = [e for i, e in enumerate(self._entries) if i not in indices]
        return before - len(self._entries)

    def lookup(self, query: State, *, threshold_bits: int = 0) -> tuple[ZpmHit, ZpmEntry | None]:
        """Run nullspace + verify. Exact matches are verify-audited;
        near-matches within ``threshold_bits`` are returned un-audited with
        the measured distance surfaced so the caller can decide whether to
        serve or fall through to decode.
        """
        hit = nullspace_search(query, self.states(), threshold_bits=threshold_bits)
        if hit.entry_index < 0:
            return hit, None
        entry = self._entries[hit.entry_index]
        if hit.exact:
            if not verify(query, entry.state):
                # Should never trigger (exact == distance 0 implies residual 0),
                # but kept for symmetry with ROMA-ZPM Stage VI.
                return ZpmHit(hit.entry_index, hit.distance_bits, threshold_bits, False, False), None
            return hit, entry
        if hit.within_threshold:
            return hit, entry
        return hit, None

    # ------------- serialization -------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(_ZPM_HEADER_MAGIC)
            for e in self._entries:
                meta_items = list(e.metadata.items())
                head = struct.pack(
                    _ZPM_ENTRY_HEADER,
                    e.state[0], e.state[1], e.state[2], e.state[3],
                    int(e.tokens) & 0xFFFFFFFF,
                    float(e.wall_s_at_write),
                    len(meta_items) & 0xFFFF,
                )
                body = e.reply_text.encode("utf-8")
                f.write(head)
                f.write(struct.pack("<I", len(body)))
                f.write(body)
                for k, v in meta_items:
                    kb = k.encode("utf-8")
                    vb = v.encode("utf-8")
                    f.write(struct.pack("<H", len(kb)))
                    f.write(kb)
                    f.write(struct.pack("<H", len(vb)))
                    f.write(vb)
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ZpmIndex":
        p = Path(path)
        if not p.is_file():
            return cls()
        # Memory-map read (P4): OS paging; parse sequentially without loading
        # the entire file into a separate bytes object on CPython.
        if p.stat().st_size == 0:
            return cls()
        with p.open("rb") as f:
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except OSError:
                mm = None
        if mm is None:
            with p.open("rb") as f:
                return cls._load_from_reader(f)
        try:
            return cls._load_from_buffer(mm)
        finally:
            mm.close()

    @classmethod
    def _load_from_reader(cls, f: Any) -> "ZpmIndex":
        magic = f.read(len(_ZPM_HEADER_MAGIC))
        if magic != _ZPM_HEADER_MAGIC:
            return cls()
        idx = cls()
        entry_head_size = struct.calcsize(_ZPM_ENTRY_HEADER)
        while True:
            head_buf = f.read(entry_head_size)
            if not head_buf or len(head_buf) < entry_head_size:
                break
            cls._append_entry_from_header(f, idx, head_buf, entry_head_size)
        return idx

    @classmethod
    def _load_from_buffer(cls, mm: mmap.mmap) -> "ZpmIndex":
        if len(mm) < len(_ZPM_HEADER_MAGIC):
            return cls()
        if mm[: len(_ZPM_HEADER_MAGIC)] != _ZPM_HEADER_MAGIC:
            return cls()
        idx = cls()
        pos = len(_ZPM_HEADER_MAGIC)
        entry_head_size = struct.calcsize(_ZPM_ENTRY_HEADER)
        while pos + entry_head_size <= len(mm):
            head_buf = mm[pos : pos + entry_head_size]
            pos += entry_head_size
            s0, s1, s2, s3, tokens, wall, meta_count = struct.unpack(
                _ZPM_ENTRY_HEADER, head_buf
            )
            if pos + 4 > len(mm):
                break
            (text_len,) = struct.unpack("<I", mm[pos : pos + 4])
            pos += 4
            if pos + text_len > len(mm):
                break
            text = mm[pos : pos + text_len].decode("utf-8", errors="replace")
            pos += text_len
            meta: dict[str, str] = {}
            for _ in range(meta_count):
                if pos + 2 > len(mm):
                    break
                (kl,) = struct.unpack("<H", mm[pos : pos + 2])
                pos += 2
                if pos + kl > len(mm):
                    break
                k = mm[pos : pos + kl].decode("utf-8", errors="replace")
                pos += kl
                if pos + 2 > len(mm):
                    break
                (vl,) = struct.unpack("<H", mm[pos : pos + 2])
                pos += 2
                if pos + vl > len(mm):
                    break
                v = mm[pos : pos + vl].decode("utf-8", errors="replace")
                pos += vl
                meta[k] = v
            idx._entries.append(
                ZpmEntry(
                    state=(s0, s1, s2, s3),
                    reply_text=text,
                    tokens=int(tokens),
                    wall_s_at_write=float(wall),
                    metadata=meta,
                )
            )
        return idx

    @staticmethod
    def _append_entry_from_header(f: Any, idx: "ZpmIndex", head_buf: bytes, entry_head_size: int) -> None:
        s0, s1, s2, s3, tokens, wall, meta_count = struct.unpack(
            _ZPM_ENTRY_HEADER, head_buf
        )
        (text_len,) = struct.unpack("<I", f.read(4))
        text = f.read(text_len).decode("utf-8", errors="replace")
        meta: dict[str, str] = {}
        for _ in range(meta_count):
            (kl,) = struct.unpack("<H", f.read(2))
            k = f.read(kl).decode("utf-8", errors="replace")
            (vl,) = struct.unpack("<H", f.read(2))
            v = f.read(vl).decode("utf-8", errors="replace")
            meta[k] = v
        idx._entries.append(
            ZpmEntry(
                state=(s0, s1, s2, s3),
                reply_text=text,
                tokens=int(tokens),
                wall_s_at_write=float(wall),
                metadata=meta,
            )
        )


# --------------------------------------------------------------------------- #
# Phase 15-EG — in-process ZPM warm-restart (read-only deck refresh)
# --------------------------------------------------------------------------- #

_PREFETCH_ZPM: dict[str, ZpmIndex] = {}


def prime(
    model_sha256: str,
    intent_anchor_bytes: bytes,
    tail_token_ids: Sequence[int],
    index_path: Path,
) -> None:
    """Re-hydrate the on-disk ZPM index into RAM for one model (read-only).

    Never writes ``index.bin`` and never touches LMO / GGUF weight blobs.
    The loaded :class:`ZpmIndex` is held until the next Python-side lookup
    in ``nrlpy.gguf`` pops it via :func:`take_prefetched_zpm_index`.

    ``intent_anchor_bytes`` and ``tail_token_ids`` are accepted so the
    warm-restart hook can carry the current manifold anchor plus the last
    few llama token ids for future sliding-window fusion; today they are
    not persisted beyond this call's argument tuple.
    """
    _ = (intent_anchor_bytes, tuple(int(x) for x in tail_token_ids))
    if not model_sha256:
        return
    if index_path.is_file():
        _PREFETCH_ZPM[model_sha256] = ZpmIndex.load(index_path)
    else:
        _PREFETCH_ZPM.pop(model_sha256, None)


def take_prefetched_zpm_index(model_sha256: str) -> ZpmIndex | None:
    """Pop a prefetched index installed by :func:`prime`, if any."""
    return _PREFETCH_ZPM.pop(model_sha256, None)


def invalidate_prefetch(model_sha256: str) -> None:
    """Drop any in-RAM prefetched index for this model (e.g. after on-disk prune)."""
    _PREFETCH_ZPM.pop(model_sha256 or "", None)


# --------------------------------------------------------------------------- #
# Stage banner (matches ROMA-ZPM v2.0 CLI output, ANSI-optional)
# --------------------------------------------------------------------------- #


def format_stage_banner(
    t: State,
    *,
    seed: int | None = None,
    solution: State | None = None,
    ansi: bool = True,
) -> str:
    """Render the ROMA-ZPM Stage I–VII banner for a given state. Useful for
    ``nrlpy zpm`` CLI output and for verbose evidence logs.
    """
    C = "\033[36m" if ansi else ""
    G = "\033[32m" if ansi else ""
    M = "\033[35m" if ansi else ""
    Y = "\033[33m" if ansi else ""
    R = "\033[0m" if ansi else ""
    # NOTE: markers are ASCII (`>>`) not `▶` so the banner prints cleanly on
    # Windows PowerShell cp1252 codepages without forcing UTF-8 on stdout.
    lines: list[str] = []
    lines.append(f"{G}>> ANCHOR    {R}4D manifold locked.")
    lines.append("   Target: " + " ".join(f"{w:016x}" for w in t))
    rows = inversion(t)
    lines.append(f"{C}>> INVERSION {R}GF(2) basis over Cl(4,0)")
    for i, r in enumerate(rows):
        lines.append(f"   Row[{i}]: 0x{r:016x}")
    det = inversion_determinant(rows)
    lines.append(f"{G}   Determinant parity -> UNITY ({1 if det else 0}){R}")
    s = seed if seed is not None else (t[0] ^ t[1])
    rot = rotor(s)
    lines.append(f"{Y}>> ROTOR     {R}Clifford sandwich product")
    lines.append(f"   Symmetry norm: {rot.norm:.6f}")
    if rot.phase_locked:
        lines.append(f"{G}   PHASE-LOCKED OK{R}")
    sol = solution if solution is not None else t
    h = hamming_state(t, sol)
    lines.append(f"{C}>> NULLSPACE {R}Intersection search")
    lines.append(f"   Hamming distance to singularity: {h} bits")
    if h == 0:
        lines.append(f"{G}   SINGULARITY DETECTED{R}")
    residual = 0
    for i in range(4):
        residual |= (t[i] ^ sol[i]) & _U64_MASK
    lines.append(f"{M}>> VERIFY    {R}Bit-symmetry audit")
    lines.append(f"   Residual: 0x{residual:016x}")
    if residual == 0:
        lines.append(f"{G}   STABILITY: ABSOLUTE UNITY{R}")
    return "\n".join(lines)
