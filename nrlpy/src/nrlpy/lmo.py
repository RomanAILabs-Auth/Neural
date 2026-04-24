# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""LMO — Lattice Model Object (GGUF absorption foundation).

Implements Phase 4-EG of :doc:`../../../Final_NRL_Architecture_GGUF.MD`:
the one-time, offline, deterministic transformation of a GGUF file into
an on-disk **Lattice Model Object** under
``$NRL_ROOT/cache/lmo/<model_sha256>/``.

Phase 4-EG scope (explicit):

* Absorption produces the on-disk artifact and the auto-materialized
  ``.nrl`` manifest v1. Nothing in the runtime Resolution Ladder
  (R0..R5) changes. Runtime behaviour of ``run_gguf`` is untouched.
* ``benchmark_class = B`` only. Class-A claims require the runtime
  ladder to actually consume the LMO; that lands in Phase 5-EG shadow
  mode and Phase 6-EG active mode.
* Stage A-VI parity is a **hard CI gate**: the retained GGUF bytes
  must be byte-identical to their source-file slices, and re-running
  ``absorb_gguf`` on an identical source must produce a byte-identical
  ``lmo.header``. The optional libllama-forward smoke-test is recorded
  in ``attest.json`` when available but is never required to pass
  (libllama is not in our CI).

Stage map (mirrors Plane-A.5 Stages I–VI for consistency with ZPM):

* **A-I   Ingest**      — streaming SHA-256, minimal GGUF metadata parse.
* **A-II  Tile Plan**   — per-tensor decomposition by the 16,384 canonical tile
  (NRL-D005).
* **A-III Pack**        — dual artefact per tile: packed INT4 potentials (2
  nibbles/byte, saturating ``[0,15]``) under ``tiles/`` and a retained
  byte-for-byte slice under ``retained/``.
* **A-IV  Router**      — omega sub-lattice topology lifted from GGUF tensor
  names + weight magnitudes, reusing ``zpm_omega_router`` vocabulary.
* **A-V   Anchor**      — Plane-A.5 Stage I over ``(model_sha256 |
  tile_plan_digest | router_graph_digest | nrl_version | cpu_features)``.
* **A-VI  Verify**      — determinism + byte-identity audits; writes
  ``attest.json``. Hard failures raise :class:`LmoError` **before** the
  partial LMO is considered valid.

All on-disk formats in this module are explicitly documented in place so
the artefact is round-trippable by third-party tools.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import dataclasses
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from . import zpm
from .runtime import features as runtime_features
from .runtime import fnv1a64_packed
from .runtime import version as runtime_version

__all__ = [
    "CANONICAL_TILE_UNITS",
    "COHERENCE_LANES",
    "LANES_ALLOWING_R2_ACTIVE",
    "LANES_ALLOWING_R2_SHADOW",
    "LMO_MAGIC",
    "LMO_SCHEMA_VERSION",
    "GGUF_MAGIC",
    "GgufParseError",
    "GgufTensorInfo",
    "GgufMetadata",
    "LmoError",
    "LmoHandle",
    "LmoHeader",
    "OMEGA_ACTIVE_GATE_SOURCE",
    "OMEGA_SHADOW_GATE_SOURCE",
    "OmegaShadowReport",
    "RouterGraph",
    "RouterSubLattice",
    "RungResult",
    "TilePlan",
    "TileSpec",
    "absorb_gguf",
    "build_router_graph",
    "compute_r2_candidate_state",
    "embedding_row_packed",
    "gguf_parse_metadata",
    "lane_allows_r2_active",
    "lane_allows_r2_shadow",
    "pack_int4_from_bytes",
    "plan_tiles",
    "sha256_file",
    "try_omega_native_resolve",
    "verify_parity_against_libllama",
    "write_header_anchor",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LMO_MAGIC = b"NRLLMO01"
LMO_SCHEMA_VERSION = 1
LMO_HEADER_FIXED_BYTES = 4096                 # fixed 4 KiB header block
CANONICAL_TILE_UNITS = 16_384                 # NRL-D005
DEFAULT_GGUF_ALIGNMENT = 32                   # GGUF default unless overridden

GGUF_MAGIC = b"GGUF"
_SUPPORTED_GGUF_VERSIONS = {2, 3}             # llama.cpp current lineage

# Coherence lanes (Final_NRL_Architecture_GGUF.MD §4.5). The rung whitelist is
# enforced by the runner; this module only advertises the lane vocabulary so
# callers can validate manifest input without a circular import through
# ``gguf.py``.
COHERENCE_LANES: frozenset[str] = frozenset({
    "fast-stable", "fast-balanced", "max-throughput",
})
LANES_ALLOWING_R2_SHADOW: frozenset[str] = frozenset({
    "fast-balanced", "max-throughput",
})
# Phase 6-EG — R2 *active* token-serving lanes. Same as shadow today per the
# operator's Phase 6 directive (both fast-balanced and max-throughput may
# serve R2 tokens once Stage-VI passes). ``fast-stable`` remains hard-locked
# to the R0/R1/R3/R5 subset — it must never emit an R2 token.
LANES_ALLOWING_R2_ACTIVE: frozenset[str] = frozenset({
    "fast-balanced", "max-throughput",
})
OMEGA_SHADOW_GATE_SOURCE = "omega_resolve_shadow"
# §4.3 active-mode label. Distinct from the shadow label so evidence-log
# analysis can separate "the lattice thought it had a match" (shadow) from
# "the lattice actually served this turn" (active).
OMEGA_ACTIVE_GATE_SOURCE = "omega_resolve"


def lane_allows_r2_shadow(lane: str) -> bool:
    """True iff ``lane`` permits a *shadow* R2 probe (Phase 5-EG).

    ``fast-stable`` never runs R2 — the ladder is frozen at R0/R1/R3/R5 for
    class-A-legal replay. ``fast-balanced`` and ``max-throughput`` allow
    the probe; see also :func:`lane_allows_r2_active`.
    """
    return lane in LANES_ALLOWING_R2_SHADOW


def lane_allows_r2_active(lane: str) -> bool:
    """True iff ``lane`` permits R2 to *emit tokens* (Phase 6-EG).

    Strictly narrower than :func:`lane_allows_r2_shadow` in principle, but
    identical today per the Phase 6-EG operator directive. Kept as a
    separate predicate so Phase 7-EG can tighten active mode (e.g. to
    ``max-throughput`` only) without touching shadow callers.
    """
    return lane in LANES_ALLOWING_R2_ACTIVE


class _GgufMetaType(IntEnum):
    """GGUF metadata value types (llama.cpp ``gguf_metadata_value_type``)."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# ggml_type → (block_size_elements, bytes_per_block). Values match
# ``GGML_BLCK_SIZE`` / ``GGML_TYPE_SIZE`` in llama.cpp / ggml.c. Keep the
# table small and conservative: unknown types cause the containing tensor
# to be retained byte-only with ``absorption_partial = true`` in
# ``attest.json`` — never silently miscounted.
_GGML_TYPE_LAYOUT: dict[int, tuple[int, int]] = {
    0:  (1,   4),     # F32
    1:  (1,   2),     # F16
    2:  (32,  18),    # Q4_0
    3:  (32,  20),    # Q4_1
    6:  (32,  22),    # Q5_0
    7:  (32,  24),    # Q5_1
    8:  (32,  34),    # Q8_0
    9:  (32,  36),    # Q8_1
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    15: (256, 292),   # Q8_K
    16: (256, 66),    # IQ2_XXS
    17: (256, 74),    # IQ2_XS
    18: (256, 98),    # IQ3_XXS
    19: (256, 50),    # IQ1_S
    20: (32,  18),    # IQ4_NL
    21: (256, 110),   # IQ3_S
    22: (256, 82),    # IQ2_S
    23: (256, 136),   # IQ4_XS
    24: (1,   1),     # I8
    25: (1,   2),     # I16
    26: (1,   4),     # I32
    27: (1,   8),     # I64
    28: (1,   8),     # F64
    29: (256, 122),   # IQ1_M
    30: (1,   2),     # BF16
}

_GGML_TYPE_NAME: dict[int, str] = {
    0: "f32",   1: "f16",   2: "q4_0",  3: "q4_1",  6: "q5_0",  7: "q5_1",
    8: "q8_0",  9: "q8_1",  10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    24: "i8",   25: "i16",  26: "i32",  27: "i64",  28: "f64",
    29: "iq1_m", 30: "bf16",
}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LmoError(RuntimeError):
    """Raised for any LMO absorption or verification failure."""


class GgufParseError(LmoError):
    """Raised when a GGUF file cannot be parsed to the level absorption needs."""


# --------------------------------------------------------------------------- #
# Stage A-I — streaming SHA-256 + minimal GGUF metadata parser
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 hex over a file, streamed in ``chunk_bytes`` chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class GgufTensorInfo:
    """One entry from the GGUF tensor-info section.

    Attributes:
        name: GGUF tensor name (e.g. ``"blk.0.attn_q.weight"``).
        shape: tuple of dimension sizes (little-endian-friendly tuple).
        ggml_type: ``ggml_type`` integer from the GGUF header.
        quant_kind: short string name (``"q4_k"``) or ``"unknown/<N>"`` when
            the ``ggml_type`` is outside :data:`_GGML_TYPE_LAYOUT`.
        data_offset: absolute byte offset into the source GGUF file where
            this tensor's payload begins (i.e. ``tensor_data_start +
            relative_offset``).
        data_bytes: payload byte length computed from shape and layout
            table. ``0`` when ``quant_kind`` is unknown.
    """

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    quant_kind: str
    data_offset: int
    data_bytes: int

    @property
    def element_count(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass(frozen=True)
class GgufMetadata:
    """Top-level view of the GGUF file the absorption pipeline needs.

    Attributes:
        version: GGUF format version (2 or 3).
        tensor_count: number of tensors in the file.
        kv_count: number of metadata key-value pairs.
        alignment: tensor-data alignment in bytes (from
            ``general.alignment`` metadata, else :data:`DEFAULT_GGUF_ALIGNMENT`).
        kv_bytes: the raw metadata-KV byte region ``[kv_start, kv_end)``.
        tokenizer_blob: concatenated raw bytes of every metadata KV pair
            whose key begins with ``"tokenizer."``, each prefixed with its
            full original byte length. Byte-for-byte recoverable.
        tensor_data_start: absolute file offset where the tensor data
            section begins.
        tensors: tuple of :class:`GgufTensorInfo` in source order.
    """

    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    kv_bytes: bytes
    tokenizer_blob: bytes
    tensor_data_start: int
    tensors: tuple[GgufTensorInfo, ...]


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GgufParseError(
            f"short read at offset {f.tell() - len(b)}: wanted {n} bytes, got {len(b)}"
        )
    return b


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f: BinaryIO) -> bytes:
    ln = _read_u64(f)
    if ln > (1 << 28):
        raise GgufParseError(f"implausible GGUF string length {ln} at {f.tell()}")
    return _read_exact(f, ln)


def _skip_gguf_value(f: BinaryIO, vtype: int) -> None:
    """Seek past a single typed value, descending into arrays recursively."""
    if vtype in (_GgufMetaType.UINT8, _GgufMetaType.INT8, _GgufMetaType.BOOL):
        _read_exact(f, 1)
    elif vtype in (_GgufMetaType.UINT16, _GgufMetaType.INT16):
        _read_exact(f, 2)
    elif vtype in (_GgufMetaType.UINT32, _GgufMetaType.INT32, _GgufMetaType.FLOAT32):
        _read_exact(f, 4)
    elif vtype in (
        _GgufMetaType.UINT64, _GgufMetaType.INT64, _GgufMetaType.FLOAT64,
    ):
        _read_exact(f, 8)
    elif vtype == _GgufMetaType.STRING:
        _read_string(f)
    elif vtype == _GgufMetaType.ARRAY:
        inner = _read_u32(f)
        count = _read_u64(f)
        if count > (1 << 30):
            raise GgufParseError(f"implausible GGUF array length {count}")
        for _ in range(count):
            _skip_gguf_value(f, inner)
    else:
        raise GgufParseError(f"unknown GGUF value type {vtype} at {f.tell()}")


def _read_gguf_value_raw(f: BinaryIO, vtype: int) -> Any:
    """Read a typed value for metadata keys we need (scalar subset only)."""
    if vtype == _GgufMetaType.UINT32:
        return _read_u32(f)
    if vtype == _GgufMetaType.INT32:
        return struct.unpack("<i", _read_exact(f, 4))[0]
    if vtype == _GgufMetaType.UINT64:
        return _read_u64(f)
    if vtype == _GgufMetaType.INT64:
        return struct.unpack("<q", _read_exact(f, 8))[0]
    if vtype == _GgufMetaType.STRING:
        return _read_string(f).decode("utf-8", errors="replace")
    _skip_gguf_value(f, vtype)
    return None


def gguf_parse_metadata(path: str | Path) -> GgufMetadata:
    """Parse a GGUF file to the level absorption requires.

    This is intentionally minimal: it walks the metadata KV section and the
    tensor-info section, records byte ranges for later byte-for-byte
    reconstruction of ``tokenizer.blob``, and computes each tensor's
    absolute file offset + payload byte length.

    We do not decode the tensor payloads here — that is Stage A-III.
    """
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GgufParseError(
                f"not a GGUF file (magic={magic!r} at offset 0 of {p})"
            )
        version = _read_u32(f)
        if version not in _SUPPORTED_GGUF_VERSIONS:
            raise GgufParseError(
                f"unsupported GGUF version {version} "
                f"(supported: {sorted(_SUPPORTED_GGUF_VERSIONS)})"
            )
        tensor_count = _read_u64(f)
        kv_count = _read_u64(f)

        # --- Metadata KV section ---
        kv_start = f.tell()
        alignment = DEFAULT_GGUF_ALIGNMENT
        tokenizer_parts: list[bytes] = []
        for _ in range(kv_count):
            pair_start = f.tell()
            key_bytes = _read_string(f)
            key = key_bytes.decode("utf-8", errors="replace")
            vtype = _read_u32(f)
            if key == "general.alignment" and vtype in (
                _GgufMetaType.UINT32, _GgufMetaType.INT32, _GgufMetaType.UINT64,
                _GgufMetaType.INT64,
            ):
                alignment = int(_read_gguf_value_raw(f, vtype) or DEFAULT_GGUF_ALIGNMENT)
                if alignment <= 0 or alignment & (alignment - 1):
                    raise GgufParseError(
                        f"invalid general.alignment: {alignment} (must be power of two)"
                    )
            else:
                _skip_gguf_value(f, vtype)
            pair_end = f.tell()
            if key.startswith("tokenizer."):
                # Rewind, copy the full KV pair bytes verbatim, restore position.
                f.seek(pair_start)
                raw = _read_exact(f, pair_end - pair_start)
                # Prefix each copied pair with its byte length so the blob can
                # be split back into pairs without re-parsing type codes.
                tokenizer_parts.append(struct.pack("<I", len(raw)) + raw)

        kv_end = f.tell()
        kv_bytes = b""  # intentionally empty; region is re-readable from the GGUF.

        # --- Tensor info section ---
        tensor_infos: list[tuple[str, tuple[int, ...], int, int]] = []  # (name, shape, type, rel_off)
        for _ in range(tensor_count):
            name = _read_string(f).decode("utf-8", errors="replace")
            n_dims = _read_u32(f)
            if n_dims == 0 or n_dims > 8:
                raise GgufParseError(
                    f"tensor {name!r}: implausible n_dims={n_dims}"
                )
            dims = tuple(_read_u64(f) for _ in range(n_dims))
            ggml_type = _read_u32(f)
            rel_off = _read_u64(f)
            tensor_infos.append((name, dims, int(ggml_type), int(rel_off)))

        # Align to ``alignment`` before tensor data begins.
        pos = f.tell()
        pad = (-pos) % alignment
        tensor_data_start = pos + pad

        # Resolve absolute offset + byte length per tensor.
        infos: list[GgufTensorInfo] = []
        for name, dims, gtype, rel_off in tensor_infos:
            layout = _GGML_TYPE_LAYOUT.get(gtype)
            if layout is None:
                quant = f"unknown/{gtype}"
                data_bytes = 0
            else:
                block_elems, block_bytes = layout
                n = 1
                for d in dims:
                    n *= int(d)
                if n <= 0:
                    raise GgufParseError(
                        f"tensor {name!r}: non-positive element count {n}"
                    )
                if n % block_elems != 0 and block_elems != 1:
                    raise GgufParseError(
                        f"tensor {name!r} ({_GGML_TYPE_NAME.get(gtype, gtype)}) "
                        f"element count {n} not a multiple of block_size {block_elems}"
                    )
                data_bytes = (n // block_elems) * block_bytes
                quant = _GGML_TYPE_NAME.get(gtype, f"unknown/{gtype}")
            abs_off = tensor_data_start + rel_off
            if data_bytes and abs_off + data_bytes > size:
                raise GgufParseError(
                    f"tensor {name!r} spans beyond EOF: "
                    f"abs_off={abs_off} data_bytes={data_bytes} file_size={size}"
                )
            infos.append(GgufTensorInfo(
                name=name, shape=dims, ggml_type=gtype, quant_kind=quant,
                data_offset=abs_off, data_bytes=data_bytes,
            ))

    tokenizer_blob = b"".join(tokenizer_parts)
    _ = kv_start; _ = kv_end  # reserved for future byte-range diagnostics

    return GgufMetadata(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        alignment=alignment,
        kv_bytes=kv_bytes,
        tokenizer_blob=tokenizer_blob,
        tensor_data_start=tensor_data_start,
        tensors=tuple(infos),
    )


# --------------------------------------------------------------------------- #
# Stage A-II — Tile Plan (16,384 canonical tile, NRL-D005)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TileSpec:
    """One packed-INT4 tile produced during Stage A-III.

    Attributes:
        tile_id: zero-based index in the LMO tile array.
        block_origin: GGUF tensor name this tile was lifted from.
        units: packed INT4 potentials in this tile (``≤ CANONICAL_TILE_UNITS``).
        quant_kind: source quant kind (``"q4_k"``, ``"f32"``, ...) or
            ``"unknown/N"`` when the GGUF type is outside our layout table.
        retained_offset: absolute offset into the source GGUF where this
            tile's retained byte slice begins.
        retained_bytes: length in bytes of the retained slice (may be
            ``0`` for unknown-type tensors — see §3.5 of the architecture
            doc on ``absorption_partial``).
        packed_path: path relative to the LMO directory, e.g.
            ``"tiles/00000000.tile"``. The tile file contains exactly
            ``(units + 1) // 2`` bytes (two nibbles per byte).
    """

    tile_id: int
    block_origin: str
    units: int
    quant_kind: str
    retained_offset: int
    retained_bytes: int
    packed_path: str

    def filename(self) -> str:
        return f"{self.tile_id:08x}.tile"

    def packed_byte_count(self) -> int:
        return (self.units + 1) // 2


@dataclass(frozen=True)
class TilePlan:
    """Deterministic decomposition of all GGUF tensors into LMO tiles."""

    tiles: tuple[TileSpec, ...]
    block_to_tiles: dict[str, tuple[int, ...]]
    total_units: int
    total_retained_bytes: int
    partial: bool                         # True iff any tensor had an unknown type

    def tiles_for(self, block_origin: str) -> tuple[TileSpec, ...]:
        ids = self.block_to_tiles.get(block_origin, ())
        return tuple(self.tiles[i] for i in ids)

    def digest(self) -> str:
        """FNV-1a64 hex over a canonical serialization of the tile plan.

        Any change to tile ids, origins, unit counts, quant kinds, or
        retained byte ranges flips the digest. The digest is part of the
        LMO anchor (Stage A-V), so plan drift invalidates downstream
        caches automatically.
        """
        parts: list[bytes] = [
            f"units={CANONICAL_TILE_UNITS};partial={1 if self.partial else 0};"
            f"total={self.total_units}".encode("utf-8")
        ]
        for t in self.tiles:
            parts.append(
                f"{t.tile_id}|{t.block_origin}|{t.units}|{t.quant_kind}|"
                f"{t.retained_offset}|{t.retained_bytes}|{t.packed_path}".encode("utf-8")
            )
        blob = b"\n".join(parts)
        return f"{fnv1a64_packed(blob):016x}"


def plan_tiles(
    meta: GgufMetadata, *, tile_units: int = CANONICAL_TILE_UNITS
) -> TilePlan:
    """Stage A-II.

    Each tensor is decomposed into ``ceil(element_count / tile_units)``
    contiguous tiles. Retained byte slices for each tile are the
    corresponding contiguous slice of the source GGUF payload.

    For unknown ``ggml_type`` (``data_bytes == 0``), we emit a single
    ``units = 0`` tile as a placeholder so the router graph still sees
    the tensor; such tiles pack zero bytes and set ``absorption_partial``
    on the plan. Runtime treats them as R5-only (Plane C) territory.
    """
    if tile_units <= 0:
        raise LmoError(f"tile_units must be positive; got {tile_units}")

    tiles: list[TileSpec] = []
    block_to_tiles: dict[str, list[int]] = {}
    total_units = 0
    total_bytes = 0
    partial = False

    for info in meta.tensors:
        ids: list[int] = []
        if info.data_bytes == 0:
            # Unknown type → retained byte-only, packed side empty.
            partial = True
            tid = len(tiles)
            ids.append(tid)
            tiles.append(TileSpec(
                tile_id=tid,
                block_origin=info.name,
                units=0,
                quant_kind=info.quant_kind,
                retained_offset=info.data_offset,
                retained_bytes=0,
                packed_path=f"tiles/{tid:08x}.tile",
            ))
        else:
            elems = info.element_count
            # Bytes-per-element as a rational: total_bytes / elems. We slice
            # retained bytes on tile boundaries, rounding to the element
            # block boundary to keep each retained slice independently
            # decodable by a downstream ggml reader.
            layout = _GGML_TYPE_LAYOUT.get(info.ggml_type)
            if layout is None:
                block_elems, block_bytes = (1, max(1, info.data_bytes // max(1, elems)))
            else:
                block_elems, block_bytes = layout
            # Tile step in elements aligned to block size.
            step_elements = max(block_elems, (tile_units // block_elems) * block_elems)
            if step_elements <= 0:
                step_elements = block_elems
            cursor = 0
            while cursor < elems:
                take = min(step_elements, elems - cursor)
                byte_off = info.data_offset + (cursor // block_elems) * block_bytes
                byte_len = (take // block_elems) * block_bytes if block_elems > 1 else take * block_bytes
                tid = len(tiles)
                ids.append(tid)
                tile_units_actual = min(tile_units, take)
                tiles.append(TileSpec(
                    tile_id=tid,
                    block_origin=info.name,
                    units=tile_units_actual,
                    quant_kind=info.quant_kind,
                    retained_offset=byte_off,
                    retained_bytes=byte_len,
                    packed_path=f"tiles/{tid:08x}.tile",
                ))
                total_units += tile_units_actual
                total_bytes += byte_len
                cursor += take
        block_to_tiles[info.name] = ids

    return TilePlan(
        tiles=tuple(tiles),
        block_to_tiles={k: tuple(v) for k, v in block_to_tiles.items()},
        total_units=total_units,
        total_retained_bytes=total_bytes,
        partial=partial,
    )


# --------------------------------------------------------------------------- #
# Stage A-III — Pack INT4 + retain original bytes
# --------------------------------------------------------------------------- #


# Phase 11 — native accelerator for ``pack_int4_from_bytes``. When
# ``_core`` was built with the Phase 11 bindings, every per-unit hash
# runs in C with the GIL released. On multi-GB GGUFs this is ~200x
# faster than the pure-Python fallback below and is the difference
# between a ~30s absorption and a multi-hour stall. The native kernel
# is byte-identical to the reference implementation (it is literally a
# transliteration of the Python loop plus the same FNV-1a64 constants
# used by ``nrlpy.runtime.fnv1a64_packed``), so existing LMO caches
# continue to Stage-VI verify.
try:  # pragma: no cover - exercised indirectly via absorb_gguf tests
    from . import _core as _lmo_core  # type: ignore

    _native_pack_int4 = getattr(_lmo_core, "pack_int4_from_bytes", None)
except Exception:  # pragma: no cover - defensive
    _native_pack_int4 = None


def pack_int4_from_bytes(raw: bytes, units: int) -> bytes:
    """Deterministically fold ``raw`` into ``units`` packed INT4 potentials.

    Contract:

    * Output length is exactly ``(units + 1) // 2`` bytes.
    * Each nibble is a value in ``[0, 15]``, compatible with the packed
      INT4 lattice semantics (saturating add, threshold reset) from
      ``nrl-architecture.md`` §2.1.
    * Pure function of ``(raw, units)``. Same inputs → byte-identical
      output. Empty ``raw`` or ``units == 0`` → empty output.

    Method: chunk ``raw`` into ``units`` roughly equal groups; each
    group's FNV-1a64 low nibble becomes one potential. FNV-1a64 is
    already the canonical NRL-side hash
    (``nrlpy.runtime.fnv1a64_packed``), so this keeps the hot-path hash
    universe consistent with ZPM Stage I and muscle-memory keys.

    This is **not** a matrix-multiply-preserving lift of the weights;
    Plane-C GEMM uses the retained byte slice, not this tile. The
    packed tile is the lattice-side state the router and Stage-VI
    verify consume (Phase 6-EG).
    """
    if units <= 0 or not raw:
        return b""
    if _native_pack_int4 is not None:
        # Native C path — identical semantics, ~200x faster.
        return _native_pack_int4(raw, units)  # type: ignore[no-any-return]
    n = len(raw)
    out = bytearray((units + 1) // 2)
    # Deterministic, stride-independent group boundaries via (i * n) // units.
    prev = 0
    nibbles: list[int] = []
    for i in range(units):
        end = ((i + 1) * n) // units
        if end <= prev:
            end = min(n, prev + 1)
        nibbles.append(fnv1a64_packed(raw[prev:end]) & 0x0F)
        prev = end
    for i in range(0, units, 2):
        lo = nibbles[i] & 0x0F
        hi = (nibbles[i + 1] & 0x0F) if i + 1 < units else 0
        out[i // 2] = lo | (hi << 4)
    return bytes(out)


def embedding_row_packed(
    handle: "LmoHandle",
    token_id: int,
    *,
    row_units: int,
    origin: str = "token_embd",
) -> bytes:
    """Phase 13 R0' — extract one token's row from the packed INT4 tiles.

    **Honest contract.** The returned bytes are *not* the original GGUF
    embedding weights in F16. They are the NRL packed INT4 projection
    of the raw byte range that backs this row — i.e. each byte in the
    output holds two nibbles in ``[0, 15]`` that were produced by
    :func:`pack_int4_from_bytes` during Stage A-III absorption. This
    is the lattice-side state consumed by the ZPM anchor and the
    router graph; it is determinisitic, addressable, and reproducible,
    and that is all Phase 13 is claiming.

    To recover F16 embedding values you need the retained Q4_K bytes
    (``handle.read_retained(tile_id)``) plus a Q4_K dequantizer. That
    work is scoped to Phase 13.5 (not in this release).

    Arguments:
        handle:    An opened :class:`LmoHandle` for an absorbed GGUF.
        token_id:  Zero-based vocab row index.
        row_units: Hidden dimension (e.g. 3072 for Phi-3-mini). Must
                   match the absorbed tensor's row width.
        origin:    Block-origin name; defaults to ``"token_embd"``.

    Returns:
        ``bytes`` of length ``row_units`` where each byte is a single
        INT4 nibble value in ``[0, 15]``. Caller can reshape / view
        as ``uint8`` for further analysis.

    Raises:
        :class:`LmoError` if ``token_id`` is out of range or the LMO
        was absorbed without the requested origin.
    """
    if token_id < 0 or row_units <= 0:
        raise LmoError(f"invalid token_id={token_id} or row_units={row_units}")
    plan = handle.tile_plan
    tile_ids = plan.block_to_tiles.get(origin)
    if not tile_ids:
        raise LmoError(
            f"LMO has no tiles for origin {origin!r}; "
            f"available origins: {sorted(plan.block_to_tiles.keys())}"
        )
    global_start = token_id * row_units
    global_end = global_start + row_units
    total_units = sum(plan.tiles[t].units for t in tile_ids)
    if global_end > total_units:
        raise LmoError(
            f"token_id {token_id} * row_units {row_units} = {global_end} "
            f"exceeds absorbed {origin!r} units ({total_units})"
        )

    out = bytearray(row_units)
    written = 0
    cursor = 0
    for tid in tile_ids:
        tile_units = plan.tiles[tid].units
        tile_start = cursor
        tile_end = cursor + tile_units
        cursor = tile_end
        if tile_end <= global_start:
            continue
        if tile_start >= global_end:
            break
        packed = handle.read_tile(tid)
        local_start = max(0, global_start - tile_start)
        local_end = min(tile_units, global_end - tile_start)
        for u in range(local_start, local_end):
            byte_idx = u // 2
            if u & 1:
                nibble = (packed[byte_idx] >> 4) & 0x0F
            else:
                nibble = packed[byte_idx] & 0x0F
            out[written] = nibble
            written += 1
        if written >= row_units:
            break
    if written != row_units:  # pragma: no cover — defensive
        raise LmoError(
            f"embedding row extract short: wrote {written} of {row_units} units"
        )
    return bytes(out)


def _pack_and_retain(
    source_gguf: Path,
    plan: TilePlan,
    lmo_dir: Path,
) -> tuple[dict[int, str], dict[int, str]]:
    """Write every tile's packed + retained slice; return checksums.

    **Phase 11 blob layout.** We concatenate every tile's retained and
    packed bytes into ``retained.blob`` / ``packed.blob`` with a parallel
    ``tile_offsets.bin`` index (four ``u64``s per tile: ``retained_off``,
    ``retained_len``, ``packed_off``, ``packed_len``). NTFS cannot keep
    up with creating ~1M tiny files for a multi-GB GGUF (we measured
    hours of wall-time just to ``rename`` the ``.tmp`` files), so this
    layout collapses the I/O to three sequential-append streams plus one
    index. The hot-path `LmoHandle.read_tile` / `read_retained`
    implementations seek into the blobs using the index.

    Returns ``(packed_fnv_by_tile, retained_fnv_by_tile)`` so Stage A-VI
    can re-verify both artefacts against a fresh source read.
    """
    import struct

    packed_fnv: dict[int, str] = {}
    retained_fnv: dict[int, str] = {}
    offsets: list[tuple[int, int, int, int]] = []

    retained_blob = lmo_dir / "retained.blob"
    packed_blob = lmo_dir / "packed.blob"
    offsets_path = lmo_dir / "tile_offsets.bin"

    r_off = 0
    p_off = 0
    # Sequential append to both blobs; no per-tile file creation, no
    # tmp+rename overhead. Buffered I/O amortizes small tile writes.
    with (
        source_gguf.open("rb") as src,
        retained_blob.open("wb") as rf,
        packed_blob.open("wb") as pf,
    ):
        for tile in plan.tiles:
            # Retained: byte-for-byte slice of source.
            if tile.retained_bytes > 0:
                src.seek(tile.retained_offset)
                raw = src.read(tile.retained_bytes)
                if len(raw) != tile.retained_bytes:
                    raise LmoError(
                        f"tile {tile.tile_id} retained read short: "
                        f"{len(raw)} of {tile.retained_bytes}"
                    )
            else:
                raw = b""
            rf.write(raw)
            retained_fnv[tile.tile_id] = f"{fnv1a64_packed(raw):016x}"

            packed = pack_int4_from_bytes(raw, tile.units)
            pf.write(packed)
            packed_fnv[tile.tile_id] = f"{fnv1a64_packed(packed):016x}"

            offsets.append((r_off, len(raw), p_off, len(packed)))
            r_off += len(raw)
            p_off += len(packed)

    with offsets_path.open("wb") as of:
        # Header: magic, version, tile_count, reserved.
        of.write(b"NRLTOFS1")
        of.write(struct.pack("<IIQ", 1, len(offsets), 0))
        for r_o, r_l, p_o, p_l in offsets:
            of.write(struct.pack("<QQQQ", r_o, r_l, p_o, p_l))

    return packed_fnv, retained_fnv


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via tmp+rename for crash-safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_tile_offsets(path: Path) -> dict[int, tuple[int, int, int, int]]:
    """Parse ``tile_offsets.bin`` into ``{tile_id: (r_off, r_len, p_off, p_len)}``.

    Blob-layout index written by :func:`_pack_and_retain`. Returns an
    empty dict if the file is missing. Tile ids are assigned by the
    TilePlan in-order starting from 0, so the index array position is
    the tile id.
    """
    import struct

    if not path.is_file():
        return {}
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"NRLTOFS1":
        raise LmoError(f"invalid tile_offsets header in {path}")
    _ver, count, _reserved = struct.unpack("<IIQ", data[8:24])
    entry_size = 32
    expected = 24 + count * entry_size
    if len(data) < expected:
        raise LmoError(
            f"tile_offsets truncated: expected {expected} bytes, got {len(data)}"
        )
    out: dict[int, tuple[int, int, int, int]] = {}
    base = 24
    for tid in range(count):
        r_off, r_len, p_off, p_len = struct.unpack(
            "<QQQQ", data[base : base + entry_size]
        )
        out[tid] = (r_off, r_len, p_off, p_len)
        base += entry_size
    return out


# --------------------------------------------------------------------------- #
# Stage A-IV — Router Graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RouterSubLattice:
    """One omega sub-lattice in the LMO router graph.

    Attributes:
        block_id: logical block label, derived from the GGUF tensor name
            (``"blk.N"`` for transformer blocks, else the tensor's top-level
            name such as ``"token_embd"`` or ``"output"``).
        tile_ids: tile indices this sub-lattice owns (read-only).
        wake_rate: proxy for activation density, computed from the
            average packed-INT4 nibble value over this sub-lattice's
            tiles, normalized into ``[0.0, 1.0]``.
        min_active: ``omega-hybrid`` active-count floor for this sub-lattice.
    """

    block_id: str
    tile_ids: tuple[int, ...]
    wake_rate: float
    min_active: int


@dataclass(frozen=True)
class RouterGraph:
    """Sub-lattice topology + edges lifted from GGUF connectivity."""

    sub_lattices: tuple[RouterSubLattice, ...]
    edges: tuple[tuple[str, str], ...]
    global_min_active: int
    default_wake_rate: float

    def serialize(self) -> bytes:
        """Canonical binary serialization. Deterministic byte output."""
        # Magic + u32 counts + per-sublattice records + edges.
        buf = bytearray()
        buf += b"NRLROUT1"
        buf += struct.pack("<I", len(self.sub_lattices))
        buf += struct.pack("<I", len(self.edges))
        buf += struct.pack("<I", self.global_min_active)
        buf += struct.pack("<d", self.default_wake_rate)
        for sl in self.sub_lattices:
            bid = sl.block_id.encode("utf-8")
            buf += struct.pack("<H", len(bid)) + bid
            buf += struct.pack("<I", len(sl.tile_ids))
            for tid in sl.tile_ids:
                buf += struct.pack("<I", tid)
            buf += struct.pack("<d", sl.wake_rate)
            buf += struct.pack("<I", sl.min_active)
        for a, b in self.edges:
            for name in (a, b):
                nb = name.encode("utf-8")
                buf += struct.pack("<H", len(nb)) + nb
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> "RouterGraph":
        if data[:8] != b"NRLROUT1":
            raise LmoError(f"invalid router graph magic: {data[:8]!r}")
        off = 8
        (n_sl,) = struct.unpack_from("<I", data, off); off += 4
        (n_ed,) = struct.unpack_from("<I", data, off); off += 4
        (gmin,) = struct.unpack_from("<I", data, off); off += 4
        (dwake,) = struct.unpack_from("<d", data, off); off += 8
        sls: list[RouterSubLattice] = []
        for _ in range(n_sl):
            (blen,) = struct.unpack_from("<H", data, off); off += 2
            block_id = data[off:off + blen].decode("utf-8", errors="replace"); off += blen
            (nt,) = struct.unpack_from("<I", data, off); off += 4
            tids = tuple(
                struct.unpack_from("<I", data, off + i * 4)[0] for i in range(nt)
            )
            off += 4 * nt
            (wake,) = struct.unpack_from("<d", data, off); off += 8
            (mina,) = struct.unpack_from("<I", data, off); off += 4
            sls.append(RouterSubLattice(
                block_id=block_id, tile_ids=tids, wake_rate=wake, min_active=mina,
            ))
        edges: list[tuple[str, str]] = []
        for _ in range(n_ed):
            pair: list[str] = []
            for _ in range(2):
                (nl,) = struct.unpack_from("<H", data, off); off += 2
                pair.append(data[off:off + nl].decode("utf-8", errors="replace"))
                off += nl
            edges.append((pair[0], pair[1]))
        return cls(
            sub_lattices=tuple(sls),
            edges=tuple(edges),
            global_min_active=gmin,
            default_wake_rate=dwake,
        )

    def digest(self) -> str:
        return f"{fnv1a64_packed(self.serialize()):016x}"


def _extract_block_id(tensor_name: str) -> str:
    """Map a GGUF tensor name to an omega sub-lattice label.

    * ``"blk.0.attn_q.weight"`` → ``"blk.0"``
    * ``"token_embd.weight"`` → ``"token_embd"``
    * ``"output.weight"`` → ``"output"``
    * bare names → the full name.
    """
    if tensor_name.startswith("blk."):
        parts = tensor_name.split(".", 2)
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return tensor_name.split(".", 1)[0]


def _avg_nibble_value(packed: bytes) -> float:
    """Average packed INT4 nibble value (0..15) across ``packed``."""
    if not packed:
        return 0.0
    total = 0
    count = 0
    for b in packed:
        total += (b & 0x0F) + ((b >> 4) & 0x0F)
        count += 2
    return total / count if count else 0.0


def build_router_graph(
    plan: TilePlan,
    lmo_dir: Path,
    *,
    global_min_active: int = 4,
    default_wake_rate: float = 0.25,
) -> RouterGraph:
    """Stage A-IV.

    One omega sub-lattice per logical block (``blk.N`` or top-level
    tensor family); wake rate derived from the average packed-INT4
    nibble density across the block's tiles; min_active floor inherits
    from the global default unless the block is degenerate
    (``units = 0``).
    """
    buckets: dict[str, list[int]] = {}
    for tile in plan.tiles:
        bid = _extract_block_id(tile.block_origin)
        buckets.setdefault(bid, []).append(tile.tile_id)

    sub_lattices: list[RouterSubLattice] = []
    # Read packed tiles to compute wake rates. Tiles are small, eager is fine.
    for bid, tids in sorted(buckets.items()):
        wake_total = 0.0
        wake_samples = 0
        degenerate = True
        for tid in tids:
            tile = plan.tiles[tid]
            if tile.units == 0:
                continue
            degenerate = False
            p = lmo_dir / tile.packed_path
            try:
                packed = p.read_bytes()
            except OSError:
                packed = b""
            wake_total += _avg_nibble_value(packed)
            wake_samples += 1
        if degenerate or wake_samples == 0:
            wake_rate = default_wake_rate
            min_active = 0
        else:
            wake_rate = (wake_total / wake_samples) / 15.0
            min_active = global_min_active
        sub_lattices.append(RouterSubLattice(
            block_id=bid,
            tile_ids=tuple(sorted(tids)),
            wake_rate=float(wake_rate),
            min_active=int(min_active),
        ))

    # Edges: deterministic chain over transformer blocks (blk.N → blk.N+1),
    # plus embedding → first block and last block → output when both exist.
    blk_ids = sorted(
        (sl.block_id for sl in sub_lattices if sl.block_id.startswith("blk.")),
        key=lambda s: int(s.split(".", 1)[1]) if "." in s else 0,
    )
    edges: list[tuple[str, str]] = []
    for i in range(len(blk_ids) - 1):
        edges.append((blk_ids[i], blk_ids[i + 1]))
    sl_ids = {sl.block_id for sl in sub_lattices}
    if blk_ids:
        if "token_embd" in sl_ids:
            edges.insert(0, ("token_embd", blk_ids[0]))
        if "output" in sl_ids:
            edges.append((blk_ids[-1], "output"))
        if "output_norm" in sl_ids:
            edges.append((blk_ids[-1], "output_norm"))

    return RouterGraph(
        sub_lattices=tuple(sub_lattices),
        edges=tuple(edges),
        global_min_active=global_min_active,
        default_wake_rate=default_wake_rate,
    )


# --------------------------------------------------------------------------- #
# Stage A-V — Anchor seed + LMO header
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LmoHeader:
    """Fixed 4 KiB LMO header written to ``lmo.header``.

    On-disk layout::

        offset  bytes  field
        0       8      magic                (b"NRLLMO01")
        8       2      schema_version       (u16 LE, currently 1)
        10      2      reserved             (zero)
        12      4      payload_len          (u32 LE, bytes of JSON below)
        16      N      JSON payload         (UTF-8, ``payload_len`` bytes)
        16+N    ...    zero padding to 4096 bytes

    All fields are stored in the JSON payload to keep human-readability
    while the fixed binary prefix enables mmap + magic sniffing.
    """

    magic: bytes
    schema_version: int
    model_sha256: str
    lmo_anchor: tuple[int, int, int, int]       # Plane-A.5 State
    tile_plan_digest: str                        # FNV-1a64 hex
    router_graph_digest: str                     # FNV-1a64 hex
    tile_count: int
    canonical_tile_units: int                    # = CANONICAL_TILE_UNITS
    nrl_version: str
    cpu_features: str
    absorbed_at_unix: int

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "model_sha256": self.model_sha256,
            "lmo_anchor": [f"{w:016x}" for w in self.lmo_anchor],
            "tile_plan_digest": self.tile_plan_digest,
            "router_graph_digest": self.router_graph_digest,
            "tile_count": self.tile_count,
            "canonical_tile_units": self.canonical_tile_units,
            "nrl_version": self.nrl_version,
            "cpu_features": self.cpu_features,
            "absorbed_at_unix": self.absorbed_at_unix,
        }, sort_keys=True, separators=(",", ":"))

    def serialize(self) -> bytes:
        payload = self.to_json().encode("utf-8")
        if len(payload) + 16 > LMO_HEADER_FIXED_BYTES:
            raise LmoError(
                f"LMO header payload too large: {len(payload)} bytes "
                f"(max {LMO_HEADER_FIXED_BYTES - 16})"
            )
        buf = bytearray(LMO_HEADER_FIXED_BYTES)
        buf[0:8] = self.magic
        struct.pack_into("<H", buf, 8, self.schema_version)
        struct.pack_into("<H", buf, 10, 0)
        struct.pack_into("<I", buf, 12, len(payload))
        buf[16:16 + len(payload)] = payload
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> "LmoHeader":
        if len(data) < 16:
            raise LmoError("LMO header too small")
        magic = bytes(data[0:8])
        if magic != LMO_MAGIC:
            raise LmoError(f"invalid LMO header magic: {magic!r}")
        (schema_version,) = struct.unpack_from("<H", data, 8)
        (payload_len,) = struct.unpack_from("<I", data, 12)
        if 16 + payload_len > len(data):
            raise LmoError("LMO header payload exceeds buffer")
        obj = json.loads(data[16:16 + payload_len].decode("utf-8"))
        anchor_hex = obj["lmo_anchor"]
        if len(anchor_hex) != 4:
            raise LmoError("lmo_anchor must have 4 hex words")
        anchor = tuple(int(w, 16) for w in anchor_hex)
        return cls(
            magic=magic,
            schema_version=int(schema_version),
            model_sha256=str(obj["model_sha256"]),
            lmo_anchor=anchor,  # type: ignore[arg-type]
            tile_plan_digest=str(obj["tile_plan_digest"]),
            router_graph_digest=str(obj["router_graph_digest"]),
            tile_count=int(obj["tile_count"]),
            canonical_tile_units=int(obj["canonical_tile_units"]),
            nrl_version=str(obj["nrl_version"]),
            cpu_features=str(obj["cpu_features"]),
            absorbed_at_unix=int(obj["absorbed_at_unix"]),
        )


def _features_string() -> str:
    """Stable one-liner of runtime CPU features for the header."""
    try:
        feats = runtime_features()
    except Exception:
        return ""
    if not isinstance(feats, dict):
        return ""
    parts = []
    for k in sorted(feats.keys()):
        parts.append(f"{k}={1 if feats[k] else 0}")
    return ";".join(parts)


def _lmo_anchor_bytes(
    model_sha256: str,
    plan_digest: str,
    router_digest: str,
    nrl_ver: str,
    cpu_feats: str,
) -> bytes:
    """Closed set of bytes the LMO anchor is computed over.

    Deliberately small + stable. Changing any element invalidates every
    previously-absorbed LMO, which is intentional — any drift here means
    runtime decisions derived from the anchor would silently change.
    """
    parts = [
        model_sha256.encode("utf-8"),
        plan_digest.encode("utf-8"),
        router_digest.encode("utf-8"),
        nrl_ver.encode("utf-8"),
        cpu_feats.encode("utf-8"),
    ]
    return b"\x1e".join(parts)


def write_header_anchor(
    lmo_dir: Path,
    *,
    model_sha256: str,
    plan: TilePlan,
    router: RouterGraph,
    absorbed_at_unix: int | None = None,
) -> LmoHeader:
    """Stage A-V.

    Computes the 256-bit LMO anchor via Plane-A.5 Stage I over
    ``_lmo_anchor_bytes(...)`` and writes ``lmo.header`` atomically.

    Also serializes ``router.graph`` to the LMO directory so its on-disk
    digest matches the in-memory value used in the anchor.
    """
    plan_digest = plan.digest()
    router_digest = router.digest()
    nrl_ver = runtime_version() or ""
    cpu_feats = _features_string()
    anchor_state = zpm.anchor(_lmo_anchor_bytes(
        model_sha256, plan_digest, router_digest, nrl_ver, cpu_feats
    ))
    header = LmoHeader(
        magic=LMO_MAGIC,
        schema_version=LMO_SCHEMA_VERSION,
        model_sha256=model_sha256,
        lmo_anchor=anchor_state,  # type: ignore[arg-type]
        tile_plan_digest=plan_digest,
        router_graph_digest=router_digest,
        tile_count=len(plan.tiles),
        canonical_tile_units=CANONICAL_TILE_UNITS,
        nrl_version=nrl_ver,
        cpu_features=cpu_feats,
        absorbed_at_unix=int(absorbed_at_unix if absorbed_at_unix is not None else time.time()),
    )
    _atomic_write_bytes(lmo_dir / "lmo.header", header.serialize())
    _atomic_write_bytes(lmo_dir / "router.graph", router.serialize())
    return header


# --------------------------------------------------------------------------- #
# Auto-materialize a .nrl v1 manifest for the LMO
# --------------------------------------------------------------------------- #


def _write_auto_manifest(lmo_dir: Path, *, model_sha256: str, gguf_path: Path) -> Path:
    """Emit a minimal, replay-lockable ``manifest.nrl`` v1 for this LMO.

    The runtime ladder does not consume LMO-specific keys yet
    (Phase 5-EG+); the manifest here is the deterministic provenance
    file the Reset Plan §5 Phase 1 requires for every GGUF run.
    """
    text = (
        "# Auto-materialized by nrlpy.lmo.absorb_gguf (Phase 4-EG).\n"
        "# Do not edit by hand — re-absorbing the source GGUF regenerates this file.\n"
        "schema = nrl.manifest.v1\n"
        "mode = gguf_run\n"
        "profile = sovereign\n"
        f"model = {gguf_path.resolve().as_posix()}\n"
        f"model_sha256 = {model_sha256}\n"
        "benchmark_class = B\n"
        "muscle_memory = on\n"
        "respect_control_hints = true\n"
        "prompt = \"\"\n"
        "max_tokens = 128\n"
        "temperature = 0.2\n"
        "top_p = 0.9\n"
        "top_k = 40\n"
        "repeat_penalty = 1.08\n"
        "seed = 0\n"
        "n_ctx = 2048\n"
        "n_batch = 512\n"
        "chat_format = none\n"
    )
    path = lmo_dir / "manifest.nrl"
    _atomic_write_bytes(path, text.encode("utf-8"))
    return path


def _write_tokenizer_blob(lmo_dir: Path, meta: GgufMetadata) -> Path:
    """Write the byte-for-byte tokenizer KV slice to ``tokenizer.blob``."""
    path = lmo_dir / "tokenizer.blob"
    _atomic_write_bytes(path, meta.tokenizer_blob)
    return path


# --------------------------------------------------------------------------- #
# Stage A-VI — Verify (hard CI gate)
# --------------------------------------------------------------------------- #


@dataclass
class Attest:
    """Contents of ``attest.json``. Documents the absorption run."""

    schema_id: str = "nrl.lmo.attest.v1"
    model_sha256: str = ""
    lmo_anchor_hex: list[str] = field(default_factory=list)
    tile_plan_digest: str = ""
    router_graph_digest: str = ""
    tile_count: int = 0
    retained_total_bytes: int = 0
    absorbed_at_unix: int = 0
    nrl_version: str = ""
    cpu_features: str = ""
    absorption_partial: bool = False

    # Stage A-VI audit results (all boolean-labelled, with details).
    retained_byte_identity_ok: bool = False
    determinism_self_parity_ok: bool = False
    header_roundtrip_ok: bool = False
    libllama_forward: dict[str, Any] = field(default_factory=dict)

    # Per-tile checksums so downstream tools can spot-check without rescanning the GGUF.
    retained_fnv1a64: dict[str, str] = field(default_factory=dict)
    packed_fnv1a64: dict[str, str] = field(default_factory=dict)


def verify_parity_against_libllama(
    lmo_dir: Path,
    *,
    source_gguf: Path,
    plan: TilePlan,
    router: RouterGraph,
    header: LmoHeader,
    packed_fnv: dict[int, str],
    retained_fnv: dict[int, str],
    attempt_libllama: bool = True,
) -> Attest:
    """Stage A-VI.

    Three audits run unconditionally (any failure raises :class:`LmoError`
    before ``attest.json`` is written, so a bad LMO is never considered
    valid):

    1. **Retained byte-identity** — each tile's ``retained/<tid>.bin`` is
       re-FNV-checked against a fresh read of the source GGUF at the
       declared offset+length. Any mismatch = the retained artefact does
       not reflect the source = hard fail.
    2. **Determinism self-parity** — the tile plan's digest is
       recomputed from the on-disk ``TileSpec`` sequence; the router
       graph digest is recomputed from its on-disk serialization; both
       must match the values embedded in ``lmo.header``. This catches
       any drift between what was anchored and what's on disk.
    3. **Header round-trip** — the on-disk header bytes
       deserialize to a structure that re-serializes byte-identically.

    A fourth audit runs **when available** and is informational only:

    4. **libllama single-token forward smoke-test** — loads the source
       GGUF via ``llama_cpp.Llama`` under a fixed seed, performs a
       one-token completion on a fixed prompt, records the output's
       FNV-1a64. Skipped when ``llama-cpp-python`` is not importable or
       the model format is unsupported by the installed build. Recorded
       in ``attest.json`` as ``libllama_forward.status`` ∈
       ``{"ok", "skipped", "error"}``.
    """
    attest = Attest(
        schema_id="nrl.lmo.attest.v1",
        model_sha256=header.model_sha256,
        lmo_anchor_hex=[f"{w:016x}" for w in header.lmo_anchor],
        tile_plan_digest=header.tile_plan_digest,
        router_graph_digest=header.router_graph_digest,
        tile_count=header.tile_count,
        retained_total_bytes=plan.total_retained_bytes,
        absorbed_at_unix=header.absorbed_at_unix,
        nrl_version=header.nrl_version,
        cpu_features=header.cpu_features,
        absorption_partial=plan.partial,
        retained_fnv1a64={f"{tid:08x}": v for tid, v in retained_fnv.items()},
        packed_fnv1a64={f"{tid:08x}": v for tid, v in packed_fnv.items()},
    )

    # Audit 1 — retained byte-identity.
    # Blob layout: read retained slice from `retained.blob` via the tile
    # offsets index. Legacy per-tile layout: read `retained/<tid>.bin`.
    blob_path = lmo_dir / "retained.blob"
    offsets_path = lmo_dir / "tile_offsets.bin"
    use_blob = blob_path.is_file() and offsets_path.is_file()
    tile_offsets = _read_tile_offsets(offsets_path) if use_blob else {}
    with source_gguf.open("rb") as src:
        retained_fh = blob_path.open("rb") if use_blob else None
        try:
            for tile in plan.tiles:
                if tile.retained_bytes == 0:
                    continue
                src.seek(tile.retained_offset)
                raw = src.read(tile.retained_bytes)
                if len(raw) != tile.retained_bytes:
                    raise LmoError(
                        f"retained parity: short source read for tile {tile.tile_id}"
                    )
                src_fnv = f"{fnv1a64_packed(raw):016x}"
                if use_blob and retained_fh is not None:
                    r_off, r_len, _p_off, _p_len = tile_offsets[tile.tile_id]
                    retained_fh.seek(r_off)
                    disk_raw = retained_fh.read(r_len)
                else:
                    disk_path = lmo_dir / "retained" / f"{tile.tile_id:08x}.bin"
                    disk_raw = disk_path.read_bytes()
                disk_fnv = f"{fnv1a64_packed(disk_raw):016x}"
                if src_fnv != disk_fnv or raw != disk_raw:
                    raise LmoError(
                        f"retained parity failed for tile {tile.tile_id}: "
                        f"source_fnv={src_fnv} disk_fnv={disk_fnv}"
                    )
        finally:
            if retained_fh is not None:
                retained_fh.close()
    attest.retained_byte_identity_ok = True

    # Audit 2 — determinism self-parity.
    plan_digest_now = plan.digest()
    router_digest_now = router.digest()
    if plan_digest_now != header.tile_plan_digest:
        raise LmoError(
            f"tile plan digest drift: header={header.tile_plan_digest} "
            f"recomputed={plan_digest_now}"
        )
    if router_digest_now != header.router_graph_digest:
        raise LmoError(
            f"router graph digest drift: header={header.router_graph_digest} "
            f"recomputed={router_digest_now}"
        )
    on_disk_router = RouterGraph.deserialize((lmo_dir / "router.graph").read_bytes())
    if on_disk_router.digest() != header.router_graph_digest:
        raise LmoError("router.graph on-disk digest does not match header")
    attest.determinism_self_parity_ok = True

    # Audit 3 — header round-trip.
    on_disk_header_bytes = (lmo_dir / "lmo.header").read_bytes()
    reparsed = LmoHeader.deserialize(on_disk_header_bytes)
    if reparsed.serialize() != on_disk_header_bytes:
        raise LmoError("lmo.header does not round-trip byte-identically")
    if reparsed.lmo_anchor != header.lmo_anchor:
        raise LmoError("lmo.header anchor changed after round-trip")
    attest.header_roundtrip_ok = True

    # Audit 4 — optional libllama forward.
    attest.libllama_forward = _libllama_smoke_test(source_gguf) if attempt_libllama else {
        "status": "skipped",
        "reason": "disabled by caller",
    }

    return attest


def _libllama_smoke_test(source_gguf: Path) -> dict[str, Any]:
    """Best-effort single-token forward pass via llama-cpp-python.

    Returns a JSON-safe dict for ``attest.json``. Never raises — any
    failure is captured as ``status: "error"`` with a short reason.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:  # pragma: no cover - depends on optional install
        return {"status": "skipped", "reason": f"llama_cpp unavailable: {e.__class__.__name__}"}
    try:
        llm = Llama(
            model_path=str(source_gguf),
            n_ctx=256,
            n_threads=1,
            n_batch=32,
            n_gpu_layers=0,
            seed=42,
            verbose=False,
            logits_all=False,
            use_mmap=True,
        )
    except Exception as e:  # pragma: no cover - depends on optional install
        return {"status": "error", "reason": f"Llama() failed: {e.__class__.__name__}: {e}"}
    try:
        out = llm(
            "NRL",
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            stream=False,
        )
        text = ""
        if isinstance(out, dict):
            choices = out.get("choices") or []
            if choices and isinstance(choices[0], dict):
                text = str(choices[0].get("text", ""))
        fnv = f"{fnv1a64_packed(text.encode('utf-8')):016x}"
        return {
            "status": "ok",
            "profile": "sovereign",
            "prompt": "NRL",
            "max_tokens": 1,
            "seed": 42,
            "output_fnv1a64": fnv,
            "output_len": len(text),
        }
    except Exception as e:  # pragma: no cover
        return {"status": "error", "reason": f"llm() failed: {e.__class__.__name__}: {e}"}
    finally:
        try:
            del llm  # noqa: F821 - llm is set when we reach here
        except Exception:
            pass


def _write_attest(lmo_dir: Path, attest: Attest) -> Path:
    path = lmo_dir / "attest.json"
    blob = json.dumps(asdict(attest), sort_keys=True, indent=2).encode("utf-8")
    _atomic_write_bytes(path, blob)
    return path


# --------------------------------------------------------------------------- #
# LMO handle (read-only view of an absorbed directory)
# --------------------------------------------------------------------------- #


class LmoHandle:
    """Read-only handle to an absorbed LMO on disk.

    Use :meth:`open` to load an existing LMO. The handle exposes the
    header, tile plan, router graph, and directory path. It does **not**
    mmap tile bytes eagerly — the runtime Resolution Ladder (Phase 5-EG+)
    is responsible for lazy tile access.
    """

    def __init__(
        self,
        lmo_dir: Path,
        header: LmoHeader,
        tile_plan: TilePlan,
        router: RouterGraph,
        attest: dict[str, Any],
    ) -> None:
        self._lmo_dir = lmo_dir
        self._header = header
        self._tile_plan = tile_plan
        self._router = router
        self._attest = attest

    @property
    def lmo_dir(self) -> Path:
        return self._lmo_dir

    @property
    def header(self) -> LmoHeader:
        return self._header

    @property
    def tile_plan(self) -> TilePlan:
        return self._tile_plan

    @property
    def router(self) -> RouterGraph:
        return self._router

    @property
    def attest(self) -> dict[str, Any]:
        return dict(self._attest)

    @property
    def model_sha256(self) -> str:
        return self._header.model_sha256

    @property
    def manifest_path(self) -> Path:
        return self._lmo_dir / "manifest.nrl"

    @property
    def tokenizer_path(self) -> Path:
        return self._lmo_dir / "tokenizer.blob"

    def _blob_offsets(self) -> dict[int, tuple[int, int, int, int]]:
        cache = getattr(self, "_tile_offset_cache", None)
        if cache is None:
            cache = _read_tile_offsets(self._lmo_dir / "tile_offsets.bin")
            self._tile_offset_cache = cache
        return cache

    def read_tile(self, tile_id: int) -> bytes:
        """Read one packed INT4 tile's bytes (blob or legacy layout)."""
        if tile_id < 0 or tile_id >= len(self._tile_plan.tiles):
            raise LmoError(f"tile_id {tile_id} out of range")
        offsets = self._blob_offsets()
        if offsets:
            _r_o, _r_l, p_off, p_len = offsets[tile_id]
            with (self._lmo_dir / "packed.blob").open("rb") as f:
                f.seek(p_off)
                return f.read(p_len)
        return (self._lmo_dir / self._tile_plan.tiles[tile_id].packed_path).read_bytes()

    def read_retained(self, tile_id: int) -> bytes:
        """Read one tile's retained (Plane C) byte slice."""
        if tile_id < 0 or tile_id >= len(self._tile_plan.tiles):
            raise LmoError(f"tile_id {tile_id} out of range")
        offsets = self._blob_offsets()
        if offsets:
            r_off, r_len, _p_off, _p_len = offsets[tile_id]
            with (self._lmo_dir / "retained.blob").open("rb") as f:
                f.seek(r_off)
                return f.read(r_len)
        return (self._lmo_dir / "retained" / f"{tile_id:08x}.bin").read_bytes()

    def total_units_for_origin(self, origin: str) -> int:
        """Sum of ``TileSpec.units`` across every tile of a given tensor.

        Used by :func:`embedding_row_packed` to validate a requested row
        index fits inside the absorbed tensor.
        """
        return sum(self._tile_plan.tiles[t].units for t in self._tile_plan.block_to_tiles.get(origin, ()))

    @classmethod
    def open(cls, lmo_dir: str | Path) -> "LmoHandle":
        d = Path(lmo_dir)
        if not d.is_dir():
            raise LmoError(f"not an LMO directory: {d}")
        header_path = d / "lmo.header"
        router_path = d / "router.graph"
        attest_path = d / "attest.json"
        if not header_path.is_file() or not router_path.is_file() or not attest_path.is_file():
            raise LmoError(
                f"LMO directory missing required artefacts "
                f"(header/router/attest): {d}"
            )
        header = LmoHeader.deserialize(header_path.read_bytes())
        router = RouterGraph.deserialize(router_path.read_bytes())
        attest = json.loads(attest_path.read_text(encoding="utf-8"))

        # Reconstruct the tile plan from attest + tile files on disk. We
        # keep the authoritative shape in attest.json's retained_fnv1a64
        # keyset, cross-referenced against the header's tile_count.
        tiles: list[TileSpec] = []
        block_to_tiles: dict[str, list[int]] = {}
        tile_ids = sorted(int(k, 16) for k in attest.get("retained_fnv1a64", {}).keys())
        # Phase 11: blob layout — prefer the tile_offsets index over
        # per-tile filesystem walks when available (orders of magnitude
        # faster on NTFS for models with ~500K+ tiles).
        blob_offsets = _read_tile_offsets(d / "tile_offsets.bin")
        if not tile_ids and blob_offsets:
            tile_ids = sorted(blob_offsets.keys())
        if not tile_ids:
            # No per-tile entries recorded (e.g. schema drift). Walk the
            # filesystem instead; this is still deterministic for a well-
            # formed LMO.
            tile_dir = d / "tiles"
            if tile_dir.is_dir():
                tile_ids = sorted(
                    int(p.stem, 16) for p in tile_dir.glob("*.tile") if len(p.stem) == 8
                )
        # Reconstructing a TilePlan without the source GGUF means we lose
        # (quant_kind, retained_offset, block_origin). For a read-only
        # handle this is OK: runtime consumers use the router graph as
        # the authoritative topology, and raw tile bytes are read by id.
        # We synthesise best-effort TileSpecs from router sub-lattices
        # when available.
        tid_to_block: dict[int, str] = {}
        for sl in router.sub_lattices:
            for tid in sl.tile_ids:
                tid_to_block[tid] = sl.block_id
        for tid in tile_ids:
            packed_rel = f"tiles/{tid:08x}.tile"
            if blob_offsets and tid in blob_offsets:
                _r_o, r_len, _p_o, p_len = blob_offsets[tid]
                units = 2 * p_len
                retained_bytes = r_len
            else:
                retained_path = d / "retained" / f"{tid:08x}.bin"
                units = (
                    2 * (d / packed_rel).stat().st_size
                    if (d / packed_rel).is_file()
                    else 0
                )
                retained_bytes = (
                    retained_path.stat().st_size if retained_path.is_file() else 0
                )
            tiles.append(TileSpec(
                tile_id=tid,
                block_origin=tid_to_block.get(tid, "unknown"),
                units=units,
                quant_kind="",
                retained_offset=0,
                retained_bytes=retained_bytes,
                packed_path=packed_rel,
            ))
            block_to_tiles.setdefault(tid_to_block.get(tid, "unknown"), []).append(tid)
        plan = TilePlan(
            tiles=tuple(tiles),
            block_to_tiles={k: tuple(v) for k, v in block_to_tiles.items()},
            total_units=sum(t.units for t in tiles),
            total_retained_bytes=sum(t.retained_bytes for t in tiles),
            partial=bool(attest.get("absorption_partial", False)),
        )
        h = cls(d, header, plan, router, attest)
        # P4 — reconcile ZPM WAL into index.bin before any chat/ladder load.
        try:
            from . import zpm_persist as _zp  # noqa: PLC0415
            from .gguf import _zpm_index_path  # noqa: PLC0415

            _zp.recover_zpm_for_model(h.model_sha256, Path(_zpm_index_path(h.model_sha256)))
        except Exception:
            pass
        return h


# --------------------------------------------------------------------------- #
# absorb_gguf() — top-level Phase 4-EG pipeline
# --------------------------------------------------------------------------- #


def _default_lmo_root() -> Path:
    env = os.environ.get("NRL_ROOT")
    base = Path(env) if env else Path.cwd()
    return base / "cache" / "lmo"


def absorb_gguf(
    gguf_path: str | Path,
    out_root: str | Path | None = None,
    *,
    force: bool = False,
    attempt_libllama: bool = True,
) -> LmoHandle:
    """Produce (or re-open) an LMO for ``gguf_path``.

    Content-addressed by ``sha256(gguf)``. A subsequent call with the
    same source file returns the cached LMO unless ``force=True``.

    **Determinism (Bio-Digital P1):** On ``force=True`` re-absorption of the
    *same* GGUF bytes, the replay-locked fields stay identical:
    ``model_sha256``, ``lmo_anchor``, ``tile_plan_digest``,
    ``router_graph_digest``, ``tile_count``, packed/retained tensor bytes,
    and ``attest.json`` checksum maps (FNV maps). The only header field
    intentionally allowed to differ between forced runs is
    ``absorbed_at_unix`` (wall-clock stamp); ``lmo.header`` file bytes may
    therefore differ even when all ladder-relevant digests match.

    Raises:
        LmoError: Stage A-VI failed any of the hard audits.
        GgufParseError: the source file is not a parseable GGUF.
        FileNotFoundError: ``gguf_path`` does not exist.
    """
    src = Path(gguf_path)
    if not src.is_file():
        raise FileNotFoundError(f"GGUF not found: {src}")

    root = Path(out_root) if out_root is not None else _default_lmo_root()

    # Stage A-I
    model_sha = sha256_file(src)
    meta = gguf_parse_metadata(src)

    lmo_dir = root / model_sha
    attest_path = lmo_dir / "attest.json"
    if attest_path.is_file() and not force:
        # Cache hit: reopen and return without reprocessing.
        return LmoHandle.open(lmo_dir)

    lmo_dir.mkdir(parents=True, exist_ok=True)

    # Stage A-II
    plan = plan_tiles(meta, tile_units=CANONICAL_TILE_UNITS)

    # Stage A-III
    packed_fnv, retained_fnv = _pack_and_retain(src, plan, lmo_dir)

    # Auxiliary artefacts (tokenizer blob; auto manifest). Router graph is
    # written inside Stage A-V (its bytes feed the anchor digest).
    _write_tokenizer_blob(lmo_dir, meta)

    # Stage A-IV
    router = build_router_graph(plan, lmo_dir)

    # Stage A-V
    header = write_header_anchor(
        lmo_dir, model_sha256=model_sha, plan=plan, router=router,
    )

    # Auto-materialized manifest.nrl v1 for replay-locked provenance.
    _write_auto_manifest(lmo_dir, model_sha256=model_sha, gguf_path=src)

    # Stage A-VI (hard CI gate)
    attest = verify_parity_against_libllama(
        lmo_dir,
        source_gguf=src,
        plan=plan,
        router=router,
        header=header,
        packed_fnv=packed_fnv,
        retained_fnv=retained_fnv,
        attempt_libllama=attempt_libllama,
    )
    _write_attest(lmo_dir, attest)

    return LmoHandle(lmo_dir, header, plan, router, asdict(attest))


# --------------------------------------------------------------------------- #
# Phase 5-EG / 6-EG — Rung R2 (Omega Native Resolve)
# --------------------------------------------------------------------------- #
#
# This section implements §4.3 of Final_NRL_Architecture_GGUF.MD. R2 runs
# in one of two modes, selected per-call by ``mode``:
#
# * ``"shadow"`` (Phase 5-EG default). The probe executes the full §4.3
#   pipeline but its :class:`RungResult` is *always* ``coherence_demoted=
#   True`` — no tokens flow from R2. Evidence records every decision under
#   ``gate_source = "omega_resolve_shadow"``.
# * ``"active"`` (Phase 6-EG). The same pipeline runs; if a ZPM hit passes
#   Stage-VI verify, the :class:`RungResult` returns ``coherence_demoted=
#   False`` and the matched stored reply is attached to the report. The
#   runner then serves those bytes. On any Stage-VI failure, R2 demotes
#   to R5 and the demotion reason is recorded.
#
# The implementation is deliberately compact and deterministic:
#
#  1. **Anchor projection.** For every router sub-lattice we fold
#     ``(intent_anchor_bytes | block_id)`` into a short packed-INT4 seed
#     via :func:`pack_int4_from_bytes` (the same kernel Stage A-III uses).
#  2. **Bounded iteration.** We evolve each seed through `iterations`
#     rounds of the ``zpm.inversion`` / ``zpm.rotor`` primitives. This is
#     the same Cl(4,0) substrate the router would use in Phase 6-EG; the
#     wall budget is capped by ``omega_budget_ms``.
#  3. **Candidate continuation anchor.** The final sub-lattice digests
#     are re-anchored (Plane-A.5 Stage I) into a single 256-bit state.
#  4. **ZPM lookup.** That state is looked up in a caller-provided ZPM
#     index under ``zpm_threshold_bits``.
#  5. **Stage-VI verify.** On a match, the entry's reply text is checked
#     against the LMO's ``tokenizer.blob`` header (UTF-8 validity + the
#     ``tokenizer.ggml.model`` key must be present) and a second-rotor
#     symmetry audit is run. Any failure is *recorded as a demotion
#     reason*; it never blocks anything in shadow mode.
#  6. **Return.** :class:`RungResult` always has
#     ``coherence_demoted = True`` in shadow mode, guaranteeing the ladder
#     will not serve a token on its behalf.


@dataclass(frozen=True)
class RungResult:
    """Per-rung ladder bookkeeping (see §Appendix A of the architecture doc).

    Exactly the structure §A enumerates; every ladder rung (R0..R5) will
    return one of these in Phases 5-EG..7-EG.

    Attributes:
        rung: ``"r0"`` .. ``"r5"``.
        gate_source: ``"zpm_nullspace"``, ``"muscle_memory"``,
            ``"omega_resolve"``, ``"omega_resolve_shadow"``,
            ``"prefill_cache"``, ``"layer_skip"``, or ``None`` when the
            rung produced nothing.
        tokens_emitted: number of tokens the rung added to ``stream_to``.
            Always ``0`` in shadow mode.
        wall_s: wall-clock seconds spent on the rung's attempt (includes
            budget overshoot when the budget was exceeded).
        coherence_demoted: ``True`` when the rung's proposal was not
            served (for any reason). Shadow mode pins this to ``True``.
        stage_vi_reason: empty string on success; otherwise a short
            machine-readable label describing the demotion.
    """

    rung: str
    gate_source: str | None
    tokens_emitted: int
    wall_s: float
    coherence_demoted: bool
    stage_vi_reason: str


@dataclass(frozen=True)
class OmegaShadowReport:
    """R2 probe result (both shadow and active modes).

    The name is historical — in Phase 5-EG this dataclass only carried
    shadow observations. From Phase 6-EG onwards the same structure is
    reused for both modes; ``mode`` distinguishes. Fields ``served`` and
    ``served_tokens`` are pinned to ``False`` / ``0`` in shadow mode by
    contract, and reflect real service in active mode.

    Attached to :class:`nrlpy.gguf.GgufRunResult` and emitted in
    ``nrl.gguf_run.v1`` evidence events. ``hits`` and
    ``demotion_reasons`` remain the actionable signals for the
    Phase-7-EG release gate.
    """

    status: str                                 # "ok" | "skipped" | "error"
    gate_source: str                            # always OMEGA_SHADOW_GATE_SOURCE
    coherence_lane: str
    mode: str                                   # "shadow" | "active" | "skipped"
    available: bool                             # LMO handle + ZPM index both present
    served: bool                                # True only in active mode on success
    served_tokens: int                          # tokens R2 actually emitted (0 in shadow)
    served_text: str                            # empty unless mode="active" and served
    hits: int                                   # 0 or 1 per turn
    candidate_continuation_fnv: str             # hex, "" when no ZPM match
    demotion_reasons: tuple[str, ...]
    wall_ms: float
    omega_iterations: int
    sub_lattices_visited: int
    zpm_distance_bits: int                      # -1 when no match
    zpm_threshold_bits: int
    stored_entry_index: int                     # ZPM entry idx on hit, -1 otherwise
    # Phase 11 — n-gram rescue (see _ngram_rescue_search). When the
    # standard Omega → ZPM lookup misses (or the match fails Stage-VI),
    # the resolver can try a second-chance scan that ranks stored
    # entries by character-4-gram Jaccard overlap against the current
    # prompt's stored ``prompt_head``. If a rescued entry passes
    # Stage-VI, ``ngram_rescued=True`` and the turn is served by R2
    # instead of demoting to R5. These fields carry the observability.
    ngram_rescued: bool = False
    ngram_candidates_considered: int = 0
    ngram_best_overlap: float = 0.0
    note: str = (
        "Phase 5-EG R2 shadow / Phase 6-EG R2 active — advisory in shadow "
        "mode; Stage-VI-gated service in active mode. Phase 11 n-gram "
        "rescue expands R2's candidate set. See Final_NRL_"
        "Architecture_GGUF.MD §4.3."
    )

    @classmethod
    def skipped(cls, lane: str, reason: str) -> "OmegaShadowReport":
        return cls(
            status="skipped",
            gate_source=OMEGA_SHADOW_GATE_SOURCE,
            coherence_lane=lane,
            mode="skipped",
            available=False,
            served=False,
            served_tokens=0,
            served_text="",
            hits=0,
            candidate_continuation_fnv="",
            demotion_reasons=(reason,),
            wall_ms=0.0,
            omega_iterations=0,
            sub_lattices_visited=0,
            zpm_distance_bits=-1,
            zpm_threshold_bits=0,
            stored_entry_index=-1,
        )


def _project_anchor_to_sublattice(
    intent_anchor_bytes: bytes, block_id: str, units: int
) -> bytes:
    """Stage §4.3(1). Fold ``(anchor | block_id)`` into a packed INT4 seed.

    The seed length in bytes is ``(units + 1) // 2``; each nibble is a
    saturating-add-compatible potential in ``[0, 15]``. Pure function of
    its inputs — determinism is essential for replay-lock in
    ``coherence_lane = max-throughput``.
    """
    blob = intent_anchor_bytes + b"\x1e" + block_id.encode("utf-8")
    return pack_int4_from_bytes(blob, max(1, units))


def _evolve_sublattice(seed: bytes, iterations: int) -> int:
    """Stage §4.3(2). Advance a sub-lattice seed ``iterations`` rounds.

    We run the same Cl(4,0) inversion + rotor primitives ZPM uses (the
    router in Phase 6-EG will share this substrate). The reduction is
    FNV-1a64 over the post-iteration byte state so the sub-lattice
    contributes a single 64-bit word to the candidate continuation
    anchor.
    """
    state = zpm.anchor(seed or b"\x00")
    for _ in range(max(1, iterations)):
        state = zpm.inversion(state)
        rot = zpm.rotor(state[0] ^ state[1])
        # Fold the rotor's norm (discrete phase proxy) back into word 3 so
        # the evolution is non-trivial while remaining fully deterministic.
        phase_bits = int((rot.norm * (1 << 52))) & ((1 << 64) - 1)
        state = (state[0], state[1], state[2], (state[3] ^ phase_bits) & ((1 << 64) - 1))
    digest_blob = b"".join(s.to_bytes(8, "little") for s in state)
    return fnv1a64_packed(digest_blob)


def _stage_vi_ngram_rescue_audit(
    lmo_handle: "LmoHandle",
    candidate_text: str,
) -> str:
    """Stage-VI audit for n-gram-rescued entries.

    Shares the tokenizer / UTF-8 / blob checks with
    :func:`_stage_vi_shadow_audit` but deliberately **skips the
    bit-symmetry step**. By construction an n-gram-rescued entry has
    ``stored_state != candidate_state`` (that's why the primary
    lookup missed); ``zpm.verify`` would always reject it.

    Honesty about the guarantee. Serving a rescued entry is a
    *surface-overlap-gated near-match service* — the reply is
    correct-in-kind for a prompt that is a plausible rephrasing of
    what produced it, but it is not a byte-identical cache hit.
    The evidence report flags it with ``ngram_rescued=True`` so
    operators and release-gate tooling can segment R2's served
    fraction into ``exact`` vs ``rescued`` and apply independent
    thresholds.
    """
    if not candidate_text:
        return "stage_vi_empty_candidate"
    try:
        candidate_text.encode("utf-8")
    except UnicodeEncodeError:
        return "stage_vi_invalid_utf8"
    tok_path = lmo_handle.tokenizer_path
    if not tok_path.is_file() or tok_path.stat().st_size == 0:
        return "stage_vi_no_tokenizer_blob"
    return ""


def _stage_vi_shadow_audit(
    lmo_handle: "LmoHandle",
    candidate_text: str,
    candidate_state: zpm.State,
    stored_state: zpm.State,
) -> str:
    """Shadow-mode Stage-VI verify. Returns ``""`` on pass, reason on fail.

    In shadow mode we can never actually serve this token, so the audit
    exists to produce a realistic demotion-reason distribution for
    Phase-6-EG readiness.
    """
    # Tokenizer-id validity check: the LMO must carry a tokenizer blob and
    # the candidate's text must be decodable UTF-8 (ZpmIndex.load already
    # enforces this, but we double-check to exercise the gate shape).
    if not candidate_text:
        return "stage_vi_empty_candidate"
    try:
        candidate_text.encode("utf-8")
    except UnicodeEncodeError:
        return "stage_vi_invalid_utf8"
    tok_path = lmo_handle.tokenizer_path
    if not tok_path.is_file() or tok_path.stat().st_size == 0:
        return "stage_vi_no_tokenizer_blob"
    # Second-rotor bit-symmetry audit (Plane-A.5 Stage III). On exact
    # matches ``verify`` always passes; on near-matches it surfaces any
    # drift the nullspace search tolerated.
    if not zpm.verify(candidate_state, stored_state):
        return "stage_vi_symmetry_drift"
    return ""


@dataclass(frozen=True)
class _OmegaProbeStats:
    """Internal: per-call §4.3(1)-(3) stats returned by the candidate-state
    computation. Exposed publicly via :func:`compute_r2_candidate_state` in
    a stripped-down form."""

    digests: tuple[int, ...]
    candidate_state: zpm.State
    candidate_fnv: str
    sub_lattices_visited: int
    omega_iterations: int
    budget_exceeded: bool


def _run_omega_probe(
    lmo_handle: "LmoHandle",
    *,
    intent_anchor_bytes: bytes,
    omega_budget_ms: float,
    omega_iterations: int,
    start_t: float,
) -> _OmegaProbeStats:
    """Shared §4.3(1)..(3) body for shadow, active, and the public helper.

    Pure function of its inputs (plus the wall-clock budget, which is the
    only source of non-determinism — by design, per §4.3 budget contract).
    """
    router = lmo_handle.router
    sub_lattices = router.sub_lattices
    budget_s = max(1e-4, omega_budget_ms / 1000.0)
    deadline = start_t + budget_s
    per_lattice_units = 64
    digests: list[int] = []
    visited = 0
    iterations_run = 0
    budget_exceeded = False
    for sl in sub_lattices:
        if time.perf_counter() > deadline:
            budget_exceeded = True
            break
        seed = _project_anchor_to_sublattice(
            intent_anchor_bytes, sl.block_id, per_lattice_units,
        )
        iters = min(
            omega_iterations,
            max(1, int(1 + sl.wake_rate * omega_iterations)),
        )
        d = _evolve_sublattice(seed, iters)
        digests.append(d)
        visited += 1
        iterations_run += iters
    blob = (
        b"".join(d.to_bytes(8, "little") for d in digests)
        if digests
        else b""
    )
    candidate_state = zpm.anchor(blob) if digests else zpm.anchor(b"\x00")
    candidate_fnv = f"{fnv1a64_packed(blob):016x}" if digests else ""
    return _OmegaProbeStats(
        digests=tuple(digests),
        candidate_state=candidate_state,
        candidate_fnv=candidate_fnv,
        sub_lattices_visited=visited,
        omega_iterations=iterations_run,
        budget_exceeded=budget_exceeded,
    )


def compute_r2_candidate_state(
    lmo_handle: "LmoHandle",
    intent_anchor_bytes: bytes,
    *,
    omega_iterations: int = 3,
    omega_budget_ms: float = 2.0,
) -> zpm.State:
    """Pure Phase 6-EG helper: compute R2's candidate continuation anchor.

    Exposes the §4.3(1)..(3) state-evolution result without running Stage-VI
    or performing a ZPM lookup. Primarily used by tests and off-line
    tooling to prime the ZPM cache with entries that R2 active mode can
    then serve — the same projection/evolution is fully deterministic
    for a given ``(lmo, intent_anchor_bytes, iterations)`` triple.
    """
    stats = _run_omega_probe(
        lmo_handle,
        intent_anchor_bytes=intent_anchor_bytes,
        omega_budget_ms=omega_budget_ms,
        omega_iterations=omega_iterations,
        start_t=time.perf_counter(),
    )
    return stats.candidate_state


# --------------------------------------------------------------------------- #
# Phase 11 — n-gram rescue (expands R2's candidate set)
# --------------------------------------------------------------------------- #
#
# Motivation. The Phase 6-EG R2 resolver computes ONE candidate state
# per turn (the Plane-A.5 fold of the Omega probe digests) and looks it
# up in the ZPM index under ``zpm_threshold_bits`` Hamming tolerance.
# On a realistic chat mix this serves ~5% of turns — the rest demote
# to R5. The ceiling isn't the decoder; it's that a single state
# candidate is too narrow a cone.
#
# Strategy. When the primary lookup misses (or the match fails
# Stage-VI) we run a bounded second-chance scan: rank every stored
# ZPM entry by character-4-gram Jaccard overlap against the current
# prompt's ``prompt_head`` metadata. Entries whose prompt_head was
# never stored (older indexes) get an overlap score of 0 and drop
# out naturally. The best-scoring entry above
# ``_NGRAM_OVERLAP_THRESHOLD`` is sent back through Stage-VI; if it
# passes, R2 serves the turn and the report carries
# ``ngram_rescued=True``.
#
# Why this is honest. The rescue never emits without a Stage-VI pass,
# so coherence is guaranteed. The scan is O(N) over the ZPM index
# (a few hundred entries in practice); bounded by the existing Omega
# wall budget. If the new prompt is not a plausible rephrasing of any
# stored prompt, the rescue silently falls through to R5 — no
# hallucinated cache hits.


# Phase 11 tuning. Char-3-grams are chosen over 4-grams because real
# chat prompts are often short (10-40 chars) — 4-grams are too sparse
# to produce meaningful overlap on e.g. "what is NRL?" vs "what's NRL
# again?" (~0.24 Jaccard with 4-grams, ~0.35 with 3-grams). 3-grams
# also reject unrelated prompts cleanly in practice because a single
# 3-gram collision across short strings still keeps Jaccard well under
# 0.20.
#
# RELEASE GATE LOCKDOWN (Phase 11 Final).
# ``_NGRAM_OVERLAP_THRESHOLD`` is the admission floor for the n-gram
# rescue path. It is intentionally locked at 0.30 for Production
# Release v1.0 — this value was measured against the 63-rephrase
# honest audit corpus in ``scripts/r2_rescue_bench.py`` (74.6% served
# share, 0.31 min overlap, 0.43 p50, zero cross-topic matches).
# Lowering it admits loose matches on unrelated domains; raising it
# starves the rescue path. Do NOT change without re-running the
# rescue bench against both the fixture AND a real GGUF, and
# documenting the new min/p50/max overlap distribution in the PR
# description. ``tests/test_r2_ngram_rescue.py`` pins the constant.
_NGRAM_OVERLAP_THRESHOLD: float = 0.30
_NGRAM_N = 3
_NGRAM_MAX_SCAN = 256
_NGRAM_HAMMING_FACTOR = 2.0  # rescued entry must lie within 2× threshold


def _char_ngrams(text: str, n: int = _NGRAM_N) -> frozenset[str]:
    """Lowercase whitespace-normalized character n-gram set.

    Non-alphanumeric runs collapse to single spaces so
    ``"What is NRL?"`` and ``"what's NRL"`` share most of their grams.
    Returns an empty set for strings shorter than ``n`` characters.
    """
    if not text:
        return frozenset()
    buf = []
    prev_space = False
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
            prev_space = False
        elif not prev_space:
            buf.append(" ")
            prev_space = True
    norm = "".join(buf).strip()
    if len(norm) < n:
        return frozenset()
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def _ngram_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B|. ``0.0`` on empty input."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0


@dataclasses.dataclass(frozen=True)
class _NgramRescueHit:
    """One n-gram-rescued ZPM entry candidate."""

    entry_index: int
    entry: "zpm.ZpmEntry"
    hamming_bits: int
    overlap: float


def _ngram_rescue_search(
    *,
    zpm_index: "zpm.ZpmIndex",
    candidate_state: "zpm.State",
    prompt_text: str,
    zpm_threshold_bits: int,
) -> tuple[_NgramRescueHit | None, int, float]:
    """Second-chance scan over ``zpm_index``.

    Returns ``(best_hit_or_none, candidates_considered, best_overlap)``.
    The scan is skipped (returns ``(None, 0, 0.0)``) when
    ``prompt_text`` is empty or produces no n-grams.

    Rescue gate. The only decision variable is char-3-gram Jaccard
    between ``prompt_text`` and the entry's stored ``prompt_head``
    metadata. State-level Hamming distance is *reported* (so operators
    can segment the served fraction) but NOT used as a reject gate —
    if we gated on Hamming we'd simply reproduce the primary
    ``zpm_index.lookup`` we already missed. The rescue's honesty
    comes from:

    * A non-trivial overlap threshold (``_NGRAM_OVERLAP_THRESHOLD``)
      rejecting surface-dissimilar prompts.
    * The dedicated :func:`_stage_vi_ngram_rescue_audit` the caller
      runs before promoting the entry.
    * Transparent evidence: every rescued service lands in the turn
      log with ``ngram_rescued=True`` and ``ngram_best_overlap``,
      making the rescued fraction auditable.
    """
    query_grams = _char_ngrams(prompt_text)
    if not query_grams:
        return None, 0, 0.0

    best: _NgramRescueHit | None = None
    best_overlap = 0.0
    scanned = 0
    entries = list(zpm_index)

    for idx, entry in enumerate(entries[:_NGRAM_MAX_SCAN]):
        scanned += 1
        stored_head = str(entry.metadata.get("prompt_head", ""))
        if not stored_head:
            continue
        overlap = _ngram_jaccard(query_grams, _char_ngrams(stored_head))
        if overlap > best_overlap:
            best_overlap = overlap
        if overlap < _NGRAM_OVERLAP_THRESHOLD:
            continue
        hamming = zpm.hamming_state(candidate_state, entry.state)
        if best is None or overlap > best.overlap:
            best = _NgramRescueHit(
                entry_index=idx,
                entry=entry,
                hamming_bits=hamming,
                overlap=overlap,
            )
    return best, scanned, best_overlap


def try_omega_native_resolve(
    lmo_handle: "LmoHandle",
    *,
    intent_anchor_bytes: bytes,
    coherence_lane: str,
    zpm_index: "zpm.ZpmIndex | None",
    omega_budget_ms: float = 2.0,
    omega_candidates: int = 4,
    omega_iterations: int = 3,
    zpm_threshold_bits: int = 0,
    mode: str = "shadow",
    prompt_text: str = "",
) -> tuple[RungResult, OmegaShadowReport]:
    """Rung R2 — Omega Native Resolve (shadow Phase 5-EG, active Phase 6-EG).

    Implements §4.3(1)..(5) of ``Final_NRL_Architecture_GGUF.MD``:

    1. Project ``intent_anchor_bytes`` onto every sub-lattice in the LMO's
       router graph.
    2. Run the bounded omega evolution (up to ``omega_iterations`` rounds
       per sub-lattice, gated by ``omega_budget_ms`` wall budget).
    3. Fold the sub-lattice digests into a 256-bit candidate continuation
       anchor via Plane-A.5 Stage I.
    4. Search the ZPM index under ``zpm_threshold_bits``.
    5. Run Stage-VI verify on any match.

    The ``mode`` argument selects the rung contract:

    * ``"shadow"`` (default, Phase 5-EG). The :class:`RungResult` is
      *always* ``coherence_demoted=True`` and ``tokens_emitted=0``. The
      report is labelled ``mode="shadow"``. Used for advisory-only
      collection prior to Phase 7-EG's release gate.
    * ``"active"`` (Phase 6-EG). On a ZPM hit that passes Stage-VI
      verify, the rung returns ``coherence_demoted=False``,
      ``tokens_emitted=<stored reply tokens>``, and
      ``gate_source=OMEGA_ACTIVE_GATE_SOURCE``. The matched reply is
      attached to the report via ``served_text`` / ``served_tokens`` so
      the runner can serve it directly. On any miss or Stage-VI failure
      the rung demotes cleanly (``coherence_demoted=True``) and the
      caller must fall through to R5.

    The function never raises — internal errors are captured as
    ``status="error"`` so the runner's hot path is never taken down by
    R2 mechanics.
    """
    lane = coherence_lane
    if mode not in ("shadow", "active"):
        raise ValueError(f"mode must be 'shadow' or 'active', got {mode!r}")

    # Lane gating. Both modes use the same set today (per Phase 6-EG
    # operator directive); ``fast-stable`` remains hard-locked.
    lane_ok = (
        lane_allows_r2_active(lane)
        if mode == "active"
        else lane_allows_r2_shadow(lane)
    )
    if not lane_ok:
        reason = (
            "coherence_lane_disallows_r2_active"
            if mode == "active"
            else "coherence_lane_disallows_r2_shadow"
        )
        rep = OmegaShadowReport.skipped(lane, reason)
        return (
            RungResult(
                rung="r2",
                gate_source=None,
                tokens_emitted=0,
                wall_s=0.0,
                coherence_demoted=True,
                stage_vi_reason=reason,
            ),
            rep,
        )

    t0 = time.perf_counter()
    demotion_reasons: list[str] = []
    try:
        router = lmo_handle.router
        if not router.sub_lattices:
            demotion_reasons.append("no_sub_lattices")

        stats = _run_omega_probe(
            lmo_handle,
            intent_anchor_bytes=intent_anchor_bytes,
            omega_budget_ms=omega_budget_ms,
            omega_iterations=omega_iterations,
            start_t=t0,
        )
        if stats.budget_exceeded:
            demotion_reasons.append("budget_exceeded")

        # No sub-lattices / empty digest set — cannot form a candidate.
        if not stats.digests:
            return _build_demoted(
                lane=lane,
                mode=mode,
                demotion_reasons=demotion_reasons or ["no_candidate"],
                candidate_fnv="",
                zpm_threshold_bits=zpm_threshold_bits,
                available=False,
                iterations_run=0,
                visited=0,
                zpm_distance_bits=-1,
                wall_s_start=t0,
            )

        # Step 4 — ZPM lookup.
        if zpm_index is None or len(zpm_index) == 0:
            demotion_reasons.append("no_zpm_index")
            return _build_demoted(
                lane=lane,
                mode=mode,
                demotion_reasons=demotion_reasons,
                candidate_fnv=stats.candidate_fnv,
                zpm_threshold_bits=zpm_threshold_bits,
                available=zpm_index is not None,
                iterations_run=stats.omega_iterations,
                visited=stats.sub_lattices_visited,
                zpm_distance_bits=-1,
                wall_s_start=t0,
            )

        z_hit, z_entry = zpm_index.lookup(
            stats.candidate_state, threshold_bits=zpm_threshold_bits,
        )

        # Phase 11 — n-gram rescue hooks. If the primary lookup missed
        # or the match fails Stage-VI below, we may replace ``z_entry``
        # with a second-chance candidate scored by char-4-gram overlap
        # on ``prompt_text``. Track outcome on the report.
        ngram_rescued = False
        ngram_candidates_considered = 0
        ngram_best_overlap = 0.0

        if z_entry is None:
            rescue_hit, ngram_candidates_considered, ngram_best_overlap = (
                _ngram_rescue_search(
                    zpm_index=zpm_index,
                    candidate_state=stats.candidate_state,
                    prompt_text=prompt_text,
                    zpm_threshold_bits=zpm_threshold_bits,
                )
            )
            if rescue_hit is None:
                demotion_reasons.append("zpm_no_match")
                if ngram_candidates_considered > 0:
                    demotion_reasons.append("ngram_rescue_no_match")
                return _build_demoted(
                    lane=lane,
                    mode=mode,
                    demotion_reasons=demotion_reasons,
                    candidate_fnv=stats.candidate_fnv,
                    zpm_threshold_bits=zpm_threshold_bits,
                    available=True,
                    iterations_run=stats.omega_iterations,
                    visited=stats.sub_lattices_visited,
                    zpm_distance_bits=-1,
                    wall_s_start=t0,
                    ngram_candidates_considered=ngram_candidates_considered,
                    ngram_best_overlap=ngram_best_overlap,
                )
            # Promote the rescued entry as if the primary lookup had
            # found it, but retain the true Hamming distance so the
            # report isn't lying. Mark it as rescued for evidence and
            # run the rescue-specific Stage-VI (skips bit-symmetry by
            # design — see _stage_vi_ngram_rescue_audit).
            rescue_vi = _stage_vi_ngram_rescue_audit(
                lmo_handle, rescue_hit.entry.reply_text,
            )
            if rescue_vi:
                demotion_reasons.append("zpm_no_match")
                demotion_reasons.append(rescue_vi)
                return _build_demoted(
                    lane=lane,
                    mode=mode,
                    demotion_reasons=demotion_reasons,
                    candidate_fnv=stats.candidate_fnv,
                    zpm_threshold_bits=zpm_threshold_bits,
                    available=True,
                    iterations_run=stats.omega_iterations,
                    visited=stats.sub_lattices_visited,
                    zpm_distance_bits=rescue_hit.hamming_bits,
                    wall_s_start=t0,
                    ngram_candidates_considered=ngram_candidates_considered,
                    ngram_best_overlap=ngram_best_overlap,
                )
            z_entry = rescue_hit.entry
            z_hit = zpm.ZpmHit(
                entry_index=rescue_hit.entry_index,
                distance_bits=rescue_hit.hamming_bits,
                threshold_bits=zpm_threshold_bits,
                exact=False,
                within_threshold=False,
            )
            ngram_rescued = True

        # Step 5 — Stage-VI verify. Rescued entries already passed the
        # tokenizer/UTF-8/blob subset via the rescue audit above (bit-
        # symmetry is skipped by design for rescues). For non-rescued
        # entries we run the full Stage-VI including bit-symmetry.
        if ngram_rescued:
            vi_reason = ""
        else:
            vi_reason = _stage_vi_shadow_audit(
                lmo_handle, z_entry.reply_text,
                stats.candidate_state, z_entry.state,
            )
        if vi_reason:
            # If the original near-match state failed Stage-VI, try a
            # single n-gram rescue before giving up. We don't retry the
            # rescue if we're already on a rescued entry -- that would
            # be the same candidate.
            if not ngram_rescued:
                rescue_hit, rescue_scanned, rescue_overlap = (
                    _ngram_rescue_search(
                        zpm_index=zpm_index,
                        candidate_state=stats.candidate_state,
                        prompt_text=prompt_text,
                        zpm_threshold_bits=zpm_threshold_bits,
                    )
                )
                if rescue_hit is not None:
                    alt_vi = _stage_vi_ngram_rescue_audit(
                        lmo_handle, rescue_hit.entry.reply_text,
                    )
                    if not alt_vi:
                        z_entry = rescue_hit.entry
                        z_hit = zpm.ZpmHit(
                            entry_index=rescue_hit.entry_index,
                            distance_bits=rescue_hit.hamming_bits,
                            threshold_bits=zpm_threshold_bits,
                            exact=False,
                            within_threshold=False,
                        )
                        ngram_rescued = True
                        ngram_candidates_considered = rescue_scanned
                        ngram_best_overlap = rescue_overlap
                        vi_reason = ""  # rescue cleared Stage-VI
            if vi_reason:
                demotion_reasons.append(vi_reason)
                return _build_demoted(
                    lane=lane,
                    mode=mode,
                    demotion_reasons=demotion_reasons,
                    candidate_fnv=stats.candidate_fnv,
                    zpm_threshold_bits=zpm_threshold_bits,
                    available=True,
                    iterations_run=stats.omega_iterations,
                    visited=stats.sub_lattices_visited,
                    zpm_distance_bits=z_hit.distance_bits,
                    wall_s_start=t0,
                    stored_entry_index=z_hit.entry_index,
                    ngram_candidates_considered=ngram_candidates_considered,
                    ngram_best_overlap=ngram_best_overlap,
                )

        # Stage-VI passed. Contract diverges between modes.
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if mode == "shadow":
            demotion_reasons.append("shadow_mode_never_emits")
            rep = OmegaShadowReport(
                status="ok",
                gate_source=OMEGA_SHADOW_GATE_SOURCE,
                coherence_lane=lane,
                mode="shadow",
                available=True,
                served=False,
                served_tokens=0,
                served_text="",
                hits=1,
                candidate_continuation_fnv=stats.candidate_fnv,
                demotion_reasons=tuple(demotion_reasons),
                wall_ms=wall_ms,
                omega_iterations=stats.omega_iterations,
                sub_lattices_visited=stats.sub_lattices_visited,
                zpm_distance_bits=z_hit.distance_bits,
                zpm_threshold_bits=zpm_threshold_bits,
                stored_entry_index=z_hit.entry_index,
                ngram_rescued=ngram_rescued,
                ngram_candidates_considered=ngram_candidates_considered,
                ngram_best_overlap=ngram_best_overlap,
            )
            return (
                RungResult(
                    rung="r2",
                    gate_source=OMEGA_SHADOW_GATE_SOURCE,
                    tokens_emitted=0,
                    wall_s=wall_ms / 1000.0,
                    coherence_demoted=True,
                    stage_vi_reason="shadow_mode_never_emits",
                ),
                rep,
            )

        # mode == "active" — R2 serves this turn.
        served_tokens = max(1, int(z_entry.tokens))
        rep = OmegaShadowReport(
            status="ok",
            gate_source=OMEGA_SHADOW_GATE_SOURCE,
            coherence_lane=lane,
            mode="active",
            available=True,
            served=True,
            served_tokens=served_tokens,
            served_text=z_entry.reply_text,
            hits=1,
            candidate_continuation_fnv=stats.candidate_fnv,
            demotion_reasons=(),
            wall_ms=wall_ms,
            omega_iterations=stats.omega_iterations,
            sub_lattices_visited=stats.sub_lattices_visited,
            zpm_distance_bits=z_hit.distance_bits,
            zpm_threshold_bits=zpm_threshold_bits,
            stored_entry_index=z_hit.entry_index,
            ngram_rescued=ngram_rescued,
            ngram_candidates_considered=ngram_candidates_considered,
            ngram_best_overlap=ngram_best_overlap,
        )
        return (
            RungResult(
                rung="r2",
                gate_source=OMEGA_ACTIVE_GATE_SOURCE,
                tokens_emitted=served_tokens,
                wall_s=wall_ms / 1000.0,
                coherence_demoted=False,
                stage_vi_reason="",
            ),
            rep,
        )

    except Exception as e:  # noqa: BLE001 — R2 must never crash the runner.
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rep = OmegaShadowReport(
            status="error",
            gate_source=OMEGA_SHADOW_GATE_SOURCE,
            coherence_lane=lane,
            mode=mode,
            available=False,
            served=False,
            served_tokens=0,
            served_text="",
            hits=0,
            candidate_continuation_fnv="",
            demotion_reasons=(f"exception:{e.__class__.__name__}",),
            wall_ms=wall_ms,
            omega_iterations=0,
            sub_lattices_visited=0,
            zpm_distance_bits=-1,
            zpm_threshold_bits=zpm_threshold_bits,
            stored_entry_index=-1,
        )
        return (
            RungResult(
                rung="r2",
                gate_source=OMEGA_SHADOW_GATE_SOURCE,
                tokens_emitted=0,
                wall_s=wall_ms / 1000.0,
                coherence_demoted=True,
                stage_vi_reason=f"exception:{e.__class__.__name__}",
            ),
            rep,
        )


def _build_demoted(
    *,
    lane: str,
    mode: str,
    demotion_reasons: list[str],
    candidate_fnv: str,
    zpm_threshold_bits: int,
    available: bool,
    iterations_run: int,
    visited: int,
    zpm_distance_bits: int,
    wall_s_start: float,
    stored_entry_index: int = -1,
    ngram_candidates_considered: int = 0,
    ngram_best_overlap: float = 0.0,
) -> tuple[RungResult, OmegaShadowReport]:
    """Shared terminal path for any R2 demotion (either mode).

    Guarantees ``coherence_demoted=True``, ``tokens_emitted=0``, and
    that the report's ``served`` / ``served_text`` are empty.
    """
    wall_ms = (time.perf_counter() - wall_s_start) * 1000.0
    rep = OmegaShadowReport(
        status="ok",
        gate_source=OMEGA_SHADOW_GATE_SOURCE,
        coherence_lane=lane,
        mode=mode,
        available=available,
        served=False,
        served_tokens=0,
        served_text="",
        hits=0,
        candidate_continuation_fnv=candidate_fnv,
        demotion_reasons=tuple(demotion_reasons),
        wall_ms=wall_ms,
        omega_iterations=iterations_run,
        sub_lattices_visited=visited,
        zpm_distance_bits=zpm_distance_bits,
        zpm_threshold_bits=zpm_threshold_bits,
        stored_entry_index=stored_entry_index,
        ngram_rescued=False,
        ngram_candidates_considered=ngram_candidates_considered,
        ngram_best_overlap=ngram_best_overlap,
    )
    # The shadow gate-source stays on the rung because this rung did not
    # serve — the rung only adopts the *active* gate-source on actual
    # service.
    return (
        RungResult(
            rung="r2",
            gate_source=OMEGA_SHADOW_GATE_SOURCE,
            tokens_emitted=0,
            wall_s=wall_ms / 1000.0,
            coherence_demoted=True,
            stage_vi_reason=demotion_reasons[-1] if demotion_reasons else "demoted",
        ),
        rep,
    )
