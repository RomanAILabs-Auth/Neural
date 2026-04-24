# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Tests for :mod:`nrlpy.lmo` — Phase 4-EG LMO absorption.

These tests exercise the full Stage A-I..A-VI pipeline against a tiny
hand-built GGUF fixture so CI never depends on a multi-GB real model.

The fixture is a minimal-but-valid GGUF v3 file with two F32 tensors
laid out as if they were transformer blocks ``blk.0.*`` and the
``token_embd.weight`` / ``output.weight`` anchors. This is sufficient
to cover every deterministic branch of the absorption pipeline.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable

import pytest

from nrlpy import lmo
from nrlpy.lmo import (
    CANONICAL_TILE_UNITS,
    GGUF_MAGIC,
    LMO_MAGIC,
    GgufParseError,
    LmoError,
    LmoHandle,
    LmoHeader,
    RouterGraph,
    TilePlan,
    _read_tile_offsets,
    absorb_gguf,
    build_router_graph,
    gguf_parse_metadata,
    pack_int4_from_bytes,
    plan_tiles,
    sha256_file,
    verify_parity_against_libllama,
    write_header_anchor,
)


def _lmo_uses_blob_layout(lmo_dir: Path) -> bool:
    return (lmo_dir / "retained.blob").is_file() and (lmo_dir / "packed.blob").is_file()


def _first_nonempty_retained_tile(plan: TilePlan) -> int | None:
    for tile in plan.tiles:
        if tile.retained_bytes > 0:
            return tile.tile_id
    return None


# --------------------------------------------------------------------------- #
# Fixture GGUF builder
# --------------------------------------------------------------------------- #


_GGUF_TYPE_F32 = 0

_GGUF_VAL_UINT32 = 4
_GGUF_VAL_STRING = 8


def _write_gguf_string(buf: bytearray, s: bytes | str) -> None:
    data = s.encode("utf-8") if isinstance(s, str) else s
    buf += struct.pack("<Q", len(data))
    buf += data


def _write_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", _GGUF_VAL_UINT32)
    buf += struct.pack("<I", int(value) & 0xFFFFFFFF)


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_gguf_string(buf, key)
    buf += struct.pack("<I", _GGUF_VAL_STRING)
    _write_gguf_string(buf, value)


def _write_tensor_info(
    buf: bytearray,
    name: str,
    shape: tuple[int, ...],
    ggml_type: int,
    rel_offset: int,
) -> None:
    _write_gguf_string(buf, name)
    buf += struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<Q", int(d))
    buf += struct.pack("<I", int(ggml_type))
    buf += struct.pack("<Q", int(rel_offset))


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def _build_fixture_gguf(path: Path, *, n_blocks: int = 2) -> None:
    """Write a minimal valid GGUF v3 file with F32 tensors.

    Tensor layout:
        * token_embd.weight  — shape (16,),  F32
        * blk.N.attn_q.weight — shape (8,),   F32 (for N in [0, n_blocks))
        * output.weight      — shape (16,),  F32

    All tensor data is deterministic (values are just the element index
    as float) so absorption outcomes are stable run-to-run.
    """
    alignment = 32

    tensors: list[tuple[str, tuple[int, ...]]] = [("token_embd.weight", (16,))]
    for i in range(n_blocks):
        tensors.append((f"blk.{i}.attn_q.weight", (8,)))
    tensors.append(("output.weight", (16,)))

    # --- header KVs --- #
    kv_buf = bytearray()
    _write_kv_uint32(kv_buf, "general.alignment", alignment)
    _write_kv_string(kv_buf, "general.architecture", "llama")
    _write_kv_string(kv_buf, "tokenizer.ggml.model", "llama")
    # A tiny tokenizer KV to exercise the ``tokenizer.`` blob extraction.
    _write_kv_string(kv_buf, "tokenizer.ggml.bos_token", "<s>")
    kv_count = 4

    # --- tensor infos with placeholder offsets --- #
    rel_offsets: list[int] = []
    cursor = 0
    tinfo_buf = bytearray()
    for name, shape in tensors:
        rel_offsets.append(cursor)
        n = 1
        for d in shape:
            n *= d
        cursor += n * 4  # F32 = 4 bytes per element
        # Each GGUF tensor data block is aligned to ``alignment``.
        cursor = _align_up(cursor, alignment)
    for (name, shape), rel in zip(tensors, rel_offsets):
        _write_tensor_info(tinfo_buf, name, shape, _GGUF_TYPE_F32, rel)

    # --- preamble --- #
    head = bytearray()
    head += GGUF_MAGIC
    head += struct.pack("<I", 3)                        # version
    head += struct.pack("<Q", len(tensors))             # tensor_count
    head += struct.pack("<Q", kv_count)                 # kv_count

    pre_data_bytes = bytes(head) + bytes(kv_buf) + bytes(tinfo_buf)
    data_start = _align_up(len(pre_data_bytes), alignment)

    # --- tensor data --- #
    data_buf = bytearray(max(data_start, len(pre_data_bytes)) + cursor)
    data_buf[: len(pre_data_bytes)] = pre_data_bytes
    for (name, shape), rel in zip(tensors, rel_offsets):
        n = 1
        for d in shape:
            n *= d
        # Fill with deterministic float values (index+1)*0.5.
        floats = [((i + 1) * 0.5) for i in range(n)]
        packed = struct.pack("<" + "f" * n, *floats)
        off = data_start + rel
        data_buf[off: off + len(packed)] = packed

    # Size the final buffer to include the last tensor's payload.
    last_rel = rel_offsets[-1]
    last_n = 1
    for d in tensors[-1][1]:
        last_n *= d
    final_end = data_start + last_rel + last_n * 4
    data_buf = data_buf[:final_end]

    path.write_bytes(bytes(data_buf))


@pytest.fixture()
def fixture_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.gguf"
    _build_fixture_gguf(path, n_blocks=2)
    return path


# --------------------------------------------------------------------------- #
# GGUF parser
# --------------------------------------------------------------------------- #


def test_gguf_parse_metadata_reads_tensors(fixture_gguf: Path) -> None:
    meta = gguf_parse_metadata(fixture_gguf)
    assert meta.version == 3
    assert meta.tensor_count == 4
    assert meta.kv_count == 4
    assert meta.alignment == 32

    names = [t.name for t in meta.tensors]
    assert "token_embd.weight" in names
    assert "blk.0.attn_q.weight" in names
    assert "blk.1.attn_q.weight" in names
    assert "output.weight" in names

    # Every F32 tensor has element_count * 4 bytes.
    for t in meta.tensors:
        assert t.quant_kind == "f32"
        assert t.data_bytes == t.element_count * 4
        assert t.data_offset >= meta.tensor_data_start


def test_gguf_parse_extracts_tokenizer_blob(fixture_gguf: Path) -> None:
    meta = gguf_parse_metadata(fixture_gguf)
    assert meta.tokenizer_blob  # non-empty
    # Two tokenizer.* KV pairs were written; each is prefixed with its u32 length.
    # We don't parse the blob's internal structure — just assert it round-trips
    # to the same bytes when we re-read the file.
    meta2 = gguf_parse_metadata(fixture_gguf)
    assert meta.tokenizer_blob == meta2.tokenizer_blob


def test_gguf_parse_rejects_non_gguf(tmp_path: Path) -> None:
    p = tmp_path / "not_gguf.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GgufParseError):
        gguf_parse_metadata(p)


def test_sha256_file_matches_hashlib(fixture_gguf: Path) -> None:
    assert sha256_file(fixture_gguf) == hashlib.sha256(
        fixture_gguf.read_bytes()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# pack_int4_from_bytes
# --------------------------------------------------------------------------- #


def test_pack_int4_output_length_and_nibble_range() -> None:
    raw = bytes(range(64))
    packed = pack_int4_from_bytes(raw, 32)
    assert len(packed) == 16  # (32 + 1) // 2
    for b in packed:
        assert 0 <= (b & 0x0F) <= 15
        assert 0 <= (b >> 4) <= 15


def test_pack_int4_deterministic() -> None:
    raw = b"hello world, NRL packed INT4 fold"
    a = pack_int4_from_bytes(raw, 64)
    b = pack_int4_from_bytes(raw, 64)
    assert a == b


def test_pack_int4_empty_cases() -> None:
    assert pack_int4_from_bytes(b"", 32) == b""
    assert pack_int4_from_bytes(b"x" * 8, 0) == b""


def test_pack_int4_odd_units_pads_high_nibble() -> None:
    raw = b"\x00" * 32
    packed = pack_int4_from_bytes(raw, 3)
    assert len(packed) == 2


# --------------------------------------------------------------------------- #
# plan_tiles
# --------------------------------------------------------------------------- #


def test_plan_tiles_covers_all_tensors(fixture_gguf: Path) -> None:
    meta = gguf_parse_metadata(fixture_gguf)
    plan = plan_tiles(meta, tile_units=CANONICAL_TILE_UNITS)

    # At least one tile per tensor.
    assert set(plan.block_to_tiles.keys()) == {t.name for t in meta.tensors}
    assert len(plan.tiles) >= len(meta.tensors)
    assert plan.partial is False

    # Retained bytes summed from tile plan equal GGUF payload sum.
    total_src_bytes = sum(t.data_bytes for t in meta.tensors)
    assert plan.total_retained_bytes == total_src_bytes


def test_plan_tiles_digest_is_deterministic(fixture_gguf: Path) -> None:
    meta = gguf_parse_metadata(fixture_gguf)
    d1 = plan_tiles(meta).digest()
    d2 = plan_tiles(meta).digest()
    assert d1 == d2
    assert len(d1) == 16  # 64-bit hex


# --------------------------------------------------------------------------- #
# absorb_gguf — full pipeline
# --------------------------------------------------------------------------- #


def _absorb(tmp_path: Path, gguf: Path) -> LmoHandle:
    return absorb_gguf(
        gguf,
        out_root=tmp_path / "cache" / "lmo",
        force=False,
        attempt_libllama=False,
    )


def test_absorb_creates_expected_layout(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    d = handle.lmo_dir
    assert d.is_dir()
    assert (d / "lmo.header").is_file()
    assert (d / "router.graph").is_file()
    assert (d / "attest.json").is_file()
    assert (d / "manifest.nrl").is_file()
    assert (d / "tokenizer.blob").is_file()
    if _lmo_uses_blob_layout(d):
        assert (d / "tile_offsets.bin").is_file()
        assert (d / "packed.blob").is_file()
        assert (d / "retained.blob").is_file()
    else:
        assert (d / "tiles").is_dir()
        assert (d / "retained").is_dir()
        tile_files = sorted(p.name for p in (d / "tiles").glob("*.tile"))
        retained_files = sorted(p.name for p in (d / "retained").glob("*.bin"))
        assert len(tile_files) == handle.header.tile_count
        assert len(retained_files) == handle.header.tile_count


def test_absorb_header_anchor_is_deterministic(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    h1 = _absorb(tmp_path / "a", fixture_gguf)
    h2 = _absorb(tmp_path / "b", fixture_gguf)
    assert h1.header.lmo_anchor == h2.header.lmo_anchor
    assert h1.header.tile_plan_digest == h2.header.tile_plan_digest
    assert h1.header.router_graph_digest == h2.header.router_graph_digest
    assert h1.header.model_sha256 == h2.header.model_sha256


def test_absorb_header_magic_and_schema(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    raw = (handle.lmo_dir / "lmo.header").read_bytes()
    assert raw[:8] == LMO_MAGIC
    assert len(raw) == 4096
    (schema,) = struct.unpack_from("<H", raw, 8)
    assert schema == 1
    # Round-trip.
    parsed = LmoHeader.deserialize(raw)
    assert parsed.serialize() == raw


def test_absorb_retained_bytes_are_byte_identical(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    src = fixture_gguf.read_bytes()
    meta = gguf_parse_metadata(fixture_gguf)
    plan = plan_tiles(meta)
    for tile in plan.tiles:
        if tile.retained_bytes == 0:
            continue
        expected = src[tile.retained_offset: tile.retained_offset + tile.retained_bytes]
        got = handle.read_retained(tile.tile_id)
        assert got == expected, f"retained byte identity failed for tile {tile.tile_id}"


def test_absorb_cache_hit_returns_same_handle_without_reprocessing(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    root = tmp_path / "cache" / "lmo"
    h1 = absorb_gguf(fixture_gguf, out_root=root, attempt_libllama=False)
    mtime1 = (h1.lmo_dir / "lmo.header").stat().st_mtime_ns

    h2 = absorb_gguf(fixture_gguf, out_root=root, attempt_libllama=False)
    mtime2 = (h2.lmo_dir / "lmo.header").stat().st_mtime_ns
    assert mtime1 == mtime2  # cache hit: nothing re-written
    assert h1.header.lmo_anchor == h2.header.lmo_anchor


def test_absorb_force_rewrites_artefacts(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    root = tmp_path / "cache" / "lmo"
    h1 = absorb_gguf(fixture_gguf, out_root=root, attempt_libllama=False)
    meta = gguf_parse_metadata(fixture_gguf)
    plan = plan_tiles(meta)
    tid = _first_nonempty_retained_tile(plan)
    assert tid is not None
    original = h1.read_retained(tid)
    d = h1.lmo_dir
    if _lmo_uses_blob_layout(d):
        offs = _read_tile_offsets(d / "tile_offsets.bin")
        r_off, r_len, _p_off, _p_len = offs[tid]
        victim = d / "retained.blob"
        buf = bytearray(victim.read_bytes())
        buf[r_off] ^= 0xFF
        victim.write_bytes(bytes(buf))
    else:
        victim = d / "retained" / f"{tid:08x}.bin"
        victim.write_bytes(b"\xFF" * max(1, len(original)))
    h2 = absorb_gguf(fixture_gguf, out_root=root, force=True, attempt_libllama=False)
    assert h2.read_retained(tid) == original
    assert h1.model_sha256 == h2.model_sha256


def test_absorb_force_twice_preserves_stable_digests(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    """P1: two forced absorbs of the same GGUF must keep ladder-stable digests."""
    root = tmp_path / "cache" / "lmo"
    h1 = absorb_gguf(fixture_gguf, out_root=root, force=True, attempt_libllama=False)
    h2 = absorb_gguf(fixture_gguf, out_root=root, force=True, attempt_libllama=False)
    assert h1.header.lmo_anchor == h2.header.lmo_anchor
    assert h1.header.tile_plan_digest == h2.header.tile_plan_digest
    assert h1.header.router_graph_digest == h2.header.router_graph_digest
    assert h1.header.model_sha256 == h2.header.model_sha256
    assert h1.header.tile_count == h2.header.tile_count
    assert h1.attest.get("packed_fnv1a64") == h2.attest.get("packed_fnv1a64")
    assert h1.attest.get("retained_fnv1a64") == h2.attest.get("retained_fnv1a64")
    reopened = LmoHandle.open(h1.lmo_dir)
    assert reopened.read_tile(0) == h1.read_tile(0)


def test_absorb_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        absorb_gguf(tmp_path / "nope.gguf", out_root=tmp_path / "cache")


def test_absorb_rejects_non_gguf(tmp_path: Path) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 128)
    with pytest.raises(GgufParseError):
        absorb_gguf(p, out_root=tmp_path / "cache")


# --------------------------------------------------------------------------- #
# Router graph
# --------------------------------------------------------------------------- #


def test_router_graph_has_sub_lattice_per_block(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    block_ids = {sl.block_id for sl in handle.router.sub_lattices}
    assert "blk.0" in block_ids
    assert "blk.1" in block_ids
    assert "token_embd" in block_ids
    assert "output" in block_ids
    # Wake rates are bounded to [0.0, 1.0].
    for sl in handle.router.sub_lattices:
        assert 0.0 <= sl.wake_rate <= 1.0


def test_router_graph_edges_chain_consecutive_blocks(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    edges = set(handle.router.edges)
    assert ("blk.0", "blk.1") in edges
    assert ("token_embd", "blk.0") in edges
    assert ("blk.1", "output") in edges


def test_router_graph_serialization_round_trip(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    raw = (handle.lmo_dir / "router.graph").read_bytes()
    parsed = RouterGraph.deserialize(raw)
    assert parsed.serialize() == raw
    assert parsed.digest() == handle.header.router_graph_digest


# --------------------------------------------------------------------------- #
# Stage A-VI — hard CI gate
# --------------------------------------------------------------------------- #


def test_stage_a_vi_audit_flags_ok(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    attest_path = handle.lmo_dir / "attest.json"
    attest = json.loads(attest_path.read_text(encoding="utf-8"))
    assert attest["schema_id"] == "nrl.lmo.attest.v1"
    assert attest["retained_byte_identity_ok"] is True
    assert attest["determinism_self_parity_ok"] is True
    assert attest["header_roundtrip_ok"] is True
    assert attest["libllama_forward"]["status"] in {"skipped", "ok", "error"}
    assert attest["tile_count"] == handle.header.tile_count
    # Every tile has a recorded FNV-1a64 checksum for retained + packed.
    assert len(attest["retained_fnv1a64"]) == handle.header.tile_count
    assert len(attest["packed_fnv1a64"]) == handle.header.tile_count


def test_stage_a_vi_rejects_tampered_retained(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    meta = gguf_parse_metadata(fixture_gguf)
    plan = plan_tiles(meta)
    router = handle.router

    tid = _first_nonempty_retained_tile(plan)
    if tid is None:
        pytest.skip("no non-empty retained tiles")
    original = handle.read_retained(tid)
    d = handle.lmo_dir
    if _lmo_uses_blob_layout(d):
        offs = _read_tile_offsets(d / "tile_offsets.bin")
        r_off, r_len, _p_off, _p_len = offs[tid]
        victim = d / "retained.blob"
        buf = bytearray(victim.read_bytes())
        buf[r_off] ^= 0xFF
        victim.write_bytes(bytes(buf))
    else:
        victim = d / "retained" / f"{tid:08x}.bin"
        victim.write_bytes(b"\x00" * len(original))

    with pytest.raises(LmoError, match="retained parity failed"):
        verify_parity_against_libllama(
            handle.lmo_dir,
            source_gguf=fixture_gguf,
            plan=plan,
            router=router,
            header=handle.header,
            packed_fnv={t.tile_id: "" for t in plan.tiles},
            retained_fnv={t.tile_id: "" for t in plan.tiles},
            attempt_libllama=False,
        )


def test_stage_a_vi_rejects_tampered_router_graph(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    handle = _absorb(tmp_path, fixture_gguf)
    meta = gguf_parse_metadata(fixture_gguf)
    plan = plan_tiles(meta)

    # Overwrite router.graph with a different graph → digest must drift.
    tampered = RouterGraph(
        sub_lattices=(),
        edges=(),
        global_min_active=9999,
        default_wake_rate=0.999,
    )
    (handle.lmo_dir / "router.graph").write_bytes(tampered.serialize())

    with pytest.raises(LmoError, match="router.graph on-disk digest"):
        verify_parity_against_libllama(
            handle.lmo_dir,
            source_gguf=fixture_gguf,
            plan=plan,
            router=handle.router,
            header=handle.header,
            packed_fnv={t.tile_id: "" for t in plan.tiles},
            retained_fnv={t.tile_id: "" for t in plan.tiles},
            attempt_libllama=False,
        )


# --------------------------------------------------------------------------- #
# LmoHandle.open
# --------------------------------------------------------------------------- #


def test_lmo_handle_open_roundtrip(fixture_gguf: Path, tmp_path: Path) -> None:
    h = _absorb(tmp_path, fixture_gguf)
    reopened = LmoHandle.open(h.lmo_dir)
    assert reopened.model_sha256 == h.model_sha256
    assert reopened.header.lmo_anchor == h.header.lmo_anchor
    assert reopened.header.tile_plan_digest == h.header.tile_plan_digest
    assert reopened.header.router_graph_digest == h.header.router_graph_digest


def test_lmo_handle_reads_tile_and_retained(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    h = _absorb(tmp_path, fixture_gguf)
    t0_packed = h.read_tile(0)
    t0_retained = h.read_retained(0)
    assert isinstance(t0_packed, bytes)
    assert isinstance(t0_retained, bytes)
    # Packed tile length matches the units-in-nibbles invariant.
    units = h.tile_plan.tiles[0].units
    assert len(t0_packed) == (units + 1) // 2


def test_lmo_handle_rejects_non_lmo_dir(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(LmoError):
        LmoHandle.open(d)


# --------------------------------------------------------------------------- #
# Auto-materialized manifest
# --------------------------------------------------------------------------- #


def test_auto_manifest_is_v1_and_locks_sha(
    fixture_gguf: Path, tmp_path: Path
) -> None:
    h = _absorb(tmp_path, fixture_gguf)
    text = h.manifest_path.read_text(encoding="utf-8")
    assert "schema = nrl.manifest.v1" in text
    assert "mode = gguf_run" in text
    assert "profile = sovereign" in text
    assert "benchmark_class = B" in text
    assert f"model_sha256 = {h.model_sha256}" in text


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #


def test_cli_absorb_prints_banner(
    fixture_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nrlpy.cli import main as cli_main

    code = cli_main([
        "absorb", str(fixture_gguf),
        "--out-root", str(tmp_path / "cache" / "lmo"),
        "--no-libllama",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "LMO ready:" in out
    assert "model_sha256" in out
    assert "stage A-VI" in out


def test_cli_absorb_json_output(
    fixture_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nrlpy.cli import main as cli_main

    code = cli_main([
        "absorb", str(fixture_gguf),
        "--out-root", str(tmp_path / "cache" / "lmo"),
        "--no-libllama",
        "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retained_byte_identity_ok"] is True
    assert payload["determinism_self_parity_ok"] is True
    assert payload["header_roundtrip_ok"] is True
    assert payload["tile_count"] >= 4
