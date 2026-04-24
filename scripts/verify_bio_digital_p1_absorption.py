#!/usr/bin/env python3
# Copyright (c) 2026 Daniel Harding - RomanAILabs
"""Bio-Digital Blueprint P1 — operator verification for GGUF → LMO absorption.

Usage (from NRL repo root, with nrlpy on PYTHONPATH or installed)::

    python scripts/verify_bio_digital_p1_absorption.py path/to/model.gguf

Uses ``NRL_ROOT`` (or cwd) for ``cache/lmo/<sha>/``. Runs two forced absorbs,
checks Stage A-VI attest flags, stable digests, and ``LmoHandle.open`` read-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    nrlpy_src = repo / "nrlpy" / "src"
    if nrlpy_src.is_dir():
        sys.path.insert(0, str(nrlpy_src))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf", help="Path to .gguf model file")
    p.add_argument(
        "--out-root",
        default="",
        help="LMO cache root (default: $NRL_ROOT/cache/lmo or ./cache/lmo)",
    )
    args = p.parse_args()
    gguf = Path(args.gguf)
    if not gguf.is_file():
        print(f"error: not a file: {gguf}", file=sys.stderr)
        return 2

    from nrlpy import lmo  # noqa: PLC0415

    out_root = Path(args.out_root) if args.out_root else None
    if out_root is None:
        nrl = os.environ.get("NRL_ROOT", "").strip()
        base = Path(nrl) if nrl else Path.cwd()
        out_root = base / "cache" / "lmo"

    print(f"P1 verify: gguf={gguf}")
    print(f"          out_root={out_root.resolve()}")

    h1 = lmo.absorb_gguf(gguf, out_root=out_root, force=True, attempt_libllama=False)
    h2 = lmo.absorb_gguf(gguf, out_root=out_root, force=True, attempt_libllama=False)

    a1, a2 = h1.attest, h2.attest
    for key in ("retained_byte_identity_ok", "determinism_self_parity_ok", "header_roundtrip_ok"):
        if not a1.get(key) or not a2.get(key):
            print(f"error: attest[{key!r}] must be true after A-VI", file=sys.stderr)
            return 1

    if h1.header.lmo_anchor != h2.header.lmo_anchor:
        print("error: lmo_anchor drift across forced re-absorb", file=sys.stderr)
        return 1
    if h1.header.tile_plan_digest != h2.header.tile_plan_digest:
        print("error: tile_plan_digest drift", file=sys.stderr)
        return 1
    if h1.header.router_graph_digest != h2.header.router_graph_digest:
        print("error: router_graph_digest drift", file=sys.stderr)
        return 1

    h3 = lmo.LmoHandle.open(h1.lmo_dir)
    if h3.read_tile(0) != h1.read_tile(0):
        print("error: read_tile(0) mismatch after LmoHandle.open", file=sys.stderr)
        return 1

    print("P1 OK: Stage A-VI attest flags true; stable digests; LmoHandle.open read-only smoke passed.")
    print(f"        lmo_dir={h1.lmo_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
