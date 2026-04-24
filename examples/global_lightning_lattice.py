#!/usr/bin/env python3
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Global Real-Time Lightning Propagation Lattice — NRL control-plane PoC.

Maps **live** lightning strike samples (Blitzortung-style WebSocket JSON) to a
deterministic INT4 workload (neurons / iterations / threshold), then shells out
to the real ``nrl`` binary for ``assimilate`` and ``bench`` — same contract as
the rest of Nrlpy.

**Important (honest scope):** ``nrl bench`` profiles use fixed synthetic packed
drives inside ``engine/src/main.c`` (sovereign / ZPM / omega). This script does
**not** claim to inject lat/lon into those tensors; lightning feeds the
**control plane** (workload sizing + bounded learn-store tokens + audit JSONL).

Live feed (default): Blitzortung community WebSocket (same wire format as
``@simonschick/blitzortungapi``): connect, send ``{"time":0}``, receive
JSON strike objects with ``lat``, ``lon``, ``time``.

Fallbacks (no fake physics numbers):

* ``NRL_LIGHTNING_HTTP`` — HTTP(S) URL whose body is **newline-delimited JSON**
  objects (``lat``, ``lon``, ``time`` …), same schema as Blitzortung strikes.
* ``NRL_LIGHTNING_JSONL`` — local path to the same NDJSON (for air-gapped replay).

Install WebSocket client::

    pip install -e \"nrlpy/[lightning]\"

**Run from the NRL repository root** (Windows)::

    cd C:\\Users\\Asus\\Desktop\\Documents\\NRL
    python -m nrlpy.cli run examples\\global_lightning_lattice.py

POSIX::

    cd /path/to/NRL && python3 -m nrlpy.cli run examples/global_lightning_lattice.py

Optional flags: ``--interval 5 --collect-seconds 2.5 --bench-reps 3``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "nrlpy" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nrlpy.evidence import append_jsonl  # noqa: E402
from nrlpy.learn_store import LearnStore  # noqa: E402
from nrlpy.runtime import assimilate_cli, bench_cli, nrl_binary_path, version  # noqa: E402

_DEFAULT_WS = [
    "wss://ws1.blitzortung.org:3000/",
    "wss://ws2.blitzortung.org:3000/",
    "wss://ws5.blitzortung.org:3000/",
    "wss://ws6.blitzortung.org:3000/",
    "wss://ws7.blitzortung.org:3000/",
    "wss://ws8.blitzortung.org:3000/",
    "wss://ws1.blitzortung.org/",
    "wss://ws7.blitzortung.org/",
]


def _parse_strike_payload(raw: str) -> list[dict[str, Any]]:
    try:
        d: Any = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(d, dict) and "lat" in d and "lon" in d:
        return [d]
    if isinstance(d, list):
        out: list[dict[str, Any]] = []
        for x in d:
            if isinstance(x, dict) and "lat" in x and "lon" in x:
                out.append(x)
        return out
    return []


def _cells_from_strikes(strikes: Iterable[dict[str, Any]]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for s in strikes:
        try:
            lat = float(s["lat"])
            lon = float(s["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        # ~0.5° geodesic binning on the sphere (sparse manifold index).
        cells.add((int(round(lat * 2.0)), int(round(lon * 2.0))))
    return sorted(cells)


def derive_lattice_params(
    strikes: list[dict[str, Any]], tick: int
) -> tuple[int, int, int, str]:
    """Return (neurons, iterations, threshold, digest_hex) — all ``nrl``-valid."""
    cells = _cells_from_strikes(strikes)
    payload = json.dumps(
        {"cells": cells, "n_strikes": len(strikes), "tick": tick},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    h = bytes.fromhex(digest[:32])
    span = (1 << 19) // 8192  # 64 steps between 8192 and 524288 neurons
    idx = int.from_bytes(h[0:2], "big") % max(1, span)
    neurons = 8192 + idx * 8192
    iterations = 128 + (int.from_bytes(h[2:4], "big") % 3840)
    iterations = max(64, min(iterations, 8192))
    threshold = 4 + (h[4] % 12)
    return neurons, iterations, threshold, digest


def fetch_strikes_http(url: str, max_lines: int = 500) -> list[dict[str, Any]]:
    req = urlrequest.Request(url, headers={"User-Agent": "NRL-global-lightning-lattice/1.0"})
    with urlrequest.urlopen(req, timeout=45) as resp:  # noqa: S310 - user-controlled audit URL
        text = resp.read().decode("utf-8", errors="replace")
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if len(out) >= max_lines:
            break
        line = line.strip()
        if not line:
            continue
        out.extend(_parse_strike_payload(line))
    return out


def fetch_strikes_file(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(out) >= max_lines:
            break
        line = line.strip()
        if not line:
            continue
        out.extend(_parse_strike_payload(line))
    return out


async def fetch_strikes_websocket(uris: list[str], collect_seconds: float) -> list[dict[str, Any]]:
    try:
        import websockets
    except ImportError as e:  # pragma: no cover - exercised manually
        raise RuntimeError(
            "Install websockets: pip install 'websockets>=12' "
            "or pip install -e nrlpy/[lightning]"
        ) from e

    deadline = time.monotonic() + collect_seconds
    strikes: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    for uri in uris:
        if time.monotonic() >= deadline or len(strikes) >= 400:
            break
        try:
            async with websockets.connect(uri, ping_interval=25, close_timeout=5) as ws:
                await ws.send(json.dumps({"time": 0}))
                while time.monotonic() < deadline and len(strikes) < 400:
                    try:
                        timeout = max(0.1, deadline - time.monotonic())
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(3.0, timeout))
                    except asyncio.TimeoutError:
                        break
                    raw_s = raw if isinstance(raw, str) else str(raw, "utf-8", errors="replace")
                    for s in _parse_strike_payload(raw_s):
                        key = (
                            int(float(s["time"])),
                            int(float(s["lat"]) * 1000),
                            int(float(s["lon"]) * 1000),
                        )
                        if key not in seen:
                            seen.add(key)
                            strikes.append(s)
        except Exception:
            continue
        if strikes:
            break
    return strikes


def _fmt_sublattices(b: dict[str, Any]) -> str:
    if "avg_active_sublattices" not in b:
        return "n/a (profile has no omega-style sub-lattice stats)"
    return (
        f"active={b.get('avg_active_sublattices', 0):.2f}  "
        f"total={b.get('avg_total_sublattices', 0):.2f}  "
        f"pruned={b.get('avg_pruned_sublattices', 0):.2f}"
    )


def _print_banner(title: str) -> None:
    line = "=" * min(78, max(40, len(title) + 8))
    print(f"\n{line}\n  {title}\n{line}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between cycles (default 5)")
    ap.add_argument(
        "--collect-seconds",
        type=float,
        default=2.5,
        help="Per-cycle WebSocket listen budget (default 2.5)",
    )
    ap.add_argument("--bench-reps", type=int, default=3, help="nrl bench reps (default 3)")
    ap.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Exit after N cycles (0 = run until Ctrl+C; use 1 for a quick smoke test)",
    )
    ap.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Append-only JSONL audit path (default: build/lightning_lattice_audit.jsonl)",
    )
    args = ap.parse_args()

    audit_path = args.audit_log or (_REPO_ROOT / "build" / "lightning_lattice_audit.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # Dedicated learn store so demos do not mix with chat vocab.
    learn_root = _REPO_ROOT / "build" / "nrlpy_learn_lightning"
    learn_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NRL_LEARN_DIR", str(learn_root))
    store = LearnStore(root=learn_root)

    http_url = os.environ.get("NRL_LIGHTNING_HTTP")
    jsonl_path = os.environ.get("NRL_LIGHTNING_JSONL")
    ws_env = os.environ.get("NRL_BLITZ_WS_URIS")
    uris = [u.strip() for u in ws_env.split(",")] if ws_env else list(_DEFAULT_WS)

    exe = nrl_binary_path()
    _print_banner("NRL - Global Real-Time Lightning Propagation Lattice")
    print(f"  nrl binary . . . . . . {exe}")
    print(f"  nrlpy version  . . . {version()}")
    print(f"  audit JSONL  . . . . {audit_path}")
    print(f"  learn store  . . . . {learn_root}")
    print(f"  cycle interval . . . {args.interval:.1f}s")
    if http_url:
        print(f"  strike source  . . . HTTP {http_url}")
    elif jsonl_path:
        print(f"  strike source  . . . file {jsonl_path}")
    else:
        print(f"  strike source  . . . WebSocket ({len(uris)} candidate URIs)")

    tick = 0
    print("\nCtrl+C to stop. All bench/assimilate numbers come from subprocess ``nrl``.\n")

    while True:
        tick += 1
        t0 = time.time()
        strikes: list[dict[str, Any]] = []
        source = "websocket"
        try:
            if http_url:
                strikes = fetch_strikes_http(http_url)
                source = "http"
            elif jsonl_path:
                jp = Path(jsonl_path)
                if not jp.is_file():
                    raise RuntimeError(f"NRL_LIGHTNING_JSONL not a file: {jp}")
                strikes = fetch_strikes_file(jp)
                source = "jsonl"
            else:
                strikes = asyncio.run(fetch_strikes_websocket(uris, args.collect_seconds))
        except RuntimeError as e:
            print(f"[cycle {tick}] strike ingest error: {e}")
            strikes = []
        except (urlerror.URLError, OSError) as e:
            print(f"[cycle {tick}] strike ingest transport error: {e}")
            strikes = []

        cells = _cells_from_strikes(strikes)
        neurons, iterations, threshold, digest = derive_lattice_params(strikes, tick)

        learn_line = " ".join(f"cell_{la}_{lo}" for la, lo in cells[:200])
        if learn_line:
            store.observe_text(learn_line + " strike_pulse")
        else:
            store.observe_text(f"idle_tick_{tick} strike_pulse")

        stats = store.stats()

        _print_banner(f"Cycle {tick} - ingest: {source}")
        if not strikes and source == "websocket":
            print(
                "  (no strikes this window - firewall may block Blitzortung; set "
                "NRL_LIGHTNING_HTTP or NRL_LIGHTNING_JSONL, or: pip install websockets)"
            )
        print(f"  Strikes this sample . {len(strikes)}")
        print(f"  Distinct 0.5 deg cells . {len(cells)}")
        print(f"  Workload digest . . . {digest[:24]}...")
        print(f"  Derived N / I / T . . {neurons} / {iterations} / {threshold}")
        print(f"  Learn store words . . {stats.unique_words} (disk {stats.used_bytes:,} B)")

        # --- Sovereign assimilate parity (two independent CLI runs must match).
        try:
            a1 = assimilate_cli(neurons, iterations, threshold)
            a2 = assimilate_cli(neurons, iterations, threshold)
        except RuntimeError as e:
            print(f"\n  assimilate_cli ERROR: {e}")
            return 1
        c1 = int(a1["checksum_fnv1a64"])
        c2 = int(a2["checksum_fnv1a64"])
        match = c1 == c2
        print("\n  --- assimilate (sovereign lane, real ``nrl assimilate``) ---")
        print(f"  checksum run A . . . . {c1}")
        print(f"  checksum run B . . . . {c2}")
        print(f"  parity (A==B)  . . . . {'YES' if match else 'NO'}")

        # --- Bench profiles (real ``nrl bench``).
        reps = max(1, min(int(args.bench_reps), 32))
        try:
            b_sov1 = bench_cli(neurons, iterations, reps, threshold, "sovereign")
            b_sov2 = bench_cli(neurons, iterations, reps, threshold, "sovereign")
            b_zpm = bench_cli(neurons, iterations, reps, threshold, "zpm")
            b_omega = bench_cli(neurons, iterations, reps, threshold, "omega")
            b_hybrid = bench_cli(neurons, iterations, reps, threshold, "omega-hybrid")
        except RuntimeError as e:
            print(f"\n  bench_cli ERROR: {e}")
            return 1

        r_elapsed = float(b_sov2["elapsed_s"]) / max(1e-12, float(b_sov1["elapsed_s"]))
        print("\n  --- bench: sovereign (repeat for wall-time witness) ---")
        print(f"  elapsed_s 1st . . . . . {float(b_sov1['elapsed_s']):.6f}")
        print(f"  elapsed_s 2nd . . . . . {float(b_sov2['elapsed_s']):.6f}")
        print(f"  ratio (2nd/1st) . . . . {r_elapsed:.3f}x  (same argv; not ML - OS/CPU variance)")
        print(
            f"  skip_ratio / GOPS . . . skip={float(b_sov1['skip_ratio']):.4f}  "
            f"exec={float(b_sov1['executed_gops']):.3f}  virt={float(b_sov1['virtual_gops']):.3f}"
        )

        print("\n  --- bench: ZPM (static-collapse lane) ---")
        print(
            f"  skip_ratio / GOPS . . . skip={float(b_zpm['skip_ratio']):.4f}  "
            f"exec={float(b_zpm['executed_gops']):.3f}  virt={float(b_zpm['virtual_gops']):.3f}"
        )

        print("\n  --- bench: omega (sparse virtual wave) ---")
        print(
            f"  skip_ratio / GOPS . . . skip={float(b_omega['skip_ratio']):.4f}  "
            f"exec={float(b_omega['executed_gops']):.3f}  virt={float(b_omega['virtual_gops']):.3f}"
        )
        print(f"  sub-lattices (avg) . . {_fmt_sublattices(b_omega)}")

        print("\n  --- bench: omega-hybrid ---")
        print(
            f"  skip_ratio / GOPS . . . skip={float(b_hybrid['skip_ratio']):.4f}  "
            f"exec={float(b_hybrid['executed_gops']):.3f}  virt={float(b_hybrid['virtual_gops']):.3f}"
        )
        print(f"  sub-lattices (avg) . . {_fmt_sublattices(b_hybrid)}")

        print("\n  --- checksum parity: sovereign vs omega ---")
        print(
            "  assimilate checksum applies to the sovereign INT4 digest contract only.\n"
            "  Omega/ZPM use different execution lanes; their observables are skip_ratio,\n"
            "  executed_updates, and virtual_gops - not the assimilate FNV digest."
        )

        record = {
            "schema_id": "nrl.lightning_lattice_cycle.v1",
            "ts_wall": time.time(),
            "tick": tick,
            "source": source,
            "n_strikes": len(strikes),
            "n_cells": len(cells),
            "digest": digest,
            "neurons": neurons,
            "iterations": iterations,
            "threshold": threshold,
            "assimilate_checksum_a": c1,
            "assimilate_checksum_b": c2,
            "assimilate_parity": match,
            "sovereign_elapsed_s_1": float(b_sov1["elapsed_s"]),
            "sovereign_elapsed_s_2": float(b_sov2["elapsed_s"]),
            "sovereign_skip": float(b_sov1["skip_ratio"]),
            "zpm_skip": float(b_zpm["skip_ratio"]),
            "omega_skip": float(b_omega["skip_ratio"]),
            "learn_unique_words": stats.unique_words,
        }
        append_jsonl(audit_path, record)

        if int(args.max_cycles) > 0 and tick >= int(args.max_cycles):
            print(f"\nFinished {tick} cycle(s) (--max-cycles).")
            return 0

        dt = time.time() - t0
        sleep_for = max(0.5, float(args.interval) - dt)
        print(f"\n  cycle wall {dt:.2f}s - sleeping {sleep_for:.2f}s")
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
