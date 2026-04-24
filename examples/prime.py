# -*- coding: utf-8 -*-
# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""
Smallest prime strictly greater than one trillion.

Run (from NRL repo root, ``nrlpy`` on PATH) — **no** ``import nrlpy`` in this file::

    nrlpy examples/prime.py

``nrlpy`` injects ``next_prime`` / ``is_prime`` / ``fabric_pulse`` like assimilated
builtins; primality itself is deterministic Python (Miller–Rabin). A one-time
``braincore_int4`` pulse ties the run to the NRL extension (see ``nrlpy.seamless``).
"""

from __future__ import annotations


def main() -> None:
    np = globals().get("next_prime")
    if np is None:
        raise SystemExit(
            "This script expects assimilated globals (next_prime).\n"
            "Run from repo root:\n"
            "  nrlpy examples/prime.py\n"
            "Install nrlpy on PATH (e.g. pip install -e nrlpy)."
        )
    trillion = 1_000_000_000_000
    p = np(trillion)
    print(f"next_prime({trillion}) = {p}")
    fp = globals().get("fabric_pulse")
    if fp is not None:
        pulse = fp(neurons=65_536, iterations=128, threshold=8)
        gn = pulse.get("giga_neurons_per_sec", 0.0)
        print(f"fabric_pulse: {gn:.4f} GN/s ({pulse.get('variant', '?')} lane)")


if __name__ == "__main__":
    main()
