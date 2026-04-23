"""Seamless assimilated helpers injected into ``nrlpy``-launched scripts.

When you run ``nrlpy yourscript.py``, :func:`nrlpy.compat.llm_globals` merges these
names into the script's global namespace so **plain Python** can call ``next_prime``,
``is_prime``, etc. without ``import nrlpy``.

**Limits (honest):** arbitrary Python is not auto-translated into lattice machine
code. Integer primality and ``next_prime`` run as **deterministic Python** (Miller–Rabin
with a fixed 64-bit witness set). The NRL **extension** is still exercised once per
process via :func:`fabric_pulse` / first-use warmup so assimilated runs touch the
same machine stack as ``braincore_int4``.
"""

from __future__ import annotations

from typing import Any

# Deterministic Miller–Rabin for n < 2^64 (Jim Sinclair witness set).
_MR_WITNESSES_64: tuple[int, ...] = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)

_fabric_once: bool = False


def _fabric_boot() -> None:
    """One micro ``braincore_int4`` call so assimilated scripts hit the NRL extension."""
    global _fabric_once
    if _fabric_once:
        return
    _fabric_once = True
    try:
        from nrlpy import runtime  # noqa: PLC0415

        runtime.braincore_int4(neurons=4096, iterations=64, threshold=8)
    except Exception:
        # Extension or CPU unavailable — seamless math still works in pure Python.
        pass


def _miller_rabin_pass(n: int, a: int) -> bool:
    """Strong probable prime to base ``a`` (``n`` odd, ``n`` > 2)."""
    if a % n == 0:
        return True
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    x = pow(a, d, n)
    if x in (1, n - 1):
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


def is_prime(n: int) -> bool:
    """Return True iff ``n`` is prime (deterministic for all ``n`` with ``|n|`` < 2**64)."""
    _fabric_boot()
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n >= (1 << 64):
        raise ValueError("is_prime: deterministic check only supports n < 2**64")
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    for a in _MR_WITNESSES_64:
        if not _miller_rabin_pass(n, a):
            return False
    return True


def next_prime(n: int) -> int:
    """Smallest prime *strictly greater than* ``n`` (for ``n`` < 2**63)."""
    _fabric_boot()
    if n < 2:
        return 2
    cand = n + 1
    if cand <= 2:
        return 2
    if cand % 2 == 0:
        cand += 1
    limit = 1 << 63
    while cand < limit:
        if is_prime(cand):
            return cand
        cand += 2
    raise ValueError("next_prime: exceeded safe 63-bit scan window")


def fabric_pulse(
    neurons: int = 65_536,
    iterations: int = 256,
    threshold: int = 8,
) -> dict[str, Any]:
    """Explicit INT4 lattice timing — same kernel family as ``nrl bench`` / assimilate."""
    from nrlpy import runtime  # noqa: PLC0415

    return dict(runtime.braincore_int4(neurons=neurons, iterations=iterations, threshold=threshold))


def injected_globals() -> dict[str, Any]:
    """Names merged into ``run_path`` / ``nrlpy script.py`` globals (no import in user file)."""
    return {
        "is_prime": is_prime,
        "next_prime": next_prime,
        "fabric_pulse": fabric_pulse,
    }
