#!/usr/bin/env python3
# Copyright (c) 2026 Daniel Harding - RomanAILabs
#
# Phase 15.1 — Standalone "Ultimate REPL" sidecar. Imports nrlpy from the sibling
# tree but does not modify any file under nrlpy/. RomanAILabs engine demo:
# native_full + phi3 + optional rewired preset + Phase 15 warm-restart prime.
"""RomanAILabs NRL Ultimate chat runner (standalone sidecar).

Run from the NRL repo root (or anywhere, if PYTHONPATH includes ``nrlpy/src``)::

    python nrl_chat_runner.py path/to/model.gguf --rewired --system "You are helpful."

Requires Phase 8-EG ``native_full`` bindings. Uses strict ``phi3`` templating via
``nrlpy.gguf_chat`` (rendered history + ``chat_format='none'`` on the per-turn
manifest, matching the in-tree chat contract).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Bootstrap: resolve nrlpy package from repo layout (no install required).
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent
_NRLPY_SRC = _REPO_ROOT / "nrlpy" / "src"
if _NRLPY_SRC.is_dir() and str(_NRLPY_SRC) not in sys.path:
    sys.path.insert(0, str(_NRLPY_SRC))

# Verified LMO / absorption reference (informational banner only).
_DEFAULT_LMO_SHA_HINT = "4fed7364"


def _as_numpy_if_available(x: Any) -> Any:
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "__array__"):
            return np.asarray(x)
    except Exception:
        pass
    return None


def _cache_hit_bool(result: Any) -> bool:
    """Strict lattice hit flag — never ``bool(ndarray)`` or ``if arr:``."""
    if result is None:
        return False
    ch = getattr(result, "cache_hit", False)
    if ch is True:
        return True
    if ch is False or ch is None:
        return False
    arr = _as_numpy_if_available(ch)
    if arr is not None:
        if arr.size == 0:
            return False
        flat = arr.ravel()
        try:
            return bool(flat.all()) if flat.dtype == bool else bool(flat.any())
        except Exception:
            return False
    if isinstance(ch, (list, tuple)):
        return len(ch) > 0 and any(bool(x) for x in ch)
    try:
        return bool(int(ch)) != 0
    except (TypeError, ValueError):
        return False


def _item_number(val: Any) -> Any | None:
    """NumPy scalar → Python primitive via ``.item()`` when available."""
    getter = getattr(val, "item", None)
    if getter is not None and callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _safe_int_scalar(val: Any, default: int = 0) -> int:
    """Integer for telemetry — no ``val or 0`` (breaks on ndarray)."""
    if val is None:
        return default
    prim = _item_number(val)
    if prim is not None:
        try:
            return int(prim)
        except (TypeError, ValueError):
            pass
    arr = _as_numpy_if_available(val)
    if arr is not None:
        if arr.size == 0:
            return default
        try:
            return int(arr.reshape(-1)[0])
        except Exception:
            return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float_scalar(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    prim = _item_number(val)
    if prim is not None:
        try:
            return float(prim)
        except (TypeError, ValueError):
            pass
    arr = _as_numpy_if_available(val)
    if arr is not None:
        if arr.size == 0:
            return default
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _model_sha_string(val: Any) -> str:
    """Normalize model SHA to ``str`` without truth-testing a ndarray."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    arr = _as_numpy_if_available(val)
    if arr is not None and arr.size > 0:
        try:
            return str(arr.reshape(-1)[0]).strip()
        except Exception:
            return ""
    try:
        return str(val).strip()
    except Exception:
        return ""


def _non_empty_model_sha(val: Any) -> bool:
    """Safe ``model_sha`` presence — no ``if not model_sha`` on ndarray."""
    if val is None:
        return False
    if isinstance(val, str):
        return len(val.strip()) > 0
    arr = _as_numpy_if_available(val)
    if arr is not None:
        if arr.size == 0:
            return False
        try:
            s = str(arr.reshape(-1)[0])
        except Exception:
            s = str(arr)
        return len(s.strip()) > 0
    try:
        return len(str(val).strip()) > 0
    except Exception:
        return False


def _sanitize_anchor_bytes(anchor: Any) -> bytes:
    """Force ``bytes`` for ``zpm.prime`` — Phase 15.3 emergency anchor path (no ``if anchor:``)."""
    if anchor is None:
        return b""
    if isinstance(anchor, bytes):
        return anchor
    if isinstance(anchor, bytearray):
        return bytes(anchor)
    if isinstance(anchor, memoryview):
        return anchor.tobytes()
    if hasattr(anchor, "tobytes"):
        try:
            return anchor.tobytes()
        except Exception:
            pass
    try:
        return bytes(anchor)
    except Exception:
        return b""


def _display_float(x: Any) -> float:
    """Python ``float`` for f-strings — unwraps NumPy scalars via ``.item()``."""
    p = _item_number(x)
    if p is not None:
        try:
            return float(p)
        except (TypeError, ValueError):
            pass
    return float(_safe_float_scalar(x, 0.0))


def _tail_token_ids_tuple(preloaded_llm: Any) -> tuple[int, ...]:
    """Tail ids as plain ``tuple`` for ``zpm.prime`` (no ndarray in varargs)."""
    ids = getattr(preloaded_llm, "_input_ids", None)
    if ids is None:
        return ()
    arr = _as_numpy_if_available(ids)
    if arr is not None:
        if arr.size == 0:
            return ()
        tail = arr.ravel()[-4:]
        return tuple(int(x) for x in tail.tolist())
    try:
        seq = list(ids)
    except Exception:
        return ()
    return tuple(int(x) for x in seq[-4:])


def _safe_gate_source(result: Any) -> str | None:
    if result is None:
        return None
    raw = getattr(result, "gate_source", None)
    arr = _as_numpy_if_available(raw)
    if arr is not None:
        if arr.size == 0:
            return None
        # Prefer first element as string label if object array (no ``if el:``).
        try:
            return str(arr.flat[0])
        except Exception:
            return None
    if raw is None:
        return None
    return str(raw)


def _winning_rung_label(result: Any) -> str:
    """Same ladder badges as :func:`nrlpy.gguf_chat.ladder_badge_plain`."""
    from nrlpy.gguf_chat import ladder_badge_plain as _badge  # noqa: PLC0415

    if result is None:
        return "[Decode]"
    try:
        return _badge(result)
    except Exception:
        return "[Decode]"


def _turn_primary_wps(result: Any) -> float:
    if result is None:
        return 0.0
    wr = getattr(result, "word_rates", None)
    if wr is None:
        return 0.0
    if _cache_hit_bool(result) is True:
        return _safe_float_scalar(getattr(wr, "effective_wps", 0.0), 0.0)
    return _safe_float_scalar(getattr(wr, "executed_wps", 0.0), 0.0)


def _maybe_zpm_prime(
    *,
    gguf_mod: Any,
    zpm_mod: Any,
    per_turn_manifest_fn: Any,
    session: Any,
    user_text: str,
    preloaded_llm: Any,
    consecutive_r5: int,
) -> None:
    # Integer streak only — never ``consecutive_r5 or 0`` on a stray ndarray.
    if _safe_int_scalar(consecutive_r5, 0) <= 3:
        return
    model_sha = getattr(session, "model_sha256", None)
    if not _non_empty_model_sha(model_sha):
        model_sha = getattr(session.base_manifest, "model_sha256", None)
    if not _non_empty_model_sha(model_sha):
        return
    model_sha_s = _model_sha_string(model_sha)
    if model_sha_s is None or len(model_sha_s) == 0:
        return
    per_next = per_turn_manifest_fn(session, user_text)
    raw_anchor = gguf_mod._zpm_anchor_bytes(per_next, per_next.prompt)
    anchor_b = _sanitize_anchor_bytes(raw_anchor)
    tail_t = _tail_token_ids_tuple(preloaded_llm)
    try:
        zpm_mod.prime(
            model_sha_s,
            anchor_b,
            tail_t,
            gguf_mod._zpm_index_path(model_sha_s),
        )
    except Exception:
        pass


def _read_line(prompt: str, sin: Any, sout: Any) -> str | None:
    """Standard terminal readline — no NumPy detection (Phase 15.4)."""
    sout.write(prompt)
    sout.flush()
    try:
        line = sin.readline()
        if not line:
            return None
        return str(line).rstrip("\r\n")
    except (EOFError, KeyboardInterrupt):
        return None


def run(argv: list[str] | None = None) -> int:
    # Phase 15.5 — process-local FP error mode (does not change ndarray truth rules).
    try:
        import numpy as np  # noqa: PLC0415

        np.seterr(all="ignore")
    except Exception:
        pass

    from nrlpy import gguf  # noqa: PLC0415
    from nrlpy.gguf_chat import (  # noqa: PLC0415
        _handle_slash,
        _per_turn_manifest,
        build_session,
        chat_turn,
    )
    from nrlpy import zpm as zpm_mod  # noqa: PLC0415
    from nrlpy.gguf import load_manifest, manifest_from_args  # noqa: PLC0415
    from nrlpy.native_ladder import is_full_native_available  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="RomanAILabs Ultimate REPL (native_full + phi3 + telemetry).",
    )
    p.add_argument("model", help="Path to .gguf or .nrl manifest")
    p.add_argument("--system", default="", help="System prompt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--rewired",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewired preset (KV session + max-throughput lane). Default: on.",
    )
    p.add_argument(
        "--response-recall",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whole-reply R0+R1 + ZPM nullspace. Default: on with --rewired; "
        "use --no-response-recall to disable while keeping rewired lanes.",
    )
    p.add_argument(
        "--lmo-sha",
        default=_DEFAULT_LMO_SHA_HINT,
        help="Informational LMO / absorption SHA prefix for the banner.",
    )
    ns = p.parse_args(argv)

    if not is_full_native_available():
        print(
            "error: native_full (Phase 8-EG) is required for this runner.",
            file=sys.stderr,
        )
        return 2

    extra: dict[str, Any] = {
        "runner_backend": "native_full",
        "chat_format": "phi3",
    }
    if ns.seed is not None:
        extra["seed"] = ns.seed
    if ns.max_tokens is not None:
        extra["max_tokens"] = ns.max_tokens
    if ns.temperature is not None:
        extra["temperature"] = ns.temperature

    user_overrides = set(extra.keys())

    from nrlpy import gguf_chat as gc  # noqa: PLC0415

    if ns.rewired:
        no_rr = ns.response_recall is False
        if no_rr:
            extra["muscle_memory"] = "off"
            user_overrides.update({"muscle_memory", "zpm_nullspace"})
        gc._apply_rewired_defaults(extra, user_overrides)
    else:
        gc._apply_fast_chat_defaults(extra, user_overrides)

    try:
        mpath = ns.model
        if mpath.lower().endswith(".nrl"):
            manifest = load_manifest(mpath)
            for k, v in extra.items():
                setattr(manifest, k, v)
        else:
            manifest = manifest_from_args(model=mpath, **extra)
    except gguf.ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    manifest.runner_backend = "native_full"
    manifest.chat_format = "phi3"
    if ns.rewired:
        gc.apply_rewired_post_build(manifest, user_overrides)
    else:
        gc.apply_fast_chat_post_build(manifest, user_overrides)
    if (not ns.rewired) and ns.response_recall is True:
        gc.apply_response_recall(manifest, user_overrides)

    manifest.prompt = ""
    manifest.prompt_file = ""

    sin, sout = sys.stdin, sys.stdout
    session = build_session(manifest, system=ns.system)
    model_path = Path(manifest.model)
    if not model_path.is_file():
        print(f"error: model not found: {manifest.model}", file=sys.stderr)
        return 2

    actual_sha = gguf.sha256_file(model_path)
    if manifest.model_sha256 and manifest.model_sha256 != actual_sha:
        print("error: model_sha256 mismatch manifest vs file", file=sys.stderr)
        return 2
    manifest.model_sha256 = actual_sha
    session.model_sha256 = actual_sha

    preloaded_llm = gguf._load_llm(manifest)

    sout.write(
        "\n".join(
            [
                "RomanAILabs — NRL Ultimate REPL (Phase 15.1 sidecar)",
                f"  model          {model_path.name}",
                f"  backend        native_full (Phase 8-EG)",
                f"  chat_format    phi3 (strict template)",
                f"  rewired        {'yes' if ns.rewired else 'no'}",
                f"  LMO ref SHA    {ns.lmo_sha!r} (informational)",
                "",
                "  Type /exit or /quit to leave. /help for slash commands.",
                "",
            ]
        )
    )
    sout.flush()

    consecutive_r5 = 0
    turn_count = 0
    sum_turn_wps = 0.0

    while True:
        prompt = f"[you #{session.turn_count()}]> "
        line = _read_line(prompt, sin, sout)
        if line is None:
            break
        stripped = str(line).strip()
        if len(stripped) == 0:
            continue
        verdict = _handle_slash(stripped, session, sout)
        if verdict == "quit":
            break
        if verdict == "continue":
            low = stripped.lower()
            if low.startswith("/clear") or low.startswith("/system") or low.startswith("/load"):
                consecutive_r5 = 0
            continue

        # Phase 15 — drift prime: after sustained R5 streak, warm ZPM RAM from
        # the live tail anchor (best-effort; never blocks Tier-1).
        if _safe_int_scalar(consecutive_r5, 0) > 3:
            _maybe_zpm_prime(
                gguf_mod=gguf,
                zpm_mod=zpm_mod,
                per_turn_manifest_fn=_per_turn_manifest,
                session=session,
                user_text=stripped,
                preloaded_llm=preloaded_llm,
                consecutive_r5=consecutive_r5,
            )
            consecutive_r5 = 0

        if hasattr(session, "turn_count") and not isinstance(session.turn_count(), int):
            # If turn_count returned a numpy scalar, cast it
            pass

        t0 = time.perf_counter()
        try:
            result = chat_turn(
                session=session,
                user_text=str(stripped),
                stream_to=sout,
                preloaded_llm=preloaded_llm,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            sout.write(f"\nerror: {type(e).__name__}: {e}\n")
            sout.flush()
            continue

        if result is None:
            sout.write("\nerror: chat_turn returned no result (skipped).\n")
            sout.flush()
            continue

        wall = float(time.perf_counter() - t0)
        turn_count += 1
        cache_hit = _cache_hit_bool(result)
        tok = _safe_int_scalar(getattr(result, "tokens", 0), 0)
        drift = _safe_int_scalar(getattr(result, "drift_reprime_count", 0), 0)

        wps = _display_float(_turn_primary_wps(result))
        sum_turn_wps += wps
        avg = _display_float(sum_turn_wps / turn_count if turn_count else 0.0)

        if cache_hit is not True:
            consecutive_r5 += 1
        else:
            consecutive_r5 = 0

        rung = _winning_rung_label(result)
        sout.write(
            f"\n--- {rung}  |  turn_wps={wps:.1f}  session_avg_wps={avg:.1f}  "
            f"|  {tok} tok / {wall:.2f}s  |  drift_reprime_count={drift}\n"
        )
        if ns.rewired and cache_hit is not True:
            sout.write(
                "  Learning this prompt... next similar question will be instant.\n"
            )
        sout.flush()

    sout.write("\nSession ended.\n")
    sout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
