"""LM/AI opt-in consent file + optional Windows user env (matches ``nrl -ai``)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def consent_json_path() -> Path:
    """``~/.nrl/consent.json`` (``%USERPROFILE%\\.nrl`` on Windows)."""
    return Path.home() / ".nrl" / "consent.json"


def lm_ai_cli_toggle(switch: str) -> int:
    """Handle ``nrlpy -ai on`` / ``nrlpy --ai off`` style switches."""
    s = switch.strip().lower()
    if s in ("on", "-on", "--on"):
        enable = True
    elif s in ("off", "-off", "--off"):
        enable = False
    else:
        return 2
    path = consent_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lm_ai_opt_in": enable,
        "source": "nrlpy -ai",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"nrlpy -ai: wrote {path} (lm_ai_opt_in={enable})")
    if os.name == "nt":
        subprocess.run(
            ["setx", "NRL_LM_AI_OPT_IN", "1" if enable else "0"],
            check=False,
            capture_output=True,
        )
        print("nrlpy -ai: NRL_LM_AI_OPT_IN persisted for new terminals (open a new shell)")
    else:
        print(f"nrlpy -ai: export NRL_LM_AI_OPT_IN={'1' if enable else '0'}  # add to ~/.profile to persist")
    return 0
