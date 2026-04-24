# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""NRL GGUF runner — clean reference example.

This **replaces** the twenty-odd ``nrl_gguf_runner_v*.py`` /
``romanai_nrl_*.py`` / ``*_phi3*.py`` experimental sketches in this folder.
They were exploration; this is the production Python path.

Run (from repo root, with nrlpy on PYTHONPATH or installed editable):

    python -m nrlpy.cli run examples/nrl_run_gguf.py

Or directly with native launcher sugar:

    nrlpy run C:\\path\\to\\phi-3-mini-4k-instruct.Q4_K_M.gguf --prompt "Hi" --seed 42

What this demonstrates (no fairy tales):

  * libllama does the numerics. NRL supervises.
  * Four TPS metrics printed with labels (executed / virtual / cache / effective).
  * Muscle-memory cache: second run with same prompt+sampler is an instant replay.
  * ``nrl assimilate`` attestation runs ONCE at init, not per token.
  * §15-style ``virtual_tps`` footnote printed inline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nrlpy import gguf


DEFAULT_MODEL = Path(
    r"C:\Users\Asus\Desktop\Documents\RomaPy Engine\phi-3-mini-4k-instruct.Q4_K_M.gguf"
)


def main() -> int:
    model = os.environ.get("NRL_GGUF_MODEL", str(DEFAULT_MODEL))
    prompt = os.environ.get(
        "NRL_GGUF_PROMPT",
        "Tell me one short, surprising fact about space.",
    )

    if not Path(model).is_file():
        print(f"model not found: {model}", file=sys.stderr)
        print("Set NRL_GGUF_MODEL=<path>.gguf and try again.", file=sys.stderr)
        return 2

    manifest = gguf.manifest_from_args(
        model=model,
        prompt=prompt,
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        seed=42,
        n_ctx=2048,
        n_threads=0,            # auto = os.cpu_count()
        n_batch=512,            # bigger batch = better TPS on CPU
        chat_format="phi3",     # Phi-3 template; use "none" for base models
        profile="omega-hybrid", # telemetry label only in P1; gating lands in P2
        muscle_memory="on",
        respect_control_hints=True,
        benchmark_class="A",    # requires seed != 0, which we set above
    )

    print(f"> {prompt}\n")
    result = gguf.run_gguf(manifest, stream_to=sys.stdout)
    print(gguf.format_banner(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
