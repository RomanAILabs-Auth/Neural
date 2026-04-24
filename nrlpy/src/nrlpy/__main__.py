# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""``python -m nrlpy`` entry point.

Delegates directly to :func:`nrlpy.cli.main` so that the native ``nrl`` binary
can spawn ``python -m nrlpy run <model.gguf>`` / ``python -m nrlpy chat
<model.gguf>`` without guessing internal module layout.

The exit code matches ``cli.main`` (0 on success, non-zero on error), and we
import ``cli`` lazily so failures in heavy submodules (e.g. optional
``llama_cpp``) surface as honest runtime errors rather than import-time
tracebacks.
"""

from __future__ import annotations

import sys


def _main() -> int:
    from .cli import main

    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_main())
