# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Execute a Python file with NRL assimilation globals pre-bound."""



from __future__ import annotations



import runpy

import sys

from pathlib import Path



from .compat import llm_globals





def run_path(script_path: str, argv: list[str] | None = None) -> None:

    path = Path(script_path).resolve()

    rest = list(argv) if argv is not None else []

    sys.argv = [str(path)] + rest

    init = llm_globals()

    init["__file__"] = str(path)

    init["__name__"] = "__main__"

    init["__package__"] = None

    init["__cached__"] = None

    runpy.run_path(str(path), init_globals=init, run_name="__main__")


