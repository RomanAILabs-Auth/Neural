# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Bounded on-disk vocabulary store (control-plane; not kernel training).

Default cap: 4 GiB (``max_bytes`` in ``config.json``). Words are tokenized from
free text; counts accumulate in ``vocab.json``. When the serialized store would
exceed the cap, lowest-frequency entries are pruned until usage is under the
target ratio.

Environment:
  ``NRL_LEARN_DIR`` — override directory (default ``<cwd>/build/nrlpy_learn``).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import evidence

SCHEMA_V1 = "nrl.learn_store.v1"
DEFAULT_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB default cap for on-disk growth
ABS_MIN_MAX_BYTES = 4096  # hard floor (tests); production deployments should use ≥ 1 MiB
PRUNE_TARGET_RATIO = 0.85  # after prune, aim below cap * this


def _root() -> Path:
    env = os.environ.get("NRL_LEARN_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "build" / "nrlpy_learn"


@dataclass
class LearnStats:
    max_bytes: int
    used_bytes: int
    unique_words: int
    total_observations: int
    path: Path

    def summary(self) -> str:
        pct = (100.0 * self.used_bytes / self.max_bytes) if self.max_bytes else 0.0
        return (
            f"Learn store at {self.path}\n"
            f"  Budget cap: {self.max_bytes:,} bytes ({self.max_bytes / (1024**3):.2f} GiB)\n"
            f"  Disk used:  {self.used_bytes:,} bytes ({pct:.2f}% of cap)\n"
            f"  Unique words: {self.unique_words:,}\n"
            f"  Total observations (sum of counts): {self.total_observations:,}"
        )


class LearnStore:
    """Light JSON vocabulary file + append-only growth log; prunes by count."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or _root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._config_path = self.root / "config.json"
        self._vocab_path = self.root / "vocab.json"
        self._growth_path = self.root / "growth.jsonl"
        self._load_or_init_config()

    def _load_or_init_config(self) -> None:
        if self._config_path.is_file():
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self._max_bytes = int(data.get("max_bytes", DEFAULT_MAX_BYTES))
            if self._max_bytes < ABS_MIN_MAX_BYTES:
                self._max_bytes = ABS_MIN_MAX_BYTES
        else:
            self._max_bytes = DEFAULT_MAX_BYTES
            self._write_config()

    def _write_config(self) -> None:
        payload = {
            "schema_id": "nrl.learn_config.v1",
            "max_bytes": self._max_bytes,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def set_max_bytes(self, n: int) -> None:
        """Raise or lower the cap; persists to config."""
        if n < ABS_MIN_MAX_BYTES:
            raise ValueError(f"max_bytes must be at least {ABS_MIN_MAX_BYTES}")
        if n < 1024 * 1024:
            warnings.warn(
                "max_bytes below 1 MiB is for testing only; production should use >= 1 MiB",
                stacklevel=2,
            )
        self._max_bytes = int(n)
        self._write_config()
        self._maybe_prune_vocab()

    def disk_usage(self) -> int:
        total = 0
        if self.root.is_dir():
            for p in self.root.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        return total

    def _read_vocab(self) -> dict[str, int]:
        if not self._vocab_path.is_file():
            return {}
        data = json.loads(self._vocab_path.read_text(encoding="utf-8"))
        raw = data.get("words", {})
        return {str(k): int(v) for k, v in raw.items() if int(v) > 0}

    def _vocab_blob_size(self, words: dict[str, int]) -> int:
        payload = {"schema_id": SCHEMA_V1, "words": words}
        return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def _write_vocab(self, words: dict[str, int]) -> None:
        payload = {
            "schema_id": SCHEMA_V1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "words": dict(sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self._vocab_path.write_text(text + "\n", encoding="utf-8")

    def _maybe_prune_vocab(self) -> int:
        """Prune lowest-count words until blob under cap * PRUNE_TARGET_RATIO. Returns pruned count."""
        words = self._read_vocab()
        if not words:
            return 0
        pruned = 0
        last_drop = ""
        target = int(self._max_bytes * PRUNE_TARGET_RATIO)
        while self._vocab_blob_size(words) > target and len(words) > 1:
            # drop single lowest count (ties: lexicographically last key among min count)
            min_c = min(words.values())
            candidates = [w for w, c in words.items() if c == min_c]
            last_drop = max(candidates)
            del words[last_drop]
            pruned += 1
        if pruned:
            self._write_vocab(words)
            self._log_growth("prune", {"removed_words": pruned, "last_removed": last_drop})
        return pruned

    def _log_growth(self, kind: str, extra: dict[str, Any]) -> None:
        evidence.append_jsonl(
            self._growth_path,
            {
                "schema_id": "nrl.learn_growth.v1",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                **extra,
            },
        )

    def observe_text(self, text: str) -> tuple[int, int]:
        """Tokenize ``text``; update counts. Returns (tokens_seen, new_unique_this_call)."""
        tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
        if not tokens:
            return (0, 0)
        words = self._read_vocab()
        before_keys = set(words)
        for t in tokens:
            words[t] = words.get(t, 0) + 1
        new_unique = len(set(words) - before_keys)
        self._write_vocab(words)
        self._maybe_prune_vocab()
        self._log_growth(
            "observe",
            {"tokens": len(tokens), "new_unique": new_unique, "sample": tokens[:12]},
        )
        return (len(tokens), new_unique)

    def stats(self) -> LearnStats:
        words = self._read_vocab()
        total_obs = sum(words.values())
        return LearnStats(
            max_bytes=self._max_bytes,
            used_bytes=self.disk_usage(),
            unique_words=len(words),
            total_observations=total_obs,
            path=self.root,
        )


def default_store() -> LearnStore:
    return LearnStore()
