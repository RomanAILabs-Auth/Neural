# Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved.
"""Canonical workload identity for harnesses and specialization keys.

Implements ``nrl.workload_descriptor.v1`` (see ``docs/schemas/workload_descriptor_v1.schema.json``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_ID = "nrl.workload_descriptor.v1"


def build_workload_descriptor(
    *,
    harness_id: str,
    neurons: int,
    iterations: int,
    reps: int,
    threshold: int,
    profiles: Sequence[str],
    nrl_version: str | None = None,
    static_drive_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return a v1 workload descriptor (identity fields only — no timings)."""
    profiles_norm = sorted(str(p).strip() for p in profiles if str(p).strip())
    body: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "harness_id": str(harness_id).strip(),
        "neurons": int(neurons),
        "iterations": int(iterations),
        "reps": int(reps),
        "threshold": int(threshold),
        "profiles": profiles_norm,
    }
    if nrl_version is not None:
        body["nrl_version"] = str(nrl_version)
    if static_drive_fingerprint is not None:
        body["static_drive_fingerprint"] = str(static_drive_fingerprint)
    return body


def canonical_json_bytes(descriptor: Mapping[str, Any]) -> bytes:
    """UTF-8 JSON with stable key order for hashing."""
    return json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode("utf-8")


def structural_hash(descriptor: Mapping[str, Any]) -> str:
    """SHA-256 hex of the canonical JSON encoding of ``descriptor``."""
    return hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()


def workload_id(harness_id: str, structural_hash_hex: str) -> str:
    """Opaque primary key: ``harness_id|structural_hash``."""
    return f"{str(harness_id).strip()}|{structural_hash_hex}"


def workload_identity_block(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptor + derived ``structural_hash`` and ``workload_id`` for JSON artifacts."""
    sh = structural_hash(descriptor)
    hid = str(descriptor["harness_id"])
    return {
        "descriptor": dict(descriptor),
        "structural_hash": sh,
        "workload_id": workload_id(hid, sh),
    }
