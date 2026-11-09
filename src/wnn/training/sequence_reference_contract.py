from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wnn.training.sequence_reference_pins import (
    ANCESTRY_LIMITATION,
    ARITHMETIC,
    COMPOSITION,
    PR49_SOURCE_COMMIT,
    PRIMARY_SPLIT_SHA256,
    PUBLISHED_RUNS,
)

# This boundary validates heterogeneous serialized provenance, not tensor state.


def canonical_sha256(value: object) -> str:
    """Hash JSON without whitespace, NaN values or mapping-order dependence."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: object) -> bool:
    """Recognize a lowercase SHA256 digest without coercing other types."""
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def members_sha256(members: list[tuple[str, int]]) -> str:
    """Bind source-qualified groups independently of export traversal order."""
    if any(s not in ("train", "valid") or type(g) is not int or g < 0 for s, g in members):
        raise ValueError("reference membership has malformed source/group identities")
    if len(set(members)) != len(members):
        raise ValueError("reference membership repeats a sequence")
    return canonical_sha256(sorted(members))


def read_object(path: Path, maximum: int = 32_000_000) -> tuple[dict[str, Any], str]:
    """Read bounded non-symlink JSON and return its exact byte digest."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"invalid reference JSON path or size: {path}")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("reference JSON exceeded its size bound during read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("reference JSON must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def validate_reference_identity(
    identity: dict[str, Any], split_hash: str, fit_members: list[tuple[str, int]]
) -> None:
    """Validate a hash-bound PR49 reference without reading tensor payloads.

    The known published provenance/checkpoint hashes exclude arbitrary or all-labels
    teachers. This proves identity, not independent historical cache reconstruction.
    """
    if not isinstance(identity, dict) or not is_sha256(identity.get("manifest_sha256")):
        raise ValueError("invalid reference manifest identity")
    producer = identity.get("producer")
    if not isinstance(producer, dict) or producer.get("kind") != "pr49-tl3-reference-v1":
        raise ValueError("unsupported reference producer")
    if split_hash != PRIMARY_SPLIT_SHA256 or producer.get("split_sha256") != split_hash:
        raise ValueError("reference requires the frozen PR49 primary split")
    if len(fit_members) != 10325 or any(source != "train" for source, _ in fit_members):
        raise ValueError("reference fit membership must be the10325 source-qualified train groups")
    fitting_hash = members_sha256(fit_members)
    if producer.get("fit_members_sha256") != fitting_hash:
        raise ValueError("reference identity has different fitting membership")
    if not is_sha256(producer.get("evaluation_members_sha256")):
        raise ValueError("reference identity lacks evaluation membership")
    required = {
        "composition": COMPOSITION,
        "arithmetic": ARITHMETIC,
        "inference_source_commit": PR49_SOURCE_COMMIT,
        "training_predictions": "in-sample",
        "ancestry_limitation": ANCESTRY_LIMITATION,
    }
    if any(producer.get(key) != value for key, value in required.items()):
        raise ValueError("reference composition, source or declared ancestry changed")
    ancestors = producer.get("ancestors")
    if not isinstance(ancestors, list) or len(ancestors) != len(PUBLISHED_RUNS):
        raise ValueError("reference must bind every one of the17 published training roles")
    names = [record.get("run") for record in ancestors if isinstance(record, dict)]
    if (
        len(names) != len(ancestors)
        or len(set(names)) != len(names)
        or set(names) != set(PUBLISHED_RUNS)
    ):
        raise ValueError("reference ancestors are missing, repeated or unknown")
    for record in ancestors:
        expected = PUBLISHED_RUNS[record["run"]]
        if type(record.get("epoch")) is not int or any(
            record.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("reference ancestor differs from its published fixed endpoint")
        if (
            record.get("fit_members_sha256") != fitting_hash
            or record.get("evaluation_members_sha256") != producer["evaluation_members_sha256"]
            or not is_sha256(record.get("trainer_source_sha256"))
        ):
            raise ValueError("reference ancestor fitting/evaluation/source identity mismatch")
    if identity.get("reference_sha256") != canonical_sha256(producer):
        raise ValueError("reference composition/ancestry digest mismatch")
