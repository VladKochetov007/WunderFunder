from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# JSON and torch checkpoint metadata are heterogeneous; every boundary is checked below.
JsonObject = dict[str, Any]


def object_value(value: object, label: str) -> JsonObject:
    """Require a string-keyed object at an external metadata boundary."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def sha256(path: Path) -> str:
    """Hash a file incrementally without executing its content."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    """Read bounded JSON metadata, rejecting non-object roots."""
    if path.stat().st_size > 8_000_000:
        raise ValueError(f"metadata too large: {path}")
    return object_value(json.loads(path.read_text(encoding="utf-8")), str(path))


def write_json(path: Path, value: JsonObject) -> None:
    """Create a new JSON artifact, never replacing an existing file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_identity(path: Path) -> JsonObject:
    """Pin a regular, nonsymlink artifact to its absolute path and byte digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular nonsymlink file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def pinned_path(value: object, label: str) -> Path:
    """Reject absent placeholders or changed files before any checkpoint loading."""
    identity = object_value(value, label)
    digest = identity.get("sha256")
    raw_path = identity.get("path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label}: a concrete lowercase SHA256 is required")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label}: an absolute artifact path is required")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{label}: artifact is missing, a symlink, or has changed")
    if "bytes" in identity and identity["bytes"] != path.stat().st_size:
        raise ValueError(f"{label}: byte size differs")
    return path
