"""Verify Kaggle raw mounts against source-qualified cache metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wnn.splits import file_sha256

from .age_readout_portable_builder import verify_parquet_source

JsonObject = dict[str, Any]


def verify_raw_sources(
    repo_root: Path,
    source_fold: JsonObject,
    sources: dict[str, Path],
    original_sources: JsonObject,
) -> JsonObject:
    """Check Parquet size/footer while preserving original and actual identities."""
    split_path = Path(source_fold["split_manifest"])
    split_path = split_path if split_path.is_absolute() else repo_root / split_path
    if file_sha256(split_path) != source_fold["split_manifest_sha256"]:
        raise ValueError("frozen campaign split SHA changed")
    campaign = _json(split_path)
    identities: JsonObject = {}
    for split, source_path in sources.items():
        identity = campaign["sources"][f"datasets/{split}.parquet"]
        original = original_sources[split]
        if original.get("campaign_identity") != identity:
            raise ValueError(f"{split} cache identity differs from the frozen campaign source")
        verify_parquet_source(source_path, identity)
        stat = source_path.stat()
        identities[split] = {
            "logical_path": f"datasets/{split}.parquet",
            "original_path": original["path"],
            "original_mtime_ns": original["mtime_ns"],
            "actual_path": str(source_path),
            "actual_mtime_ns": stat.st_mtime_ns,
            "bytes": stat.st_size,
            "footer_sha256": identity["footer_sha256"],
        }
    return identities


def _json(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
