#!/usr/bin/env python3
"""Pin graph-regeneration source files and verify the raw-input preflight receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GRAPH_SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
GRAPH_SOURCE_FILES = {
    "scripts/sequence_chains.py": "09bf23adc40d948bf5459ceca1e6da9825eea35f6172918ff68e4f3cb2ddbfb1",
    "src/wnn/data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
}
RAW_PARQUET_IDENTITIES = {
    "train": (29_240_219_011, "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81"),
    "valid": (5_615_509_128, "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"),
}


def validate_graph_source(source_root: Path, executed_script: Path) -> dict[str, object]:
    """Check the exact a266 graph generator and feature-column source before derivation."""
    files = {}
    paths = {
        "pinned_script": source_root / "scripts/sequence_chains.py",
        "executed_script": executed_script,
        "feature_source": source_root / "src/wnn/data.py",
    }
    for name, path in paths.items():
        relative = "src/wnn/data.py" if name == "feature_source" else "scripts/sequence_chains.py"
        expected = GRAPH_SOURCE_FILES[relative]
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned graph source mismatch: {name} ({path})")
        files[name] = {
            "path": str(path.resolve()),
            "sha256_lf_normalized": digest,
            "bytes": len(payload),
        }
    return {"revision": GRAPH_SOURCE_REVISION, "files": files}


def validate_input_preflight(path: Path) -> dict[str, object]:
    """Require exact whole-file input hashes and the valid-mask hash before fold derivation."""
    payload = path.read_bytes()
    receipt = json.loads(payload)
    if receipt.get("preflight_status") != "verified_r4_raw_parquets":
        raise ValueError("input preflight does not verify the frozen R4 raw Parquets")
    if receipt.get("manifest_generation_allowed") is not True:
        raise ValueError("input preflight does not authorize manifest generation")
    for split in ("train", "valid"):
        source = receipt.get(split, {}).get("parquet", {})
        expected_bytes, expected_sha = RAW_PARQUET_IDENTITIES[split]
        if (
            source.get("bytes") != expected_bytes
            or source.get("sha256") != expected_sha
            or source.get("expected_r4_sha256") != expected_sha
        ):
            raise ValueError(f"input preflight {split} raw Parquet hash mismatch")
    mask_source = receipt.get("valid", {}).get("valid_mask_parquet")
    if not isinstance(mask_source, dict) or len(mask_source.get("sha256", "")) != 64:
        raise ValueError("input preflight lacks the whole-file valid-mask Parquet identity")
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
