from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_source_package(
    source_root: Path, manifest: dict, *, manifest_sha256: str
) -> dict:
    """Verify every deployed LF source byte, including extra-file rejection."""
    if manifest.get("schema") != "tabdata-deployed-subset-manifest.v1":
        raise ValueError("unsupported source package")
    expected = {k: v["deployed_sha256"] for k, v in manifest["files"].items()}
    if manifest["file_count"] != len(expected):
        raise ValueError("source package file count mismatch")
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe source package path")
    actual = {}
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source package symlink")
        if path.is_file():
            actual[path.relative_to(source_root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if actual != expected:
        raise ValueError("deployed source package bytes or membership differ")
    identity = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != manifest["deployed_subset_identity_sha256"]:
        raise ValueError("source package aggregate mismatch")
    return {
        "schema": manifest["schema"],
        "source_revision": manifest["source_revision"],
        "manifest_sha256": manifest_sha256,
        "archive_sha256": manifest["archive_sha256"],
        "deployed_tree_identity_sha256": identity,
        "file_count": len(actual),
        "whole_upstream_tree": False,
        "normalization": manifest["normalization"],
    }
