"""Verify the root-sealed flat successor controller bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from frozen_panel import sha256_file


def bundle_identity(directory: Path) -> tuple[str, dict[str, str]]:
    """Hash every regular Python file immediately in the controller directory."""
    files: dict[str, str] = {}
    for path in sorted(directory.glob("*.py"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"controller source is not a regular file: {path.name}")
        files[path.name] = sha256_file(path)
    if not files:
        raise ValueError("successor control bundle is empty")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), files


def verify_control_manifest(
    directory: Path, manifest_path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Require the complete source file set and canonical digest sealed by root."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest, actual = bundle_identity(directory)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "tabdata-r01-density4-control-bundle.v1"
        or manifest.get("files") != actual
        or manifest.get("sha256") != digest
        or digest != expected_sha256
    ):
        raise ValueError("successor control source differs from the root-sealed bundle")
    return manifest
