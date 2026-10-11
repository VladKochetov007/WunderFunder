"""Read-only validation for the frozen H377 profile source and inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_digest(path: Path, expected: str, label: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"{label} sha256 differs: {path}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve(strict=True)
    snapshot = args.snapshot_root.resolve(strict=True)
    if os.environ.get("WNN_PROJECT_ROOT") != str(root):
        raise ValueError("WNN_PROJECT_ROOT differs from the declared project root")
    if os.environ.get("PYTHONPATH") != str(snapshot / "src"):
        raise ValueError("PYTHONPATH differs from the frozen source package")
    if os.environ.get("WNN_PORTABLE_DATA_MANIFEST") != str(
        args.portable_manifest.resolve(strict=True)
    ):
        raise ValueError("WNN_PORTABLE_DATA_MANIFEST differs from the declared registry")

    source_manifest_path = snapshot / "SOURCE_MANIFEST.json"
    _expect_digest(source_manifest_path, args.source_manifest_sha256, "source manifest")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema") != "wnn.frozen-source-snapshot.v1" or source_manifest.get(
        "remote_root"
    ) != str(snapshot):
        raise ValueError("source manifest schema or frozen root differs")
    sources = source_manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source manifest has no source map")

    observed = {
        str(path.relative_to(snapshot))
        for base in (snapshot / "scripts", snapshot / "src/wnn")
        for path in base.rglob("*.py")
    }
    if observed != set(sources):
        raise ValueError("source snapshot Python file set differs from its manifest")
    for relative, expected in sorted(sources.items()):
        path = snapshot / relative
        if not path.is_file() or not path.resolve().is_relative_to(snapshot):
            raise ValueError(f"source file is absent or escapes the snapshot: {relative}")
        _expect_digest(path, expected, f"source {relative}")

    for name in ("artifacts", "cache", "datasets", "research"):
        alias = snapshot / name
        if not alias.is_symlink() or alias.resolve() != (root / name).resolve():
            raise ValueError(f"source project alias differs: {name}")

    _expect_digest(args.portable_manifest, args.portable_manifest_sha256, "portable registry")
    registry = json.loads(args.portable_manifest.read_text())
    if not isinstance(registry.get("sources"), dict):
        raise ValueError("portable registry has no source inventory")
    _expect_digest(args.split_manifest, args.split_manifest_sha256, "split manifest")
    return {
        "format": "wnn.h377-profile-preflight-result",
        "source_files": len(sources),
        "source_manifest_sha256": args.source_manifest_sha256,
        "portable_manifest_sha256": args.portable_manifest_sha256,
        "split_manifest_sha256": args.split_manifest_sha256,
        "registry_sources": sorted(registry["sources"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--portable-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"H377_SOURCE_INPUT_PREFLIGHT_FAIL: {exc}", file=sys.stderr)
        return 1
    print("H377_SOURCE_INPUT_PREFLIGHT_OK", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
