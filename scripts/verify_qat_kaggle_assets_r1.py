from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the bounded pending asset manifest before validating staged files."""
    if path.stat().st_size > 2_000_000:
        raise ValueError("pending asset manifest exceeds 2 MB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("private") is not True or not isinstance(value.get("files"), list):
        raise ValueError("manifest is not a private file inventory")
    return value


def sha256_file(path: Path) -> str:
    """Stream an immutable asset using an 8 MiB block and constant memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(manifest: dict[str, Any], asset_dir: Path, data_root: Path) -> dict[str, Any]:
    """Check source/stage hardlink identity and hash every future upload byte."""
    observed: list[dict[str, Any]] = []
    for item in manifest["files"]:
        logical = str(item["path"])
        source = data_root / logical
        staged = asset_dir / str(item["filename"])
        source_stat, staged_stat = source.stat(), staged.stat()
        if source_stat.st_size != item["size"] or staged_stat.st_size != item["size"]:
            raise ValueError(f"staged length differs from source manifest: {logical}")
        if (source_stat.st_dev, source_stat.st_ino) != (staged_stat.st_dev, staged_stat.st_ino):
            raise ValueError(f"staged file is not the original immutable hardlink: {logical}")
        started = time.monotonic()
        digest = sha256_file(staged)
        observation = {
            "path": logical,
            "filename": staged.name,
            "bytes": staged_stat.st_size,
            "expected_sha256": item["sha256"],
            "observed_sha256": digest,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": "match" if digest == item["sha256"] else "mismatch",
        }
        observed.append(observation)
        if digest != item["sha256"]:
            raise ValueError(f"staged payload differs from trusted identity: {logical}")
    return {
        "schema": "qat-kaggle-staged-asset-verification-r1",
        "status": "pass",
        "dataset": manifest["dataset"],
        "files_verified": len(observed),
        "bytes_verified": sum(item["bytes"] for item in observed),
        "block_bytes": 8 * 1024**2,
        "hardlinks_preserved": True,
        "files": observed,
    }


def write_json(path: Path, value: object) -> None:
    """Write a finite, atomic verification receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Verify every future upload file and optionally create Kaggle's final manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    try:
        receipt = verify_assets(manifest, args.asset_dir, args.data_root)
    except Exception as error:
        receipt = {
            "schema": "qat-kaggle-staged-asset-verification-r1",
            "status": "fail",
            "failure": f"{type(error).__name__}: {error}",
        }
        write_json(args.receipt, receipt)
        raise
    write_json(args.receipt, receipt)
    if args.promote:
        final_manifest = {
            "version": 1,
            "dataset": manifest["dataset"],
            "private": True,
            "files": manifest["files"],
        }
        write_json(args.asset_dir / "assets-manifest.json", final_manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
