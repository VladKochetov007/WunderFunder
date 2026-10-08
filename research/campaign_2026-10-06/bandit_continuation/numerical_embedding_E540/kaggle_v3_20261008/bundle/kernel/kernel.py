from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = 'e540-v3-warm-20261008'
EXPECTED_ARCHIVE_SHA256 = '399401077888caa5547cf4398b5fee409d00f14a369f1298d7d7eb2e1580e673'
EXPECTED_EXTERNAL_MOUNTS = {'datasets/train.parquet': 'train.parquet', 'datasets/valid.parquet': 'valid.parquet', 'cache/train_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn'}, 'cache/valid_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn'}}


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def main() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest = json.loads(matches[0].read_text())
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    archive = matches[0].parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != manifest["archive_sha256"]
        or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("Private bundle archive identity mismatch")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Archive path escapes experiment directory")
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unsupported, undeclared or duplicate archive member")
            seen.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no file content")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024**2)
    for logical, mapping in manifest["external_mounts"].items():
        path = PurePosixPath(logical)
        if path.is_absolute() or ".." in path.parts or logical not in allowed:
            raise ValueError("Unsafe external input mapping")
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/")[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(candidates[0])
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        "/kaggle/working/results",
    ]
    for name in manifest["spec_paths"]:
        command.extend(["--spec", str(root / name)])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run(command, check=True)
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()
