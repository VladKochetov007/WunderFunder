"""Launch the reviewed H232 export worker from a pinned private Kaggle asset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

# Keep the original one-GPU visibility and four-thread cap before worker imports.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

ASSET_DATASET_ID = "__ASSET_DATASET_ID__"
ASSET_DATASET_VERSION = 0
ASSET_FILENAME = "__ASSET_FILENAME__"
ASSET_BYTES = 0
ASSET_SHA256 = "__ASSET_SHA256__"
PAYLOAD_ZIP_BYTES = 0
PAYLOAD_ZIP_SHA256 = "__PAYLOAD_ZIP_SHA256__"
PAYLOAD_MEMBER_SHA256: dict[str, str] = {}
KERNEL_ID = "__KERNEL_ID__"
KERNEL_TITLE = "__KERNEL_TITLE__"
KERNEL_WALL_SECONDS_MAX = 6_000
WORKER_WALL_SECONDS_MAX = 4_500
BOOTSTRAP_STARTED_MONOTONIC = time.monotonic()


def sha256_file(path: Path) -> str:
    """Hash a file using bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_verified_asset(
    input_root: Path,
    dataset_id: str,
    filename: str,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    """Find one nested mounted asset and verify its manifest and bytes."""
    if dataset_id.count("/") != 1 or Path(filename).name != filename:
        raise ValueError("asset identity is malformed")
    slug = dataset_id.split("/")[1]
    matches = list(input_root.glob(f"**/{slug}/{filename}"))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"expected exactly one mounted payload for {dataset_id}")
    asset = matches[0]
    manifest_path = asset.parent / "assets-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if (
        manifest.get("version") != 1
        or manifest.get("private") is not True
        or manifest.get("dataset") != dataset_id
        or not isinstance(files, list)
        or len(files) != 1
    ):
        raise ValueError("mounted private asset manifest differs from the pinned contract")
    entry = files[0]
    if (
        entry.get("filename") != filename
        or entry.get("size") != expected_size
        or entry.get("sha256") != expected_sha256
    ):
        raise ValueError("mounted private asset manifest has unexpected file identity")
    if asset.stat().st_size != expected_size or sha256_file(asset) != expected_sha256:
        raise ValueError("mounted private asset bytes differ from the pinned identity")
    return asset


def extract_verified_payload(
    archive_path: Path,
    destination: Path,
    expected_archive_bytes: int,
    expected_archive_sha256: str,
    expected_members: dict[str, str],
) -> dict[str, str]:
    """Extract only declared payload members after archive and member hash checks."""
    if (
        archive_path.stat().st_size != expected_archive_bytes
        or sha256_file(archive_path) != expected_archive_sha256
    ):
        raise ValueError("H232 payload archive bytes differ from the reviewed identity")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite payload extraction directory: {destination}")
    observed: dict[str, str] = {}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if len(names) != len(set(names)) or set(names) != set(expected_members):
                raise ValueError("H232 payload archive has duplicate or unexpected members")
            for item in members:
                relative = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in item.filename
                    or item.is_dir()
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError("H232 payload archive contains an unsafe member path")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(item, "r") as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)
                        digest.update(block)
                observed[item.filename] = digest.hexdigest()
                if observed[item.filename] != expected_members[item.filename]:
                    raise ValueError(f"H232 payload member hash mismatch: {item.filename}")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return observed


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Persist a small runtime transport receipt before launching the worker."""
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    """Verify the private asset, then invoke the exact reviewed export worker."""
    receipt_path = Path("/kaggle/working/h232_export_asset_transport_receipt_r5.json")
    receipt: dict[str, Any] = {
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "asset_dataset_id": ASSET_DATASET_ID,
        "asset_dataset_version": ASSET_DATASET_VERSION,
        "asset_filename": ASSET_FILENAME,
        "expected_asset_bytes": ASSET_BYTES,
        "expected_asset_sha256": ASSET_SHA256,
        "expected_payload_zip_bytes": PAYLOAD_ZIP_BYTES,
        "expected_payload_zip_sha256": PAYLOAD_ZIP_SHA256,
        "status": "verifying_private_asset",
    }
    write_receipt(receipt_path, receipt)
    destination = Path("/kaggle/working/h232_v7b_ep3_verified_payload_r5")
    try:
        asset = find_verified_asset(
            Path("/kaggle/input"),
            ASSET_DATASET_ID,
            ASSET_FILENAME,
            ASSET_BYTES,
            ASSET_SHA256,
        )
        observed = extract_verified_payload(
            asset,
            destination,
            PAYLOAD_ZIP_BYTES,
            PAYLOAD_ZIP_SHA256,
            PAYLOAD_MEMBER_SHA256,
        )
    except Exception as error:
        receipt.update({"status": "asset_or_payload_verification_failed", "error": str(error)})
        write_receipt(receipt_path, receipt)
        raise
    receipt.update(
        {
            "status": "payload_verified",
            "observed_asset_bytes": asset.stat().st_size,
            "observed_asset_sha256": sha256_file(asset),
            "observed_payload_member_sha256": observed,
            "extracted_payload_path": str(destination),
            "kernel_elapsed_seconds_before_worker": time.monotonic() - BOOTSTRAP_STARTED_MONOTONIC,
        }
    )
    write_receipt(receipt_path, receipt)
    elapsed = time.monotonic() - BOOTSTRAP_STARTED_MONOTONIC
    if elapsed + WORKER_WALL_SECONDS_MAX > KERNEL_WALL_SECONDS_MAX:
        raise TimeoutError("verified payload staging leaves insufficient worker time")
    sys.path.insert(0, str(destination))
    from export_worker import main as run_export

    run_export(destination)


if __name__ == "__main__":
    main()
