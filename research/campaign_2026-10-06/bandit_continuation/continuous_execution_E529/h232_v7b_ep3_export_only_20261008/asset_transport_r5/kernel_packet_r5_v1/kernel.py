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

ASSET_DATASET_ID = "vladkochetov07/h232-v7b-ep3-payload-b9e14e8509df"
ASSET_DATASET_VERSION = 1
ASSET_FILENAME = "b9e14e8509dff6abc18f4e3aaffe4af031d78883e91fb98f28fb4a845ed1490e.wnn"
ASSET_BYTES = 8321018
ASSET_SHA256 = "b9e14e8509dff6abc18f4e3aaffe4af031d78883e91fb98f28fb4a845ed1490e"
PAYLOAD_ZIP_BYTES = 8321018
PAYLOAD_ZIP_SHA256 = "b9e14e8509dff6abc18f4e3aaffe4af031d78883e91fb98f28fb4a845ed1490e"
PAYLOAD_MEMBER_SHA256: dict[str, str] = {
    "H232_AMP4096_SUCCESSOR_RECIPE.json": "d5f418cfd4dca2501e8d95f53b7615e7cc2663ad96fd5cdd1db4a0d06e957601",
    "H232_PROFILE_PLAN.json": "3b20466afd1762606d02532f10d6d576e1a88fa33a087265ae532fbb76d34952",
    "cache_pressure_recovery.py": "ef413deb589fefec411a76b9e74a1557e3316449b81887a517844c4b2e73db2a",
    "checkpoint_source_review.json": "de67fb148436b055d4d63269d2849cf429671e986abffbc0e2ecb345e0c0cdbc",
    "export_worker.py": "e5568c871f6c918330c424dafc03ece7018e2059d019c6f87ac72352fdf82201",
    "external_review.json": "3d00f9cf9796c1511625af368fb59a3c4574a505ca9bdeaa88e4fe14a32e5949",
    "gpu_memory_domains.py": "285b42f24630cea7b033d97f9fd3b70236eca508f1fc8bad38fc291259260042",
    "membership_audit.py": "3f179f00b496ff177f609d9f9f6af3a665af781fb83fd120d70b66fdd3a3dc14",
    "model_ep3.pt": "aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829",
    "packet_meta.json": "bd89d56514f5a18b2077cec738fc8a9d15f92b86dda6e77802305f7952e63378",
    "prior_input_identity_v2.json": "cbea42868303968f7ee84105990a25848ce17add175b7161e116c80dfa63e46e",
    "profile_worker.py": "6943a0e688af3fc320bc9809a1567e37ed576acc7f8cc92f57cd09391918c5c3",
    "provenance.json": "28372b5b9ed20b619073e1e2c7e21ac4d25c45867aa9a01b4f10bdc993fb1526",
    "source_archive.tar": "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4",
    "source_archive_manifest.json": "6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42",
    "terminal_receipt.json": "4ba1fc63e8119951b9a625f93f701a88dbac69a5b92d16c57be84d6def9c5434",
    "verified_digest_handoff.py": "c351f88ab9f040867da8e30f767133b8017cc8d15aefdf4c8af65040fb51688f",
}
KERNEL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r5-asset"
KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R5 Asset"
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
