"""Build a small Kaggle bootstrap backed by an exact private payload asset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKET_DIRECTORY = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().parent
R3_BUILD_RECEIPT = ASSET_ROOT / "basis/H232_V7B_EP3_EXPORT_R3_BUILD_RECEIPT.json"
R3_TITLE_ID = "H232 GRU384 V7B EP3 Export 20261008 R3"
R3_CANONICAL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r3"
R3_CODE_SHA256 = "a307809730010f04766cfa8c9f18923ad7df6809bd7c14ddc4c2745db2f4b1ee"
KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R5 Asset"
KERNEL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r5-asset"
DATASET_VERSION = 1
DATASET_SOURCES = (
    "vladkochetov07/wunder-funder",
    "vladkochetov07/wnn-c20261004-fold0-assets",
)
SOURCE_ROOT = Path(__file__).resolve().parents[6]
ROOT_REVIEW = ASSET_ROOT / "basis/H232_EP3_EXPORT_ROOT_INDEPENDENT_REVIEW_20261008_R1.json"


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file using bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object documents."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def kernel_slug(title: str) -> str:
    """Normalize the alphanumeric title tokens used by Kaggle kernel IDs."""
    return "-".join(re.findall(r"[a-z0-9]+", title.casefold()))


def validate_kernel_identity(kernel_id: str, title: str) -> None:
    """Require a kernel title whose normalized slug exactly matches its ID."""
    owner, separator, slug = kernel_id.partition("/")
    if not separator or not owner or not slug or kernel_slug(title) != slug:
        raise ValueError(f"kernel title {title!r} does not resolve to ID {kernel_id!r}")


def read_frozen_payload(path: Path, expected_sha256: str, expected_bytes: int) -> bytes:
    """Read a local copy of the private frozen ZIP and verify its identity."""
    payload = path.read_bytes()
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("private frozen payload ZIP differs from the reviewed identity")
    return payload


def render_bootstrap(
    template: str,
    *,
    dataset_id: str,
    dataset_version: int,
    asset_filename: str,
    asset_bytes: int,
    asset_sha256: str,
    payload_bytes: int,
    payload_sha256: str,
    payload_members: dict[str, str],
) -> str:
    """Bind private asset and unchanged payload identities into a tiny bootstrap."""
    replacements = {
        'ASSET_DATASET_ID = "__ASSET_DATASET_ID__"': f"ASSET_DATASET_ID = {dataset_id!r}",
        "ASSET_DATASET_VERSION = 0": f"ASSET_DATASET_VERSION = {dataset_version}",
        'ASSET_FILENAME = "__ASSET_FILENAME__"': f"ASSET_FILENAME = {asset_filename!r}",
        "ASSET_BYTES = 0": f"ASSET_BYTES = {asset_bytes}",
        'ASSET_SHA256 = "__ASSET_SHA256__"': f"ASSET_SHA256 = {asset_sha256!r}",
        "PAYLOAD_ZIP_BYTES = 0": f"PAYLOAD_ZIP_BYTES = {payload_bytes}",
        'PAYLOAD_ZIP_SHA256 = "__PAYLOAD_ZIP_SHA256__"': (
            f"PAYLOAD_ZIP_SHA256 = {payload_sha256!r}"
        ),
        "PAYLOAD_MEMBER_SHA256: dict[str, str] = {}": (
            f"PAYLOAD_MEMBER_SHA256: dict[str, str] = {payload_members!r}"
        ),
        'KERNEL_ID = "__KERNEL_ID__"': f"KERNEL_ID = {KERNEL_ID!r}",
        'KERNEL_TITLE = "__KERNEL_TITLE__"': f"KERNEL_TITLE = {KERNEL_TITLE!r}",
    }
    rendered = template
    for old, new in replacements.items():
        if rendered.count(old) != 1:
            raise ValueError(f"bootstrap template binding is not unique: {old}")
        rendered = rendered.replace(old, new)
    ast.parse(rendered)
    return rendered


def build(
    *,
    payload_path: Path,
    asset_source: Path,
    asset_upload: Path,
    kernel_packet: Path,
) -> dict[str, Any]:
    """Stage the exact reviewed payload and build its private-asset kernel packet."""
    previous = read_json(R3_BUILD_RECEIPT)
    if previous.get("kernel_id") != R3_CANONICAL_ID or previous.get("kernel_title") != R3_TITLE_ID:
        raise ValueError("R3 source receipt has an unexpected canonical identity")
    if previous.get("code_file_sha256") != R3_CODE_SHA256:
        raise ValueError("R3 source receipt differs from the root-reviewed code identity")
    root_review = read_json(ROOT_REVIEW)
    if root_review.get("checkpoint_sha256") != previous.get("checkpoint_sha256"):
        raise ValueError("root checkpoint review differs from the R3 payload identity")
    if root_review.get("source_archive_sha256") != previous.get("source_archive_sha256"):
        raise ValueError("root source review differs from the R3 payload identity")
    if any(path.exists() for path in (asset_source, asset_upload, kernel_packet)):
        raise FileExistsError("refusing to overwrite asset or kernel transport artifacts")

    payload = read_frozen_payload(
        payload_path,
        previous["payload_zip_sha256"],
        previous["payload_zip_bytes"],
    )
    asset_sha256 = sha256_bytes(payload)
    asset_filename = f"{asset_sha256}.wnn"
    asset_dataset = f"vladkochetov07/h232-v7b-ep3-payload-{asset_sha256[:12]}"
    asset_source.mkdir(parents=True, exist_ok=False)
    (asset_source / "payload.wnn").write_bytes(payload)

    sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
    from kaggle_assets import prepare

    prepare(asset_source, ["payload.wnn"], asset_upload, asset_dataset)
    asset_manifest = read_json(asset_upload / "assets-manifest.json")
    asset_path = asset_upload / asset_filename
    dataset_metadata = read_json(asset_upload / "dataset-metadata.json")
    if (
        asset_manifest.get("dataset") != asset_dataset
        or asset_manifest.get("private") is not True
        or asset_manifest.get("files")
        != [
            {
                "path": "payload.wnn",
                "size": len(payload),
                "sha256": asset_sha256,
                "filename": asset_filename,
            }
        ]
        or dataset_metadata.get("id") != asset_dataset
        or dataset_metadata.get("isPrivate") is not True
        or asset_path.stat().st_size != len(payload)
        or sha256_file(asset_path) != asset_sha256
    ):
        raise ValueError("private asset helper staged an unexpected payload identity")

    members = previous["payload_member_sha256"]
    if len(members) != 17:
        raise ValueError("reviewed R3 payload must contain exactly 17 members")
    template_path = ASSET_ROOT / "h232_export_asset_bootstrap.py"
    code = render_bootstrap(
        template_path.read_text(encoding="utf-8"),
        dataset_id=asset_dataset,
        dataset_version=DATASET_VERSION,
        asset_filename=asset_filename,
        asset_bytes=len(payload),
        asset_sha256=asset_sha256,
        payload_bytes=len(payload),
        payload_sha256=previous["payload_zip_sha256"],
        payload_members=members,
    )
    validate_kernel_identity(KERNEL_ID, KERNEL_TITLE)
    kernel_packet.mkdir(parents=True, exist_ok=False)
    code_path = kernel_packet / "kernel.py"
    code_path.write_text(code, encoding="utf-8")
    subprocess.run(["ruff", "format", str(code_path)], check=True, capture_output=True, text=True)
    metadata = {
        "code_file": code_path.name,
        "competition_sources": [],
        "dataset_sources": [*DATASET_SOURCES, f"{asset_dataset}/{DATASET_VERSION}"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": KERNEL_ID,
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": KERNEL_TITLE,
        "kernel_sources": [],
    }
    (kernel_packet / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if code_path.stat().st_size >= 1_000_000:
        raise ValueError("bootstrap source exceeds Kaggle's 1 MB source limit")
    if metadata["dataset_sources"][-1] != f"{asset_dataset}/1":
        raise ValueError("kernel metadata does not pin the exact private dataset version")
    receipt = {
        "schema": "h232-export-asset-transport-build-v1",
        "status": "prepared_pending_root_bootstrap_review_and_kernel_admission",
        "packet_id": "E529_H232_V7B_EP3_FP32_EXPORT_ONLY_20261008_R4_ASSET_R5",
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "kernel_packet_path": str(kernel_packet),
        "kernel_source_bytes": code_path.stat().st_size,
        "kernel_source_sha256": sha256_file(code_path),
        "dataset_sources": metadata["dataset_sources"],
        "asset_dataset_id": asset_dataset,
        "asset_dataset_version": DATASET_VERSION,
        "asset_upload_path": str(asset_upload),
        "asset_manifest_sha256": sha256_file(asset_upload / "assets-manifest.json"),
        "dataset_metadata_sha256": sha256_file(asset_upload / "dataset-metadata.json"),
        "asset_file": asset_filename,
        "asset_bytes": len(payload),
        "asset_sha256": asset_sha256,
        "payload_zip_sha256": previous["payload_zip_sha256"],
        "payload_zip_bytes": len(payload),
        "payload_member_sha256": members,
        "payload_member_count": len(members),
        "checkpoint_sha256": previous["checkpoint_sha256"],
        "source_archive_sha256": previous["source_archive_sha256"],
        "source_archive_manifest_sha256": previous["source_archive_manifest_sha256"],
        "r3_packet_code_sha256": previous["code_file_sha256"],
        "r3_parent_build_receipt": str(R3_BUILD_RECEIPT),
        "frozen_payload_input_path": str(payload_path),
        "worker_science_profile_and_checkpoint_changed": False,
        "fit_enabled": False,
        "profile_enabled": False,
        "scoring_enabled": False,
        "provider_kernel_push_authorized": False,
        "asset_dataset_create_authorized": True,
        "asset_dataset_version_pin": f"{asset_dataset}/{DATASET_VERSION}",
    }
    receipt_path = ASSET_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R5_BUILD_RECEIPT_R1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Build unique upload and kernel directories without provider calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload-path",
        type=Path,
        required=True,
        help="local copy of the private version-1 payload ZIP, downloaded and hash checked",
    )
    parser.add_argument("--asset-source", type=Path, default=ASSET_ROOT / "asset_source_r1")
    parser.add_argument("--asset-upload", type=Path, default=ASSET_ROOT / "asset_upload_v1")
    parser.add_argument("--kernel-packet", type=Path, default=ASSET_ROOT / "kernel_packet_r5_v1")
    args = parser.parse_args()
    receipt = build(
        payload_path=args.payload_path.resolve(),
        asset_source=args.asset_source.resolve(),
        asset_upload=args.asset_upload.resolve(),
        kernel_packet=args.kernel_packet.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
