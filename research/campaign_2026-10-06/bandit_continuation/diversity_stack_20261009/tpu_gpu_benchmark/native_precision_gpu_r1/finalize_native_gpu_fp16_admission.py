"""Build a separately sealed native FP16 packet from a root admission receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]
R5_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
R5_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
R4_KERNEL = (
    REPO_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    / "cloud_delivery/gpu_loss_pair_r4/kernel/wnn-loss-pair-gpu-20261009-v4.py"
)


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streamed SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    """Serialize a JSON object in the campaign's canonical format."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _load_kernel_builder(package: Path):
    """Load the reviewed native bootstrap builder without running its CLI."""
    path = package / "seal_native_gpu_fp16_packet.py"
    spec = importlib.util.spec_from_file_location("native_fp16_packet_sealer", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import sealed packet builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._native_kernel


def _validate_root_admission(admission: dict[str, Any], *, source_manifest_sha256: str) -> None:
    if (
        admission.get("schema") != "wnn.native-gpu-fp16.root-admission.v1"
        or admission.get("state") != "admitted"
        or admission.get("push_allowed") is not True
        or admission.get("profile_allowed") is not True
        or admission.get("source_manifest_sha256") != source_manifest_sha256
        or admission.get("r5_training_contract_sha256") != R5_CONTRACT_SHA256
        or admission.get("r5_root_review_sha256") != R5_REVIEW_SHA256
        or admission.get("compute_precision") != "fp16"
        or admission.get("amp_initial_scale") != 1024.0
        or not isinstance(admission.get("conditional_fit_allowed"), bool)
    ):
        raise ValueError("root admission is missing or mismatches a required packet pin")
    expected_training_contract = (
        R5_CONTRACT_SHA256 if admission["conditional_fit_allowed"] else None
    )
    if admission.get("training_contract_sha256") != expected_training_contract:
        raise ValueError("conditional fit authority must bind the exact R5 training contract")


def finalize(
    *,
    pending_package: Path,
    admission_path: Path,
    output_package: Path,
    r4_kernel_path: Path = R4_KERNEL,
) -> dict[str, Any]:
    """Copy a pending packet, embed root authority, and reseal its bootstrap."""
    pending_package = pending_package.resolve()
    output_package = output_package.resolve()
    if output_package.exists():
        raise FileExistsError(f"finalized output already exists: {output_package}")

    packet_path = pending_package / "native_gpu_fp16.packet.json"
    pending_packet_bytes = packet_path.read_bytes()
    packet = json.loads(pending_packet_bytes)
    if (
        packet.get("authorized_for_execution") is not False
        or packet.get("root_admission", {}).get("state") != "pending"
        or packet.get("root_admission", {}).get("push_allowed") is not False
        or packet.get("root_admission", {}).get("profile_allowed") is not False
    ):
        raise ValueError("input packet is not the immutable pending candidate")

    source_manifest_path = pending_package / "source_manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest_sha256 = sha256_bytes(source_manifest_bytes)
    if source_manifest_sha256 != packet.get("source_manifest_sha256"):
        raise ValueError("pending packet source-manifest hash is invalid")
    source_manifest = json.loads(source_manifest_bytes)
    if source_manifest.get("source_commit") != packet.get("source_commit"):
        raise ValueError("pending packet source commit differs from its source manifest")

    source_asset = pending_package / "source_asset"
    asset_manifest_path = source_asset / "assets-manifest.json"
    asset_manifest_bytes = asset_manifest_path.read_bytes()
    if sha256_bytes(asset_manifest_bytes) != packet.get("source_asset_manifest_sha256"):
        raise ValueError("pending source-asset manifest hash is invalid")
    asset_manifest = json.loads(asset_manifest_bytes)
    for row in asset_manifest.get("files", []):
        payload_path = source_asset / str(row["filename"])
        if (
            payload_path.stat().st_size != int(row["size"])
            or sha256_file(payload_path) != row["sha256"]
        ):
            raise ValueError(f"pending source payload differs from manifest: {row['path']}")
    admission_rows = [
        row
        for row in asset_manifest["files"]
        if row.get("path") == "native_controls/root_admission.json"
    ]
    if len(admission_rows) != 1:
        raise ValueError("source asset must contain exactly one root-admission payload")
    pending_admission_bytes = (source_asset / admission_rows[0]["filename"]).read_bytes()
    pending_admission = json.loads(pending_admission_bytes)
    if pending_admission.get("state") != "pending":
        raise ValueError("source asset does not preserve the pending root-admission payload")

    admission_bytes = admission_path.read_bytes()
    admission = read_object(admission_path)
    _validate_root_admission(admission, source_manifest_sha256=source_manifest_sha256)
    if admission_bytes == pending_admission_bytes:
        raise ValueError("root admission is identical to the pending payload")

    metadata_path = pending_package / "kernel" / "kernel-metadata.json"
    kernel_path = pending_package / "kernel" / str(packet["kernel"]["code_file"])
    if sha256_file(kernel_path) != packet["kernel"]["code_file_sha256"]:
        raise ValueError("pending bootstrap differs from its packet pin")
    if sha256_file(metadata_path) != packet["kernel"]["metadata_sha256"]:
        raise ValueError("pending kernel metadata differs from its packet pin")
    if not r4_kernel_path.is_file():
        raise FileNotFoundError(f"reviewed R4 bootstrap is missing: {r4_kernel_path}")

    output_package.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_asset, output_package / "source_asset")
    (output_package / "kernel").mkdir()
    shutil.copyfile(metadata_path, output_package / "kernel" / "kernel-metadata.json")
    shutil.copyfile(source_manifest_path, output_package / "source_manifest.json")

    output_asset = output_package / "source_asset"
    updated_manifest = json.loads((output_asset / "assets-manifest.json").read_text())
    root_row = next(
        row
        for row in updated_manifest["files"]
        if row.get("path") == "native_controls/root_admission.json"
    )
    old_payload = output_asset / root_row["filename"]
    opaque_name = (
        hashlib.sha256(
            (root_row["path"] + "\0" + sha256_bytes(admission_bytes)).encode()
        ).hexdigest()
        + ".wnn"
    )
    (output_asset / opaque_name).write_bytes(admission_bytes)
    old_payload.unlink()
    root_row["filename"] = opaque_name
    root_row["sha256"] = sha256_bytes(admission_bytes)
    root_row["size"] = len(admission_bytes)
    updated_asset_manifest_bytes = canonical(updated_manifest)
    (output_asset / "assets-manifest.json").write_bytes(updated_asset_manifest_bytes)

    asset_paths = [str(row["path"]) for row in updated_manifest["files"]]
    kernel_builder = _load_kernel_builder(PACKAGE)
    generated_kernel = kernel_builder(
        r4_kernel_path.read_bytes(),
        asset_manifest_sha=sha256_bytes(updated_asset_manifest_bytes),
        source_manifest_sha=source_manifest_sha256,
        asset_paths=asset_paths,
    )
    finalized_kernel_path = output_package / "kernel" / kernel_path.name
    finalized_kernel_path.write_bytes(generated_kernel)

    packet["status"] = "root_admitted_ready_for_profile"
    packet["authorized_for_execution"] = True
    packet["root_admission_required_before_asset_upload_or_kernel_push"] = False
    packet["source_asset_manifest_sha256"] = sha256_bytes(updated_asset_manifest_bytes)
    packet["kernel"]["code_file_sha256"] = sha256_bytes(generated_kernel)
    packet["root_admission"] = admission
    inventory = packet["source_asset_upload_inventory"]
    for row in inventory:
        if row.get("path") == "native_controls/root_admission.json":
            row["filename"] = opaque_name
            row["sha256"] = sha256_bytes(admission_bytes)
            row["size_bytes"] = len(admission_bytes)
        elif row.get("path") == "assets-manifest.json":
            row["sha256"] = sha256_bytes(updated_asset_manifest_bytes)
            row["size_bytes"] = len(updated_asset_manifest_bytes)
    packet.pop("packet_sha256_before_self_pin", None)
    packet["packet_sha256_before_self_pin"] = sha256_bytes(canonical(packet))
    finalized_packet_bytes = canonical(packet)
    (output_package / "native_gpu_fp16.packet.json").write_bytes(finalized_packet_bytes)

    receipt = {
        "schema": "wnn.native-gpu-fp16-root-admission-finalization.v1",
        "state": "finalized_pending_provider_action",
        "pending_packet_sha256": sha256_bytes(pending_packet_bytes),
        "final_packet_sha256": sha256_bytes(finalized_packet_bytes),
        "root_admission_sha256": sha256_bytes(admission_bytes),
        "source_commit": packet["source_commit"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_asset_manifest_sha256": sha256_bytes(updated_asset_manifest_bytes),
        "kernel_sha256": sha256_bytes(generated_kernel),
        "provider_mutation": False,
        "device_fit": False,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (output_package / "FINALIZATION_RECEIPT_R1.json").write_bytes(canonical(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pending-package", type=Path, default=PACKAGE)
    parser.add_argument("--r4-kernel", type=Path, default=R4_KERNEL)
    args = parser.parse_args()
    result = finalize(
        pending_package=args.pending_package,
        admission_path=args.admission,
        output_package=args.output_dir,
        r4_kernel_path=args.r4_kernel,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
