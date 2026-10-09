from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
R2 = ROOT.parent / "d02_muon_loss_successor_r5_profile_export_r2"
KERNEL_FILENAME = "wnn-d02-muon-loss-gpu-20261009-v5.py"
SOURCE_ADMISSION_PATH = ROOT / "ROOT_D02_R5_SOURCE_ADMISSION_R1.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> bytes:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return payload


def seal() -> dict[str, Any]:
    """Reseal only the bootstrap guard while reusing the reviewed R2 source asset."""
    prior_packet_path = R2 / "gpu_loss_pair.packet.json"
    prior_packet_bytes = prior_packet_path.read_bytes()
    prior_packet = json.loads(prior_packet_bytes)
    source = prior_packet["source"]
    kernel_id = prior_packet["kernel"]["metadata"]["id"]
    if kernel_id != prior_packet["packet_id"]:
        raise ValueError("R2 kernel metadata and packet IDs differ")

    asset_root = R2 / "source_asset"
    asset_manifest_path = asset_root / "assets-manifest.json"
    asset_manifest_bytes = asset_manifest_path.read_bytes()
    asset_manifest_sha256 = _sha256(asset_manifest_bytes)
    if asset_manifest_sha256 != source["asset_manifest_sha256"]:
        raise ValueError("R2 source asset manifest differs from its sealed packet")
    manifest = json.loads(asset_manifest_bytes)
    if manifest.get("dataset") != source["source_dataset"] or manifest.get("private") is not True:
        raise ValueError("R2 source asset manifest identity differs from its packet")
    entries = {row["path"]: row for row in manifest["files"]}
    if len(entries) != 5:
        raise ValueError("R2 source asset manifest must list exactly five payloads")
    for row in manifest["files"]:
        payload = asset_root / row["filename"]
        payload_bytes = payload.read_bytes()
        if len(payload_bytes) != int(row["size"]) or _sha256(payload_bytes) != row["sha256"]:
            raise ValueError(f"R2 source payload differs from its manifest: {row['path']}")
    archive_path = asset_root / entries["wnn_loss_pair_src.tar.gz"]["filename"]
    code_manifest_path = asset_root / entries["source-code-manifest.json"]["filename"]
    archive_sha256 = _sha256(archive_path.read_bytes())
    code_manifest_sha256 = _sha256(code_manifest_path.read_bytes())
    if archive_sha256 != source["archive_sha256"]:
        raise ValueError("R2 source archive differs from its packet")
    if code_manifest_sha256 != source["code_manifest_sha256"]:
        raise ValueError("R2 source code manifest differs from its packet")

    source_admission_bytes = SOURCE_ADMISSION_PATH.read_bytes()
    source_admission = json.loads(source_admission_bytes)
    admission = source_admission.get("root_admission_payload")
    expected_admission = {
        "schema": "wnn.d02.root-admission.v1",
        "kernel_id": kernel_id,
        "push_allowed": True,
        "conditional_fit_allowed": True,
        "source_code_manifest_sha256": code_manifest_sha256,
        "gpu_control_contract_sha256": prior_packet["r5_control_parent"]["training_contract"][
            "sha256"
        ],
        "control_gpu_kernel": prior_packet["control_gpu_kernel"],
        "runtime_seconds": prior_packet["kernel"]["worker_wall_seconds"],
    }
    if admission != expected_admission:
        raise ValueError("root source admission payload differs from the sealed R2 source")
    admission_bytes = _write_json(ROOT / "root_admission.json", admission)

    template = (ROOT / "kernel_template.py").read_text(encoding="utf-8")
    replacements = {
        "__ASSET_MANIFEST_SHA256__": asset_manifest_sha256,
        "__ARCHIVE_SHA256__": archive_sha256,
        "__CODE_MANIFEST_SHA256__": code_manifest_sha256,
        "__SOURCE_COMMIT__": source["source_commit"],
        "__D02_ADMISSION_B64__": base64.b64encode(admission_bytes).decode(),
    }
    for placeholder, replacement in replacements.items():
        template = template.replace(placeholder, replacement)
    if any(placeholder in template for placeholder in replacements):
        raise ValueError("kernel template contains unresolved content pins")

    kernel_dir = ROOT / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = kernel_dir / KERNEL_FILENAME
    bootstrap_bytes = template.encode()
    bootstrap_path.write_bytes(bootstrap_bytes)
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_bytes = (R2 / "kernel" / "kernel-metadata.json").read_bytes()
    metadata_path.write_bytes(metadata_bytes)

    packet = copy.deepcopy(prior_packet)
    packet["kernel"]["code_file"] = KERNEL_FILENAME
    packet["kernel"]["code_file_sha256"] = _sha256(bootstrap_bytes)
    packet["kernel"]["code_file_size_bytes"] = len(bootstrap_bytes)
    packet["kernel"]["metadata_sha256"] = _sha256(metadata_bytes)
    packet["kernel"]["metadata"] = json.loads(metadata_bytes)
    packet["root_admission_sha256"] = _sha256(admission_bytes)
    packet["bootstrap_successor"] = {
        "parent_packet_sha256": _sha256(prior_packet_bytes),
        "reason": "Correct the runtime admission kernel ID from the rejected R4 ID to the sealed R5 ID.",
        "source_asset_manifest_sha256": asset_manifest_sha256,
        "source_archive_sha256": archive_sha256,
        "code_manifest_sha256": code_manifest_sha256,
        "source_commit": source["source_commit"],
        "source_asset_reused_from": str(asset_root.relative_to(ROOT.parent.parent)),
        "science_recipe_changed": False,
    }
    packet["root_source_admission"] = {
        "path": "ROOT_D02_R5_SOURCE_ADMISSION_R1.json",
        "sha256": _sha256(source_admission_bytes),
    }
    packet_path = ROOT / "gpu_loss_pair.packet.json"
    packet_bytes = _write_json(packet_path, packet)
    return {
        "packet_sha256": _sha256(packet_bytes),
        "bootstrap_sha256": _sha256(bootstrap_bytes),
        "metadata_sha256": _sha256(metadata_bytes),
        "root_admission_sha256": _sha256(admission_bytes),
        "source_code_manifest_sha256": code_manifest_sha256,
        "source_asset_manifest_sha256": asset_manifest_sha256,
        "source_archive_sha256": archive_sha256,
        "kernel_id": kernel_id,
        "push_allowed": admission["push_allowed"],
        "conditional_fit_allowed": admission["conditional_fit_allowed"],
    }


if __name__ == "__main__":
    print(json.dumps(seal(), sort_keys=True))
