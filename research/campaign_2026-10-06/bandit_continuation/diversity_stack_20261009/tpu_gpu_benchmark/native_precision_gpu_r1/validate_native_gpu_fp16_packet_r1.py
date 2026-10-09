"""Run offline producer-consumer checks on the generated native FP16 packet."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
KERNEL_DIR = PACKAGE / "kernel"
ASSET_DIR = PACKAGE / "source_asset"
PACKET_PATH = PACKAGE / "native_gpu_fp16.packet.json"


def sha256_file(path: Path) -> str:
    """Return a streamed SHA-256 digest for a regular packet file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from the sealed packet directory."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import generated module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate() -> dict[str, Any]:
    """Validate packet pins, actual overlay extraction, and pending admission."""
    packet = read_json(PACKET_PATH)
    metadata = read_json(KERNEL_DIR / "kernel-metadata.json")
    kernel_path = KERNEL_DIR / str(packet["kernel"]["code_file"])
    source_text = kernel_path.read_text(encoding="utf-8")
    compile(source_text, str(kernel_path), "exec")
    if (
        packet.get("authorized_for_execution") is not False
        or packet.get("root_admission_required_before_asset_upload_or_kernel_push") is not True
        or packet.get("root_admission", {}).get("state") != "pending"
        or packet.get("root_admission", {}).get("push_allowed") is not False
        or packet.get("root_admission", {}).get("profile_allowed") is not False
        or packet.get("root_admission", {}).get("conditional_fit_allowed") is not False
    ):
        raise ValueError("packet does not remain pending root admission")
    if packet["kernel"]["code_file_sha256"] != sha256_file(kernel_path):
        raise ValueError("generated kernel hash differs from packet pin")
    if metadata["id"] != packet["packet_id"] or metadata["code_file"] != kernel_path.name:
        raise ValueError("kernel metadata does not bind the packet ID and code file")
    if sha256_file(KERNEL_DIR / "kernel-metadata.json") != packet["kernel"]["metadata_sha256"]:
        raise ValueError("kernel metadata hash differs from packet pin")

    asset_manifest_path = ASSET_DIR / "assets-manifest.json"
    if sha256_file(asset_manifest_path) != packet["source_asset_manifest_sha256"]:
        raise ValueError("source asset manifest hash differs from packet pin")
    asset_manifest = read_json(asset_manifest_path)
    if asset_manifest.get("dataset") != packet["source_asset_dataset"]:
        raise ValueError("source asset manifest is bound to a different dataset ID")
    for row in asset_manifest.get("files", []):
        payload = ASSET_DIR / str(row["filename"])
        if payload.stat().st_size != row["size"] or sha256_file(payload) != row["sha256"]:
            raise ValueError(f"sealed source payload differs from manifest: {row['path']}")

    with tempfile.TemporaryDirectory(prefix="native-fp16-bootstrap-", dir=PACKAGE) as temp:
        temporary_root = Path(temp)
        kernel = _load_module(kernel_path, "native_fp16_packet_bootstrap_validation")
        kernel.INPUT = ASSET_DIR
        kernel.REPO = temporary_root / "wnn"
        native_root, discovered_manifest = kernel.discover_native_asset_manifest()
        if discovered_manifest != asset_manifest:
            raise ValueError("bootstrap discovers a different native asset manifest")
        source_manifest = kernel.unpack_native_overlay(native_root, discovered_manifest)
        extracted_source_manifest = kernel.REPO / "native_controls" / "source_manifest.json"
        if sha256_file(extracted_source_manifest) != packet["source_manifest_sha256"]:
            raise ValueError("bootstrap extraction differs from the packet source-manifest pin")
        if source_manifest.get("source_commit") != packet["source_commit"]:
            raise ValueError("runtime source manifest commit differs from the packet")

        scripts = kernel.REPO / "scripts"
        sys.path.insert(0, str(scripts))
        worker = _load_module(
            scripts / "native_gpu_fp16_worker.py", "native_fp16_packet_worker_validation"
        )
        bootstrap = {
            "native_root_admission_path": str(
                kernel.REPO / "native_controls" / "root_admission.json"
            ),
            "native_controls_root": str(kernel.REPO / "native_controls"),
            "native_source_manifest_sha256": packet["source_manifest_sha256"],
        }
        try:
            worker._validate_admission(bootstrap)
        except worker.ContractError:
            pending_admission_rejected = True
        else:
            raise ValueError("runtime accepted the sealed pending admission")
        finally:
            sys.path.remove(str(scripts))

    return {
        "schema": "wnn.native-gpu-fp16-offline-bootstrap-validation.r1",
        "state": "passed",
        "packet_id": packet["packet_id"],
        "packet_source_commit": packet["source_commit"],
        "kernel_sha256": sha256_file(kernel_path),
        "asset_manifest_sha256": sha256_file(asset_manifest_path),
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "native_payload_count": len(asset_manifest["files"]),
        "bootstrap_extraction_verified": True,
        "pending_admission_rejected_by_actual_worker": pending_admission_rejected,
        "provider_mutation": False,
        "device_fit": False,
    }


def main() -> int:
    receipt = validate()
    receipt_path = PACKAGE / "NATIVE_BOOTSTRAP_VALIDATION_R1.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
