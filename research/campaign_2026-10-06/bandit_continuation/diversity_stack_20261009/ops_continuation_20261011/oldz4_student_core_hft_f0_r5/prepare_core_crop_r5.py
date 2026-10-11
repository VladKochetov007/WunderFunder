from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

PACKET_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKET_ROOT / "CORE_CROP_CONTRACT_R5.json"
SOURCE_BINDINGS_PATH = PACKET_ROOT / "SOURCE_BINDINGS_R5.json"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other JSON values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def import_crop_source(path: Path) -> ModuleType:
    """Import the pinned crop helper without copying or rewriting its logic."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 before importing frozen source")
    spec = importlib.util.spec_from_file_location("oldz4_core_crop_snapshot", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import crop helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_crop(
    source_checkpoint: Path,
    source_provenance: Path,
    output_checkpoint: Path,
    output_provenance: Path,
    *,
    source_run: str,
    source_epoch: int,
) -> dict[str, Any]:
    """Check that the actual crop removes only eight family output rows."""
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    output = torch.load(output_checkpoint, map_location="cpu", weights_only=False)
    source_state = source["model"]
    output_state = output["model"]
    expected_keys = set(source_state)
    if set(output_state) != expected_keys:
        raise ValueError("crop changed the serialized model tensor names")

    weight_key, bias_key = "head.2.weight", "head.2.bias"
    if tuple(source_state[weight_key].shape)[0] != 31:
        raise ValueError("source head does not have 31 rows")
    if tuple(source_state[bias_key].shape) != (31,):
        raise ValueError("source head bias does not have 31 values")
    if tuple(output_state[weight_key].shape)[0] != 23:
        raise ValueError("cropped head does not have 23 rows")
    if tuple(output_state[bias_key].shape) != (23,):
        raise ValueError("cropped head bias does not have 23 values")
    if not torch.equal(output_state[weight_key], source_state[weight_key][:23]):
        raise ValueError("cropped head weight rows differ from the original first 23")
    if not torch.equal(output_state[bias_key], source_state[bias_key][:23]):
        raise ValueError("cropped head bias differs from the original first 23")

    unchanged_tensor_count = 0
    for key in sorted(expected_keys - {weight_key, bias_key}):
        if not torch.equal(output_state[key], source_state[key]):
            raise ValueError(f"crop changed a non-family model tensor: {key}")
        unchanged_tensor_count += 1

    derivation = output.get("core_derivation")
    if not isinstance(derivation, dict):
        raise ValueError("cropped checkpoint lacks its derivation record")
    expected_source_hash = sha256_file(source_checkpoint)
    expected_provenance_hash = sha256_file(source_provenance)
    if derivation != {
        "source_run": source_run,
        "source_epoch": source_epoch,
        "source_checkpoint_sha256": expected_source_hash,
        "source_provenance_sha256": expected_provenance_hash,
        "dropped_family_rows": 8,
    }:
        raise ValueError("crop derivation record does not bind its exact parent")
    if output.get("provenance", {}).get("core_derivation") != derivation:
        raise ValueError("embedded checkpoint provenance disagrees with crop derivation")

    source_norm = source["model"]
    output_norm = output["model"]
    normalization_keys = [
        key
        for key in source_norm
        if key in {"mean", "std"} or key.endswith(".mean") or key.endswith(".std")
    ]
    for key in normalization_keys:
        if not torch.equal(source_norm[key], output_norm[key]):
            raise ValueError(f"crop changed inherited normalization tensor: {key}")

    sidecar = read_object(output_provenance)
    if sidecar.get("core_derivation") != derivation:
        raise ValueError("provenance sidecar disagrees with crop derivation")
    checkpoint_name = output_checkpoint.name
    if sidecar.get("checkpoint_sha256", {}).get(checkpoint_name) != sha256_file(output_checkpoint):
        raise ValueError("cropped sidecar checkpoint hash is stale")
    return {
        "head_rows_before": 31,
        "head_rows_after": 23,
        "dropped_family_rows": 8,
        "model_tensor_count": len(expected_keys),
        "unchanged_non_head_tensor_count": unchanged_tensor_count,
        "normalization_tensor_count": len(normalization_keys),
        "source_checkpoint_sha256": expected_source_hash,
        "source_provenance_sha256": expected_provenance_hash,
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "output_provenance_sha256": sha256_file(output_provenance),
    }


def run(project_root: Path) -> dict[str, Any]:
    """Create the write-once crop after checking exact parent identity."""
    project_root = project_root.resolve()
    contract = read_object(CONTRACT_PATH)
    bindings = read_object(SOURCE_BINDINGS_PATH)
    if sha256_file(SOURCE_BINDINGS_PATH) != contract["source_snapshot_bindings_sha256"]:
        raise ValueError("source snapshot binding manifest differs from the crop contract")
    source_bindings = bindings["snapshots"]["core_crop"]
    source_script = (
        PACKET_ROOT / source_bindings["root_relative_path"] / "scripts/make_core_checkpoint.py"
    )
    if sha256_file(source_script) != source_bindings["script_sha256"]:
        raise ValueError("frozen crop source hash mismatch")

    source_checkpoint = project_root / contract["source_checkpoint_relative_path"]
    source_provenance = project_root / contract["source_provenance_relative_path"]
    if sha256_file(source_checkpoint) != contract["source_checkpoint_sha256"]:
        raise ValueError("actual CP1 differs from the pinned parent checkpoint")
    if sha256_file(source_provenance) != contract["source_provenance_sha256"]:
        raise ValueError("actual CP1 provenance differs from the pinned parent")
    admission_path = PACKET_ROOT / "ROOT_CORE_CROP_ADMISSION_R5.json"
    if not admission_path.is_file():
        raise FileNotFoundError("root-owned core crop execution admission is missing")
    admission = read_object(admission_path)
    if (
        admission.get("schema") != "wnn.oldz4-core-crop-admission.v1"
        or admission.get("decision") != "PASS"
        or admission.get("core_crop_contract_sha256") != sha256_file(CONTRACT_PATH)
        or admission.get("source_checkpoint_sha256") != contract["source_checkpoint_sha256"]
    ):
        raise ValueError("root crop admission does not release this exact CP1 crop")
    destination = project_root / "artifacts/runs" / contract["destination_run"]
    if destination.exists():
        raise FileExistsError(f"crop destination already exists: {destination}")

    module = import_crop_source(source_script)
    module.ROOT = project_root
    result = module.derive_core(
        project_root,
        contract["source_run"],
        contract["source_checkpoint_epoch"],
        contract["destination_run"],
    )
    output_checkpoint = destination / f"model_ep{contract['source_checkpoint_epoch']}.pt"
    output_provenance = destination / "provenance.json"
    verified = verify_crop(
        source_checkpoint,
        source_provenance,
        output_checkpoint,
        output_provenance,
        source_run=contract["source_run"],
        source_epoch=contract["source_checkpoint_epoch"],
    )
    if result.get("checkpoint_sha256") != verified["output_checkpoint_sha256"]:
        raise ValueError("crop primitive result hash differs from the verified output")

    receipt_path = PACKET_ROOT / "CORE_CROP_PRODUCER_RECEIPT_R5.json"
    receipt = {
        "schema": "wnn.oldz4-core-crop-producer-receipt.v1",
        "status": "PASS_ACTUAL_CORE_CROP",
        "created_utc": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "root_admission_sha256": sha256_file(admission_path),
        "crop_source_sha256": source_bindings["script_sha256"],
        "destination_run": contract["destination_run"],
        "output_checkpoint_relative_path": str(output_checkpoint.relative_to(project_root)),
        "output_provenance_relative_path": str(output_provenance.relative_to(project_root)),
        "verification": verified,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    """Run the CPU-only exact CP1 crop and emit its bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), sort_keys=True))


if __name__ == "__main__":
    main()
