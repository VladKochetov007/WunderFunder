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

PACKET_ROOT = Path(__file__).resolve().parent
R5_PACKET_ROOT = PACKET_ROOT.parent / "oldz4_student_core_hft_f0_r5"
RETRY_CONTRACT_PATH = PACKET_ROOT / "CORE_CROP_RETRY_CONTRACT_R6.json"
R5_CONTRACT_PATH = R5_PACKET_ROOT / "CORE_CROP_CONTRACT_R5.json"
R5_BINDINGS_PATH = R5_PACKET_ROOT / "SOURCE_BINDINGS_R5.json"
R5_FAILURE_COPY_PATH = PACKET_ROOT / "R5_FAILED_ATTEMPT_RECEIPT_COPY_R1.json"
R5_RECEIPT_PATH = R5_PACKET_ROOT / "CORE_CROP_PRODUCER_RECEIPT_R5.json"
R6_ADMISSION_PATH = PACKET_ROOT / "ROOT_CORE_CROP_ADMISSION_R6.json"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read and validate a JSON object with heterogeneous fields."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def import_from_path(path: Path, module_name: str) -> ModuleType:
    """Import one exact frozen module from the R5 source or verifier."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 before importing frozen source")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pins(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate retry, original source bindings, and the preserved R5 failure."""
    retry = read_object(RETRY_CONTRACT_PATH)
    base = read_object(R5_CONTRACT_PATH)
    bindings = read_object(R5_BINDINGS_PATH)
    if sha256_file(R5_CONTRACT_PATH) != retry["r5_core_crop_contract_sha256"]:
        raise ValueError("the frozen R5 core-crop contract changed")
    if sha256_file(R5_BINDINGS_PATH) != retry["r5_source_bindings_sha256"]:
        raise ValueError("the frozen R5 source-binding manifest changed")
    consumer_path = PACKET_ROOT / "prepare_core_crop_r6.py"
    if sha256_file(consumer_path) != retry["consumer_source_sha256"]:
        raise ValueError("R6 consumer source differs from the retry contract")

    crop_binding = bindings["snapshots"]["core_crop"]
    crop_relative = "scripts/make_core_checkpoint.py"
    crop_sha = crop_binding["files"].get(crop_relative)
    if crop_sha != retry["crop_source_sha256"]:
        raise ValueError("R6 must use the exact crop source bound by R5")
    crop_path = R5_PACKET_ROOT / crop_binding["root_relative_path"] / crop_relative
    if sha256_file(crop_path) != crop_sha:
        raise ValueError("frozen R5 crop primitive bytes changed")

    verifier_path = R5_PACKET_ROOT / "prepare_core_crop_r5.py"
    if sha256_file(verifier_path) != retry["r5_verifier_source_sha256"]:
        raise ValueError("frozen R5 actual-output verifier changed")

    source_checkpoint = project_root / base["source_checkpoint_relative_path"]
    source_provenance = project_root / base["source_provenance_relative_path"]
    if sha256_file(source_checkpoint) != retry["source_checkpoint_sha256"]:
        raise ValueError("actual parent checkpoint differs from the R6 binding")
    if sha256_file(source_provenance) != retry["source_provenance_sha256"]:
        raise ValueError("actual parent provenance differs from the R6 binding")
    if (
        base["source_checkpoint_sha256"] != retry["source_checkpoint_sha256"]
        or base["source_provenance_sha256"] != retry["source_provenance_sha256"]
        or base["source_run"] != retry["source_run"]
        or base["source_checkpoint_epoch"] != retry["source_epoch"]
        or base["destination_run"] != retry["destination_run"]
    ):
        raise ValueError("R6 retry changed the scientific parent or crop destination")

    failure_path = project_root / retry["failed_r5_receipt_relative_path"]
    if sha256_file(failure_path) != retry["failed_r5_receipt_sha256"]:
        raise ValueError("the original failed R5 receipt is missing or changed")
    if sha256_file(R5_FAILURE_COPY_PATH) != retry["failed_r5_receipt_sha256"]:
        raise ValueError("the preserved R5 failure copy is missing or changed")
    failure = read_object(failure_path)
    if (
        failure.get("attempt") != retry["superseded_attempt"]
        or failure.get("phase") != "source_binding_validation_before_derive"
        or failure.get("error_type") != "KeyError"
        or failure.get("error") != "script_sha256"
        or failure.get("crop_destination_exists") is not False
        or failure.get("original_CP1_unchanged") is not True
        or failure.get("scientific_negative") is not False
    ):
        raise ValueError("R5 failure receipt does not establish the expected pre-write error")
    return retry, base, bindings


def _require_r6_admission(retry: dict[str, Any], retry_contract_sha: str) -> str:
    """Require a root-owned R6 release pinned to this exact corrective consumer."""
    admission = read_object(R6_ADMISSION_PATH)
    expected = {
        "schema": "wnn.oldz4-core-crop-r6-admission.v1",
        "decision": "PASS",
        "attempt": retry["attempt"],
        "retry_contract_sha256": retry_contract_sha,
        "consumer_source_sha256": retry["consumer_source_sha256"],
        "r5_core_crop_contract_sha256": retry["r5_core_crop_contract_sha256"],
        "r5_source_bindings_sha256": retry["r5_source_bindings_sha256"],
        "source_checkpoint_sha256": retry["source_checkpoint_sha256"],
        "source_provenance_sha256": retry["source_provenance_sha256"],
        "crop_source_sha256": retry["crop_source_sha256"],
        "failed_r5_receipt_sha256": retry["failed_r5_receipt_sha256"],
        "destination_run": retry["destination_run"],
    }
    for key, value in expected.items():
        if admission.get(key) != value:
            raise ValueError(f"R6 root admission differs at {key}")
    return sha256_file(R6_ADMISSION_PATH)


def run(project_root: Path) -> dict[str, Any]:
    """Run the corrected consumer on the bound CP1 and write the R5-compatible receipt."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 for the frozen source snapshots")
    project_root = project_root.resolve()
    retry, base, bindings = _load_pins(project_root)
    retry_contract_sha = sha256_file(RETRY_CONTRACT_PATH)
    admission_sha = _require_r6_admission(retry, retry_contract_sha)
    source_checkpoint = project_root / base["source_checkpoint_relative_path"]
    source_provenance = project_root / base["source_provenance_relative_path"]
    destination = project_root / "artifacts/runs" / retry["destination_run"]
    if destination.exists():
        raise FileExistsError(f"crop destination already exists: {destination}")
    if R5_RECEIPT_PATH.exists():
        raise FileExistsError(
            f"preserve existing R5-compatible producer receipt: {R5_RECEIPT_PATH}"
        )

    crop_binding = bindings["snapshots"]["core_crop"]
    crop_source_path = (
        R5_PACKET_ROOT / crop_binding["root_relative_path"] / "scripts/make_core_checkpoint.py"
    )
    crop_module = import_from_path(crop_source_path, "oldz4_core_crop_r6_primitive")
    crop_module.ROOT = project_root
    result = crop_module.derive_core(
        project_root,
        retry["source_run"],
        retry["source_epoch"],
        retry["destination_run"],
    )

    output_checkpoint = destination / f"model_ep{retry['source_epoch']}.pt"
    output_provenance = destination / "provenance.json"
    verifier_path = R5_PACKET_ROOT / "prepare_core_crop_r5.py"
    verifier = import_from_path(verifier_path, "oldz4_core_crop_r5_verifier")
    verification = verifier.verify_crop(
        source_checkpoint,
        source_provenance,
        output_checkpoint,
        output_provenance,
        source_run=retry["source_run"],
        source_epoch=retry["source_epoch"],
    )
    if result.get("checkpoint_sha256") != verification["output_checkpoint_sha256"]:
        raise ValueError("crop primitive output differs from the actual-output verifier")

    post_crop_binding = bindings["snapshots"]["core_crop"]
    post_crop_sha = post_crop_binding["files"].get("scripts/make_core_checkpoint.py")
    if (
        post_crop_sha != retry["crop_source_sha256"]
        or sha256_file(crop_source_path) != post_crop_sha
    ):
        raise ValueError("post-derive crop source differs from the frozen R5 binding")

    receipt = {
        "schema": "wnn.oldz4-core-crop-producer-receipt.v1",
        "status": "PASS_ACTUAL_CORE_CROP",
        "created_utc": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "contract_sha256": retry["r5_core_crop_contract_sha256"],
        "root_admission_sha256": admission_sha,
        "crop_source_sha256": post_crop_sha,
        "destination_run": retry["destination_run"],
        "output_checkpoint_relative_path": str(output_checkpoint.relative_to(project_root)),
        "output_provenance_relative_path": str(output_provenance.relative_to(project_root)),
        "verification": verification,
        "consumer_revision": "R6",
        "consumer_source_sha256": retry["consumer_source_sha256"],
        "retry_contract_sha256": retry_contract_sha,
        "attempt": retry["attempt"],
        "supersedes_attempt": retry["superseded_attempt"],
        "superseded_failure_receipt_sha256": retry["failed_r5_receipt_sha256"],
    }
    with R5_RECEIPT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    """Run the R6 crop successor under root's separate one-attempt admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), sort_keys=True))


if __name__ == "__main__":
    main()
