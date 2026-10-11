from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

PACKET_ROOT = Path(__file__).resolve().parent
R5_PACKET_ROOT = PACKET_ROOT.parent / "oldz4_student_core_hft_f0_r5"
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from the frozen consumer packet."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def import_from_path(path: Path, module_name: str) -> ModuleType:
    """Import one copied consumer so all resolved packet paths are temporary."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_r5_inputs(project_root: Path, r5_packet: Path, retry: dict[str, Any]) -> None:
    """Copy only the R5 files consumed by R6 and its downstream receipt reader."""
    for filename in (
        "CORE_CROP_CONTRACT_R5.json",
        "SOURCE_BINDINGS_R5.json",
        "prepare_core_crop_r5.py",
        "run_hft_stage_r5.py",
    ):
        shutil.copy2(R5_PACKET_ROOT / filename, r5_packet / filename)
    source_binding = read_object(R5_PACKET_ROOT / "SOURCE_BINDINGS_R5.json")["snapshots"][
        "core_crop"
    ]
    primitive_relative = (
        Path(source_binding["root_relative_path"]) / "scripts/make_core_checkpoint.py"
    )
    primitive = R5_PACKET_ROOT / primitive_relative
    target = r5_packet / primitive_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primitive, target)

    failure_relative = Path(retry["failed_r5_receipt_relative_path"])
    failure_source = PROJECT_ROOT / failure_relative
    failure_target = project_root / failure_relative
    failure_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(failure_source, failure_target)

    for relative_key in (
        "source_checkpoint_relative_path",
        "source_provenance_relative_path",
    ):
        relative_path = Path(retry[relative_key])
        source = PROJECT_ROOT / relative_path
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_synthetic_admission(r6_packet: Path, retry: dict[str, Any]) -> str:
    """Create a test-only release that binds this exact isolated fixture."""
    contract_path = r6_packet / "CORE_CROP_RETRY_CONTRACT_R6.json"
    consumer_path = r6_packet / "prepare_core_crop_r6.py"
    admission = {
        "schema": "wnn.oldz4-core-crop-r6-admission.v1",
        "decision": "PASS",
        "attempt": retry["attempt"],
        "retry_contract_sha256": sha256_file(contract_path),
        "consumer_source_sha256": sha256_file(consumer_path),
        "r5_core_crop_contract_sha256": retry["r5_core_crop_contract_sha256"],
        "r5_source_bindings_sha256": retry["r5_source_bindings_sha256"],
        "source_checkpoint_sha256": retry["source_checkpoint_sha256"],
        "source_provenance_sha256": retry["source_provenance_sha256"],
        "crop_source_sha256": retry["crop_source_sha256"],
        "failed_r5_receipt_sha256": retry["failed_r5_receipt_sha256"],
        "destination_run": retry["destination_run"],
        "fixture_only": True,
    }
    path = r6_packet / "ROOT_CORE_CROP_ADMISSION_R6.json"
    path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def run() -> dict[str, Any]:
    """Exercise R6 preflight, actual CP1 derivation, and the unchanged R5 reader."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    retry = read_object(PACKET_ROOT / "CORE_CROP_RETRY_CONTRACT_R6.json")
    with tempfile.TemporaryDirectory(prefix="oldz4-core-crop-r6-actual-") as temp:
        project_root = Path(temp)
        ops_root = project_root / (
            "research/campaign_2026-10-06/bandit_continuation/"
            "diversity_stack_20261009/ops_continuation_20261011"
        )
        r5_packet = ops_root / "oldz4_student_core_hft_f0_r5"
        r6_packet = ops_root / "oldz4_student_core_hft_f0_r6"
        r5_packet.mkdir(parents=True)
        shutil.copytree(PACKET_ROOT, r6_packet)
        copy_r5_inputs(project_root, r5_packet, retry)
        admission_sha = write_synthetic_admission(r6_packet, retry)

        runner = import_from_path(
            r6_packet / "prepare_core_crop_r6.py", "oldz4_core_crop_r6_fixture"
        )
        producer_receipt = runner.run(project_root)
        hft = import_from_path(
            r5_packet / "run_hft_stage_r5.py", "oldz4_hft_r5_receipt_consumer_fixture"
        )
        downstream_receipt, downstream_receipt_sha = hft._load_crop_receipt(project_root)
        if downstream_receipt != producer_receipt:
            raise ValueError("unchanged R5 HFT reader returned a different crop receipt")
        if downstream_receipt["root_admission_sha256"] != admission_sha:
            raise ValueError("R6 producer receipt does not bind the fixture admission")

        return {
            "schema": "wnn.oldz4-core-crop-r6-actual-producer-fixture.v1",
            "status": "PASS_FULL_CONSUMER_AND_R5_DOWNSTREAM_READER",
            "fixture_only_admission": True,
            "actual_parent_checkpoint_sha256": retry["source_checkpoint_sha256"],
            "actual_parent_provenance_sha256": retry["source_provenance_sha256"],
            "r5_contract_sha256": retry["r5_core_crop_contract_sha256"],
            "r5_source_bindings_sha256": retry["r5_source_bindings_sha256"],
            "r6_consumer_sha256": sha256_file(PACKET_ROOT / "prepare_core_crop_r6.py"),
            "retry_contract_sha256": sha256_file(PACKET_ROOT / "CORE_CROP_RETRY_CONTRACT_R6.json"),
            "output_verification": producer_receipt["verification"],
            "r6_attempt": producer_receipt["attempt"],
            "superseded_r5_failure_sha256": producer_receipt["superseded_failure_receipt_sha256"],
            "downstream_r5_reader_status": "PASS",
            "downstream_receipt_sha256": downstream_receipt_sha,
            "temporary_project_root_removed": True,
        }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
