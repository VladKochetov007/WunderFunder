#!/usr/bin/env python3
"""Seal the callback-compatible R2 contract from the immutable R1 producers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
R1_PACKET = PACKET.parent / "native_shared_cleanfold_q8_r1"
WEIGHTS = Path("/tmp/wnn_shared_cleanfold_q8_r3/weights.npz")
WEIGHTS_RECEIPT = Path("/tmp/wnn_shared_cleanfold_q8_r3/WEIGHTS_RECEIPT_R1.json")


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict[str, str]:
    """Return an absolute path and the exact current file digest."""
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, rejecting non-object roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_contract() -> dict[str, Any]:
    """Bind R1 producer bytes and the corrected R2 callback consumer."""
    r1_contract_path = R1_PACKET / "CONTRACT_R1.json"
    r1_validate_path = R1_PACKET / "VALIDATE_PACKET_R1.json"
    r1_consumer_path = R1_PACKET / "NATIVE_CONSUMER_PREFLIGHT_R1.json"
    r1_contract = read_json(r1_contract_path)
    r1_validate = read_json(r1_validate_path)
    r1_preflight = read_json(r1_consumer_path)
    contract_sha = sha256_file(r1_contract_path)
    if contract_sha != "b1e69c6b8184fec4843c443783492fbdb2d7becb8d4323bfcc7891ba21986897":
        raise ValueError("R1 producer contract is not the reviewed immutable contract")
    if r1_validate.get("contract_sha256") != contract_sha:
        raise ValueError("R1 packet-validation receipt is bound to another contract")
    if r1_validate.get("status") != "PASS_PREFLIGHT_ONLY":
        raise ValueError("R1 packet-validation receipt is not a preflight pass")
    if r1_preflight.get("contract_sha256") != contract_sha:
        raise ValueError("R1 native-consumer receipt is bound to another contract")
    if r1_preflight.get("status") != "PASS_ACTUAL_WEIGHT_CONSUMER_PREFLIGHT":
        raise ValueError("R1 native-consumer preflight is not a pass")
    if r1_preflight.get("weights_archive_sha256") != sha256_file(WEIGHTS):
        raise ValueError("R1 consumer preflight does not bind the staged weights")
    if r1_preflight.get("weights_receipt_sha256") != sha256_file(WEIGHTS_RECEIPT):
        raise ValueError("R1 consumer preflight does not bind the weight receipt")

    source_bindings = {
        "r2_consumer": bind(PACKET / "run_quality_r2.py"),
        "r2_contract_builder": bind(PACKET / "build_contract_r2.py"),
        "r1_quality_consumer": r1_contract["source_bindings"]["quality_consumer"],
        "callback_runner": r1_contract["source_bindings"]["callback_runner"],
        "wrapped_predictor_source": r1_contract["source_bindings"]["checkpoint_loader"],
        "official_metric": r1_contract["source_bindings"]["official_metric"],
        "project_utils": r1_contract["source_bindings"]["project_utils"],
        "native_elf": bind(Path(r1_contract["runtime"]["loaded_elf_path"])),
        "native_package_init": bind(Path(r1_contract["runtime"]["package_init_path"])),
        "native_wheel": bind(Path(r1_contract["runtime"]["wheel_path"])),
    }
    if source_bindings["native_elf"]["sha256"] != r1_contract["runtime"]["loaded_elf_sha256"]:
        raise ValueError("native ELF differs from the reviewed R1 runtime")

    folds = {}
    for fold, binding in r1_contract["folds"].items():
        folds[fold] = {
            "evaluation_groups": binding["split_manifest"]["evaluation_valid_count"],
            "evaluation_groups_sorted_sha256": binding["split_manifest"][
                "evaluation_row_groups_sorted_canonical_sha256"
            ],
            "split_manifest_sha256": binding["split_manifest"]["sha256"],
            "exposure": binding["exposure"],
        }
    valid_data = r1_contract["folds"]["0"]["valid_dataset"]
    producer_binding = {
        "r1_contract": bind(r1_contract_path),
        "r1_validation_receipt": bind(r1_validate_path),
        "r1_native_preflight_receipt": bind(r1_consumer_path),
        "r1_weights_archive": bind(WEIGHTS),
        "r1_weights_receipt": bind(WEIGHTS_RECEIPT),
        "r1_weights_contract_sha256": contract_sha,
    }
    contract = {
        "schema": "wnn.native.shared-cleanfold-q8-quality-contract.r2",
        "status": "prepared_pending_root_execution_admission",
        "created_utc": datetime.now(UTC).isoformat(),
        "candidate_id": "oldz4_f0_f1_student_hft_native_q8_shared_half_r2_callback_fixed",
        "scope": {
            "claim": "native Q8 fixed-output blend quality on complete source-qualified fold-0 and fold-1 outer evaluation partitions",
            "folds_are_exposed": True,
            "fresh_holdout": False,
            "full_valid_in_sample_score": False,
            "public_forecast": False,
            "submission_recommendation": False,
            "training": False,
        },
        "producer_binding": producer_binding,
        "endpoint": {
            "member_order": ["student", "hft"],
            "coefficients": [0.5, 0.5],
            "coefficient_sum": 1.0,
            "blend": "fixed arithmetic mean of member output predictions; no parameter averaging",
            "shared_backbone": True,
            "fast_activations": True,
            "precision_request": "q8vnni",
            "required_actual_precision": "q8vnni",
        },
        "metric": r1_contract["metric"],
        "folds": folds,
        "valid_dataset": {
            key: valid_data[key]
            for key in (
                "path",
                "bytes",
                "footer_bytes",
                "footer_sha256_including_8_byte_trailer",
                "row_groups",
                "rows",
            )
        },
        "callback_contract": {
            "sequence_rows": 20_000,
            "feature_columns": 112,
            "rows_advanced": "all rows including warmup; warmup calls must return None",
            "state_reset": "native EnsemblePredictor reset before every complete sequence through the pinned callback runner",
            "output": "fixed two-member arithmetic mean; predictions only on need_prediction rows",
            "progress": "emit flushed fold/sequence-count/UTC heartbeat every 10 completed sequences",
            "fixture": "one complete feature-only sequence per fold, run twice with reset; labels are not loaded",
        },
        "runtime": {
            "python_executable": r1_contract["runtime"]["python_executable"],
            "python_requirement": r1_contract["runtime"]["python_requirement"],
            "loaded_elf_path": r1_contract["runtime"]["loaded_elf_path"],
            "loaded_elf_sha256": r1_contract["runtime"]["loaded_elf_sha256"],
            "package_init_path": r1_contract["runtime"]["package_init_path"],
            "package_init_sha256": r1_contract["runtime"]["package_init_sha256"],
            "native_package_dir": r1_contract["runtime"]["native_package_dir"],
            "wheel_path": r1_contract["runtime"]["wheel_path"],
            "wheel_sha256": r1_contract["runtime"]["wheel_sha256"],
            "thread_count": 1,
            "cpu_quota_percent": 100,
            "memory_max_bytes": 1_000_000_000,
            "wall_timeout_seconds": 1_800,
            "callback_only_projection": {
                "full_fold_callback_rows": 18_500_000,
                "observed_microseconds_per_row": [28.4798432225, 29.9806670075],
                "one_core_callback_seconds_at_30us_per_row": 555.0,
                "status": "projection only; not an AMD runtime guarantee; excludes decoding and metric accumulation",
            },
        },
        "execution_gate": {
            "required_root_receipt_schema": "wnn.root.native-shared-cleanfold-quality-admission.r1",
            "required_decision": "PASS_FOR_ONE_FROZEN_FULL_CLEANFOLD_Q8_EVALUATION",
            "required_receipt_bindings": [
                "contract_sha256",
                "weights_archive_sha256",
                "weights_receipt_sha256",
                "native_elf_sha256",
                "valid_dataset_footer_identity",
            ],
            "full_evaluation_performed": False,
        },
        "source_bindings": source_bindings,
        "provenance": {
            "weights_receipt_schema": "wnn.native.shared-cleanfold-q8-weights.r1",
            "weights_sha256": producer_binding["r1_weights_archive"]["sha256"],
            "weights_receipt_sha256": producer_binding["r1_weights_receipt"]["sha256"],
            "r1_preflight_sha256": producer_binding["r1_native_preflight_receipt"]["sha256"],
            "r1_packet_validation_sha256": producer_binding["r1_validation_receipt"]["sha256"],
        },
    }
    return contract


def main() -> None:
    """Create one uniquely named contract without replacing an earlier receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKET / "CONTRACT_R2.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace R2 contract: {output}")
    contract = build_contract()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": contract["status"], "contract": str(output)}))


if __name__ == "__main__":
    main()
