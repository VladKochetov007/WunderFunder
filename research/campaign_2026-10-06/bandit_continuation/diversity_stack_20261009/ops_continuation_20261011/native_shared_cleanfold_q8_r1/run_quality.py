#!/usr/bin/env python3
"""Preflight or score the frozen shared-Q8 student/HFT output blend on clean folds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PACKET = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_preflight(contract: dict[str, Any], contract_path: Path) -> tuple[Any, Any, Any]:
    if contract.get("status") != "prepared_pending_root_execution_admission":
        raise ValueError("contract is not pending root execution admission")
    for item in contract["source_bindings"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen source changed: {path}")
    validator = load_source("shared_cleanfold_packet_validator", PACKET / "validate_packet.py")
    fold_results = {
        fold: validator.validate_fold(int(fold), binding)
        for fold, binding in contract["folds"].items()
    }
    native = validator.validate_native(contract)
    runner = load_source(
        "shared_cleanfold_callback_runner",
        Path(contract["source_bindings"]["callback_runner"]["path"]),
    )
    metric = load_source(
        "shared_cleanfold_official_metric", Path(contract["metric"]["implementation_path"])
    )
    if contract["metric"]["id"] != metric.METRIC_ID or metric.METRIC_CLIP != 2.0:
        raise ValueError("loaded official metric identifier or clipping differs")
    dataset_identity = {
        key: contract["folds"]["0"]["valid_dataset"][key]
        for key in ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
    }
    other_identity = {
        key: contract["folds"]["1"]["valid_dataset"][key]
        for key in ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
    }
    if dataset_identity != other_identity:
        raise ValueError("fold contracts point at different validation data identities")
    return {"folds": fold_results, "native": native}, runner, metric


def load_weights(
    contract: dict[str, Any], weights_path: Path, receipt_path: Path, contract_path: Path
) -> tuple[dict[str, dict[str, list[dict[str, np.ndarray]]]], dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract_sha = sha256_file(contract_path)
    archive_sha = sha256_file(weights_path)
    if receipt.get("schema") != "wnn.native.shared-cleanfold-q8-weights.r1":
        raise ValueError("unexpected weights receipt schema")
    if receipt.get("contract_sha256") != contract_sha:
        raise ValueError("weights were exported from another contract")
    if receipt.get("status") != "PASS_SOURCE_PINNED_WEIGHTS_EXPORTED":
        raise ValueError("weights receipt is not a successful source-pinned export")
    if receipt.get("member_order") != ["student", "hft"] or receipt.get("coefficients") != [
        0.5,
        0.5,
    ]:
        raise ValueError("weights receipt changes the fixed member order or blend")
    if receipt.get("weights_sha256") != archive_sha:
        raise ValueError("weights archive differs from its receipt")
    if Path(receipt["weights_path"]).resolve() != weights_path.resolve():
        raise ValueError("weights receipt names another archive")
    loaded: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {}
    with np.load(weights_path, allow_pickle=False) as archive:
        for fold in ("0", "1"):
            members = []
            for role in ("student", "hft"):
                mapping = receipt["array_map"][fold][role]
                member = {}
                for tensor_name, archive_key in mapping.items():
                    if archive_key not in archive.files:
                        raise ValueError(f"missing array in weights archive: {archive_key}")
                    value = np.asarray(archive[archive_key], dtype=np.float32)
                    if value.size == 0 or not np.isfinite(value).all():
                        raise ValueError(f"empty/nonfinite tensor: {fold}/{role}/{tensor_name}")
                    member[tensor_name] = np.ascontiguousarray(value)
                members.append(member)
            if set(members[0]) != set(members[1]):
                raise ValueError(f"fold {fold} exported member tensor names differ")
            fold_receipt = receipt["folds"][fold]
            if (
                set(fold_receipt.get("changed_raw_tensors", []))
                != {"head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"}
                or fold_receipt.get("shared_backbone_verified_bitwise") is not True
            ):
                raise ValueError(f"fold {fold} receipt does not prove exact common trunk")
            if fold_receipt.get("raw_head_output_rows") != {"student": 31, "hft": 23}:
                raise ValueError(f"fold {fold} raw head geometry differs")
            loaded[fold] = {"members": members}
    return loaded, receipt


def make_predictors(native: Any, weights: dict[str, Any]) -> dict[str, Any]:
    predictors: dict[str, Any] = {}
    for fold in ("0", "1"):
        members = weights[fold]["members"]
        model = native.EnsemblePredictor(
            members,
            [0.5, 0.5],
            fast_activations=True,
            precision="q8vnni",
            shared_backbone=True,
        )
        actual_precision = str(model.precision)
        if not actual_precision.startswith("q8vnni"):
            raise RuntimeError(f"fold {fold} selected unexpected precision: {actual_precision}")
        predictors[fold] = model
    return predictors


def admission_gate(
    admission_path: Path,
    contract_path: Path,
    contract: dict[str, Any],
    weights_path: Path,
    args_weights_receipt: Path,
) -> dict[str, Any]:
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("schema") != contract["execution_gate"]["required_root_receipt_schema"]:
        raise ValueError("unexpected root execution-admission schema")
    if admission.get("status") != "admitted" or admission.get("authorized") is not True:
        raise ValueError("root has not admitted this evaluation")
    if admission.get("decision") != contract["execution_gate"]["required_decision"]:
        raise ValueError("root decision does not authorize the frozen evaluation")
    binding = admission.get("bindings", {})
    expected_data = next(iter(contract["folds"].values()))["valid_dataset"]
    expected_footer = {
        "bytes": expected_data["bytes"],
        "footer_bytes": expected_data["footer_bytes"],
        "footer_sha256_including_8_byte_trailer": expected_data[
            "footer_sha256_including_8_byte_trailer"
        ],
    }
    if binding.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("root receipt is bound to another contract")
    if binding.get("weights_archive_sha256") != sha256_file(weights_path):
        raise ValueError("root receipt is bound to another weight archive")
    if binding.get("weights_receipt_sha256") != sha256_file(args_weights_receipt):
        raise ValueError("root receipt is bound to another weights receipt")
    if binding.get("native_elf_sha256") != contract["runtime"]["loaded_elf_sha256"]:
        raise ValueError("root receipt is bound to another native library")
    if binding.get("valid_dataset_footer_identity") != expected_footer:
        raise ValueError("root receipt is bound to another validation dataset")
    return admission


def verify_valid_metadata(path: Path, expected: dict[str, Any], runner: Any) -> None:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups != expected["row_groups"]:
        raise ValueError("valid parquet row-group count differs")
    if parquet.metadata.num_rows != expected["rows"]:
        raise ValueError("valid parquet row count differs")
    if any(parquet.metadata.row_group(i).num_rows != 20_000 for i in range(parquet.num_row_groups)):
        raise ValueError("valid parquet contains a non-complete sequence row group")
    required = {
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        "t0",
        "t1",
        *runner.FEATURE_COLUMNS,
    }
    if not required <= set(parquet.schema.names):
        raise ValueError("valid parquet lacks callback or scoring columns")


def evaluate_fold(
    fold: str,
    binding: dict[str, Any],
    model: Any,
    runner: Any,
    metric: Any,
    output_dir: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    groups = binding["split_manifest"]["evaluation_row_groups_sorted"]
    data_path = Path(binding["valid_dataset"]["path"])
    parquet = pq.ParquetFile(data_path)
    columns = ["seq_ix", "is_scored", "t0", "t1"]
    accumulator = metric.GlobalAccumulator()
    prediction_path = output_dir / f"predictions_fold{fold}.npy"
    if prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {prediction_path}")
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(groups), 20_000, 2)
    )
    seen_ids: set[int] = set()
    callback_ns = 0
    scored_sequences = 0
    for index, group in enumerate(groups):
        sequence = runner.load_sequences(data_path, int(group), 1)[0]
        sequence_id = int(sequence["sequence_id"])
        if sequence_id in seen_ids:
            raise ValueError(f"duplicate sequence ID in fold {fold}: {sequence_id}")
        seen_ids.add(sequence_id)
        labels = parquet.read_row_group(int(group), columns=columns, use_threads=False)
        seq_ids = labels["seq_ix"].to_numpy(zero_copy_only=False)
        if len(seq_ids) != 20_000 or np.any(seq_ids != sequence_id):
            raise ValueError(f"row-group identity mismatch at fold {fold}, group {group}")
        target = np.column_stack(
            [
                labels["t0"].to_numpy(zero_copy_only=False),
                labels["t1"].to_numpy(zero_copy_only=False),
            ]
        ).astype(np.float32, copy=False)
        scored = labels["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        need = np.asarray(sequence["need"], dtype=bool)
        if target.shape != (20_000, 2) or need.shape != (20_000,) or scored.shape != (20_000,):
            raise ValueError(f"unexpected target/mask shape for fold {fold}, group {group}")
        elapsed_ns, raw_prediction = runner.run_callbacks(model, sequence)
        callback_ns += elapsed_ns
        if not np.isfinite(raw_prediction[need]).all():
            raise ValueError(f"nonfinite required prediction for fold {fold}, group {group}")
        prediction = np.zeros((20_000, 2), dtype=np.float32)
        prediction[need] = raw_prediction[need]
        mask = need & scored
        accumulator.add(target, prediction, mask, group=int(group))
        predictions[index] = prediction
        scored_sequences += 1
    predictions.flush()
    del predictions
    grade = accumulator.result()
    return {
        "fold": int(fold),
        "exposure": binding["exposure"],
        "evaluation_groups_sorted": groups,
        "evaluation_groups_canonical_sha256": binding["split_manifest"][
            "evaluation_row_groups_sorted_canonical_sha256"
        ],
        "complete_sequences_scored": scored_sequences,
        "unique_sequence_ids": len(seen_ids),
        "metric": grade,
        "callback_ns": callback_ns,
        "callback_rows": scored_sequences * 20_000,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": sha256_file(prediction_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PACKET / "CONTRACT_R1.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-receipt", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--root-admission", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    preflight_mode = args.preflight_output is not None
    if preflight_mode:
        if args.root_admission is not None or args.output_dir is not None:
            parser.error("preflight mode cannot include evaluation arguments")
    elif args.root_admission is None or args.output_dir is None:
        parser.error("evaluation requires both --root-admission and --output-dir")
    contract_path = args.contract.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validated, runner, metric = source_preflight(contract, contract_path)
    runtime = contract["runtime"]
    verify_valid_metadata(
        Path(contract["folds"]["0"]["valid_dataset"]["path"]),
        contract["folds"]["0"]["valid_dataset"],
        runner,
    )
    weights, weights_receipt = load_weights(
        contract,
        args.weights.resolve(strict=True),
        args.weights_receipt.resolve(strict=True),
        contract_path,
    )
    native = __import__("wnn_native")
    predictors = make_predictors(native, weights)
    for fold, model in predictors.items():
        if str(model.precision) != "q8vnni":
            raise ValueError(f"fold {fold}: native precision string is not exact q8vnni")
    if preflight_mode:
        output = args.preflight_output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite preflight receipt: {output}")
        receipt = {
            "schema": "wnn.native.shared-cleanfold-q8-consumer-preflight.r1",
            "status": "PASS_ACTUAL_WEIGHT_CONSUMER_PREFLIGHT",
            "created_utc": datetime.now(UTC).isoformat(),
            "contract_sha256": sha256_file(contract_path),
            "weights_receipt_sha256": sha256_file(args.weights_receipt),
            "weights_archive_sha256": sha256_file(args.weights),
            "native_elf_sha256": runtime["loaded_elf_sha256"],
            "folds": {
                fold: {
                    "actual_precision": str(model.precision),
                    "shared_backbone": True,
                    "coefficients": [0.5, 0.5],
                    "member_count": 2,
                    "actual_pair_tensors": len(weights[fold]["members"][0]),
                    "producer_check": validated["folds"][fold],
                }
                for fold, model in predictors.items()
            },
            "parquet_metadata_verified": True,
            "labels_loaded": False,
            "predictions_generated": False,
            "quality_evaluation_performed": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "contract_sha256": receipt["contract_sha256"],
                    "weights_sha256": receipt["weights_archive_sha256"],
                    "output": str(output),
                }
            )
        )
        return

    admission_path = args.root_admission.resolve(strict=True)
    admission_gate(
        admission_path,
        contract_path,
        contract,
        args.weights.resolve(strict=True),
        args.weights_receipt.resolve(strict=True),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    results = {
        fold: evaluate_fold(fold, contract["folds"][fold], model, runner, metric, output_dir)
        for fold, model in predictors.items()
    }
    receipt = {
        "schema": "wnn.native.shared-cleanfold-q8-quality-result.r1",
        "status": "PASS_FULL_EXPOSED_CROSSFOLD_Q8_OUTPUT_BLEND",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "root_admission_path": str(admission_path),
        "root_admission_sha256": sha256_file(admission_path),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "weights_receipt_sha256": sha256_file(args.weights_receipt),
        "native_elf_sha256": runtime["loaded_elf_sha256"],
        "endpoint": contract["endpoint"],
        "metric": contract["metric"],
        "folds": results,
        "elapsed_wall_seconds": time.monotonic() - start,
        "training_performed": False,
        "public_forecast": False,
        "fresh_holdout_claim": False,
    }
    result_path = output_dir / "QUALITY_RESULT_R1.json"
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "fold_wp": {f: r["metric"]["wp"] for f, r in results.items()},
                "result": str(result_path),
            }
        )
    )


if __name__ == "__main__":
    main()
