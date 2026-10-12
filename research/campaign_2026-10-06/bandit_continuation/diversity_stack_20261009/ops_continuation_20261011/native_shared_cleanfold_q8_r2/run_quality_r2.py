#!/usr/bin/env python3
"""Exercise and evaluate the frozen shared-Q8 endpoint with the real callback API."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash one contiguous array representation."""
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def load_source(name: str, path: Path) -> Any:
    """Load the source file at one already hash-bound path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_contract(path: Path) -> dict[str, Any]:
    """Read the R2 contract and enforce all R2 source bindings."""
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "wnn.native.shared-cleanfold-q8-quality-contract.r2":
        raise ValueError("unexpected R2 contract schema")
    if contract.get("status") != "prepared_pending_root_execution_admission":
        raise ValueError("R2 contract is not pending root admission")
    for name, item in contract["source_bindings"].items():
        source_path = Path(item["path"])
        if not source_path.is_file() or sha256_file(source_path) != item["sha256"]:
            raise ValueError(f"R2 source binding changed: {name}")
    return contract


def load_producer(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any, Any, Any, Any]:
    """Load and validate the R1 producers, then wrap their native callback."""
    producer = contract["producer_binding"]
    r1_contract_path = Path(producer["r1_contract"]["path"])
    r1_contract_sha = producer["r1_contract"]["sha256"]
    if sha256_file(r1_contract_path) != r1_contract_sha:
        raise ValueError("R1 source/producer contract changed")
    r1_contract = json.loads(r1_contract_path.read_text(encoding="utf-8"))
    if r1_contract.get("schema") != "wnn.native.shared-cleanfold-q8-quality-contract.r1":
        raise ValueError("unexpected R1 producer contract schema")
    if (
        sha256_file(Path(producer["r1_validation_receipt"]["path"]))
        != producer["r1_validation_receipt"]["sha256"]
    ):
        raise ValueError("R1 packet validation receipt changed")
    if (
        sha256_file(Path(producer["r1_native_preflight_receipt"]["path"]))
        != producer["r1_native_preflight_receipt"]["sha256"]
    ):
        raise ValueError("R1 native preflight receipt changed")
    weights_path = Path(producer["r1_weights_archive"]["path"])
    weights_receipt_path = Path(producer["r1_weights_receipt"]["path"])
    if sha256_file(weights_path) != producer["r1_weights_archive"]["sha256"]:
        raise ValueError("R1 weights archive changed")
    if sha256_file(weights_receipt_path) != producer["r1_weights_receipt"]["sha256"]:
        raise ValueError("R1 weights receipt changed")

    r1_consumer_path = Path(contract["source_bindings"]["r1_quality_consumer"]["path"])
    r1_consumer = load_source("native_shared_cleanfold_quality_r1", r1_consumer_path)
    validated, runner, metric = r1_consumer.source_preflight(r1_contract, r1_contract_path)
    weights, weights_receipt = r1_consumer.load_weights(
        r1_contract, weights_path, weights_receipt_path, r1_contract_path
    )
    if weights_receipt["contract_sha256"] != r1_contract_sha:
        raise ValueError("R1 exported weights are not bound to the R1 contract")
    native = importlib.import_module("wnn_native")
    wrapper_source = load_source(
        "native_shared_cleanfold_wrapped_predictor",
        Path(contract["source_bindings"]["wrapped_predictor_source"]["path"]),
    )
    return r1_contract, validated, runner, metric, (native, wrapper_source, weights), r1_consumer


def make_predictors(native: Any, wrapper_source: Any, weights: dict[str, Any]) -> dict[str, Any]:
    """Create the exact Q8 shared ensembles and apply the pinned callback adapter."""
    predictors = {}
    for fold in ("0", "1"):
        model = native.EnsemblePredictor(
            weights[fold]["members"],
            [0.5, 0.5],
            fast_activations=True,
            precision="q8vnni",
            shared_backbone=True,
        )
        if str(model.precision) != "q8vnni":
            raise RuntimeError(f"fold {fold} selected non-Q8 precision {model.precision}")
        predictors[fold] = wrapper_source.WrappedPredictor(model)
    return predictors


def open_valid_data(contract: dict[str, Any], runner: Any) -> Any:
    """Open one valid Parquet reader and check full-shape metadata only."""
    import pyarrow.parquet as pq

    path = Path(contract["valid_dataset"]["path"])
    parquet = pq.ParquetFile(path)
    expected = contract["valid_dataset"]
    if parquet.metadata.num_rows != expected["rows"]:
        raise ValueError("valid Parquet row count differs from the frozen contract")
    if parquet.num_row_groups != expected["row_groups"]:
        raise ValueError("valid Parquet row-group count differs from the frozen contract")
    if any(parquet.metadata.row_group(i).num_rows != 20_000 for i in range(parquet.num_row_groups)):
        raise ValueError("valid Parquet contains a non-complete sequence row group")
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
        raise ValueError("valid Parquet does not contain the frozen callback/metric columns")
    return parquet


def load_sequence(
    parquet: Any, row_group: int, runner: Any, include_labels: bool
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Read one whole row group from the shared ParquetFile instance."""
    columns = ["seq_ix", "step_in_seq", "need_prediction", *runner.FEATURE_COLUMNS]
    if include_labels:
        columns.extend(["is_scored", "t0", "t1"])
    table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
    sequence_id, need = runner.validate_sequence(table)
    features = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in runner.FEATURE_COLUMNS]
    ).astype(np.float32, copy=False)
    if features.shape != (20_000, len(runner.FEATURE_COLUMNS)):
        raise ValueError(f"unexpected feature shape for row group {row_group}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"nonfinite feature value in row group {row_group}")
    sequence = {
        "row_group": row_group,
        "sequence_id": int(sequence_id),
        "need": np.asarray(need, dtype=bool),
        "features": features,
    }
    if not include_labels:
        return sequence, None
    seq_ids = table["seq_ix"].to_numpy(zero_copy_only=False)
    if len(seq_ids) != 20_000 or np.any(seq_ids != sequence["sequence_id"]):
        raise ValueError(f"row-group sequence identity differs at {row_group}")
    target = np.column_stack(
        [
            table["t0"].to_numpy(zero_copy_only=False),
            table["t1"].to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float32, copy=False)
    scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
    if target.shape != (20_000, 2) or scored.shape != (20_000,):
        raise ValueError(f"unexpected target/mask shape at row group {row_group}")
    return sequence, {"target": target, "scored": scored}


def canonical_prediction(raw: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Replace the intentionally undefined warmup slots by unscored zeros."""
    if raw.shape != (20_000, 2) or need.shape != (20_000,):
        raise ValueError("callback result has an unexpected shape")
    if not np.isfinite(raw[need]).all():
        raise ValueError("callback returned a nonfinite required prediction")
    result = np.zeros((20_000, 2), dtype=np.float32)
    result[need] = raw[need]
    return result


def run_callback_fixture(
    contract: dict[str, Any],
    r1_contract: dict[str, Any],
    parquet: Any,
    runner: Any,
    predictors: dict[str, Any],
) -> dict[str, Any]:
    """Run one real feature-only complete sequence per fold, including reset replay."""
    folds = {}
    for fold in ("0", "1"):
        group = int(r1_contract["folds"][fold]["split_manifest"]["evaluation_row_groups_sorted"][0])
        sequence, labels = load_sequence(parquet, group, runner, include_labels=False)
        if labels is not None:
            raise AssertionError("feature-only fixture unexpectedly loaded labels")
        model = predictors[fold]
        pass_hashes = []
        callback_ns = []
        for _replay in range(2):
            elapsed, raw = runner.run_callbacks(model, sequence)
            callback_ns.append(elapsed)
            prediction = canonical_prediction(raw, sequence["need"])
            pass_hashes.append(sha256_array(prediction))
            del raw, prediction
        if pass_hashes[0] != pass_hashes[1]:
            raise ValueError(f"fold {fold}: reset replay was not bitwise deterministic")
        folds[fold] = {
            "row_group": group,
            "sequence_id": sequence["sequence_id"],
            "complete_rows": int(sequence["features"].shape[0]),
            "feature_count": int(sequence["features"].shape[1]),
            "need_prediction_rows": int(sequence["need"].sum()),
            "warmup_rows_returned_none": int((~sequence["need"]).sum()),
            "features_sha256": sha256_array(sequence["features"]),
            "replay_prediction_sha256": pass_hashes,
            "reset_replay_bitwise_equal": True,
            "callback_rows_per_pass": 20_000,
            "callback_ns_per_pass": callback_ns,
            "actual_precision": str(model.model.precision),
            "member_count": 2,
        }
        del sequence
    return folds


def admission_gate(
    contract_path: Path,
    contract: dict[str, Any],
    admission_path: Path,
) -> dict[str, Any]:
    """Require the independent root receipt bound to exact R2 inputs."""
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    gate = contract["execution_gate"]
    if admission.get("schema") != gate["required_root_receipt_schema"]:
        raise ValueError("unexpected root evaluation-admission schema")
    if admission.get("status") != "admitted" or admission.get("authorized") is not True:
        raise ValueError("root has not admitted this evaluation")
    if admission.get("decision") != gate["required_decision"]:
        raise ValueError("root receipt does not authorize the frozen evaluation")
    binding = admission.get("bindings", {})
    valid = contract["valid_dataset"]
    expected_footer = {
        "bytes": valid["bytes"],
        "footer_bytes": valid["footer_bytes"],
        "footer_sha256_including_8_byte_trailer": valid["footer_sha256_including_8_byte_trailer"],
    }
    exact = {
        "contract_sha256": sha256_file(contract_path),
        "weights_archive_sha256": contract["producer_binding"]["r1_weights_archive"]["sha256"],
        "weights_receipt_sha256": contract["producer_binding"]["r1_weights_receipt"]["sha256"],
        "native_elf_sha256": contract["runtime"]["loaded_elf_sha256"],
        "valid_dataset_footer_identity": expected_footer,
    }
    for key, value in exact.items():
        if binding.get(key) != value:
            raise ValueError(f"root receipt binding differs: {key}")
    return admission


def evaluate_fold(
    fold: str,
    r1_binding: dict[str, Any],
    parquet: Any,
    model: Any,
    runner: Any,
    metric: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Score one complete exposed fold using the unchanged official accumulator."""
    groups = r1_binding["split_manifest"]["evaluation_row_groups_sorted"]
    prediction_path = output_dir / f"predictions_fold{fold}.npy"
    if prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {prediction_path}")
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(groups), 20_000, 2)
    )
    accumulator = metric.GlobalAccumulator()
    seen_ids: set[int] = set()
    callback_ns = 0
    for index, group_value in enumerate(groups):
        group = int(group_value)
        sequence, labels = load_sequence(parquet, group, runner, include_labels=True)
        if labels is None:
            raise AssertionError("quality pass failed to load labels")
        sequence_id = sequence["sequence_id"]
        if sequence_id in seen_ids:
            raise ValueError(f"duplicate sequence ID in fold {fold}: {sequence_id}")
        seen_ids.add(sequence_id)
        elapsed_ns, raw = runner.run_callbacks(model, sequence)
        callback_ns += elapsed_ns
        prediction = canonical_prediction(raw, sequence["need"])
        accumulator.add(
            labels["target"],
            prediction,
            sequence["need"] & labels["scored"],
            group=group,
        )
        predictions[index] = prediction
        del raw, prediction, sequence, labels
        completed = index + 1
        if completed % 10 == 0 or completed == len(groups):
            print(
                json.dumps(
                    {
                        "event": "fold_progress",
                        "fold": int(fold),
                        "completed_sequences": completed,
                        "total_sequences": len(groups),
                        "utc": datetime.now(UTC).isoformat(),
                    }
                ),
                flush=True,
            )
    predictions.flush()
    del predictions
    metric_result = accumulator.result()
    return {
        "fold": int(fold),
        "exposure": r1_binding["exposure"],
        "evaluation_groups_sorted": groups,
        "evaluation_groups_canonical_sha256": r1_binding["split_manifest"][
            "evaluation_row_groups_sorted_canonical_sha256"
        ],
        "complete_sequences_scored": len(seen_ids),
        "metric": metric_result,
        "callback_ns": callback_ns,
        "callback_rows": len(seen_ids) * 20_000,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": sha256_file(prediction_path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one immutable JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Run the feature-only R2 fixture or admitted full exposed-fold evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PACKET / "CONTRACT_R2.json")
    parser.add_argument(
        "--weights", type=Path, default=Path("/tmp/wnn_shared_cleanfold_q8_r3/weights.npz")
    )
    parser.add_argument(
        "--weights-receipt",
        type=Path,
        default=Path("/tmp/wnn_shared_cleanfold_q8_r3/WEIGHTS_RECEIPT_R1.json"),
    )
    parser.add_argument("--mode", choices=("callback-fixture", "full-quality"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-admission", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    contract = read_contract(contract_path)
    if args.mode == "callback-fixture" and args.root_admission is not None:
        parser.error("callback fixture does not accept an execution-admission receipt")
    if args.mode == "full-quality" and args.root_admission is None:
        parser.error("full-quality mode requires the independent root admission receipt")
    r1_contract, validated, runner, metric, native_parts, r1_consumer = load_producer(contract)
    native, wrapper_source, weights = native_parts
    if (
        args.weights.resolve(strict=True)
        != Path(contract["producer_binding"]["r1_weights_archive"]["path"]).resolve()
    ):
        raise ValueError("CLI weight path differs from the frozen R1 weight producer")
    if (
        args.weights_receipt.resolve(strict=True)
        != Path(contract["producer_binding"]["r1_weights_receipt"]["path"]).resolve()
    ):
        raise ValueError("CLI weight-receipt path differs from the frozen R1 producer")
    predictors = make_predictors(native, wrapper_source, weights)
    parquet = open_valid_data(contract, runner)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    if args.mode == "callback-fixture":
        fixture_folds = run_callback_fixture(contract, r1_contract, parquet, runner, predictors)
        fixture = {
            "schema": "wnn.native.shared-cleanfold-q8-callback-fixture.r2",
            "status": "PASS_ACTUAL_COMPLETE_SEQUENCE_CALLBACK_AND_RESET_REPLAY",
            "created_utc": datetime.now(UTC).isoformat(),
            "contract_sha256": sha256_file(contract_path),
            "r1_contract_sha256": contract["producer_binding"]["r1_contract"]["sha256"],
            "weights_archive_sha256": contract["producer_binding"]["r1_weights_archive"]["sha256"],
            "weights_receipt_sha256": contract["producer_binding"]["r1_weights_receipt"]["sha256"],
            "native_elf_sha256": contract["runtime"]["loaded_elf_sha256"],
            "callback_runner_sha256": contract["source_bindings"]["callback_runner"]["sha256"],
            "wrapped_predictor_source_sha256": contract["source_bindings"][
                "wrapped_predictor_source"
            ]["sha256"],
            "folds": fixture_folds,
            "labels_loaded": False,
            "quality_evaluation_performed": False,
            "training_performed": False,
            "public_forecast": False,
        }
        write_json(output, fixture)
        print(json.dumps({"status": fixture["status"], "output": str(output)}), flush=True)
        return

    admission_path = args.root_admission.resolve(strict=True)
    admission = admission_gate(contract_path, contract, admission_path)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    results = {
        fold: evaluate_fold(
            fold,
            r1_contract["folds"][fold],
            parquet,
            predictors[fold],
            runner,
            metric,
            output,
        )
        for fold in ("0", "1")
    }
    receipt = {
        "schema": "wnn.native.shared-cleanfold-q8-quality-result.r2",
        "status": "PASS_FULL_EXPOSED_CROSSFOLD_Q8_OUTPUT_BLEND",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "root_admission_path": str(admission_path),
        "root_admission_sha256": sha256_file(admission_path),
        "root_admission_decision": admission["decision"],
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": contract["producer_binding"]["r1_weights_archive"]["sha256"],
        "weights_receipt_sha256": contract["producer_binding"]["r1_weights_receipt"]["sha256"],
        "native_elf_sha256": contract["runtime"]["loaded_elf_sha256"],
        "endpoint": contract["endpoint"],
        "metric": contract["metric"],
        "folds": results,
        "elapsed_wall_seconds": time.monotonic() - start,
        "labels_loaded": True,
        "training_performed": False,
        "public_forecast": False,
        "fresh_holdout_claim": False,
    }
    result_path = output / "QUALITY_RESULT_R2.json"
    write_json(result_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "fold_wp": {fold: result["metric"]["wp"] for fold, result in results.items()},
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
