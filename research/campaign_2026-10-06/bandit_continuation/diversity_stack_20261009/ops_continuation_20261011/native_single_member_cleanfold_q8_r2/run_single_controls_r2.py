#!/usr/bin/env python3
"""Run the real R2 native callback path for standalone student/HFT controls."""

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
CONTRACT_PATH = PACKET / "CONTRACT_SINGLE_R2.json"
ROLES = ("student", "hft")
FOLDS = ("0", "1")
HEAD_TENSORS = ("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias")


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash one contiguous array representation."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_source(name: str, path: Path) -> Any:
    """Import a source file from its hash-bound path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_contract(path: Path) -> dict[str, Any]:
    """Load the frozen contract and fail if any executable/source input changed."""
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "wnn.native.single-member-cleanfold-q8-contract.r2":
        raise ValueError("unexpected standalone-control contract schema")
    if contract.get("status") != (
        "prepared_pending_py311_runtime_validation_and_root_execution_admission"
    ):
        raise ValueError("standalone-control contract is not pending Py3.11 validation/admission")
    for name, binding in contract["source_bindings"].items():
        source_path = Path(binding["path"])
        if not source_path.is_file() or sha256_file(source_path) != binding["sha256"]:
            raise ValueError(f"frozen source binding changed: {name}")
    return contract


def verify_producers(contract: dict[str, Any]) -> tuple[Any, ...]:
    """Verify actual checkpoints and completed R2 producer/metric receipts."""
    if Path(sys.executable).resolve() != Path(contract["runtime"]["python_executable"]).resolve():
        raise RuntimeError("use the exact MAIN interpreter bound by this contract")
    if list(sys.version_info[:2]) != contract["runtime"]["python_version_prefix"]:
        raise RuntimeError("MAIN interpreter version differs from the contract")
    producer = contract["producer_binding"]
    for key in (
        "r2_contract",
        "r2_terminal_handoff",
        "r2_quality_result",
        "r2_callback_fixture",
        "preparation_incident_receipt",
        "supplementary_callback_fixture",
        "python311_dependency_probe",
        "r1_contract",
        "weights_archive",
        "weights_receipt",
    ):
        binding = producer[key]
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"R2 producer input changed: {key}")
    r2_contract_path = Path(producer["r2_contract"]["path"])
    r2_consumer = load_source(
        "single_member_r2_quality_consumer",
        Path(contract["source_bindings"]["r2_consumer"]["path"]),
    )
    r2_contract = r2_consumer.read_contract(r2_contract_path)
    r1_contract_path = Path(producer["r1_contract"]["path"])
    r1_contract = json.loads(r1_contract_path.read_text(encoding="utf-8"))
    r1_consumer = load_source(
        "single_member_r1_weight_consumer",
        Path(contract["source_bindings"]["r1_quality_consumer"]["path"]),
    )
    weights, weights_receipt = r1_consumer.load_weights(
        r1_contract,
        Path(producer["weights_archive"]["path"]),
        Path(producer["weights_receipt"]["path"]),
        r1_contract_path,
    )
    if weights_receipt.get("contract_sha256") != producer["r1_contract"]["sha256"]:
        raise ValueError("R2 weights are not bound to the exact R1 producer contract")
    native_package = Path(contract["runtime"]["native_package_dir"]).resolve(strict=True)
    sys.path.insert(0, str(native_package))
    native = importlib.import_module("wnn_native")
    native_elf = importlib.import_module("wnn_native.wnn_native")
    wrapper_source = load_source(
        "native_shared_cleanfold_wrapped_predictor",
        Path(contract["source_bindings"]["wrapped_predictor_source"]["path"]),
    )
    runner = load_source(
        "single_member_native_callback_runner",
        Path(contract["source_bindings"]["callback_runner"]["path"]),
    )
    metric = load_source(
        "single_member_official_metric",
        Path(contract["source_bindings"]["official_metric"]["path"]),
    )
    expected_metric_id = contract["metric"]["id"]
    expected_clip = float(contract["metric"]["clip"][1])
    loaded_metric = (metric.METRIC_ID, metric.METRIC_CLIP)
    expected_metric = (expected_metric_id, expected_clip)
    if loaded_metric != expected_metric:
        raise ValueError("loaded official metric differs from the R2 contract")
    terminal = json.loads(Path(producer["r2_terminal_handoff"]["path"]).read_text())
    quality = json.loads(Path(producer["r2_quality_result"]["path"]).read_text())
    if quality.get("status") != "PASS_FULL_EXPOSED_CROSSFOLD_Q8_OUTPUT_BLEND":
        raise ValueError("completed R2 half-output control is not valid")
    if quality.get("endpoint") != {
        "blend": "fixed arithmetic mean of member output predictions; no parameter averaging",
        "coefficient_sum": 1.0,
        "coefficients": [0.5, 0.5],
        "fast_activations": True,
        "member_order": ["student", "hft"],
        "precision_request": "q8vnni",
        "required_actual_precision": "q8vnni",
        "shared_backbone": True,
    }:
        raise ValueError("R2 reference endpoint is not the frozen fixed half blend")
    if str(Path(native_elf.__file__).resolve()) != str(
        Path(contract["runtime"]["loaded_elf_path"]).resolve()
    ):
        raise ValueError("loaded native extension path differs from the R2 runtime")
    if sha256_file(Path(native_elf.__file__)) != contract["runtime"]["loaded_elf_sha256"]:
        raise ValueError("loaded native extension bytes differ from the R2 runtime")
    native_init_binding = contract["source_bindings"]["native_package_init"]
    if (
        Path(native.__file__).resolve() != Path(native_init_binding["path"]).resolve()
        or sha256_file(Path(native.__file__)) != native_init_binding["sha256"]
    ):
        raise ValueError("loaded native package initializer differs from the R2 runtime")
    for fold in FOLDS:
        binding = r1_contract["folds"][fold]
        terminal_fold = terminal["evaluation"]["folds"][fold]
        if terminal_fold["fit_evaluation_component_overlap_count"] != 0:
            raise ValueError(f"fold {fold} has fitting/evaluation component overlap")
        if (
            terminal_fold["evaluation_group_list_sha256"]
            != contract["folds"][fold]["evaluation_group_list_sha256"]
        ):
            raise ValueError(f"fold {fold} evaluation membership changed")
        for role in ROLES:
            frozen = contract["folds"][fold]["models"][role]
            source_model = binding["models"][role]
            actual_model = terminal_fold["source_qualified_models"][role]
            for key in ("checkpoint", "provenance"):
                item = frozen[key]
                path = Path(item["path"])
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise ValueError(f"actual fold {fold}/{role} {key} changed")
            for key in (
                "run",
                "epoch",
                "fit_source",
                "fit_membership_identity",
                "initial_checkpoint_sha256",
            ):
                if frozen[key] != source_model[key] or frozen[key] != actual_model[key]:
                    raise ValueError(f"fold {fold}/{role} producer identity changed: {key}")
            if frozen["checkpoint"]["sha256"] != actual_model["checkpoint_sha256"]:
                raise ValueError(f"fold {fold}/{role} checkpoint differs from terminal R2")
            if frozen["provenance"]["sha256"] != actual_model["provenance_sha256"]:
                raise ValueError(f"fold {fold}/{role} provenance differs from terminal R2")
        members = weights[fold]["members"]
        if len(members) != 2:
            raise ValueError(f"fold {fold} is not the reviewed two-member producer")
        if any(name not in members[0] or name not in members[1] for name in HEAD_TENSORS):
            raise ValueError(f"fold {fold} lacks a required nonlinear head tensor")
        if any(np.array_equal(members[0][name], members[1][name]) for name in HEAD_TENSORS):
            raise ValueError(f"fold {fold} student/HFT nonlinear heads are not independent")
    return r2_contract, r1_contract, runner, metric, native, wrapper_source, weights, quality


def make_single_models(
    native: Any, wrapper_source: Any, weights: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Construct one independent Q8 predictor per fold and model role."""
    predictors: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        predictors[fold] = {}
        members = weights[fold]["members"]
        for index, role in enumerate(ROLES):
            model = native.EnsemblePredictor(
                [members[index]],
                [1.0],
                fast_activations=True,
                precision="q8vnni",
                shared_backbone=False,
            )
            if str(model.precision) != "q8vnni":
                raise RuntimeError(f"fold {fold}/{role} did not load exact Q8VNNI")
            predictors[fold][role] = wrapper_source.WrappedPredictor(model)
    return predictors


def run_callback_fixture(
    contract: dict[str, Any],
    r2_consumer: Any,
    r1_contract: dict[str, Any],
    runner: Any,
    predictors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run real complete feature-only sequences with reset/replay for each control."""
    parquet = r2_consumer.open_valid_data(
        json.loads(Path(contract["producer_binding"]["r2_contract"]["path"]).read_text()),
        runner,
    )
    previous = json.loads(
        Path(contract["producer_binding"]["r2_callback_fixture"]["path"]).read_text()
    )
    results: dict[str, Any] = {}
    for fold in FOLDS:
        expected_fixture = previous["folds"][fold]
        group = int(r1_contract["folds"][fold]["split_manifest"]["evaluation_row_groups_sorted"][0])
        if group != expected_fixture["row_group"]:
            raise ValueError(f"fold {fold} callback fixture group differs from R2")
        sequence, labels = r2_consumer.load_sequence(parquet, group, runner, include_labels=False)
        if labels is not None:
            raise AssertionError("callback fixture unexpectedly loaded labels")
        fold_results: dict[str, Any] = {}
        for role in ROLES:
            model = predictors[fold][role]
            hashes: list[str] = []
            elapsed: list[int] = []
            prediction_shape: list[int] | None = None
            prediction_dtype: str | None = None
            for _ in range(2):
                callback_ns, raw = runner.run_callbacks(model, sequence)
                elapsed.append(int(callback_ns))
                prediction = r2_consumer.canonical_prediction(raw, sequence["need"])
                if not np.isfinite(prediction).all():
                    raise ValueError(f"fold {fold}/{role} fixture output is nonfinite")
                prediction_shape = list(prediction.shape)
                prediction_dtype = str(prediction.dtype)
                hashes.append(sha256_array(prediction))
                del raw, prediction
            if hashes[0] != hashes[1]:
                raise ValueError(f"fold {fold}/{role} reset replay is not bitwise stable")
            feature_sha = sha256_array(sequence["features"])
            if feature_sha != expected_fixture["features_sha256"]:
                raise ValueError(f"fold {fold}/{role} sequence features differ from R2")
            fold_results[role] = {
                "member_count": 1,
                "model_precision": str(model.model.precision),
                "shared_backbone": False,
                "row_group": group,
                "sequence_id": sequence["sequence_id"],
                "complete_rows": int(sequence["features"].shape[0]),
                "feature_count": int(sequence["features"].shape[1]),
                "need_prediction_rows": int(sequence["need"].sum()),
                "warmup_rows": int((~sequence["need"]).sum()),
                "features_sha256": feature_sha,
                "prediction_shape": prediction_shape,
                "prediction_dtype": prediction_dtype,
                "prediction_all_finite": True,
                "reset_replay_prediction_sha256": hashes,
                "reset_replay_bitwise_equal": True,
                "callback_rows_per_pass": 20_000,
                "callback_ns_per_pass": elapsed,
                "labels_loaded": False,
            }
        results[fold] = fold_results
        del sequence
    return results


def verify_admission(
    contract: dict[str, Any],
    contract_path: Path,
    path: Path,
    callback_fixture_path: Path,
) -> dict[str, Any]:
    """Require a root receipt bound to this exact single-member evaluation."""
    admission = json.loads(path.read_text(encoding="utf-8"))
    gate = contract["execution_gate"]
    if admission.get("schema") != gate["required_root_receipt_schema"]:
        raise ValueError("root admission schema differs")
    if admission.get("status") != "admitted" or admission.get("authorized") is not True:
        raise ValueError("root did not admit this evaluation")
    if admission.get("decision") != gate["required_decision"]:
        raise ValueError("root admission decision differs")
    producer = contract["producer_binding"]
    binding = admission.get("bindings", {})
    expected = {
        "contract_sha256": sha256_file(contract_path),
        "r2_contract_sha256": producer["r2_contract"]["sha256"],
        "r2_terminal_handoff_sha256": producer["r2_terminal_handoff"]["sha256"],
        "r2_quality_result_sha256": producer["r2_quality_result"]["sha256"],
        "r2_callback_fixture_sha256": producer["r2_callback_fixture"]["sha256"],
        "preparation_incident_receipt_sha256": producer["preparation_incident_receipt"]["sha256"],
        "supplementary_callback_fixture_sha256": producer["supplementary_callback_fixture"][
            "sha256"
        ],
        "python311_dependency_probe_sha256": producer["python311_dependency_probe"]["sha256"],
        "standalone_callback_fixture_sha256": sha256_file(callback_fixture_path),
        "weights_archive_sha256": producer["weights_archive"]["sha256"],
        "weights_receipt_sha256": producer["weights_receipt"]["sha256"],
        "native_elf_sha256": contract["runtime"]["loaded_elf_sha256"],
        "valid_dataset_footer_identity": {
            key: contract["valid_dataset"][key]
            for key in ("bytes", "footer_bytes", "footer_sha256_including_8_byte_trailer")
        },
        "fold_membership_sha256s": {
            fold: contract["folds"][fold]["evaluation_group_list_sha256"] for fold in FOLDS
        },
        "checkpoint_provenance_sha256s": {
            fold: {
                role: {
                    "checkpoint": contract["folds"][fold]["models"][role]["checkpoint"]["sha256"],
                    "provenance": contract["folds"][fold]["models"][role]["provenance"]["sha256"],
                }
                for role in ROLES
            }
            for fold in FOLDS
        },
        "output_root_absent": True,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"root admission binding differs: {key}")
    if Path(contract["expected_output"]["output_root"]).exists():
        raise FileExistsError("full-quality output root already exists")
    return admission


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one create-only receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Run a bounded fixture or a root-admitted full exposed-fold score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("callback-fixture", "full-quality"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-admission", type=Path)
    parser.add_argument("--callback-fixture", type=Path)
    args = parser.parse_args()
    contract = read_contract(CONTRACT_PATH)
    if args.mode == "callback-fixture" and args.root_admission is not None:
        parser.error("callback-fixture mode does not accept a root admission")
    if args.mode == "full-quality" and args.root_admission is None:
        parser.error("full-quality mode requires the root admission receipt")
    if args.mode == "callback-fixture" and args.callback_fixture is not None:
        parser.error("callback-fixture mode writes the fixture to --output")
    if args.mode == "full-quality" and args.callback_fixture is None:
        parser.error("full-quality mode requires the completed standalone callback fixture")
    (
        _r2_contract,
        r1_contract,
        runner,
        metric,
        native,
        wrapper_source,
        weights,
        _quality_reference,
    ) = verify_producers(contract)
    predictors = make_single_models(native, wrapper_source, weights)
    r2_consumer = load_source(
        "single_member_r2_quality_consumer",
        Path(contract["source_bindings"]["r2_consumer"]["path"]),
    )
    if args.mode == "callback-fixture":
        if args.output.resolve() != Path(contract["expected_output"]["callback_fixture"]).resolve():
            raise ValueError("callback fixture output path differs from the frozen contract")
        if args.output.exists():
            raise FileExistsError(f"refusing to replace callback receipt: {args.output}")
        folds = run_callback_fixture(contract, r2_consumer, r1_contract, runner, predictors)
        receipt = {
            "schema": "wnn.native.single-member-cleanfold-q8-callback-fixture.r2",
            "status": "PASS_ACTUAL_STANDALONE_COMPLETE_SEQUENCE_RESET_REPLAY",
            "created_utc": datetime.now(UTC).isoformat(),
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "r2_contract_sha256": contract["producer_binding"]["r2_contract"]["sha256"],
            "r2_quality_result_sha256": contract["producer_binding"]["r2_quality_result"]["sha256"],
            "weights_archive_sha256": contract["producer_binding"]["weights_archive"]["sha256"],
            "native_elf_sha256": contract["runtime"]["loaded_elf_sha256"],
            "runtime": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "python_version_prefix": list(sys.version_info[:2]),
                "native_package_path": str(Path(native.__file__).resolve()),
                "native_package_init_sha256": sha256_file(Path(native.__file__).resolve()),
                "native_extension_path": str(
                    Path(importlib.import_module("wnn_native.wnn_native").__file__).resolve()
                ),
                "native_extension_sha256": contract["runtime"]["loaded_elf_sha256"],
                "qualification": "required CPython 3.11 serving-runtime callback fixture; does not by itself establish full-fold quality",
            },
            "folds": folds,
            "quality_evaluation_performed": False,
            "labels_loaded": False,
            "training_performed": False,
            "public_forecast": False,
        }
        write_json(args.output, receipt)
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
        return

    admission_path = args.root_admission.resolve(strict=True)
    callback_fixture_path = args.callback_fixture.resolve(strict=True)
    if str(callback_fixture_path) != contract["expected_output"]["callback_fixture"]:
        raise ValueError("callback fixture path differs from the frozen contract")
    callback_fixture = json.loads(callback_fixture_path.read_text(encoding="utf-8"))
    if (
        callback_fixture.get("schema")
        != "wnn.native.single-member-cleanfold-q8-callback-fixture.r2"
    ):
        raise ValueError("standalone callback fixture schema differs")
    if callback_fixture.get("status") != "PASS_ACTUAL_STANDALONE_COMPLETE_SEQUENCE_RESET_REPLAY":
        raise ValueError("standalone callback fixture did not pass")
    if callback_fixture.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("standalone callback fixture is bound to another contract")
    fixture_runtime = callback_fixture.get("runtime", {})
    if (
        Path(fixture_runtime.get("python_executable", "")).resolve()
        != Path(contract["runtime"]["python_executable"]).resolve()
        or fixture_runtime.get("python_version_prefix")
        != contract["runtime"]["python_version_prefix"]
        or fixture_runtime.get("native_extension_sha256")
        != contract["runtime"]["loaded_elf_sha256"]
    ):
        raise ValueError("standalone callback fixture runtime differs from the contract")
    for fold in FOLDS:
        for role in ROLES:
            item = callback_fixture["folds"][fold][role]
            if (
                item.get("member_count") != 1
                or item.get("model_precision") != "q8vnni"
                or item.get("reset_replay_bitwise_equal") is not True
                or item.get("labels_loaded") is not False
                or item.get("complete_rows") != 20_000
                or item.get("callback_rows_per_pass") != 20_000
                or item.get("prediction_all_finite") is not True
                or len(item.get("reset_replay_prediction_sha256", [])) != 2
            ):
                raise ValueError(f"fold {fold}/{role} actual callback fixture is invalid")
    admission = verify_admission(contract, CONTRACT_PATH, admission_path, callback_fixture_path)
    output = Path(contract["expected_output"]["output_root"]).resolve()
    if args.output.resolve() != output:
        raise ValueError("output path differs from the frozen contract")
    output.mkdir(parents=True, exist_ok=False)
    r2_contract = json.loads(Path(contract["producer_binding"]["r2_contract"]["path"]).read_text())
    parquet = r2_consumer.open_valid_data(r2_contract, runner)
    result_folds: dict[str, Any] = {}
    expected_bytes = contract["expected_output"]["prediction_files"]
    start = time.monotonic()
    for fold in FOLDS:
        result_folds[fold] = {}
        for role in ROLES:
            role_output = output / f"fold{fold}_{role}"
            role_output.mkdir(exist_ok=False)
            scored = r2_consumer.evaluate_fold(
                fold,
                r1_contract["folds"][fold],
                parquet,
                predictors[fold][role],
                runner,
                metric,
                role_output,
            )
            if scored["complete_sequences_scored"] != contract["folds"][fold]["evaluation_groups"]:
                raise ValueError(f"fold {fold}/{role} did not score every frozen group")
            if (
                scored["evaluation_groups_canonical_sha256"]
                != contract["folds"][fold]["evaluation_group_list_sha256"]
            ):
                raise ValueError(f"fold {fold}/{role} group order differs from the contract")
            reference = contract["folds"][fold]["fixed_half_output_reference"]
            if scored["metric"]["selected_rows"] != reference["selected_rows"]:
                raise ValueError(f"fold {fold}/{role} scored-row mask differs from R2")
            key = f"fold{fold}_{role}"
            if scored["prediction_bytes"] != expected_bytes[key]["bytes"]:
                raise ValueError(f"fold {fold}/{role} prediction byte count differs")
            result_folds[fold][role] = {
                **scored,
                "fixed_half_output_reference": reference,
                "delta_wp_vs_fixed_half_output": scored["metric"]["wp"] - reference["wp"],
                "precision": "q8vnni",
                "member_count": 1,
                "shared_backbone": False,
            }
    result = {
        "schema": "wnn.native.single-member-cleanfold-q8-result.r2",
        "status": "PASS_FULL_EXPOSED_CROSSFOLD_STANDALONE_Q8_CONTROLS",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "root_admission_path": str(admission_path),
        "root_admission_sha256": sha256_file(admission_path),
        "root_admission_decision": admission["decision"],
        "standalone_callback_fixture_sha256": sha256_file(callback_fixture_path),
        "metric": contract["metric"],
        "endpoint": contract["endpoint"],
        "folds": result_folds,
        "training_performed": False,
        "public_forecast": False,
        "fresh_holdout_claim": False,
        "blend_search": False,
        "parameter_averaging": False,
        "r2_fixed_half_comparison_sha256": contract["producer_binding"]["r2_quality_result"][
            "sha256"
        ],
        "elapsed_callback_seconds": sum(
            item[role]["callback_ns"] / 1e9 for item in result_folds.values() for role in ROLES
        ),
        "elapsed_wall_seconds": time.monotonic() - start,
    }
    write_json(output / "SINGLE_MEMBER_RESULT_R2.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold_wp": {
                    fold: {role: result_folds[fold][role]["metric"]["wp"] for role in ROLES}
                    for fold in FOLDS
                },
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
