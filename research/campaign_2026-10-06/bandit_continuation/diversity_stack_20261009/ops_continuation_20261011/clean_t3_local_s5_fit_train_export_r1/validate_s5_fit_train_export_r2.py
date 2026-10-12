"""Independently validate the terminal s5 clean-F0 FIT_TRAIN export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ATTEMPT_ID = "clean-t3-f0-s5-fit-train-export-r1-20261012"
RUN = "clean_t3_f0_muonall_h384_e3_s5_r5"
CONTRACT_SHA256 = "e132b48bf176f412bff2851993550dde449db3a762c88ecc811b09ee744d5945"
MANIFEST_SHA256 = "dab022235b671e3525dade86c24ffb2ee4d496a97d97863ad6b28420710fc60b"
RUNNER_SHA256 = "6c005b7a0d7718b500222490af479bf0d8afaba021123ca64be9f672872f14bb"
CHECKPOINT_SHA256 = "763fac7f0aa24507167425e9905f1514a0a28ace48036b1c1c5819c7eb02c8d1"
PROVENANCE_SHA256 = "44728c0c60981166f63038f2a6696b6714ce57c52e039e304883ffe7449def18"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SELECTOR_PREFLIGHT_SHA256 = "701642f796a5d4fc19ccebc9c0f71d126308d74f347c784dc083fc713f6f5c7a"
READINESS_SHA256 = "58db010039b4b58a8771c2da982e9d5bcba21806aaa2793331419fe2e47407a9"
SOURCE_MANIFEST_SHA256 = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
TRAINER_SHA256 = "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6"
PROPENSITY_CHECKPOINT_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
SELECTED_ORDER_SHA256 = "8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda"
EXPECTED_SHAPE = (10535, 20000, 2)
EXPECTED_BYTES = 1_685_600_128
INPUT_SIZE_BYTES = 29_240_219_011
INPUT_FOOTER_SHA256 = "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb"
PARENT_CGROUP = Path(
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice"
)
PARENT_HIGH_BYTES = 18_000_000_000
VALIDATOR_RESERVE_BYTES = 500_000_000


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a file incrementally so large prediction arrays stay out of RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_group_sha256(groups: list[int]) -> str:
    """Hash ordered group IDs in the producer's little-endian int64 format."""
    values = np.asarray(groups, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object; Any is needed for its externally defined fields."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def parent_resources() -> dict[str, int]:
    """Read campaign-slice current memory and swap at the validation boundary."""
    return {
        "memory_current_bytes": int((PARENT_CGROUP / "memory.current").read_text()),
        "memory_swap_current_bytes": int((PARENT_CGROUP / "memory.swap.current").read_text()),
    }


def require(condition: bool, message: str) -> None:
    """Fail closed when a source, membership, or output invariant differs."""
    if not condition:
        raise ValueError(message)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Check terminal execution, source-qualified ancestry, sidecar, and NPY."""
    root = args.project_root.resolve(strict=True)
    base = Path(__file__).resolve().parent
    ops = root / (
        "research/campaign_2026-10-06/bandit_continuation/"
        "diversity_stack_20261009/ops_continuation_20261011"
    )
    started = time.monotonic()
    resources_before = parent_resources()
    require(
        resources_before["memory_current_bytes"] + VALIDATOR_RESERVE_BYTES < PARENT_HIGH_BYTES,
        "campaign parent plus validation reserve is not below MemoryHigh",
    )
    require(resources_before["memory_swap_current_bytes"] == 0, "campaign swap is nonzero")

    attempt_path = args.attempt_receipt.resolve(strict=True)
    attempt = read_json(attempt_path)
    require(attempt.get("id") == ATTEMPT_ID, "attempt ID differs")
    require(
        attempt.get("status") == "finished"
        and attempt.get("phase") == "terminal"
        and attempt.get("exit_status") == 0
        and attempt.get("failure") is None,
        "the bounded export did not finish successfully",
    )
    require(attempt.get("contract_sha256") == CONTRACT_SHA256, "attempt contract SHA differs")
    for counter in ("max", "oom", "oom_group_kill", "oom_kill"):
        require(
            attempt.get("memory_event_deltas", {}).get(counter) == 0,
            f"attempt memory event {counter} is nonzero",
        )
    require(int(attempt.get("peak_ram_bytes", 0)) <= 5_500_000_000, "worker RAM cap exceeded")
    require(
        int(attempt.get("peak_observed_vram_bytes", 0)) <= 5_000_000_000,
        "observed VRAM cap exceeded",
    )
    require(
        int(attempt.get("peak_parent_ram_bytes", PARENT_HIGH_BYTES)) < PARENT_HIGH_BYTES,
        "parent memory high cap was reached",
    )

    contract_path = base / "S5_FIT_TRAIN_EXPORT_CONTRACT_R1.json"
    manifest_path = base / "S5_FIT_TRAIN_EXPORT_SUCCESSOR_R1.json"
    readiness_path = ops / "clean_t3_local_s3_terminal_r1/S4_S5_EXPORT_PREFLIGHT_R1.json"
    selector_path = ops / (
        "clean_t3_local_s3_fit_train_preflight_r1/LOCAL_S3_FIT_TRAIN_PREFLIGHT_R1.json"
    )
    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    provenance_path = root / f"artifacts/runs/{RUN}/provenance.json"
    checkpoint_path = root / f"artifacts/runs/{RUN}/model_ep2.pt"
    source_manifest_path = (
        ops / "clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/SOURCE_MANIFEST_H390.json"
    )
    predictor_path = (
        ops / "clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/scripts/predict_run.py"
    )
    trainer_path = (
        ops / "clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot/scripts/train_rnn.py"
    )
    require(sha256_file(contract_path) == CONTRACT_SHA256, "s5 contract bytes differ")
    require(sha256_file(manifest_path) == MANIFEST_SHA256, "s5 manifest bytes differ")
    manifest = read_json(manifest_path)
    require(manifest.get("contract_sha256") == CONTRACT_SHA256, "manifest contract differs")
    require(manifest.get("runner_sha256") == RUNNER_SHA256, "manifest runner hash differs")
    runner_path = Path(str(manifest["runner_path"])).resolve(strict=True)
    require(sha256_file(runner_path) == RUNNER_SHA256, "frozen export runner bytes differ")
    require(sha256_file(readiness_path) == READINESS_SHA256, "seed-specific readiness changed")
    readiness = read_json(readiness_path)
    require(
        readiness.get("status") == "PASS_PROVENANCE_AND_OUTPUT_COLLISION_PREFLIGHT_ONLY",
        "seed-specific readiness did not pass",
    )
    endpoint = next(row for row in readiness["endpoints"] if row.get("run") == RUN)
    require(
        endpoint.get("seed") == 5
        and endpoint.get("selected_epoch") == 2
        and endpoint.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and endpoint.get("provenance_sha256") == PROVENANCE_SHA256
        and endpoint.get("membership", {}).get("train_count") == EXPECTED_SHAPE[0]
        and endpoint.get("membership", {}).get("train_sorted_int64_le_sha256")
        == SELECTED_ORDER_SHA256
        and endpoint.get("membership", {}).get("valid_fit_eval_disjoint") is True,
        "readiness does not bind the intended s5 CP2/membership",
    )
    require(sha256_file(selector_path) == SELECTOR_PREFLIGHT_SHA256, "selector preflight changed")
    selector = read_json(selector_path)
    boundary = selector["actual_consumer_boundary"]
    require(
        str(selector.get("status", "")).startswith("PASS_ACTUAL_FIT_TRAIN_SELECTOR_BEFORE_CUDA")
        and boundary.get("actual_fit_train_count") == EXPECTED_SHAPE[0]
        and boundary.get("actual_fit_valid_count") == 1405
        and boundary.get("actual_evaluation_valid_count") == 468
        and boundary.get("selected_groups_equal_frozen_split_fit_train_list") is True
        and boundary.get("source_qualified_fit_evaluation_component_overlap") == 0
        and boundary.get("feature_only") is True
        and boundary.get("selected_fit_train_groups_ordered_int64_le_sha256")
        == SELECTED_ORDER_SHA256,
        "actual selector preflight membership/source gate differs",
    )
    require(sha256_file(split_path) == SPLIT_SHA256, "F0 split manifest changed")
    split = read_json(split_path)
    fit_train = list(map(int, split["fit"]["train"]))
    fit_valid = list(map(int, split["fit"]["valid"]))
    evaluation_valid = list(map(int, split["evaluation"]["valid"]))
    require(
        len(fit_train) == EXPECTED_SHAPE[0]
        and len(set(fit_train)) == len(fit_train)
        and ordered_group_sha256(fit_train) == SELECTED_ORDER_SHA256,
        "split FIT_TRAIN order or membership differs",
    )
    require(
        len(fit_valid) == 1405
        and len(set(fit_valid)) == 1405
        and len(evaluation_valid) == 468
        and not (set(fit_valid) & set(evaluation_valid)),
        "source-qualified FIT_VALID/evaluation membership overlaps or changed",
    )

    require(sha256_file(provenance_path) == PROVENANCE_SHA256, "s5 provenance bytes differ")
    provenance = read_json(provenance_path)
    require(
        provenance.get("seed") == 5
        and provenance.get("split_manifest_sha256") == SPLIT_SHA256
        and provenance.get("propensity_checkpoint_sha256") == PROPENSITY_CHECKPOINT_SHA256,
        "s5 seed, split, or propensity ancestry differs",
    )
    provenance_groups = provenance.get("fit_row_groups", {})
    train_order = list(map(int, provenance_groups.get("train", [])))
    valid_order = list(map(int, provenance_groups.get("valid", [])))
    require(
        len(train_order) == len(fit_train)
        and len(set(train_order)) == len(train_order)
        and set(train_order) == set(fit_train)
        and len(valid_order) == len(fit_valid)
        and len(set(valid_order)) == len(valid_order)
        and set(valid_order) == set(fit_valid),
        "s5 source-qualified provenance membership differs from fold0 FIT sets",
    )
    for name, groups in (("train", train_order), ("valid", valid_order)):
        identity = provenance.get("fit_membership_identity", {}).get(name, {})
        require(
            identity.get("count") == len(groups)
            and identity.get("ordered_int64_le_sha256") == ordered_group_sha256(groups),
            f"s5 ordered {name} membership identity differs",
        )
    require(sha256_file(checkpoint_path) == CHECKPOINT_SHA256, "s5 selected CP2 hash differs")
    require(
        provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") == CHECKPOINT_SHA256,
        "s5 provenance does not bind selected CP2",
    )

    require(sha256_file(source_manifest_path) == SOURCE_MANIFEST_SHA256, "source manifest changed")
    require(sha256_file(predictor_path) == PREDICTOR_SHA256, "frozen predictor bytes differ")
    require(sha256_file(trainer_path) == TRAINER_SHA256, "frozen trainer bytes differ")
    source_closure = readiness["source_closure"]
    require(
        source_closure.get("snapshot_manifest") == SOURCE_MANIFEST_SHA256
        and source_closure.get("predictor") == PREDICTOR_SHA256
        and source_closure.get("trainer") == TRAINER_SHA256,
        "readiness source closure differs from consumed snapshot",
    )

    observation_path = base / "S5_FULL_EXPORT_OBSERVATION_R1.json"
    observation = read_json(observation_path)
    require(
        observation.get("status") == "FULL_EXPORT_COMPLETE"
        and observation.get("contract_sha256") == CONTRACT_SHA256
        and observation.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and observation.get("full_selected_group_count") == EXPECTED_SHAPE[0]
        and observation.get("full_selected_group_sha256") == SELECTED_ORDER_SHA256
        and observation.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256
        and observation.get("predictor_sha256") == PREDICTOR_SHA256
        and observation.get("trainer_sha256") == TRAINER_SHA256
        and observation.get("output_expected_shape") == list(EXPECTED_SHAPE)
        and observation.get("output_dtype") == "float32"
        and observation.get("output_size_bytes") == EXPECTED_BYTES
        and bool(observation.get("output_sha256")),
        "runner terminal observation differs from frozen s5 contract/output",
    )
    require(observation.get("loader_workers") == 0, "export used loader workers")
    metadata = observation.get("output_metadata", {})
    require(
        metadata.get("run") == RUN
        and metadata.get("epoch") == 2
        and metadata.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and metadata.get("campaign_fit_scope") == "train"
        and metadata.get("split") == "train"
        and metadata.get("inference_precision") == "fp32"
        and metadata.get("evaluation_membership_sha256") is None
        and metadata.get("campaign_evaluation_membership_sha256") is None,
        "runner output metadata contains mismatched scope or evaluation ancestry",
    )

    sidecar_path = (
        root / "cache/pred_clean_t3_f0_muonall_h384_e3_s5_r5_local_fit_train_r1_train.json"
    )
    sidecar = read_json(sidecar_path)
    require(sha256_file(sidecar_path) == observation.get("sidecar_sha256"), "sidecar hash differs")
    sidecar_scope = {
        "run": RUN,
        "split": "train",
        "epoch": 2,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
        "campaign_split_manifest_sha256": SPLIT_SHA256,
        "campaign_fit_scope": "train",
        "inference_precision": "fp32",
    }
    require(
        all(sidecar.get(key) == value for key, value in sidecar_scope.items()),
        "sidecar source, checkpoint, split, or fit scope differs",
    )
    groups = list(map(int, sidecar.get("row_groups", [])))
    require(
        len(groups) == EXPECTED_SHAPE[0]
        and len(set(groups)) == len(groups)
        and set(groups) == set(fit_train)
        and ordered_group_sha256(groups) == SELECTED_ORDER_SHA256,
        "producer sidecar row order/membership differs from the selected source groups",
    )

    output_path = root / "cache/pred_clean_t3_f0_muonall_h384_e3_s5_r5_local_fit_train_r1_train.npy"
    partial_path = output_path.with_name(output_path.name + ".partial")
    require(
        output_path.is_file() and not partial_path.exists(), "output missing or partial remains"
    )
    require(output_path.stat().st_size == EXPECTED_BYTES, "output byte count differs")
    predictions = np.load(output_path, mmap_mode="r", allow_pickle=False)
    require(
        predictions.shape == EXPECTED_SHAPE and predictions.dtype == np.float32,
        "NPY header differs",
    )
    finite_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, len(predictions), 16):
        block = predictions[start : start + 16]
        require(bool(np.isfinite(block).all()), f"nonfinite prediction at sequence row {start}")
        finite_count += int(block.size)
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
    output_sha256 = sha256_file(output_path)
    require(
        output_sha256 == observation.get("output_sha256"),
        "completed prediction SHA differs from runner observation",
    )
    require(sidecar.get("prediction_sha256") == output_sha256, "sidecar content SHA differs")

    raw_source = selector.get("raw_train_source_identity", {})
    require(
        raw_source.get("verified_size_bytes") == INPUT_SIZE_BYTES
        and raw_source.get("verified_footer_plus_length_plus_PAR1_sha256") == INPUT_FOOTER_SHA256
        and raw_source.get("full_content_rehashed_locally") is False,
        "bound train parquet source identity differs",
    )
    train_path = root / "datasets/train.parquet"
    require(
        train_path.stat().st_size == INPUT_SIZE_BYTES
        and train_path.stat().st_mtime_ns == observation.get("dataset_mtime_ns"),
        "point-of-use train parquet size/mtime differs from the export observation",
    )
    resources_after = parent_resources()
    require(
        resources_after["memory_current_bytes"] + VALIDATOR_RESERVE_BYTES < PARENT_HIGH_BYTES,
        "campaign parent plus validation reserve reached MemoryHigh",
    )
    require(resources_after["memory_swap_current_bytes"] == 0, "campaign swap became nonzero")

    return {
        "schema": "wnn.clean-t3-f0-s5-fit-train-export-terminal-validation.r2",
        "status": "PASS_TERMINAL_ACTUAL_PRODUCER_VALIDATION",
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt": {
            "path": str(attempt_path),
            "sha256": sha256_file(attempt_path),
            "id": ATTEMPT_ID,
            "status": attempt["status"],
            "phase": attempt["phase"],
            "exit_status": attempt["exit_status"],
            "started_utc": attempt.get("created_utc"),
            "finished_utc": attempt.get("finished_utc"),
            "elapsed_seconds": attempt.get("elapsed_seconds"),
            "memory_event_deltas": attempt.get("memory_event_deltas"),
            "memory_event_unit": "counter events; not bytes or pages",
            "peak_worker_ram_bytes": attempt.get("peak_ram_bytes"),
            "peak_parent_ram_bytes": attempt.get("peak_parent_ram_bytes"),
            "peak_observed_vram_bytes": attempt.get("peak_observed_vram_bytes"),
            "worker_memory_hard_cap_bytes": 5_500_000_000,
            "worker_memory_high_bytes": 4_950_000_000,
            "worker_swap_cap_bytes": 0,
            "parent_memory_high_bytes": PARENT_HIGH_BYTES,
            "parent_memory_max_bytes": 20_000_000_000,
            "parent_swap_cap_bytes": 0,
        },
        "source_qualified_membership": {
            "seed": 5,
            "run": RUN,
            "selected_epoch": 2,
            "split_manifest_sha256": SPLIT_SHA256,
            "fit_train_count": len(fit_train),
            "fit_train_order_sha256": ordered_group_sha256(fit_train),
            "teacher_train_order_sha256": ordered_group_sha256(train_order),
            "prediction_row_order_sha256": ordered_group_sha256(groups),
            "fit_valid_count": len(fit_valid),
            "evaluation_valid_count": len(evaluation_valid),
            "fit_valid_evaluation_overlap": 0,
            "actual_selector_preflight_sha256": SELECTOR_PREFLIGHT_SHA256,
            "seed_specific_readiness_sha256": READINESS_SHA256,
            "provenance_sha256": PROVENANCE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "source_identity": {
            "train_parquet_bytes": INPUT_SIZE_BYTES,
            "train_footer_length_par1_sha256": INPUT_FOOTER_SHA256,
            "train_full_content_sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
            "train_full_rehash_performed": False,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "predictor_sha256": PREDICTOR_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "runner_sha256": RUNNER_SHA256,
        },
        "producer_output": {
            "prediction_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "prediction_sha256": output_sha256,
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "file_bytes_including_header": output_path.stat().st_size,
            "finite_value_count": finite_count,
            "minimum": minimum,
            "maximum": maximum,
            "full_array_copy_created": False,
            "scan_block_sequences": 16,
            "feature_only": True,
            "label_derived_targets": False,
            "cache_clipping": False,
            "disk_free_bytes_after_validation": os.statvfs(output_path.parent).f_bavail
            * os.statvfs(output_path.parent).f_frsize,
        },
        "validator_resources": {
            "parent_before": resources_before,
            "parent_after": resources_after,
            "reserve_bytes": VALIDATOR_RESERVE_BYTES,
            "elapsed_seconds": time.monotonic() - started,
            "method": "read-only mmap finite scan and incremental SHA256",
        },
    }


def main() -> int:
    """Parse paths, validate the immutable producer output, and write its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "output": result["producer_output"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
