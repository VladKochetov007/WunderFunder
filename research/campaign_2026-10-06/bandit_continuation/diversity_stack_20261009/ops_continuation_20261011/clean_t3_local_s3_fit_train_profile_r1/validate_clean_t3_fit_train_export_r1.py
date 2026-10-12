"""Validate the terminal s3 clean-F0 FIT_TRAIN prediction export in bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "attempt_id": "clean-t3-f0-s3-fit-train-export-r1-20261012",
    "contract_sha256": "2f73150c825d1ede5c401051fdb5cfaa2896abb5060ac31006ab6fff87856d37",
    "runner_sha256": "6c005b7a0d7718b500222490af479bf0d8afaba021123ca64be9f672872f14bb",
    "run": "clean_t3_f0_muonall_h384_e3_s3_r5",
    "checkpoint_sha256": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    "propensity_checkpoint_sha256": "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7",
    "split_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "preflight_sha256": "701642f796a5d4fc19ccebc9c0f71d126308d74f347c784dc083fc713f6f5c7a",
    "selected_order_sha256": "8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda",
    "shape": [10535, 20000, 2],
    "npy_bytes": 1685600128,
    "input_size_bytes": 29240219011,
    "input_footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
}
PARENT_CGROUP = Path(
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice"
)
RESERVE_BYTES = 500_000_000
PARENT_HIGH_BYTES = 18_000_000_000


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a file incrementally without allocating a whole-file copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_group_sha256(groups: list[int]) -> str:
    """Hash ordered group IDs using the export's signed little-endian int64 convention."""
    values = np.asarray(groups, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level types."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def parent_resources() -> dict[str, int]:
    """Read actual campaign-slice memory and swap counters."""
    return {
        "memory_current_bytes": int((PARENT_CGROUP / "memory.current").read_text()),
        "memory_swap_current_bytes": int((PARENT_CGROUP / "memory.swap.current").read_text()),
    }


def validate_and_write(args: argparse.Namespace) -> dict[str, Any]:
    """Check receipts, CP/membership, producer sidecar and streamed NPY payload."""
    root = args.project_root.resolve(strict=True)
    base = Path(__file__).resolve().parent
    started = time.monotonic()
    resources_before = parent_resources()
    if resources_before["memory_current_bytes"] + RESERVE_BYTES >= PARENT_HIGH_BYTES:
        raise RuntimeError("parent memory plus the validator reserve is not below MemoryHigh")

    attempt_path = args.attempt_receipt.resolve(strict=True)
    attempt = read_json(attempt_path)
    if attempt.get("id") != EXPECTED["attempt_id"]:
        raise ValueError("attempt receipt ID differs")
    if attempt.get("status") != "finished" or attempt.get("phase") != "terminal":
        raise ValueError("the export attempt is not terminal")
    if attempt.get("exit_status") != 0 or attempt.get("failure"):
        raise ValueError("the export attempt did not finish successfully")
    runner_path = base / "run_clean_t3_fit_train_export_r1.py"
    contract_path = base / "R1_FULL_EXPORT_CONTRACT_PENDING_ROOT_ADMISSION.json"
    if (
        sha256_file(runner_path) != EXPECTED["runner_sha256"]
        or sha256_file(contract_path) != EXPECTED["contract_sha256"]
        or attempt.get("contract_sha256") != EXPECTED["contract_sha256"]
    ):
        raise ValueError("frozen full-export contract or runner identity differs")
    event_deltas = attempt.get("final_memory_event_deltas", {})
    for counter in ("max", "oom", "oom_group_kill", "oom_kill"):
        if event_deltas.get(counter) != 0:
            raise ValueError(f"worker memory event {counter} is nonzero")
    if int(attempt.get("peak_ram_bytes", 0)) > 5_500_000_000:
        raise ValueError("worker peak RAM exceeds its hard cap")
    if int(attempt.get("peak_parent_ram_bytes", 0)) >= PARENT_HIGH_BYTES:
        raise ValueError("parent peak RAM reached its high cap")

    preflight_path = root / (
        "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
        "ops_continuation_20261011/clean_t3_local_s3_fit_train_preflight_r1/"
        "LOCAL_S3_FIT_TRAIN_PREFLIGHT_R1.json"
    )
    if sha256_file(preflight_path) != EXPECTED["preflight_sha256"]:
        raise ValueError("selector preflight bytes changed")
    preflight = read_json(preflight_path)
    boundary = preflight["actual_consumer_boundary"]
    if (
        not str(preflight.get("status", "")).startswith(
            "PASS_ACTUAL_FIT_TRAIN_SELECTOR_BEFORE_CUDA"
        )
        or boundary.get("actual_cp_metadata_before_cuda", {}).get("epoch") != 2
        or boundary.get("actual_cp_metadata_before_cuda", {}).get("name") != EXPECTED["run"]
        or boundary.get("actual_cp_metadata_before_cuda", {}).get("seed") != 3
        or boundary.get("actual_fit_train_count") != EXPECTED["shape"][0]
        or boundary.get("actual_fit_valid_count") != 1405
        or boundary.get("actual_evaluation_valid_count") != 468
        or boundary.get("selected_groups_equal_frozen_split_fit_train_list") is not True
        or boundary.get("source_qualified_fit_evaluation_component_overlap") != 0
        or boundary.get("feature_only") is not True
    ):
        raise ValueError("actual CP2/source-qualified selector preflight gate differs")
    if boundary.get("selected_fit_train_groups_ordered_int64_le_sha256") != EXPECTED[
        "selected_order_sha256"
    ]:
        raise ValueError("preflight ordered FIT_TRAIN group fingerprint differs")

    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    if sha256_file(split_path) != EXPECTED["split_sha256"]:
        raise ValueError("F0 split manifest bytes changed")
    split = read_json(split_path)
    fit_train = list(map(int, split["fit"]["train"]))
    fit_valid = list(map(int, split["fit"]["valid"]))
    evaluation_valid = list(map(int, split["evaluation"]["valid"]))
    if len(set(fit_train)) != 10535 or len(set(fit_valid)) != 1405:
        raise ValueError("F0 fit membership contains duplicates or changed counts")
    if len(set(evaluation_valid)) != 468 or set(fit_valid) & set(evaluation_valid):
        raise ValueError("F0 valid fit/evaluation source-qualified membership overlaps")

    provenance_path = root / f"artifacts/runs/{EXPECTED['run']}/provenance.json"
    provenance_sha256 = sha256_file(provenance_path)
    provenance = read_json(provenance_path)
    if provenance_sha256 != preflight.get("actual_provenance_sha256"):
        raise ValueError("teacher provenance is not the one in the selector preflight")
    if provenance.get("seed") != 3 or provenance.get("split_manifest_sha256") != EXPECTED[
        "split_sha256"
    ]:
        raise ValueError("teacher provenance seed or split identity differs")
    provenance_groups = provenance.get("fit_row_groups", {})
    train_order = list(map(int, provenance_groups.get("train", [])))
    valid_order = list(map(int, provenance_groups.get("valid", [])))
    if (
        len(set(train_order)) != len(train_order)
        or len(set(valid_order)) != len(valid_order)
        or set(train_order) != set(fit_train)
        or set(valid_order) != set(fit_valid)
    ):
        raise ValueError("teacher source-qualified F0 fitting sets changed")
    if set(fit_valid) & set(evaluation_valid):
        raise ValueError("teacher fit-valid membership overlaps evaluation membership")
    membership = provenance.get("fit_membership_identity", {})
    for name, order in (("train", train_order), ("valid", valid_order)):
        identity = membership.get(name, {})
        if identity.get("count") != len(order) or identity.get(
            "ordered_int64_le_sha256"
        ) != ordered_group_sha256(order):
            raise ValueError(f"teacher {name} ordered fit provenance fingerprint differs")
    checkpoint_path = root / f"artifacts/runs/{EXPECTED['run']}/model_ep2.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        checkpoint_sha256 != EXPECTED["checkpoint_sha256"]
        or provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != checkpoint_sha256
        or provenance.get("propensity_checkpoint_sha256")
        != EXPECTED["propensity_checkpoint_sha256"]
        or boundary["actual_cp_metadata_before_cuda"].get("propensity")
        != "c20260929_prop_f0"
    ):
        raise ValueError("teacher CP2 or propensity ancestry hash differs")

    sidecar_path = root / (
        "cache/pred_clean_t3_f0_muonall_h384_e3_s3_r5_local_fit_train_r1_train.json"
    )
    sidecar = read_json(sidecar_path)
    expected_scope = {
        "run": EXPECTED["run"],
        "split": "train",
        "epoch": 2,
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": EXPECTED["split_sha256"],
        "campaign_split_manifest_sha256": EXPECTED["split_sha256"],
        "campaign_fit_scope": "train",
        "inference_precision": "fp32",
    }
    if any(sidecar.get(key) != value for key, value in expected_scope.items()):
        raise ValueError("prediction sidecar source, split, checkpoint, or scope differs")
    groups = list(map(int, sidecar.get("row_groups", [])))
    if (
        len(groups) != EXPECTED["shape"][0]
        or len(set(groups)) != len(groups)
        or set(groups) != set(fit_train)
        or ordered_group_sha256(groups) != EXPECTED["selected_order_sha256"]
    ):
        raise ValueError("prediction sidecar row order/membership differs from actual preflight")

    output_path = root / (
        "cache/pred_clean_t3_f0_muonall_h384_e3_s3_r5_local_fit_train_r1_train.npy"
    )
    partial_path = output_path.with_name(output_path.name + ".partial")
    if not output_path.is_file() or partial_path.exists():
        raise ValueError("terminal output is absent or a partial remains")
    if output_path.stat().st_size != EXPECTED["npy_bytes"]:
        raise ValueError("completed NPY file byte count differs")
    predictions = np.load(output_path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != tuple(EXPECTED["shape"]) or predictions.dtype != np.float32:
        raise ValueError("completed NPY shape or dtype differs")

    # Each block spans 16 sequences (~2.56 MiB); no full-array materialization occurs.
    finite_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, len(predictions), 16):
        block = predictions[start : start + 16]
        if not np.isfinite(block).all():
            raise ValueError(f"prediction contains nonfinite values in rows {start}: {start + 16}")
        finite_count += int(block.size)
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
    output_sha256 = sha256_file(output_path)
    if sidecar.get("prediction_sha256") != output_sha256:
        raise ValueError("completed NPY content hash differs from producer sidecar")

    raw_source = preflight.get("raw_train_source_identity", {})
    if (
        raw_source.get("verified_size_bytes") != EXPECTED["input_size_bytes"]
        or raw_source.get("verified_footer_plus_length_plus_PAR1_sha256")
        != EXPECTED["input_footer_sha256"]
        or raw_source.get("full_content_rehashed_locally") is not False
    ):
        raise ValueError("frozen train source identity receipt differs")
    if not sidecar.get("prediction_sha256"):
        raise ValueError("producer sidecar is missing its output content binding")

    resources_after = parent_resources()
    if resources_after["memory_current_bytes"] + RESERVE_BYTES >= PARENT_HIGH_BYTES:
        raise RuntimeError("parent memory plus validator reserve is not below MemoryHigh after scan")
    if resources_after["memory_swap_current_bytes"] != 0:
        raise RuntimeError("campaign slice reports swap in use")

    receipt = {
        "schema": "wnn.clean-t3-f0-s3-fit-train-export-terminal-validation.r1",
        "status": "PASS_TERMINAL_ACTUAL_PRODUCER_VALIDATION",
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt": {
            "path": str(attempt_path),
            "sha256": sha256_file(attempt_path),
            "id": attempt["id"],
            "status": attempt["status"],
            "phase": attempt["phase"],
            "exit_status": attempt["exit_status"],
            "started_utc": attempt["started_utc"],
            "finished_utc": attempt["finished_utc"],
            "elapsed_seconds": attempt["elapsed_seconds"],
            "final_memory_event_deltas": event_deltas,
            "memory_event_unit": "counter events; not bytes or pages",
            "peak_worker_ram_bytes": attempt["peak_ram_bytes"],
            "peak_parent_ram_bytes": attempt["peak_parent_ram_bytes"],
            "peak_observed_vram_bytes": attempt["peak_observed_vram_bytes"],
            "observed_vram_cap_bytes": attempt["observed_vram_cap_bytes"],
            "worker_hard_memory_cap_bytes": 5_500_000_000,
            "worker_memory_high_bytes": 4_950_000_000,
            "worker_swap_cap_bytes": 0,
            "parent_memory_high_bytes": PARENT_HIGH_BYTES,
            "parent_memory_max_bytes": 20_000_000_000,
            "parent_swap_cap_bytes": 0,
            "controller_command": attempt["command"],
            "contract_sha256": attempt["contract_sha256"],
            "runner_path": str(runner_path),
            "runner_sha256": EXPECTED["runner_sha256"],
        },
        "source_qualified_membership": {
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": EXPECTED["split_sha256"],
            "fit_train_source": "datasets/train.parquet",
            "fit_train_groups": len(fit_train),
            "fit_train_split_order_sha256": ordered_group_sha256(fit_train),
            "fit_train_teacher_provenance_order_sha256": ordered_group_sha256(train_order),
            "fit_train_prediction_row_order_sha256": ordered_group_sha256(groups),
            "fit_train_prediction_membership_equals_split": True,
            "fit_valid_groups_on_distinct_source": len(fit_valid),
            "evaluation_valid_source": "datasets/valid.parquet",
            "evaluation_valid_groups": len(evaluation_valid),
            "fit_valid_evaluation_overlap": 0,
            "preflight_path": str(preflight_path),
            "preflight_sha256": EXPECTED["preflight_sha256"],
            "preflight_component_overlap": 0,
            "input_data_size_bytes": EXPECTED["input_size_bytes"],
            "input_content_identity_scope": raw_source["local_identity_scope"],
            "input_full_content_sha256": raw_source[
                "expected_full_content_sha256_from_immutable_R3_input_binding_and_R5_spec"
            ],
            "input_footer_length_par1_sha256": EXPECTED["input_footer_sha256"],
            "input_full_rehash_performed": False,
        },
        "producer": {
            "run": EXPECTED["run"],
            "seed": 3,
            "epoch": 2,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "provenance_path": str(provenance_path),
            "provenance_sha256": provenance_sha256,
            "prediction_path": str(output_path),
            "prediction_sidecar_path": str(sidecar_path),
            "prediction_sidecar_sha256": sha256_file(sidecar_path),
            "prediction_sha256": output_sha256,
            "campaign_fit_scope": sidecar["campaign_fit_scope"],
            "source_split": sidecar["split"],
            "inference_precision": sidecar["inference_precision"],
            "prediction_channels": 2,
            "prediction_treatment": "feature-only model outputs; no label-derived target channels; no cache clipping",
        },
        "output": {
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "npy_file_bytes_including_header": output_path.stat().st_size,
            "finite_value_count": finite_count,
            "minimum": minimum,
            "maximum": maximum,
            "full_array_copy_created": False,
            "validation_block_sequences": 16,
        },
        "disk": {
            "output_bytes": output_path.stat().st_size,
            "output_directory_free_bytes_after_validation": os.statvfs(output_path.parent).f_bavail
            * os.statvfs(output_path.parent).f_frsize,
        },
        "validator_resources": {
            "parent_before": resources_before,
            "parent_after": resources_after,
            "reserve_bytes": RESERVE_BYTES,
            "parent_memory_high_bytes": PARENT_HIGH_BYTES,
            "elapsed_seconds": time.monotonic() - started,
            "method": "read-only mmap finite scan plus 8 MiB incremental SHA256 under a 1 CPU / 500 MB / 120 s limited scope",
        },
        "authorization_boundary": {
            "fit_launched": False,
            "student_cache_built": False,
            "student_fit_launched": False,
            "scorer_or_predictor_modified": False,
        },
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Run the bounded validator and print a compact success record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate_and_write(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt": str(args.receipt),
                "prediction_sha256": result["producer"]["prediction_sha256"],
                "shape": result["output"]["shape"],
                "finite_value_count": result["output"]["finite_value_count"],
                "elapsed_seconds": result["validator_resources"]["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
