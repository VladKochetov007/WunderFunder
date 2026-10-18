"""Capture the guarded attempt and validate its exact feature-export sidecar."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import struct
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from bindings import (
    B_GUARD_STATE,
    EPOCH,
    EXECUTION_DIR,
    OUTPUT_NAME,
    PINS,
    RESOURCE_ENVELOPE,
    RUN_NAME,
    captured_preinference_record_path,
    output_paths,
    read_json,
    require_project_cache_directory,
    sha256_file,
    worker_preinference_record_path,
    write_bytes_exclusive,
    write_json_exclusive,
)
from launch import worker_command
from prepare import build_contract

H390_SIDECAR_FIELDS = {
    "run",
    "split",
    "epoch",
    "checkpoint_sha256",
    "prediction_sha256",
    "split_manifest_sha256",
    "quote_cache",
    "gate_mode",
    "inference_precision",
    "quantization_preset",
    "recurrent_rank",
    "sentinel_preprocessing",
    "numeric_preprocessing",
    "row_groups",
    "evaluation_membership_sha256",
    "fit_subset_manifest_sha256",
    "campaign_split_manifest_sha256",
    "campaign_fit_scope",
    "campaign_evaluation_membership_sha256",
}


def _archive_preinference_record(attempt_id: str) -> tuple[Path, bytes]:
    """Copy the worker's original pre-inference record to a distinct create-only path."""
    source = worker_preinference_record_path(attempt_id)
    captured = captured_preinference_record_path(attempt_id)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError("worker strict-FP32 pre-inference evidence is missing")
    if os.path.lexists(captured):
        raise FileExistsError(f"capture output already exists: {captured}")
    raw = source.read_bytes()
    write_bytes_exclusive(captured, raw)
    return captured, raw


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], str, bool, int]:
    """Read a bounded NPY header and return shape, dtype, order, payload offset."""
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    version = stream.read(2)
    if len(version) != 2:
        raise ValueError("truncated NPY version")
    if version[0] == 1:
        length_bytes = stream.read(2)
        length_format = "<H"
    elif version[0] in (2, 3):
        length_bytes = stream.read(4)
        length_format = "<I"
    else:
        raise ValueError("unsupported NPY version")
    if len(length_bytes) != struct.calcsize(length_format):
        raise ValueError("truncated NPY header length")
    header_length = struct.unpack(length_format, length_bytes)[0]
    if header_length <= 0 or header_length > 1024 * 1024:
        raise ValueError("NPY header length is outside the small-header bound")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated NPY header")
    encoding = "latin1" if version[0] < 3 else "utf-8"
    header = ast.literal_eval(header_bytes.decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError("NPY header is not a dictionary")
    return (
        tuple(header["shape"]),
        str(header["descr"]),
        bool(header["fortran_order"]),
        stream.tell(),
    )


def _ordered_row_groups(path: Path) -> tuple[list[int], str]:
    """Read only the small row_group member from valid_targets.npz."""
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo("row_group.npy")
        except KeyError as error:
            raise ValueError("valid_targets.npz lacks row_group.npy") from error
        if info.file_size > 128 * 1024:
            raise ValueError("row_group member exceeds its small-array bound")
        payload = archive.read(info)
    stream = io.BytesIO(payload)
    shape, descr, fortran, offset = _read_npy_header(stream)
    if len(shape) != 1 or shape[0] != 1873 or descr not in {"<i8", "=i8"} or fortran:
        raise ValueError("valid row_group array has unexpected dtype, order, or shape")
    raw_values = payload[offset:]
    if len(raw_values) != shape[0] * 8:
        raise ValueError("row_group payload length differs from its header")
    digest = hashlib.sha256(raw_values).hexdigest()
    values = list(struct.unpack(f"<{shape[0]}q", raw_values))
    return values, digest


def _validate_h390_sidecar(
    contract: dict[str, Any],
    sidecar: dict[str, Any],
    row_groups: list[int],
    prediction_sha256: str,
) -> None:
    """Enforce the frozen predictor's existing metadata schema and values."""
    if set(sidecar) != H390_SIDECAR_FIELDS:
        raise ValueError("sidecar field inventory differs from the frozen H390 schema")
    if sidecar.get("run") != RUN_NAME or sidecar.get("split") != "valid":
        raise ValueError("sidecar run or split differs from the contract")
    if sidecar.get("epoch") != EPOCH:
        raise ValueError("sidecar epoch differs from the contract")
    if sidecar.get("checkpoint_sha256") != contract["artifacts"]["control_checkpoint"]["sha256"]:
        raise ValueError("sidecar checkpoint digest differs from the restored control")
    if sidecar.get("split_manifest_sha256") != PINS["old_split_sha256"]:
        raise ValueError("sidecar split digest differs from frozen Main F0")
    if sidecar.get("campaign_split_manifest_sha256") != PINS["old_split_sha256"]:
        raise ValueError("sidecar campaign split digest differs from frozen Main F0")
    if sidecar.get("campaign_fit_scope") != "full-valid-feature-only":
        raise ValueError("sidecar campaign_fit_scope differs from the requested feature-only mode")
    if sidecar.get("inference_precision") != "fp32":
        raise ValueError("sidecar inference precision is not fp32")
    if sidecar.get("row_groups") != row_groups:
        raise ValueError("sidecar row order differs from valid_targets.npz")
    if sidecar.get("evaluation_membership_sha256") is not None:
        raise ValueError("full-valid export must not claim evaluation membership")
    if sidecar.get("campaign_evaluation_membership_sha256") is not None:
        raise ValueError("full-valid export must not claim evaluation-only membership")
    if sidecar.get("prediction_sha256") != prediction_sha256:
        raise ValueError("sidecar prediction digest differs from output bytes")


def _check_outputs(contract: dict[str, Any], sidecar: dict[str, Any]) -> dict[str, Any]:
    """Verify the expected H390 sidecar fields, row order and NPY header."""
    project_root = Path(contract["project_root"])
    require_project_cache_directory(project_root)
    paths = output_paths(project_root)
    for key in ("partial", "stale_seqsum", "family_prediction"):
        if os.path.lexists(paths[key]):
            raise FileExistsError(f"unexpected leftover predictor path: {paths[key]}")
    prediction = paths["prediction"]
    sidecar_path = paths["sidecar"]
    if prediction.is_symlink() or sidecar_path.is_symlink():
        raise ValueError("prediction and sidecar must be regular output files")
    expected_row_groups, row_group_sha256 = _ordered_row_groups(
        project_root / "cache/valid_targets.npz"
    )
    if prediction.stat().st_size != 299680128:
        raise ValueError("prediction byte count differs from the full-valid float32 shape")
    with prediction.open("rb") as stream:
        shape, descr, fortran, offset = _read_npy_header(stream)
    if shape != (1873, 20000, 2) or descr not in {"<f4", "=f4"} or fortran:
        raise ValueError("prediction NPY header differs from (1873, 20000, 2) float32")
    prediction_sha256 = sha256_file(prediction)
    _validate_h390_sidecar(contract, sidecar, expected_row_groups, prediction_sha256)
    return {
        "prediction": {
            "path": str(prediction),
            "bytes": prediction.stat().st_size,
            "sha256": prediction_sha256,
            "shape": list(shape),
            "dtype": "float32",
            "header_bytes": offset,
        },
        "sidecar": {
            "path": str(sidecar_path),
            "bytes": sidecar_path.stat().st_size,
            "sha256": sha256_file(sidecar_path),
            "schema_fields": sorted(sidecar),
            "campaign_fit_scope": sidecar["campaign_fit_scope"],
        },
        "row_groups": {
            "count": len(expected_row_groups),
            "ordered_int64_le_sha256": row_group_sha256,
            "first": expected_row_groups[0],
            "last": expected_row_groups[-1],
        },
    }


def _verify_guard_attempt(
    contract: dict[str, Any], contract_path: Path, attempt_record: dict[str, Any]
) -> None:
    """Require successful one-GPU guarded execution with clean resource events."""
    attempt = contract["attempt_id"]
    if (
        attempt_record.get("id") != attempt
        or attempt_record.get("status") != "finished"
        or attempt_record.get("phase") != "terminal"
        or attempt_record.get("exit_status") != 0
        or attempt_record.get("failure") not in (None, "")
    ):
        raise ValueError("bandit_resources attempt did not finish successfully")
    if (
        attempt_record.get("kind") != "gpu"
        or attempt_record.get("gpu_device") != contract["gpu_uuid"]
    ):
        raise ValueError("guard attempt GPU identity differs from root admission")
    if attempt_record.get("contract") != str(contract_path):
        raise ValueError("guard attempt contract path differs")
    if attempt_record.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("guard attempt contract digest differs")
    if attempt_record.get("unit") != f"wnn-bandit-{attempt}.service":
        raise ValueError("guard attempt systemd unit differs from the frozen controller")
    if attempt_record.get("reservation") != {
        "cpus": 1,
        "ram_bytes": RESOURCE_ENVELOPE["reserve_bytes"],
    }:
        raise ValueError("guard attempt resource reservation differs")
    if attempt_record.get("profile") != RESOURCE_ENVELOPE["profile"]:
        raise ValueError("guard attempt profile label differs")
    if (
        attempt_record.get("gpu_memory_safety_reserve_bytes")
        != (RESOURCE_ENVELOPE["gpu_free_memory_safety_reserve_bytes"])
    ):
        raise ValueError("guard attempt free-VRAM safety reserve differs")
    if float(attempt_record.get("elapsed_seconds", 7201)) > 7200:
        raise ValueError("guard attempt exceeded the 2-hour runtime cap")
    for key in (
        "final_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    ):
        deltas = attempt_record.get(key)
        if not isinstance(deltas, dict) or any(
            int(deltas.get(event, 0)) != 0 for event in ("oom", "oom_kill", "oom_group_kill")
        ):
            raise ValueError(f"guard attempt OOM counters are missing or nonzero: {key}")
    command = attempt_record.get("command")
    expected_command = worker_command(
        contract_path,
        sha256_file(contract_path),
        contract,
    )
    if command != expected_command:
        raise ValueError("guard attempt command differs from the frozen export recipe")
    sources = attempt_record.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("guard attempt lacks source hashes")
    predictor_rel = "frozen_source_snapshot/scripts/predict_run.py"
    if sources.get(predictor_rel) != PINS["predictor_sha256"]:
        raise ValueError("guard's frozen source manifest does not bind the exact predictor")
    for name, identity in contract["runtime"]["export_utility_code"].items():
        relative = str((EXECUTION_DIR / name).relative_to(Path(contract["project_root"])))
        if sources.get(relative) != identity["sha256"]:
            raise ValueError(f"guard's frozen source manifest does not bind utility: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--attempt-record", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    contract = read_json(contract_path)
    attempt = str(contract["attempt_id"])
    record_dir = EXECUTION_DIR / "records" / attempt
    output_receipt = record_dir / "EXPORT_RECEIPT.json"
    raw_attempt_copy = record_dir / "BANDIT_ATTEMPT.json"
    raw_sidecar_copy = record_dir / "PREDICTION_SIDECAR.json"
    raw_preinference_copy = captured_preinference_record_path(attempt)
    for path in (output_receipt, raw_attempt_copy, raw_sidecar_copy, raw_preinference_copy):
        if os.path.lexists(path):
            raise FileExistsError(f"capture output already exists: {path}")
    admission_path = Path(contract["root_admission"]["path"])
    actual = build_contract(
        attempt,
        admission_path,
        require_fresh_admission=False,
        check_outputs=False,
    )
    saved = dict(contract)
    saved.pop("contract_path", None)
    if actual != saved:
        raise ValueError("current inputs differ from the prepared export contract")
    attempt_path = args.attempt_record or (B_GUARD_STATE / "attempts" / f"{attempt}.json")
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise FileNotFoundError(f"guard attempt record is missing or linked: {attempt_path}")
    raw_attempt = attempt_path.read_bytes()
    attempt_sha = hashlib.sha256(raw_attempt).hexdigest()
    attempt_record = json.loads(raw_attempt)
    if not isinstance(attempt_record, dict):
        raise ValueError("guard attempt record is not a JSON object")
    _verify_guard_attempt(contract, contract_path, attempt_record)
    paths = output_paths(Path(contract["project_root"]))
    if not paths["sidecar"].is_file() or not paths["prediction"].is_file():
        raise FileNotFoundError("successful attempt is missing prediction or sidecar")
    sidecar_raw = paths["sidecar"].read_bytes()
    sidecar = json.loads(sidecar_raw)
    if not isinstance(sidecar, dict):
        raise ValueError("predictor sidecar is not a JSON object")
    outputs = _check_outputs(contract, sidecar)
    preinference_path = worker_preinference_record_path(attempt)
    archived_preinference_path, preinference_raw = _archive_preinference_record(attempt)
    preinference = json.loads(preinference_raw)
    if (
        preinference.get("schema") != "quality_gap.strong_control_pre_inference_runtime.r1"
        or preinference.get("status") != "PRE_INFERENCE_SETTINGS_VERIFIED"
        or preinference.get("attempt_id") != attempt
        or preinference.get("cuda_visible_devices") != contract["gpu_uuid"]
        or preinference.get("wnn_gpu_device") != contract["gpu_uuid"]
        or preinference.get("cuda_matmul_allow_tf32") is not False
        or preinference.get("cudnn_allow_tf32") is not False
        or preinference.get("cudnn_benchmark") is not False
        or preinference.get("cudnn_deterministic") is not True
        or preinference.get("float32_matmul_precision") != "highest"
        or preinference.get("nvidia_tf32_override") != "0"
        or preinference.get("inference_precision") != "fp32"
    ):
        raise ValueError("worker strict-FP32 runtime evidence failed validation")
    now = datetime.now(UTC).isoformat()
    receipt = {
        "schema": "quality_gap.f0_strong_control_export_receipt.r1",
        "status": "PASS_STRONG_CONTROL_FEATURE_EXPORT_CAPTURE",
        "created_utc": now,
        "attempt_id": attempt,
        "owner": "root",
        "contract": {
            "path": str(contract_path),
            "bytes": contract_path.stat().st_size,
            "sha256": sha256_file(contract_path),
        },
        "preflight_receipt": contract["preflight_receipt"],
        "root_admission": contract["root_admission"],
        "gpu_uuid": contract["gpu_uuid"],
        "guard_attempt": {
            "path": str(attempt_path),
            "bytes": len(raw_attempt),
            "sha256": attempt_sha,
            "status": attempt_record["status"],
            "exit_status": attempt_record["exit_status"],
            "unit": attempt_record["unit"],
            "elapsed_seconds": attempt_record["elapsed_seconds"],
            "reservation": attempt_record["reservation"],
            "peak_ram_bytes": attempt_record.get("peak_ram_bytes"),
            "peak_parent_ram_bytes": attempt_record.get("peak_parent_ram_bytes"),
            "peak_observed_vram_bytes": attempt_record.get("peak_observed_vram_bytes"),
            "final_memory_event_deltas": attempt_record["final_memory_event_deltas"],
            "final_parent_memory_event_deltas": attempt_record["final_parent_memory_event_deltas"],
            "final_parent_hierarchical_memory_event_deltas": attempt_record[
                "final_parent_hierarchical_memory_event_deltas"
            ],
        },
        "source_pins": contract["pins"],
        "execution": contract["execution"],
        "pre_inference_runtime": {
            "path": str(preinference_path),
            "bytes": len(preinference_raw),
            "sha256": hashlib.sha256(preinference_raw).hexdigest(),
            "torch_version": preinference["torch_version"],
            "torch_cuda_version": preinference["torch_cuda_version"],
            "cuda_matmul_allow_tf32": preinference["cuda_matmul_allow_tf32"],
            "cudnn_allow_tf32": preinference["cudnn_allow_tf32"],
            "cudnn_benchmark": preinference["cudnn_benchmark"],
            "cudnn_deterministic": preinference["cudnn_deterministic"],
            "float32_matmul_precision": preinference["float32_matmul_precision"],
            "cuda_visible_devices": preinference["cuda_visible_devices"],
        },
        "output_name": OUTPUT_NAME,
        "prediction": outputs["prediction"],
        "sidecar": outputs["sidecar"],
        "ordered_rows": outputs["row_groups"],
        "scope": {
            "feature_only_full_valid": True,
            "training": False,
            "grading": False,
            "metric_computed": False,
            "evaluation_only_export": False,
        },
        "runtime_notes": contract["unverified_gates"],
    }
    write_bytes_exclusive(raw_attempt_copy, raw_attempt)
    write_bytes_exclusive(raw_sidecar_copy, sidecar_raw)
    receipt["captured_raw_evidence"] = {
        "bandit_attempt": {
            "path": str(raw_attempt_copy),
            "bytes": raw_attempt_copy.stat().st_size,
            "sha256": sha256_file(raw_attempt_copy),
        },
        "prediction_sidecar": {
            "path": str(raw_sidecar_copy),
            "bytes": raw_sidecar_copy.stat().st_size,
            "sha256": sha256_file(raw_sidecar_copy),
        },
        "pre_inference_runtime": {
            "path": str(archived_preinference_path),
            "bytes": archived_preinference_path.stat().st_size,
            "sha256": sha256_file(archived_preinference_path),
        },
    }
    write_json_exclusive(output_receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
