"""Hash-bound native GPU FP16 controls and artifact checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

R5_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
R5_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
R5_REVIEW_STATE = "verified_for_independent_grade_and_native_control_transport"
AMP_INITIAL_SCALE = 1024.0
R4_TRAINER_SOURCE_SHA256 = "971c0336a71d374b507b0a5b83c312afc9ae8dbae8262cb492f0611e6d5bcc0b"
R4_PREDICTOR_SOURCE_SHA256 = "74d1eac17add3f9461e1205bac0eb5b6a78facd567898d679b15bdfa5bed4f5c"
UPDATE_COUNT = 220
PROFILE_UPDATE_COUNT = 20
PRECISION_GUARD_BEFORE = (
    b'            or args.precision != "fp32"\n            or args.resume_state\n'
)
PRECISION_GUARD_AFTER = (
    b'            or args.precision not in {"fp32", "fp16"}\n            or args.resume_state\n'
)
REQUIRED_SHARED_ARTIFACTS = {
    "cache_manifest": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
    "cache_reader_profile": "fbac76445ff91dbe8d8336bc3ab2bc3e2ddddd3c4be90d25550c478ea774091e",
    "cache_validation_receipt": "bf43ade9f08fb0447fc1df18dbb91f1d4723c828eaa3c14a7f5bffd73049f6ed",
    "device_profile_gate": "70b8b40e994a74387a0b7e9055a532f21fbbb2ea516ead3ef3201d1ad3c575ae",
    "fold0_asset_validation_receipt": "93ecabd180a608d72f111b9886e8804cb084d4ace694de57d188b745c33dbd43",
    "initial_state": "10e4b56002b52faaaf55dbfcbd28b37f479d2dbd69e727740250dbb90d873af1",
    "normalization": "d096115cb59a7462608d5d11df29ba77142aac03527f26fbb2a8534077e57849",
    "normalization_metadata": "0e4762ebb51f95589b93e79545c769376b42afaa2e7069f81b199164dd59cf96",
    "onecycle_trace": "a46a5beb2484b3549b6fdb926b885ecbcaf4a61b1cec60dd3750760eafff27de",
    "profile_inputs": "f042ae97ddebac87c1023758ccd84d9b2923e5cc6ae3ac878596e84bfde9a1c6",
    "profile_inputs_receipt": "6620c5f34c13fd9310a4aaa7d27e0631b9bdf8d05ceb0459863cfed1746596ab",
    "sample_orders": "cbb972a29278617890ac3e51d9f4d3cb954d392192aef3ebe044b8a71a8ec523",
    "source_stage_receipt": "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7",
    "source_verification": "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567",
}
REQUIRED_PREDICTION_SHAPE = (468, 20_000, 2)
MINIMUM_HOST_AVAILABLE_BYTES = 4_000_000_000
MINIMUM_CGROUP_HEADROOM_BYTES = 4_000_000_000
MINIMUM_GPU_FREE_BYTES = 2_000_000_000
MINIMUM_DISK_FREE_BYTES = 4_000_000_000


class ContractError(ValueError):
    """A control, command, or producer artifact is outside the frozen contract."""


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Hash a regular file in bounded chunks."""
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def native_train_argv(base_argv: Iterable[str]) -> list[str]:
    """Change only the trainer precision and its predeclared GradScaler start."""
    command = list(base_argv)
    indexes = [index for index, token in enumerate(command) if token == "--precision"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ContractError("base command must have exactly one --precision value")
    if command[indexes[0] + 1] != "fp32":
        raise ContractError("native arm must derive from the frozen FP32 D01 command")
    if "--amp-init-scale" in command:
        raise ContractError("base command unexpectedly sets a scaler; reject ambiguous override")
    if "--fused-adam" not in command or "--grad-stats" not in command:
        raise ContractError("FP16 command must retain fused AdamW and per-update gradient checks")
    command[indexes[0] + 1] = "fp16"
    command.extend(("--amp-init-scale", str(AMP_INITIAL_SCALE)))
    return command


def native_trainer_argv(
    base_argv: Iterable[str], source_script: Path, overlay_script: Path
) -> list[str]:
    """Bind FP16 train commands to the exact guard-compatible R4 trainer overlay."""
    command = native_train_argv(base_argv)
    if len(command) < 2 or Path(command[1]) != source_script:
        raise ContractError("native train command does not invoke the pinned R4 trainer")
    command[1] = str(overlay_script)
    return command


def native_prediction_argv(base_argv: Iterable[str]) -> list[str]:
    """Remove R4's unsupported device flag and request common FP32 predictions."""
    command = list(base_argv)
    if len(command) < 3 or Path(command[1]).name != "predict_run.py":
        raise ContractError("prediction command does not use the sealed R4 predictor")
    device_indexes = [index for index, token in enumerate(command) if token == "--device"]
    if (
        len(device_indexes) != 1
        or device_indexes[0] + 1 >= len(command)
        or command[device_indexes[0] + 1] != "cuda"
    ):
        raise ContractError("R4 prediction command must contain exactly one unsupported CUDA flag")
    if "--inference-precision" in command:
        raise ContractError("R4 prediction command has an ambiguous inference precision")
    for option in (
        "--parsed-cache-manifest",
        "--parsed-cache-validation-receipt",
        "--expected-parsed-cache-manifest-sha256",
        "--expected-parsed-cache-validation-sha256",
        "--expected-parsed-cache-key",
    ):
        if command.count(option) != 1:
            raise ContractError(f"R4 prediction command lost its cache binding: {option}")
    del command[device_indexes[0] : device_indexes[0] + 2]
    command.extend(("--inference-precision", "fp32"))
    return command


def create_native_trainer_overlay(source_path: Path, destination: Path) -> dict[str, Any]:
    """Create a hash-bound R4 trainer copy allowing FP16 in its loss-pair prefit guard."""
    source_sha256 = sha256_file(source_path)
    if source_sha256 != R4_TRAINER_SOURCE_SHA256:
        raise ContractError("R4 train_rnn.py differs from the reviewed precision-guard source")
    if destination.exists() or destination.is_symlink():
        raise ContractError("native trainer overlay destination already exists")
    source_bytes = source_path.read_bytes()
    if source_bytes.count(PRECISION_GUARD_BEFORE) != 1:
        raise ContractError("R4 trainer precision guard is not the exact single reviewed target")
    overlay_bytes = source_bytes.replace(PRECISION_GUARD_BEFORE, PRECISION_GUARD_AFTER, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(overlay_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema": "wnn.native-gpu-fp16-trainer-overlay.v1",
        "state": "created",
        "base_trainer_sha256": source_sha256,
        "overlay_trainer_sha256": sha256_file(destination),
        "overlay_path": destination.as_posix(),
        "exact_guard_replacement_count": 1,
        "change": "loss-pair prefit export accepts fp32 or fp16; all other guard terms are unchanged",
    }


def validate_native_profile_evidence(evidence: dict[str, Any]) -> None:
    """Require a finite 20-update CUDA profile with active GradScaler and no skips."""
    expected = {
        "valid_for_throughput": True,
        "optimizer_attempts": PROFILE_UPDATE_COUNT,
        "optimizer_completed_updates": PROFILE_UPDATE_COUNT,
        "amp_skipped_updates": 0,
        "loss_finite": True,
        "gradient_finite": True,
        "grad_scaler_enabled": True,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ContractError(f"native FP16 profile evidence differs at {key}")


def validate_fit_headroom(sample: dict[str, Any], required_gpu_indices: set[int]) -> dict[str, Any]:
    """Require measured host, bounded-cgroup, disk, and actual-GPU reserves."""
    memory = sample.get("memory")
    if not isinstance(memory, dict):
        raise ContractError("conditional fit lacks a live memory snapshot")
    host = memory.get("mem_available_bytes")
    current = memory.get("cgroup_current_bytes")
    maximum = memory.get("cgroup_max_bytes")
    disk = sample.get("working_disk_free_bytes")
    if not isinstance(host, int) or host < MINIMUM_HOST_AVAILABLE_BYTES:
        raise ContractError("conditional fit requires at least 4 GB observed host MemAvailable")
    if not isinstance(current, int):
        raise ContractError("conditional fit requires an observed cgroup current memory value")
    unbounded = memory.get("cgroup_max_unbounded") is True
    if maximum is not None and not isinstance(maximum, int):
        raise ContractError("cgroup memory maximum is neither bounded nor explicitly unbounded")
    if maximum is None and not unbounded:
        raise ContractError("cgroup memory limit is missing or was not verified unbounded")
    cgroup_free = maximum - current if isinstance(maximum, int) else None
    if cgroup_free is not None and cgroup_free < MINIMUM_CGROUP_HEADROOM_BYTES:
        raise ContractError("conditional fit requires 4 GB observed cgroup headroom")
    if not isinstance(disk, int) or disk < MINIMUM_DISK_FREE_BYTES:
        raise ContractError("conditional fit requires at least 4 GB observed working-disk free")
    devices = sample.get("gpus")
    if not isinstance(devices, list):
        raise ContractError("conditional fit requires live per-device CUDA memory samples")
    by_index = {int(row.get("physical_index", -1)): row for row in devices if isinstance(row, dict)}
    if set(by_index) != required_gpu_indices:
        raise ContractError("conditional fit GPU inventory differs from the admitted device set")
    gpu_free: dict[str, int] = {}
    for index, row in sorted(by_index.items()):
        total = row.get("memory_total_mib")
        used = row.get("memory_used_mib")
        if not isinstance(total, int) or not isinstance(used, int) or total < used:
            raise ContractError(f"GPU {index} memory sample is incomplete or inconsistent")
        free_bytes = (total - used) * (1 << 20)
        if free_bytes < MINIMUM_GPU_FREE_BYTES:
            raise ContractError(f"GPU {index} has less than 2 GB observed free memory")
        gpu_free[str(index)] = free_bytes
    return {
        "state": "passed",
        "host_mem_available_bytes": host,
        "cgroup_current_bytes": current,
        "cgroup_max_bytes": maximum,
        "cgroup_headroom_bytes": cgroup_free,
        "disk_free_bytes": disk,
        "device_free_bytes": gpu_free,
        "policy": {
            "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
            "minimum_bounded_cgroup_headroom_bytes": MINIMUM_CGROUP_HEADROOM_BYTES,
            "minimum_free_per_gpu_bytes": MINIMUM_GPU_FREE_BYTES,
            "minimum_working_disk_free_bytes": MINIMUM_DISK_FREE_BYTES,
        },
    }


def verify_r5_control_tree(control_root: Path, review_path: Path | None = None) -> dict[str, Any]:
    """Verify the exact R5 producer and all shared roles needed by native training."""
    contract_path = control_root / "training_contract.json"
    review_path = review_path or control_root / "ROOT_GPU_R5_TERMINAL_REVIEW_R1.json"
    if sha256_file(contract_path) != R5_CONTRACT_SHA256:
        raise ContractError("mounted R5 training contract hash differs from the reviewed producer")
    review = _read_json(review_path)
    if (
        sha256_file(review_path) != R5_REVIEW_SHA256
        or review.get("schema") != "wnn.root-r5-terminal-review.r1"
        or review.get("state") != R5_REVIEW_STATE
        or review.get("training_contract_sha256") != R5_CONTRACT_SHA256
        or review.get("quality_claim") != "pending independent fixed grader"
    ):
        raise ContractError("R5 control review receipt is not the reviewed pending-grade receipt")

    contract = _read_json(contract_path)
    if contract.get("schema") != "wnn.loss-pair.training-contract.v1":
        raise ContractError("unsupported R5 training-contract schema")
    expected = {
        "state": "complete",
        "cache": {
            "fold": 0,
            "fit_train_count": 350,
            "fit_valid_count_not_used_for_fit": 28,
            "evaluation_valid_count": 468,
            "membership_sha256": "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd",
        },
        "evaluation": {
            "fixed_evaluation_count": 468,
            "fold": 0,
            "metric_id": "global_wp_clip2_v1",
            "state": "pending_independent_grade",
        },
        "fit": {
            "batch_size": 32,
            "chunk_rows": 2000,
            "epochs": 2,
            "seed": 20261009,
            "updates_per_arm": 220,
            "optimizer": "AdamW",
            "optimizer_defaults": {"beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.01},
            "scheduler": "OneCycleLR",
            "learning_rate_max": 0.001,
            "global_gradient_clip": 1.0,
            "warmup_row_mask": "row_index >= 99",
            "loss_normalization": "per chunk",
        },
        "model": {
            "family": "two-layer GRU",
            "input_layout": {"raw_features": 112, "causal_event_features": 99},
            "input_width": 211,
            "hidden_size": 256,
            "layers": 2,
            "activation": "GELU",
            "precision": "float32",
            "tf32": False,
        },
    }
    for parent, values in expected.items():
        observed = contract.get(parent)
        if isinstance(values, dict):
            if not isinstance(observed, dict):
                raise ContractError(f"R5 contract section is missing: {parent}")
            for key, value in values.items():
                if isinstance(value, dict):
                    child = observed.get(key)
                    if not isinstance(child, dict) or any(
                        child.get(k) != v for k, v in value.items()
                    ):
                        raise ContractError(f"R5 contract differs at {parent}.{key}")
                elif observed.get(key) != value:
                    raise ContractError(f"R5 contract differs at {parent}.{key}")
        elif observed != values:
            raise ContractError(f"R5 contract differs at {parent}")

    roles = contract.get("artifacts")
    if not isinstance(roles, dict):
        raise ContractError("R5 contract omits artifact-role bindings")
    for role, expected_sha in REQUIRED_SHARED_ARTIFACTS.items():
        record = roles.get(role)
        if not isinstance(record, dict) or record.get("sha256") != expected_sha:
            raise ContractError(f"R5 contract has a different shared role: {role}")
        path = control_root / str(record.get("path", ""))
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ContractError(f"R5 shared role size differs: {role}")
        if sha256_file(path) != expected_sha:
            raise ContractError(f"R5 shared role bytes differ: {role}")
    return contract


def compare_npz_arrays(expected_path: Path, actual_path: Path, role: str) -> None:
    """Compare producer/consumer NPZ values, shape, dtype, and member inventory."""
    import numpy as np

    try:
        with (
            np.load(expected_path, allow_pickle=False) as expected,
            np.load(actual_path, allow_pickle=False) as actual,
        ):
            if set(expected.files) != set(actual.files):
                raise ContractError(f"{role} array names differ from the R5 producer")
            for name in expected.files:
                if (
                    expected[name].dtype != actual[name].dtype
                    or expected[name].shape != actual[name].shape
                    or not np.array_equal(expected[name], actual[name])
                ):
                    raise ContractError(f"{role} array differs from the R5 producer: {name}")
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot consume {role} NPZ: {error}") from error


def summarize_update_trace(path: Path, expected_attempts: int) -> dict[str, Any]:
    """Count completed AMP updates separately from attempted schedule rows."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid update trace row {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ContractError(f"update trace row {line_number} is not an object")
        rows.append(row)
    if len(rows) != expected_attempts:
        raise ContractError(f"expected {expected_attempts} update attempts, saw {len(rows)}")
    if [row.get("attempt_index") for row in rows] != list(range(expected_attempts)):
        raise ContractError("update trace indices are not the exact contiguous schedule")
    completed = 0
    skipped = 0
    scales = []
    for row in rows:
        skip = row.get("scaler_skipped")
        if not isinstance(skip, bool):
            raise ContractError("FP16 update trace omits the GradScaler skip flag")
        skipped += int(skip)
        completed += int(not skip)
        for key in ("amp_scale_before", "amp_scale_after"):
            scale = row.get(key)
            if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
                raise ContractError(f"FP16 update trace has an invalid {key}")
            scales.append(float(scale))
        for key in ("unclipped_global_grad_norm",):
            value = row.get(key)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ContractError(f"FP16 update trace has a nonfinite {key}")
        for group in row.get("optimizer_groups", []):
            if not math.isfinite(float(group["lr"])) or not math.isfinite(float(group["beta1"])):
                raise ContractError("FP16 update trace has nonfinite schedule values")
    return {
        "optimizer_attempts": expected_attempts,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "all_attempts_completed": completed == expected_attempts and skipped == 0,
        "scale_min": min(scales) if scales else None,
        "scale_max": max(scales) if scales else None,
        "throughput_update_count": completed,
    }


def validate_prediction_npy(path: Path) -> dict[str, Any]:
    """Check the actual producer's NPY shape, dtype, and finite chunks without loading all rows."""
    import numpy as np

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot load producer prediction NPY: {error}") from error
    if array.shape != REQUIRED_PREDICTION_SHAPE or array.dtype != np.dtype("<f4"):
        raise ContractError(
            "producer prediction header differs from the fixed 468 x 20,000 x 2 FP32 contract"
        )
    for start in range(0, array.shape[0], 8):
        if not np.isfinite(array[start : start + 8]).all():
            raise ContractError("producer prediction contains nonfinite rows")
    return {"shape": list(array.shape), "dtype": array.dtype.str, "finite": True}


def verify_prediction_membership(
    output_root: Path, arm: str, contract: dict[str, Any]
) -> dict[str, Any]:
    """Bind actual producer prediction rows to its cache membership/order sidecars."""
    if arm not in {"wmse", "huber_d1"}:
        raise ContractError(f"unsupported loss-pair arm: {arm}")
    arm_root = output_root / "arms" / arm
    array_record = next(
        (
            row
            for row in contract["arms"][arm]["artifacts"]
            if row.get("path") == f"arms/{arm}/predictions_valid.npy"
        ),
        None,
    )
    if not isinstance(array_record, dict):
        raise ContractError(f"R5 contract omits {arm} prediction NPY")
    array_path = output_root / array_record["path"]
    if array_path.stat().st_size != int(array_record["size_bytes"]):
        raise ContractError(f"R5 prediction byte size differs for {arm}")
    if sha256_file(array_path) != array_record["sha256"]:
        raise ContractError(f"R5 prediction hash differs for {arm}")
    array = validate_prediction_npy(array_path)

    orders_path = output_root / contract["artifacts"]["sample_orders"]["path"]
    membership_path = arm_root / "prediction_membership.json"
    sidecar_path = arm_root / "predictions_valid.json"
    orders = _read_json(orders_path)
    membership = _read_json(membership_path)
    sidecar = _read_json(sidecar_path)
    expected_ids = orders.get("evaluation_ids")
    rows = membership.get("rows")
    if (
        membership.get("schema") != "wnn.loss-pair.prediction-membership.v1"
        or membership.get("cache_key") != contract["cache"]["cache_key"]
        or membership.get("membership_sha256") != contract["cache"]["membership_sha256"]
        or not isinstance(expected_ids, list)
        or not isinstance(rows, list)
        or len(expected_ids) != 468
        or len(rows) != 468
        or [row.get("sequence_id") for row in rows] != expected_ids
        or [row.get("position") for row in rows] != list(range(468))
    ):
        raise ContractError(f"R5 {arm} prediction rows differ from the frozen evaluation order")
    if sidecar.get("parsed_sequence_cache", {}).get("cache_key") != contract["cache"][
        "cache_key"
    ] or sidecar.get("row_groups") != [int(row["row_group"]) for row in rows]:
        raise ContractError(f"R5 {arm} predictor sidecar differs from membership")
    return {
        "prediction": array,
        "evaluation_count": len(rows),
        "membership_sha256": membership["membership_sha256"],
        "cache_key": membership["cache_key"],
        "sidecar_bound": True,
    }


def verify_r5_full_output_artifacts(output_root: Path) -> dict[str, Any]:
    """Read the completed R5 producer exactly as a downstream consumer would."""
    contract = verify_r5_control_tree(
        output_root,
        review_path=output_root.parents[5] / "ROOT_GPU_R5_TERMINAL_REVIEW_R1.json",
    )
    verified = 0
    model_states = {}
    for arm in ("wmse", "huber_d1"):
        arm_contract = contract["arms"][arm]
        arm_root = output_root / "arms" / arm
        for record in arm_contract["artifacts"]:
            path = output_root / record["path"]
            if path.stat().st_size != int(record["size_bytes"]):
                raise ContractError(f"R5 artifact size differs: {record['path']}")
            if sha256_file(path) != record["sha256"]:
                raise ContractError(f"R5 artifact hash differs: {record['path']}")
            verified += 1
        verify_prediction_membership(output_root, arm, contract)
        update_summary = summarize_update_trace(
            arm_root / "update_trace.jsonl", expected_attempts=UPDATE_COUNT
        )
        if not update_summary["all_attempts_completed"]:
            raise ContractError(f"R5 FP32 control has incomplete schedule rows: {arm}")
        # The R5 export recovery intentionally retained the final inference state
        # but not each original optimizer continuation file.
        model_checkpoint = arm_root / "model" / "model.pt"
        model_states[arm] = inspect_fp32_model_checkpoint(model_checkpoint)
    return {
        "training_contract_sha256": R5_CONTRACT_SHA256,
        "root_review_sha256": R5_REVIEW_SHA256,
        "verified_arm_artifact_count": verified,
        "arms": ["wmse", "huber_d1"],
        "model_states": model_states,
        "producer_consumer": "actual R5 contract, checkpoints, update traces, NPY arrays and row sidecars",
    }


def inspect_fp32_model_checkpoint(model_path: Path) -> dict[str, Any]:
    """Inspect an actual inference checkpoint for FP32 master parameter tensors."""
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(model_state, dict) or not model_state:
        raise ContractError("model checkpoint consumer did not find its state_dict")
    model_tensors = [value for value in model_state.values() if isinstance(value, torch.Tensor)]
    if not model_tensors or any(
        tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in model_tensors
    ):
        raise ContractError("model checkpoint does not retain FP32 master tensors")
    return {"model_parameter_dtype": "torch.float32", "floating_tensor_count": len(model_tensors)}


def verify_fp32_master_checkpoint(model_path: Path, training_state_path: Path) -> dict[str, Any]:
    """Inspect serialized model/optimizer tensors and reject low-precision master state."""
    import torch

    model_evidence = inspect_fp32_model_checkpoint(model_path)
    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    training_model = training_state.get("model") if isinstance(training_state, dict) else None
    if not isinstance(training_model, dict) or not training_model:
        raise ContractError("full training-state consumer omitted its model master weights")
    training_model_tensors = [
        value for value in training_model.values() if isinstance(value, torch.Tensor)
    ]
    if not training_model_tensors or any(
        tensor.is_floating_point() and tensor.dtype != torch.float32
        for tensor in training_model_tensors
    ):
        raise ContractError("full training state does not retain FP32 master weights")
    optimizer = training_state.get("optimizer") if isinstance(training_state, dict) else None
    state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(state, dict):
        raise ContractError("full training-state consumer omitted optimizer state")
    floating_states = [
        value
        for param_state in state.values()
        if isinstance(param_state, dict)
        for value in param_state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not floating_states or any(value.dtype != torch.float32 for value in floating_states):
        raise ContractError("native FP16 AdamW accumulators are not FP32")
    return {
        "model_parameter_dtype": "torch.float32",
        "optimizer_accumulator_dtype": "torch.float32",
        "floating_model_tensors": max(
            model_evidence["floating_tensor_count"], len(training_model_tensors)
        ),
        "floating_optimizer_tensors": len(floating_states),
    }
