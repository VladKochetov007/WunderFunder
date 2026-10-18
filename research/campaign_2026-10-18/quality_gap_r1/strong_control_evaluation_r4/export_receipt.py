"""Validate the exact capture receipt, guarded attempt and H390 sidecar."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from export_contract import OUTPUT_NAME, _mapping
from pins import PREDICTOR_SHA256, ROW_GROUPS
from source_alignment import read_pinned_json, require_sha256

RECEIPT_SCHEMA = "quality_gap.f0_strong_control_export_receipt.r1"


def _expected_worker_command(
    contract: dict[str, object], contract_path: Path, contract_sha256: str
) -> list[str]:
    """Reconstruct the exact child command sealed by export_execution_r1."""
    project = Path(str(contract["project_root"]))
    execution = _mapping(contract["execution"], "contract execution")
    resources = _mapping(contract["resource_envelope"], "contract resource envelope")
    runtime = _mapping(contract["runtime"], "contract runtime")
    python = str(execution["python"])
    execution_dir = Path(str(contract["execution_dir"]))
    export_prep = project / "research/campaign_2026-10-18/quality_gap_r1/export_preparation"
    env_values = {
        "WNN_PROJECT_ROOT": str(project),
        "WNN_PORTABLE_DATA_MANIFEST": str(
            project / "research/campaign_2026-10-04/portable_inputs.json"
        ),
        "WNN_F0_COMPATIBILITY_RECEIPT": str(
            export_prep / "compatibility_overlay_r2/COMPATIBILITY_RECEIPT_R2.json"
        ),
        "WNN_F0_MEMBERSHIP_RECEIPT": str(export_prep / "F0_MEMBERSHIP_COMPARISON.json"),
        "PYTHONPATH": str(execution["pythonpath"]),
        "WNN_GPU_MEMORY_BYTES": str(resources["allocator_bytes"]),
        "WNN_GPU_MEMORY_RESERVE_BYTES": str(resources["gpu_free_memory_safety_reserve_bytes"]),
        "WNN_LOADER_WORKERS": "0",
        "WNN_LOADER_RAM_GB": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        "/usr/bin/env",
        *(f"{key}={value}" for key, value in env_values.items()),
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{resources['runtime_limit_seconds']}s",
        python,
        "-B",
        str(execution_dir / "worker.py"),
        "--contract",
        str(contract_path),
        "--contract-sha256",
        contract_sha256,
        "--predictor",
        str(execution["predictor"]),
    ]
    utilities = _mapping(runtime.get("export_utility_code"), "runtime utility hashes")
    for name in sorted(utilities):
        command.extend(["--sealed-source", str(execution_dir / name)])
    return command


def _validate_preinference_paths(
    captured_preinference: dict[str, object],
    preinference: dict[str, object],
    record_dir: Path,
) -> None:
    """Require the archived copy and worker's original evidence to stay distinct."""
    if (
        Path(str(captured_preinference.get("path")))
        != record_dir / "CAPTURED_WORKER_PRE_INFERENCE.json"
        or Path(str(preinference.get("path"))) != record_dir / "WORKER_PRE_INFERENCE.json"
    ):
        raise ValueError("export receipt captured raw evidence paths differ from the contract")


def _read_captured_json(record: dict[str, object], name: str) -> tuple[dict[str, object], bytes]:
    """Read one small captured JSON file after checking its path, size, and digest."""
    path = Path(str(record.get("path", "")))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"captured {name} path must be a regular absolute file")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"captured {name} exceeds the JSON byte bound")
    raw = path.read_bytes()
    if len(raw) != record.get("bytes"):
        raise ValueError(f"captured {name} byte count differs from its receipt")
    digest = hashlib.sha256(raw).hexdigest()
    require_sha256(digest, str(record.get("sha256", "")), f"captured {name}")
    value = json.loads(raw)
    return _mapping(value, f"captured {name}"), raw


def _check_successful_attempt(
    raw_attempt: dict[str, object],
    raw_attempt_bytes: bytes,
    guard: dict[str, object],
    contract: dict[str, object],
    contract_path: Path,
    contract_sha256: str,
) -> None:
    """Bind successful guard evidence to the contract, source inventory, and command."""
    attempt_id = str(contract["attempt_id"])
    reservation = {"cpus": 1, "ram_bytes": 8_000_000_000}
    if (
        hashlib.sha256(raw_attempt_bytes).hexdigest() != guard.get("sha256")
        or len(raw_attempt_bytes) != guard.get("bytes")
        or raw_attempt.get("id") != attempt_id
        or raw_attempt.get("status") != "finished"
        or raw_attempt.get("phase") != "terminal"
        or type(raw_attempt.get("exit_status")) is not int
        or raw_attempt.get("exit_status") != 0
        or raw_attempt.get("failure") not in (None, "")
        or raw_attempt.get("kind") != "gpu"
        or raw_attempt.get("unit") != f"wnn-bandit-{attempt_id}.service"
        or raw_attempt.get("gpu_device") != contract.get("gpu_uuid")
        or raw_attempt.get("reservation") != reservation
        or type(guard.get("exit_status")) is not int
        or raw_attempt.get("contract") != str(contract_path.resolve(strict=True))
        or raw_attempt.get("contract_sha256") != contract_sha256
        or raw_attempt.get("command")
        != _expected_worker_command(contract, contract_path, contract_sha256)
        or guard.get("status") != "finished"
        or guard.get("exit_status") != 0
        or guard.get("unit") != f"wnn-bandit-{attempt_id}.service"
        or guard.get("reservation") != reservation
        or float(guard.get("elapsed_seconds", math.inf)) > 7200
    ):
        raise ValueError("captured guard attempt is not a successful exit-0 contract run")

    resource_profile = _mapping(contract.get("resource_envelope"), "contract resource envelope")
    if (
        raw_attempt.get("profile") != resource_profile.get("profile")
        or raw_attempt.get("gpu_memory_safety_reserve_bytes")
        != resource_profile.get("gpu_free_memory_safety_reserve_bytes")
        or guard.get("elapsed_seconds") != raw_attempt.get("elapsed_seconds")
    ):
        raise ValueError("guard attempt profile differs from the export contract")
    for key in (
        "final_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    ):
        deltas = _mapping(raw_attempt.get(key), f"guard attempt {key}")
        if any(
            type(deltas.get(event)) is not int or deltas[event] != 0
            for event in ("oom", "oom_kill", "oom_group_kill")
        ):
            raise ValueError(f"guard attempt OOM counters are missing or nonzero: {key}")
        if guard.get(key) != deltas:
            raise ValueError(f"export receipt guard summary differs from raw attempt: {key}")

    sources = _mapping(raw_attempt.get("source_sha256"), "guard source hashes")
    if sources.get("frozen_source_snapshot/scripts/predict_run.py") != PREDICTOR_SHA256:
        raise ValueError("guard attempt does not bind the frozen H390 predictor")
    runtime = _mapping(contract.get("runtime"), "contract runtime")
    execution = _mapping(contract.get("execution"), "contract execution")
    for path_key, hash_key in (
        ("guard", "bandit_resources_sha256"),
        ("limited", "limited_sha256"),
    ):
        expected_path = str(Path(str(execution[path_key])))
        if sources.get(expected_path) != runtime.get(hash_key):
            raise ValueError(
                f"guard attempt does not bind the pinned guard or limited source: {path_key}"
            )
    expected_source_manifest_sha256 = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if raw_attempt.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("guard attempt source manifest digest is inconsistent")
    utilities = _mapping(runtime.get("export_utility_code"), "runtime utility hashes")
    execution_dir = Path(str(contract["execution_dir"]))
    project = Path(str(contract["project_root"]))
    for name, value in utilities.items():
        identity = _mapping(value, f"runtime utility {name}")
        relative = str((execution_dir / name).relative_to(project))
        if sources.get(relative) != identity.get("sha256"):
            raise ValueError(f"guard attempt does not bind exporter utility source: {name}")


def validate_receipt(
    receipt_path: Path,
    expected_sha256: str,
    contract_path: Path,
    contract: dict[str, object],
    contract_sha256: str,
    prediction_path: Path,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    prediction_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Check the captured export receipt, raw guard exit, sidecar and runtime evidence."""
    if receipt_path.is_symlink():
        raise ValueError("export receipt must be a regular file")
    receipt = read_pinned_json(receipt_path, expected_sha256, "export receipt")
    attempt_id = str(contract["attempt_id"])
    contract_meta = _mapping(receipt.get("contract"), "receipt contract")
    prediction_meta = _mapping(receipt.get("prediction"), "receipt prediction")
    sidecar_meta = _mapping(receipt.get("sidecar"), "receipt sidecar")
    ordered_rows = _mapping(receipt.get("ordered_rows"), "receipt ordered rows")
    guard = _mapping(receipt.get("guard_attempt"), "receipt guard attempt")
    captured = _mapping(receipt.get("captured_raw_evidence"), "receipt captured raw evidence")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "PASS_STRONG_CONTROL_FEATURE_EXPORT_CAPTURE"
        or receipt.get("owner") != "root"
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("output_name") != OUTPUT_NAME
        or receipt.get("source_pins") != contract.get("pins")
        or receipt.get("execution") != contract.get("execution")
    ):
        raise ValueError("export receipt schema, status, attempt, output or source pins mismatch")
    if (
        Path(str(contract_meta.get("path"))) != contract_path.resolve(strict=True)
        or contract_meta.get("sha256") != contract_sha256
        or contract_meta.get("bytes") != contract_path.stat().st_size
        or contract.get("contract_path") != str(contract_path.resolve(strict=True))
        or receipt.get("preflight_receipt") != contract.get("preflight_receipt")
        or receipt.get("root_admission") != contract.get("root_admission")
        or receipt.get("gpu_uuid") != contract.get("gpu_uuid")
    ):
        raise ValueError("export receipt does not bind the exact prepared contract")
    preflight = _mapping(contract.get("preflight_receipt"), "contract preflight receipt")
    pins = _mapping(contract.get("pins"), "contract pins")
    if preflight.get("status") != "PASS_CPU_LOADER_PREFLIGHT" or pins.get(
        "r2_receipt_sha256"
    ) != preflight.get("sha256"):
        raise ValueError("contract preflight receipt identity is absent or inconsistent")
    if (
        Path(str(prediction_meta.get("path"))) != prediction_path.resolve(strict=True)
        or prediction_meta.get("bytes") != prediction_path.stat().st_size
        or prediction_meta.get("sha256") != prediction_sha256
        or prediction_meta.get("shape") != [ROW_GROUPS, 20_000, 2]
        or prediction_meta.get("dtype") != "float32"
        or prediction_meta.get("bytes") != 299_680_128
        or Path(str(sidecar_meta.get("path"))) != sidecar_path.resolve(strict=True)
        or sidecar_meta.get("bytes") != sidecar_path.stat().st_size
        or sidecar_meta.get("sha256") != expected_sidecar_sha256
    ):
        raise ValueError("export receipt prediction or sidecar bytes differ from supplied files")
    if (
        ordered_rows.get("count") != ROW_GROUPS
        or not isinstance(ordered_rows.get("ordered_int64_le_sha256"), str)
        or ordered_rows.get("first") is None
        or ordered_rows.get("last") is None
    ):
        raise ValueError("export receipt lacks the explicit full-valid row-order digest")
    expected_scope = {
        "feature_only_full_valid": True,
        "training": False,
        "grading": False,
        "metric_computed": False,
        "evaluation_only_export": False,
    }
    if receipt.get("scope") != expected_scope:
        raise ValueError("export receipt does not claim the precise feature-only no-training scope")
    if float(guard.get("elapsed_seconds", math.inf)) > 7200:
        raise ValueError("export receipt guard attempt exceeded its runtime cap")

    raw_attempt, raw_attempt_bytes = _read_captured_json(
        _mapping(captured.get("bandit_attempt"), "captured guard attempt"), "guard attempt"
    )
    _check_successful_attempt(
        raw_attempt, raw_attempt_bytes, guard, contract, contract_path, contract_sha256
    )
    captured_attempt = _mapping(captured.get("bandit_attempt"), "captured guard attempt")
    raw_sidecar, raw_sidecar_bytes = _read_captured_json(
        _mapping(captured.get("prediction_sidecar"), "captured sidecar"), "prediction sidecar"
    )
    del raw_sidecar
    captured_sidecar = _mapping(captured.get("prediction_sidecar"), "captured sidecar")
    if (
        hashlib.sha256(raw_sidecar_bytes).hexdigest() != expected_sidecar_sha256
        or hashlib.sha256(raw_sidecar_bytes).hexdigest() != sidecar_meta.get("sha256")
        or captured_sidecar.get("sha256") != sidecar_meta.get("sha256")
        or captured_sidecar.get("bytes") != sidecar_meta.get("bytes")
    ):
        raise ValueError("captured raw sidecar does not match the supplied sidecar bytes")

    raw_preinference, raw_preinference_bytes = _read_captured_json(
        _mapping(captured.get("pre_inference_runtime"), "captured pre-inference runtime"),
        "pre-inference runtime",
    )
    captured_preinference = _mapping(
        captured.get("pre_inference_runtime"), "captured pre-inference runtime"
    )
    preinference = _mapping(receipt.get("pre_inference_runtime"), "receipt pre-inference runtime")
    if (
        hashlib.sha256(raw_preinference_bytes).hexdigest() != preinference.get("sha256")
        or len(raw_preinference_bytes) != preinference.get("bytes")
        or captured_preinference.get("sha256") != preinference.get("sha256")
        or captured_preinference.get("bytes") != preinference.get("bytes")
        or raw_preinference.get("schema") != "quality_gap.strong_control_pre_inference_runtime.r1"
        or raw_preinference.get("status") != "PRE_INFERENCE_SETTINGS_VERIFIED"
        or raw_preinference.get("attempt_id") != attempt_id
        or raw_preinference.get("predictor_sha256") != PREDICTOR_SHA256
        or raw_preinference.get("wnn_gpu_device") != contract.get("gpu_uuid")
        or raw_preinference.get("inference_precision") != "fp32"
        or raw_preinference.get("cuda_matmul_allow_tf32") is not False
        or raw_preinference.get("cudnn_allow_tf32") is not False
        or raw_preinference.get("cudnn_benchmark") is not False
        or raw_preinference.get("cudnn_deterministic") is not True
        or raw_preinference.get("float32_matmul_precision") != "highest"
        or raw_preinference.get("nvidia_tf32_override") != "0"
        or raw_preinference.get("cuda_visible_devices") != contract.get("gpu_uuid")
        or preinference.get("torch_version") != raw_preinference.get("torch_version")
        or preinference.get("torch_cuda_version") != raw_preinference.get("torch_cuda_version")
        or preinference.get("cuda_matmul_allow_tf32")
        != raw_preinference.get("cuda_matmul_allow_tf32")
        or preinference.get("cudnn_allow_tf32") != raw_preinference.get("cudnn_allow_tf32")
        or preinference.get("cudnn_benchmark") != raw_preinference.get("cudnn_benchmark")
        or preinference.get("cudnn_deterministic") != raw_preinference.get("cudnn_deterministic")
        or preinference.get("float32_matmul_precision")
        != raw_preinference.get("float32_matmul_precision")
        or preinference.get("cuda_visible_devices") != raw_preinference.get("cuda_visible_devices")
    ):
        raise ValueError("captured strict-FP32 pre-inference evidence is invalid")
    execution_dir = Path(str(contract["execution_dir"]))
    record_dir = execution_dir / "records" / attempt_id
    if (
        Path(str(captured_attempt["path"])) != record_dir / "BANDIT_ATTEMPT.json"
        or Path(str(captured_sidecar["path"])) != record_dir / "PREDICTION_SIDECAR.json"
    ):
        raise ValueError("export receipt captured raw evidence paths differ from the contract")
    _validate_preinference_paths(captured_preinference, preinference, record_dir)
    execution = _mapping(contract.get("execution"), "contract execution")
    guard_state = str(execution["guard_state"])
    if Path(str(guard.get("path"))) != Path(guard_state) / "attempts" / f"{attempt_id}.json":
        raise ValueError("export receipt original guard-attempt path differs from the contract")
    row_hash = str(ordered_rows["ordered_int64_le_sha256"])
    if len(row_hash) != 64 or any(char not in "0123456789abcdef" for char in row_hash):
        raise ValueError("export receipt row-order digest is not lowercase SHA-256")
    return receipt, {
        "contract_sha256": contract_sha256,
        "export_receipt_sha256": expected_sha256,
        "attempt_id": attempt_id,
        "raw_guard_attempt_sha256": str(guard["sha256"]),
        "ordered_row_groups_sha256": row_hash,
        "captured_sidecar_sha256": expected_sidecar_sha256,
        "raw_guard_attempt_path": str(captured_attempt["path"]),
        "captured_sidecar_path": str(captured_sidecar["path"]),
        "raw_preinference_sha256": str(preinference["sha256"]),
        "captured_preinference_path": str(captured_preinference["path"]),
    }
