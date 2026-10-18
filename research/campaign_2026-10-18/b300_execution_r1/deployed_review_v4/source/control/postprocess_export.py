"""Frozen epoch-5 feature-only export validation for completed fit cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .policy import EXPECTED_CUDA, EXPECTED_TORCH
    from .postprocess_common import (
        ACTOR_SHA256,
        EVAL_SHAPE_TAIL,
        GPU_BUDGET_BYTES,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _read_json,
        _sha256,
    )
except ImportError:
    from policy import EXPECTED_CUDA, EXPECTED_TORCH
    from postprocess_common import (
        ACTOR_SHA256,
        EVAL_SHAPE_TAIL,
        GPU_BUDGET_BYTES,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _read_json,
        _sha256,
    )


def _export_argv(paths: dict[str, Path], fit: dict[str, Any], output: Path) -> list[str]:
    """Build the fixed feature-only exporter command for one completed fit."""
    job = fit["job"]
    return [
        str(paths["python"]),
        str(paths["code"] / "gpu_entrypoint.py"),
        str(paths["code"] / "export_cell.py"),
        "--source-root",
        str(paths["source"]),
        "--run-dir",
        str(fit["run_dir"]),
        "--export-dir",
        str(output),
        "--split-manifest",
        str(fit["split_path"]),
        "--run-name",
        str(job["id"]),
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--grader-script",
        str(paths["grader"]),
    ]


def _validate_controller_receipt(
    attempt_root: Path,
    paths: dict[str, Path],
    fit: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Require durable successful controller evidence when reconciling old output."""
    job = fit["job"]
    path = attempt_root / "postprocess_receipt.json"
    controller = _read_json(path)
    command_path = attempt_root / "postprocess_command.json"
    command = _read_json(command_path)
    stdout = attempt_root / "postprocess.stdout.txt"
    stderr = attempt_root / "postprocess.stderr.txt"
    execution = controller.get("execution", {})
    environment = command.get("env_binding", {})
    if not isinstance(execution, dict) or not isinstance(environment, dict):
        raise ValueError(f"controller evidence is malformed for {job['id']}")
    execution_stdout, execution_stderr = execution.get("stdout"), execution.get("stderr")
    if not isinstance(execution_stdout, dict) or not isinstance(execution_stderr, dict):
        raise ValueError(f"controller logs are not receipt-bound for {job['id']}")
    if (
        controller.get("status") != "completed_validated"
        or controller.get("run_name") != job["id"]
        or controller.get("fit_receipt_sha256") != fit.get("fit_receipt_sha256")
        or controller.get("grader_sha256") != GRADER_SHA256
        or controller.get("source_tree_identity_sha256") != SOURCE_TREE_SHA256
        or command.get("run_name") != job["id"]
        or command.get("argv") != _export_argv(paths, fit, output)
        or command.get("gpu_uuid") != job.get("gpu_uuid")
        or command.get("cpu_affinity") != sorted(job.get("cpu_ids_initial", []))
        or command.get("actor_bundle_sha256") != ACTOR_SHA256
        or command.get("grader_sha256") != GRADER_SHA256
        or command.get("source_tree_identity_sha256") != SOURCE_TREE_SHA256
        or command.get("fit_receipt_sha256") != fit.get("fit_receipt_sha256")
        or command.get("checkpoint_sha256") != fit.get("checkpoint_sha256")
        or command.get("split_manifest_sha256") != fit.get("split_sha256")
        or command.get("expected_shape") != [len(fit["groups"]), *EVAL_SHAPE_TAIL]
        or environment.get("WNN_RUN_DIR") != str(output)
        or environment.get("WNN_GPU_MEMORY_BYTES") != str(GPU_BUDGET_BYTES)
        or environment.get("CUDA_VISIBLE_DEVICES") != job.get("gpu_uuid")
        or environment.get("WNN_THREADS") != "2"
        or environment.get("WNN_LOADER_WORKERS") != "0"
        or environment.get("WNN_LOADER_RAM_GB") != "1"
        or environment.get("OMP_NUM_THREADS") != "2"
        or environment.get("MKL_NUM_THREADS") != "2"
        or environment.get("OPENBLAS_NUM_THREADS") != "2"
        or environment.get("NUMEXPR_NUM_THREADS") != "2"
        or environment.get("NVIDIA_TF32_OVERRIDE") != "0"
        or environment.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "0"
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
        or controller.get("controller_command_sha256") != _sha256(command_path)
        or controller.get("stdout_sha256") != _sha256(stdout)
        or controller.get("stderr_sha256") != _sha256(stderr)
        or execution_stdout.get("sha256") != controller.get("stdout_sha256")
        or execution_stderr.get("sha256") != controller.get("stderr_sha256")
        or execution.get("exit_code") != 0
    ):
        raise ValueError(f"controller receipt does not bind a successful export for {job['id']}")
    return {
        "execution": execution,
        "controller_command_sha256": _sha256(command_path),
        "controller_receipt_sha256": _sha256(path),
        "controller_stdout": {
            "path": str(stdout),
            "bytes": stdout.stat().st_size,
            "sha256": _sha256(stdout),
        },
        "controller_stderr": {
            "path": str(stderr),
            "bytes": stderr.stat().st_size,
            "sha256": _sha256(stderr),
        },
    }


def _validate_export(
    paths: dict[str, Path],
    fit: dict[str, Any],
    output: Path,
    attempt_root: Path,
    *,
    require_controller_receipt: bool = False,
) -> dict[str, Any]:
    """Verify receipt, command, exact predictor lineage and resource admission."""
    job = fit["job"]
    controller_evidence = None
    if require_controller_receipt:
        controller_evidence = _validate_controller_receipt(attempt_root, paths, fit, output)
    receipt = _read_json(output / "export_receipt.json")
    process = _read_json(output / "export_cell_resource_receipt.json")
    command_path = output / "export_command.json"
    command = _read_json(command_path)
    prediction = output / "cache" / f"pred_{job['id']}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    shape = [len(fit["groups"]), *EVAL_SHAPE_TAIL]
    predictor = paths["source"] / "scripts" / "predict_run.py"
    expected_predictor_argv = [
        str(predictor),
        str(job["id"]),
        "valid",
        "--epoch",
        "5",
        "--campaign-split-manifest",
        str(fit["split_path"]),
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    builder = _read_json(paths["builder"])
    bundle = receipt.get("adapter_bundle_identity")
    deployed_tree = receipt.get("deployed_source_tree")
    if not isinstance(bundle, dict) or not isinstance(deployed_tree, dict):
        raise ValueError(f"export receipt has malformed source identities for {job['id']}")
    if (
        receipt.get("status") != "completed_fixed_epoch5_feature_only_fp32"
        or receipt.get("run_name") != job["id"]
        or receipt.get("fold") != job["fold"]
        or receipt.get("seed") != job["seed"]
        or receipt.get("epoch") != 5
        or receipt.get("prediction_shape") != shape
        or receipt.get("prediction_dtype") != "float32"
        or receipt.get("inference_precision") != "fp32"
        or receipt.get("feature_only") is not True
        or receipt.get("target_reads_during_export") is not False
        or receipt.get("evaluation_row_groups") != fit["groups"]
        or receipt.get("split_manifest_sha256") != fit["split_sha256"]
        or receipt.get("checkpoint_sha256") != fit["checkpoint_sha256"]
        or receipt.get("fit_receipt_sha256") != fit["fit_receipt_sha256"]
        or receipt.get("predictor_sha256") != builder.get("predictor_sha256")
        or receipt.get("r4_grader_sha256") != GRADER_SHA256
        or bundle.get("sha256") != ACTOR_SHA256
        or deployed_tree.get("deployed_tree_identity_sha256") != SOURCE_TREE_SHA256
        or receipt.get("command_sha256") != _sha256(command_path)
    ):
        raise ValueError(f"export receipt does not satisfy the fixed endpoint for {job['id']}")
    if (
        command.get("fit_run_dir") != str(fit["run_dir"])
        or command.get("export_dir") != str(output)
        or command.get("run_name") != job["id"]
        or command.get("split_manifest_sha256") != fit["split_sha256"]
        or command.get("grader_sha256") != GRADER_SHA256
        or command.get("argv") != expected_predictor_argv
        or command.get("predictor_sha256") != builder.get("predictor_sha256")
        or command.get("WNN_GPU_MEMORY_BYTES") != str(GPU_BUDGET_BYTES)
    ):
        raise ValueError(f"export command does not bind the completed fit {job['id']}")
    if (
        process.get("status") != "actor_returned"
        or process.get("script") != "export_cell.py"
        or process.get("torch") != EXPECTED_TORCH
        or process.get("cuda") != EXPECTED_CUDA
        or "B300" not in str(process.get("gpu", ""))
        or process.get("visible_devices") != job.get("gpu_uuid")
        or process.get("cpu_affinity") != sorted(job.get("cpu_ids_initial", []))
        or process.get("allocator_budget_bytes") != GPU_BUDGET_BYTES
        or type(process.get("cuda_peak_allocated_bytes")) is not int
        or type(process.get("cuda_peak_reserved_bytes")) is not int
        or max(process["cuda_peak_allocated_bytes"], process["cuda_peak_reserved_bytes"])
        > GPU_BUDGET_BYTES
    ):
        raise ValueError(
            f"export resource receipt fails the exact-path L4 admission gate for {job['id']}"
        )
    for path, expected in (
        (prediction, receipt.get("prediction_sha256")),
        (sidecar, receipt.get("sidecar_sha256")),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"export output bytes differ from their receipt: {path}")
    result = {
        "status": "completed_validated",
        "run_name": job["id"],
        "arm": job["arm"],
        "fold": job["fold"],
        "seed": job["seed"],
        "gpu_uuid": job["gpu_uuid"],
        "cpu_affinity": sorted(job["cpu_ids_initial"]),
        "prediction_batch_size": 16,
        "inference_chunk_size": 4000,
        "shape": shape,
        "dtype": "float32",
        "prediction": {
            "path": str(prediction),
            "bytes": prediction.stat().st_size,
            "sha256": _sha256(prediction),
        },
        "sidecar": {
            "path": str(sidecar),
            "bytes": sidecar.stat().st_size,
            "sha256": _sha256(sidecar),
        },
        "export_receipt_sha256": _sha256(output / "export_receipt.json"),
        "resource_receipt_sha256": _sha256(output / "export_cell_resource_receipt.json"),
        "fit_receipt_sha256": fit["fit_receipt_sha256"],
        "source_tree_identity_sha256": SOURCE_TREE_SHA256,
        "grader_sha256": GRADER_SHA256,
    }
    if controller_evidence is not None:
        result.update(controller_evidence)
    return result
