"""Run one isolated export for a completed fit; preserve every attempt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .policy import ACTOR_SHA256
    from .postprocess_common import (
        EVAL_SHAPE_TAIL,
        GPU_BUDGET_BYTES,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _check_disk,
        _child_environment,
        _execute,
        _remaining,
        _sha256,
        _write_new_json,
    )
    from .postprocess_export import _export_argv, _validate_export
except ImportError:
    from policy import ACTOR_SHA256
    from postprocess_common import (
        EVAL_SHAPE_TAIL,
        GPU_BUDGET_BYTES,
        GRADER_SHA256,
        SOURCE_TREE_SHA256,
        _check_disk,
        _child_environment,
        _execute,
        _remaining,
        _sha256,
        _write_new_json,
    )
    from postprocess_export import _export_argv, _validate_export


def _write_failure_receipt(
    attempt_root: Path, fit: dict[str, Any], error: Exception, execution: dict[str, Any] | None
) -> None:
    """Persist failed admission evidence so this export path is never retried."""
    command = attempt_root / "postprocess_command.json"
    stdout, stderr = (
        attempt_root / "postprocess.stdout.txt",
        attempt_root / "postprocess.stderr.txt",
    )
    receipt = attempt_root / "postprocess_receipt.json"
    if receipt.exists():
        return
    evidence: dict[str, Any] = {
        "schema": "tabdata-b300-export-controller-receipt.v1",
        "status": "attempted_failed",
        "run_name": fit["job"]["id"],
        "error_type": type(error).__name__,
        "error": str(error),
        "execution": execution,
        "fit_receipt_sha256": fit["fit_receipt_sha256"],
        "grader_sha256": GRADER_SHA256,
        "source_tree_identity_sha256": SOURCE_TREE_SHA256,
    }
    if command.is_file():
        evidence["controller_command_sha256"] = _sha256(command)
    if stdout.is_file():
        evidence["stdout_sha256"] = _sha256(stdout)
    if stderr.is_file():
        evidence["stderr_sha256"] = _sha256(stderr)
    _write_new_json(receipt, evidence)


def _export_one(
    root: Path, paths: dict[str, Path], fit: dict[str, Any], deadline: float
) -> dict[str, Any]:
    """Make one fresh isolated feature export, or reconcile without retrying."""
    job = fit["job"]
    run_name = str(job["id"])
    attempt_root = root / "exports" / run_name
    output = attempt_root / "export-r1"
    if attempt_root.exists():
        try:
            entry = _validate_export(
                paths, fit, output, attempt_root, require_controller_receipt=True
            )
            entry["output_root"] = output
            entry["reconciled"] = True
            return entry
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            return {
                "run_name": run_name,
                "status": "existing_attempt_preserved",
                "resource_admitted": False,
                "error": str(error),
            }

    attempt_root.mkdir(parents=True)
    tmp_dir = attempt_root / "tmp"
    tmp_dir.mkdir()
    stdout = attempt_root / "postprocess.stdout.txt"
    stderr = attempt_root / "postprocess.stderr.txt"
    execution: dict[str, Any] | None = None
    try:
        _remaining(deadline)
        _check_disk(root)
        argv = _export_argv(paths, fit, output)
        env = _child_environment(root, str(job["gpu_uuid"]), output, grading=False)
        env.update(
            {
                "TMPDIR": str(tmp_dir),
                "WNN_GPU_MEMORY_BYTES": str(GPU_BUDGET_BYTES),
            }
        )
        command = {
            "schema": "tabdata-b300-export-controller-command.v1",
            "argv": argv,
            "env_binding": {
                key: env[key]
                for key in sorted(env)
                if key.startswith(
                    (
                        "WNN_",
                        "OMP_",
                        "MKL_",
                        "OPENBLAS_",
                        "NUMEXPR_",
                        "CUDA_",
                        "NVIDIA_",
                        "TORCH_",
                        "PYTHON",
                    )
                )
            },
            "actor_bundle_sha256": ACTOR_SHA256,
            "grader_sha256": GRADER_SHA256,
            "source_tree_identity_sha256": SOURCE_TREE_SHA256,
            "fit_receipt_sha256": fit["fit_receipt_sha256"],
            "checkpoint_sha256": fit["checkpoint_sha256"],
            "split_manifest_sha256": fit["split_sha256"],
            "expected_shape": [len(fit["groups"]), *EVAL_SHAPE_TAIL],
            "run_name": run_name,
            "fold": job["fold"],
            "seed": job["seed"],
            "gpu_uuid": job["gpu_uuid"],
            "cpu_affinity": sorted(job["cpu_ids_initial"]),
        }
        _write_new_json(attempt_root / "postprocess_command.json", command)
        _check_disk(root)
        execution = _execute(
            argv, env, root, stdout, stderr, deadline, list(job["cpu_ids_initial"])
        )
        if execution.get("exit_code") != 0:
            raise RuntimeError(f"export actor did not return successfully: {execution}")
        entry = _validate_export(paths, fit, output, attempt_root)
        _remaining(deadline)
        receipt = {
            "schema": "tabdata-b300-export-controller-receipt.v1",
            **entry,
            "execution": execution,
            "controller_command_sha256": _sha256(attempt_root / "postprocess_command.json"),
            "stdout_sha256": _sha256(stdout),
            "stderr_sha256": _sha256(stderr),
            "fit_receipt_sha256": fit["fit_receipt_sha256"],
            "grader_sha256": GRADER_SHA256,
            "source_tree_identity_sha256": SOURCE_TREE_SHA256,
        }
        _write_new_json(attempt_root / "postprocess_receipt.json", receipt)
        _remaining(deadline)
        _check_disk(root)
        entry = _validate_export(paths, fit, output, attempt_root, require_controller_receipt=True)
        entry["output_root"] = output
        entry["resource_admitted"] = True
        return entry
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        TimeoutError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        _write_failure_receipt(attempt_root, fit, error, execution)
        return {
            "run_name": run_name,
            "status": "attempted_failed",
            "resource_admitted": False,
            "execution": execution,
            "error": str(error),
        }
