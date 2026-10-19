"""One-shot fit validation and fixed-epoch export phases."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from delivery_watch_core import (
    EXPECTED_DEADLINE_UNIX,
    EXPORT_TIMEOUT_SECONDS,
    POLL_SECONDS,
    VALIDATE_TIMEOUT_SECONDS,
    actor_still_running,
    append_event,
    fresh_attempt_allowed,
    read_object,
    remaining,
    sha256_file,
)
from delivery_watch_receipts import verify_export_receipt, verify_fit_receipt
from delivery_watch_state import _base_cli, _cell_write, _run_child


def _fit_one(
    config: dict[str, Any],
    state: Path,
    recovery: Path,
    job: dict[str, Any],
    cell: dict[str, Any],
    stable: dict[str, tuple[str, float]],
    delivery: Any,
    placement_verifier: Any,
) -> None:
    """Wait for a final train receipt, then validate the existing fit once."""
    job_id = str(job["id"])
    run_dir = Path(job["run_dir"])
    fit_receipt_path = recovery / "receipts" / f"{job_id}.json"
    view = recovery / "fit_views" / job_id
    if fit_receipt_path.exists():
        try:
            receipt = verify_fit_receipt(
                delivery,
                recovery,
                config["packet_path"],
                config["root_path"],
                job,
                placement_verifier,
            )
            _cell_write(
                state,
                cell,
                "fit",
                status="validated",
                receipt_sha256=sha256_file(fit_receipt_path),
                actor_phase=receipt.get("original_actor_phase"),
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            quality_status = "not_scored"
            producer_path = run_dir / "producer_receipt.json"
            if producer_path.is_file() and not producer_path.is_symlink():
                try:
                    producer_status = read_object(producer_path).get("status")
                except (OSError, ValueError, TypeError):
                    producer_status = None
                if producer_status == "trainer_failed":
                    quality_status = "not_evaluable_training_failure"
            _cell_write(
                state,
                cell,
                "fit",
                status="rejected",
                error_type=type(error).__name__,
                error=str(error),
                quality_status=quality_status,
            )
        return
    if not fresh_attempt_allowed(
        fit_receipt_path.exists(), view.exists(), cell.get("fit", {}).get("status", "pending")
    ):
        _cell_write(
            state,
            cell,
            "fit",
            status="preserved_partial",
            reason="prior validation attempt has no verified receipt; no retry",
        )
        return
    resource_path = run_dir / "train_cell_resource_receipt.json"
    if not resource_path.is_file() or resource_path.is_symlink():
        return
    try:
        resource = read_object(resource_path)
        digest = sha256_file(resource_path)
        previous, since = stable.get(job_id, ("", time.time()))
        if digest != previous:
            stable[job_id] = (digest, time.time())
            return
        if time.time() - since < POLL_SECONDS or actor_still_running(resource):
            return
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        append_event(
            state / "events.jsonl",
            {
                "time_unix": time.time(),
                "job_id": job_id,
                "phase": "fit_wait",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return
    _cell_write(state, cell, "fit", status="validation_started", resource_sha256=digest)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        WNN_PROJECT_ROOT=str(config["root_path"]),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
    )
    argv = _base_cli(config, recovery, "validate") + ["--job-id", job_id]
    try:
        result = _run_child(
            argv,
            env,
            state / "logs" / f"validate-{job_id}.log",
            min(VALIDATE_TIMEOUT_SECONDS, remaining(EXPECTED_DEADLINE_UNIX)),
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"fit validation exited {result['returncode']}")
        receipt = verify_fit_receipt(
            delivery,
            recovery,
            config["packet_path"],
            config["root_path"],
            job,
            placement_verifier,
        )
        _cell_write(
            state,
            cell,
            "fit",
            status="validated",
            receipt_sha256=sha256_file(fit_receipt_path),
            actor_phase=receipt.get("original_actor_phase"),
            validator_log_sha256=result["log_sha256"],
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        RuntimeError,
        TimeoutError,
    ) as error:
        quality_status = "not_scored"
        producer_path = run_dir / "producer_receipt.json"
        if producer_path.is_file() and not producer_path.is_symlink():
            try:
                producer_status = read_object(producer_path).get("status")
            except (OSError, ValueError, TypeError):
                producer_status = None
            if producer_status == "trainer_failed":
                quality_status = "not_evaluable_training_failure"
        _cell_write(
            state,
            cell,
            "fit",
            status="rejected",
            error_type=type(error).__name__,
            error=str(error),
            quality_status=quality_status,
        )


def _export_one(
    config: dict[str, Any],
    state: Path,
    recovery: Path,
    job: dict[str, Any],
    cell: dict[str, Any],
    delivery: Any,
    placement_verifier: Any,
) -> None:
    """Export one verified fixed-epoch checkpoint; never retry an existing attempt."""
    job_id = str(job["id"])
    fit_path = recovery / "receipts" / f"{job_id}.json"
    export_path = recovery / "receipts" / f"{job_id}.export.json"
    output = recovery / "exports" / job_id
    if cell.get("fit", {}).get("status") != "validated":
        if cell.get("fit", {}).get("status") in {"rejected", "preserved_partial"}:
            _cell_write(
                state, cell, "export", status="not_applicable", reason="fit did not validate"
            )
        return
    if export_path.exists():
        try:
            fit = verify_fit_receipt(
                delivery,
                recovery,
                config["packet_path"],
                config["root_path"],
                job,
                placement_verifier,
            )
            verify_export_receipt(
                recovery, config["root_path"], job_id, fit, str(config["gpu_uuid"])
            )
            _cell_write(
                state, cell, "export", status="exported", receipt_sha256=sha256_file(export_path)
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            _cell_write(
                state,
                cell,
                "export",
                status="rejected",
                error_type=type(error).__name__,
                error=str(error),
            )
        return
    if not fresh_attempt_allowed(
        export_path.exists(), output.exists(), cell.get("export", {}).get("status", "pending")
    ):
        _cell_write(
            state,
            cell,
            "export",
            status="preserved_partial",
            reason="prior export attempt has no verified receipt; no retry",
        )
        return
    _cell_write(
        state, cell, "export", status="export_started", fit_receipt_sha256=sha256_file(fit_path)
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
        WNN_PROJECT_ROOT=str(config["root_path"]),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
    )
    argv = _base_cli(config, recovery, "export") + ["--job-id", job_id]
    try:
        result = _run_child(
            argv,
            env,
            state / "logs" / f"export-{job_id}.log",
            min(EXPORT_TIMEOUT_SECONDS, remaining(EXPECTED_DEADLINE_UNIX)),
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"export exited {result['returncode']}")
        fit = verify_fit_receipt(
            delivery,
            recovery,
            config["packet_path"],
            config["root_path"],
            job,
            placement_verifier,
        )
        verify_export_receipt(recovery, config["root_path"], job_id, fit, str(config["gpu_uuid"]))
        _cell_write(
            state,
            cell,
            "export",
            status="exported",
            receipt_sha256=sha256_file(export_path),
            actor_log_sha256=result["log_sha256"],
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        RuntimeError,
        TimeoutError,
    ) as error:
        _cell_write(
            state,
            cell,
            "export",
            status="rejected",
            error_type=type(error).__name__,
            error=str(error),
        )
