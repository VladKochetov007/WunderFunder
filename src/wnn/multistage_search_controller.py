from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import append_event, atomic_write_json, file_lock, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_runtime import (
    base_environment,
    launch_detached_worker,
    process_identity,
    terminate_owned_process,
)
from wnn.multistage_search_study import open_study

_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def launch_worker(
    run_dir: Path,
    mode: str,
    worker_id: str,
    gpu_key: str,
    trial_number: int | None = None,
    *,
    stage: int = 1,
    result_path: Path | None = None,
) -> dict[str, object]:
    """Detach one bounded profile or fit worker on an admitted GPU slot."""
    if (
        mode not in {"profile", "fit", "teacher-export"}
        or stage not in {1, 2, 3}
        or not _WORKER_ID.fullmatch(worker_id)
    ):
        raise ValueError("worker mode or id is invalid")
    if mode == "teacher-export" and stage not in {1, 2}:
        raise ValueError("teacher-export stage denotes a stage1/2 source trial")
    if result_path is not None:
        if mode != "teacher-export":
            raise ValueError("--result-path is available only for a teacher export")
        exact_result_path = result_path.resolve(strict=True)
        if not exact_result_path.is_relative_to(run_dir.resolve(strict=True)):
            raise ValueError("teacher result path must remain inside the frozen campaign")
        result = read_json(exact_result_path)
        if result.get("status") != "COMPLETE" or result.get("stage") != stage:
            raise ValueError("teacher result path must identify a complete result of this stage")
        if type(result.get("trial_number")) is not int:
            raise ValueError("teacher result path must bind an exact trial number")
        if trial_number is not None and result.get("trial_number") != trial_number:
            raise ValueError("teacher result path and trial number disagree")
        trial_number = int(result["trial_number"])
    elif mode == "teacher-export" and trial_number is None:
        raise ValueError("teacher export requires either an exact trial number or result path")
    run_manifest = read_json(run_dir / "run.json")
    frozen_admission = read_json(Path(str(run_manifest["admission"]["path"])))
    devices = frozen_admission["profile_policy"]["devices"]
    if not isinstance(devices, Mapping) or gpu_key not in {str(key) for key in devices}:
        raise ValueError("worker GPU is not in the admitted measured device set")
    reserved = frozen_admission.get("reserved_gpu_keys", [])
    if not isinstance(reserved, list) or any(not isinstance(value, str) for value in reserved):
        raise ValueError("reserved_gpu_keys must be a list of GPU keys")
    if gpu_key in reserved:
        raise ValueError(f"GPU {gpu_key} is reserved for an already-owned campaign worker")
    device_spec = next(value for key, value in devices.items() if str(key) == gpu_key)
    if not isinstance(device_spec, Mapping) or not isinstance(
        device_spec.get("cuda_visible_devices"), str
    ):
        raise ValueError("admitted GPU needs a CUDA_VISIBLE_DEVICES selector")
    owner_dir = run_dir / "workers"
    owner_path = owner_dir / f"{worker_id}.json"
    if owner_path.exists():
        raise FileExistsError(f"worker id already has an ownership record: {worker_id}")
    with file_lock(run_dir / "launch.lock"):
        for previous in owner_dir.glob("*.json") if owner_dir.exists() else []:
            try:
                record = read_json(previous)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            pid = record.get("pid")
            if str(record.get("gpu_key")) == gpu_key and type(pid) is int:
                identity = process_identity(pid)
                if identity is not None and identity.get("birth") == record.get("birth"):
                    raise RuntimeError(f"GPU {gpu_key} already has an owned live worker")
        repo_root = Path(str(run_manifest["repo_root"]))
        script = repo_root / "scripts" / "multistage_search.py"
        argv = [
            sys.executable,
            str(script),
            "worker",
            "--run-dir",
            str(run_dir.resolve()),
            "--mode",
            mode,
            "--worker-id",
            worker_id,
            "--gpu-key",
            gpu_key,
            "--stage",
            str(stage),
        ]
        if trial_number is not None:
            argv.extend(["--trial-number", str(trial_number)])
        if result_path is not None:
            argv.extend(["--result-path", str(exact_result_path)])
        record = launch_detached_worker(
            argv,
            cwd=repo_root,
            env=base_environment(str(device_spec["cuda_visible_devices"]), source_root=repo_root),
            receipt_path=owner_path,
            metadata={
                "worker_id": worker_id,
                "mode": mode,
                "gpu_key": gpu_key,
                "stage": stage,
                "trial_number": trial_number,
            },
        )
        launch_receipt_path = owner_dir / f"{worker_id}.launch_receipt.json"
        try:
            atomic_write_json(launch_receipt_path, record, replace=False)
        except OSError:
            terminate_owned_process(record, grace_seconds=0.0)
            raise
    append_event(
        run_dir / "events.jsonl",
        {"event": "worker_detached", "worker_id": worker_id, "mode": mode, **record},
    )
    return record


def status_summary(run_dir: Path) -> dict[str, object]:
    """Report durable campaign limits, attempt counts, Optuna states, and owned workers."""
    run_manifest = read_json(run_dir / "run.json")
    budget = read_json(run_dir / "budget.json")
    deadline = read_json(run_dir / "campaign_deadline.json")
    worker_records: list[dict[str, object]] = []
    workers_dir = run_dir / "workers"
    if workers_dir.is_dir():
        for path in sorted(workers_dir.glob("*.json")):
            if path.name.endswith(".launch_receipt.json"):
                continue
            record = read_json(path)
            pid = record.get("pid")
            identity = process_identity(pid) if type(pid) is int else None
            worker_records.append(
                {
                    "path": str(path),
                    "worker_id": record.get("worker_id", path.stem),
                    "mode": record.get("mode"),
                    "stage": record.get("stage", 1),
                    "gpu_key": record.get("gpu_key"),
                    "trial_number": record.get("trial_number"),
                    "status": record.get("status"),
                    "pid": pid,
                    "birth": record.get("birth"),
                    "pgid": record.get("pgid"),
                    "process_alive": identity is not None
                    and identity.get("birth") == record.get("birth"),
                    "reported_status": record.get("status")
                    if identity is not None and identity.get("birth") == record.get("birth")
                    else "external_process_exited"
                    if record.get("external") is True
                    else record.get("status"),
                    "external": record.get("external") is True,
                    "attempt_id": record.get("attempt_id"),
                }
            )
    trial_states: dict[str, int] = {}
    for stage in (1, 2, 3):
        try:
            study = open_study(run_dir, "status", stage=stage)
            for trial in study.get_trials(deepcopy=False):
                state = getattr(getattr(trial, "state", None), "name", "UNKNOWN")
                key = f"stage{stage}.{state}"
                trial_states[key] = trial_states.get(key, 0) + 1
        except (ImportError, RuntimeError, OSError):
            trial_states[f"stage{stage}.unavailable"] = 1
    active_worker = any(item.get("process_alive") is True for item in worker_records)
    if active_worker:
        dynamic_status = "running"
    elif any(key.endswith(".RUNNING") and count for key, count in trial_states.items()):
        dynamic_status = "waiting_for_profile_review_or_fit_admission"
    elif not budget.get("profile_attempts"):
        dynamic_status = "prepared"
    else:
        dynamic_status = "idle_with_durable_results_or_failures"
    return {
        "schema": "wnn.multistage-status.v1",
        "experiment_id": run_manifest["experiment_id"],
        "status": dynamic_status,
        "prepared_status": run_manifest["status"],
        "budget": budget,
        "deadline": deadline,
        "optuna_trial_states": trial_states,
        "owned_workers": worker_records,
    }


def stop_worker(run_dir: Path, worker_id: str) -> dict[str, object]:
    """Stop one process only when its current PID, birth counter, and PGID match its receipt."""
    if not _WORKER_ID.fullmatch(worker_id):
        raise ValueError("worker id is invalid")
    path = run_dir / "workers" / f"{worker_id}.json"
    record = read_json(path)
    if record.get("external") is True:
        raise ValueError("external root-owned workers cannot be stopped by this controller")
    pid = record.get("pid")
    if type(pid) is not int:
        raise ValueError("worker receipt does not bind a numeric PID")
    identity = process_identity(pid)
    if (
        identity is None
        or identity.get("birth") != record.get("birth")
        or identity.get("pgid") != record.get("pgid")
    ):
        return {
            "worker_id": worker_id,
            "status": "already_exited_or_identity_changed",
            "signaled": False,
        }
    signaled = terminate_owned_process(record)
    record.update(
        {"status": "stopped" if signaled else "already_exited", "stop_requested_at_utc": utc_now()}
    )
    atomic_write_json(path, record)
    append_event(
        run_dir / "events.jsonl",
        {"event": "worker_stop_requested", "worker_id": worker_id, "status": record["status"]},
    )
    return {"worker_id": worker_id, "status": record["status"], "signaled": signaled}
