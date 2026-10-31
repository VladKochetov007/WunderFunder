from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wnn.multistage_delivery_candidates import select_stage4_candidates
from wnn.multistage_delivery_tasks import (
    FIT_IDS,
    build_postprocess_tasks,
    fit_evidence,
    run_postprocess_task,
)
from wnn.multistage_search import atomic_write_json, file_lock, file_sha256, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner

FIXED_CUTOFF = datetime(2026, 10, 31, 4, 51, tzinfo=UTC).timestamp()


def _deadline(run_dir: Path) -> tuple[float, float]:
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve, hard = deadline.get("reserve_start_epoch"), deadline.get("deadline_epoch")
    if not isinstance(reserve, (int, float)) or not isinstance(hard, (int, float)):
        raise ValueError("campaign deadline is not started and frozen")
    return min(FIXED_CUTOFF, float(reserve) - 1800), float(hard)


def _stage4(
    run_dir: Path,
    runtime: Path,
    operator: Path,
    data_root: Path,
    auxiliary: Path,
    cutoff: float,
    state: dict[str, Any],
    state_path: Path,
    lock: threading.Lock,
    revision: str,
) -> None:
    refs, omitted = select_stage4_candidates(run_dir)
    state.update({"stage4_candidate_refs": refs, "stage4_omitted_categories": omitted})
    if len(refs) < 2 or time.time() >= cutoff:
        state["stage4_status"] = "not_admitted_insufficient_candidates_or_cutoff"
        atomic_write_json(state_path, state)
        return
    output = run_dir / "stage4"
    output.mkdir(exist_ok=True)
    receipt_name = (
        "delivery-stage4-cpu.child.json"
        if revision == "r1"
        else f"delivery-stage4-{revision}-cpu.child.json"
    )
    receipt = runtime / "receipts" / receipt_name
    result = output / "ensemble_result.json"
    runner_request = output / "runner_request.json"
    if runner_request.is_file():
        request = read_json(runner_request)
        if (
            request.get("schema") == "wnn.multistage-stage4-deferral.v1"
            and request.get("scientific_search_started") is False
        ):
            state["stage4_status"] = "deferred_by_frozen_root_marker"
            state["stage4_deferral_marker"] = {
                "path": str(runner_request),
                "sha256": file_sha256(runner_request),
            }
            atomic_write_json(state_path, state)
            return
    if receipt.is_file():
        prior = read_json(receipt)
        if prior.get("status") == "completed" and prior.get("exit_code") == 0 and result.is_file():
            state["stage4_status"] = "complete"
            state["stage4_worker_receipt"] = {
                "path": str(receipt),
                "sha256": file_sha256(receipt),
                "exit_code": 0,
                "timed_out": False,
            }
            state["stage4_result"] = {"path": str(result), "sha256": file_sha256(result)}
            atomic_write_json(state_path, state)
            return
    if receipt.exists() or runner_request.exists() or (output / "ensemble_result.json").exists():
        state["stage4_status"] = "not_retried_existing_attempt"
        atomic_write_json(state_path, state)
        return
    command = [
        sys.executable,
        str(operator / "scripts" / "multistage_stage4_runner.py"),
        "--run-dir",
        str(run_dir),
        "--data-root",
        str(data_root),
        "--auxiliary-split",
        str(auxiliary),
        "--portable-data-manifest",
        str(runtime / "inputs" / "portable-data-r1.json"),
    ]
    for ref in refs:
        command.extend(("--candidate", ref))
    env = os.environ.copy() | {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": os.pathsep.join((str(operator / "src"), str(runtime / "cpu-deps-r1"))),
        "WNN_PORTABLE_DATA_MANIFEST": str(runtime / "inputs" / "portable-data-r1.json"),
        "WNN_PROJECT_ROOT": str(data_root),
    }
    limit = min(1800, int(cutoff - time.time()) - 2)
    if limit < 1:
        state["stage4_status"] = "not_admitted_cutoff_passed"
        atomic_write_json(state_path, state)
        return
    try:
        code, timed_out, _ = run_owned_command(
            command,
            cwd=operator,
            env=env,
            log_path=runtime / "logs" / f"delivery-{revision}-stage4-cpu.log",
            receipt_path=receipt,
            owner=current_process_owner() | {"mode": "stage4_cpu_search"},
            stage=4,
            trial_number=0,
            mode="stage4_cpu_search",
            gpu_key="cpu",
            output_dir=output,
            timeout_seconds=limit,
        )
        state["stage4_status"] = (
            "complete" if code == 0 and not timed_out and result.is_file() else "failed"
        )
        state["stage4_worker_receipt"] = {
            "path": str(receipt),
            "sha256": file_sha256(receipt),
            "exit_code": code,
            "timed_out": timed_out,
        }
        if result.is_file():
            state["stage4_result"] = {"path": str(result), "sha256": file_sha256(result)}
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        state["stage4_status"] = "failed"
        state["stage4_error"] = f"{type(error).__name__}: {error}"
    with lock:
        atomic_write_json(state_path, state)


def run_delivery(
    runtime: Path,
    run_dir: Path,
    operator: Path,
    data_root: Path,
    auxiliary: Path,
    poll_seconds: float = 10.0,
    revision: str = "r1",
) -> dict[str, Any]:
    """Wait for all admitted fit receipts, postprocess once per owned GPU, and run Stage4."""
    if poll_seconds < 1:
        raise ValueError("poll interval must be at least one second")
    if revision not in {"r1", "r2"}:
        raise ValueError("delivery retry revision must be r1 or r2")
    state_path = runtime / "diagnostics" / f"multistage-delivery-{revision}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(run_dir / "delivery-driver.lock", blocking=False):
        state = (
            read_json(state_path)
            if state_path.is_file()
            else {"schema": "wnn.multistage-delivery-driver.v1", "jobs": {}}
        )
        manifest_sha = file_sha256(run_dir / "run.json")
        if state.get("run_manifest_sha256") not in (None, manifest_sha):
            raise ValueError("delivery journal belongs to a different campaign")
        state["run_manifest_sha256"] = manifest_sha
        lock = threading.Lock()
        cutoff, hard = _deadline(run_dir)
        state["stage4_admission_cutoff_epoch"] = cutoff
        state["campaign_hard_deadline_epoch"] = hard
        state["campaign_reserve_start_epoch"] = read_json(run_dir / "campaign_deadline.json")[
            "reserve_start_epoch"
        ]
        stage1_queue = runtime / "requests" / "pending-stage1-finalizers-r2.json"
        if stage1_queue.is_file():
            state["stage1_finalizer_jobs_sha256"] = file_sha256(stage1_queue)
        atomic_write_json(state_path, state)
        while time.time() < cutoff:
            pending = [
                f"stage{s}-trial-{n:06d}"
                for s, n in FIT_IDS
                if not fit_evidence(run_dir, runtime, s, n).get("terminal")
            ]
            queue = runtime / "requests" / "pending-stage1-finalizers-r2.json"
            if not queue.is_file():
                pending.append(str(queue))
            else:
                jobs = read_json(queue).get("jobs")
                if not isinstance(jobs, list) or len(jobs) != 3:
                    pending.append("three_stage1_finalizer_jobs")
            state.update(
                {
                    "status": "waiting_for_fit_receipts" if pending else "postprocessing",
                    "pending_fit_receipts": pending,
                    "updated_at_utc": utc_now(),
                }
            )
            atomic_write_json(state_path, state)
            if not pending:
                break
            time.sleep(max(1, min(poll_seconds, cutoff - time.time())))
        tasks, unresolved = build_postprocess_tasks(runtime, run_dir, operator, cutoff)
        if not (runtime / "requests" / "pending-stage1-finalizers-r2.json").is_file():
            unresolved.append("stage1_finalizer_job_manifest_missing")
        state.update(
            {"pending_fit_receipts_at_cutoff": unresolved, "postprocess_task_count": len(tasks)}
        )
        atomic_write_json(state_path, state)
        queues: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            queues.setdefault(task["gpu"], []).append(task)
        for queue in queues.values():
            queue.sort(key=lambda task: (task["stage"], task["trial"]))

        def run_queue(queue: list[dict[str, Any]]) -> None:
            for task in queue:
                run_postprocess_task(task, runtime, cutoff, state, state_path, lock, revision)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(queues)))) as pool:
            futures = [pool.submit(run_queue, queue) for queue in queues.values()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except (OSError, RuntimeError, ValueError, TypeError) as error:
                    with lock:
                        state.setdefault("driver_errors", []).append(
                            f"{type(error).__name__}: {error}"
                        )
                        atomic_write_json(state_path, state)
        _stage4(
            run_dir,
            runtime,
            operator,
            data_root,
            auxiliary,
            cutoff,
            state,
            state_path,
            lock,
            revision,
        )
        state.update(
            {
                "status": "complete"
                if state.get("stage4_status") == "complete"
                else "postprocess_complete_stage4_incomplete",
                "hard_campaign_deadline_epoch": hard,
                "finished_at_utc": utc_now(),
            }
        )
        atomic_write_json(state_path, state)
        return state
