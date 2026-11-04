from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import current_process_owner

FIT_IDS = [(2, n) for n in range(1, 7)] + [(3, n) for n in range(12)]


def fit_evidence(run_dir: Path, runtime: Path, stage: int, number: int) -> dict[str, Any]:
    """Return terminal fit evidence and eligibility for fixed-final postprocessing."""
    trial = run_dir / f"stage{stage}" / "trials" / f"{number:06d}"
    request_path = trial / "trial_request.json"
    receipt_path = runtime / "receipts" / f"stage{stage}-trial-{number:06d}.child.json"
    if not request_path.is_file() or not receipt_path.is_file():
        return {"terminal": False}
    request, receipt = read_json(request_path), read_json(receipt_path)
    attempt = f"stage{stage}-trial-{number:06d}"
    rows = [
        r
        for r in read_json(run_dir / "budget.json").get("fit_attempts", [])
        if isinstance(r, dict) and r.get("attempt_id") == attempt
    ]
    budget = rows[0] if len(rows) == 1 else {}
    terminal = (
        receipt.get("status") not in {"running", "launching"} and budget.get("status") != "reserved"
    )
    output = run_dir / f"stage{stage}" / "runs" / f"{number:06d}"
    eligible = (
        receipt.get("schema") == "wnn.multistage-owned-process.v1"
        and receipt.get("stage") == stage
        and receipt.get("trial_number") == number
        and receipt.get("status") == "completed"
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
        and receipt.get("mode") in {"fit", f"stage{stage}_fit"}
        and receipt.get("gpu_key") == request.get("gpu_key")
        and Path(str(receipt.get("output_dir", ""))).resolve() == output.resolve()
        and budget.get("status") in {"complete", "completed"}
        and budget.get("exit_code") == 0
        and request.get("primary_valid_used") is False
        and request.get("objective_scope") == "inner_train_proxy"
    )
    return {
        "terminal": terminal,
        "eligible": eligible,
        "request": request,
        "request_sha256": file_sha256(request_path),
        "result": trial / "trial_result.json",
    }


def _stage1_tasks(runtime: Path, operator: Path) -> list[dict[str, Any]]:
    path = runtime / "requests" / "pending-stage1-finalizers-r2.json"
    if not path.is_file():
        return []
    jobs = read_json(path).get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 3:
        raise ValueError("Stage1 finalizer list must contain exactly three reviewed jobs")
    tasks = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError("Stage1 finalizer job must be an object")
        command, env = job.get("command"), job.get("environment")
        result = Path(str(job.get("output_result_path", "")))
        if (
            not isinstance(command, list)
            or not command
            or not isinstance(env, dict)
            or not result.is_absolute()
        ):
            raise ValueError("Stage1 finalizer job requires argv, environment, and absolute output")
        match = re.search(r"trial[-_ ]?0*(\d+)", " ".join(command), re.I)
        number = int(match.group(1)) if match else index
        if not result.is_file():
            tasks.append(
                {
                    "id": f"stage1-finalizer-{number:06d}",
                    "stage": 1,
                    "trial": number,
                    "gpu": str(job["gpu_key"]),
                    "command": command,
                    "env": os.environ.copy() | env,
                    "cwd": str(job.get("cwd", operator)),
                    "output": str(result.parent),
                    "result": str(result),
                    "log": str(job["log_path"]),
                    "mode": "stage1_fixed_finalization",
                }
            )
    return tasks


def build_postprocess_tasks(
    runtime: Path, run_dir: Path, operator: Path, cutoff: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create tasks only for terminal exit-zero fits, preserving their original GPU assignment."""
    tasks = _stage1_tasks(runtime, operator)
    if not isinstance(read_json(run_dir / "run.json").get("evaluation_source_root"), str):
        raise ValueError("run manifest omits pinned evaluation source")
    for stage, number in FIT_IDS:
        evidence = fit_evidence(run_dir, runtime, stage, number)
        if (
            not evidence.get("terminal")
            or not evidence.get("eligible")
            or Path(evidence["result"]).exists()
        ):
            continue
        request = evidence["request"]
        devices = request.get("requested_device_set")
        if (
            not isinstance(devices, list)
            or len(devices) != 1
            or not isinstance(request.get("gpu_key"), str)
        ):
            continue
        fit_dir = run_dir / f"stage{stage}" / "runs" / f"{number:06d}"
        finalizer = operator / "scripts" / "finalize_reviewed_successor_fit.py"
        if not finalizer.is_file():
            raise FileNotFoundError(finalizer)
        outer = min(900, max(1, int(cutoff - time.time()) - 5))
        command = [
            sys.executable,
            str(finalizer),
            "--run-dir",
            str(run_dir),
            "--stage",
            str(stage),
            "--trial-number",
            str(number),
            "--timeout-seconds",
            str(max(1, outer - 15)),
            "--python",
            sys.executable,
        ]
        env = os.environ.copy() | {
            "CUDA_VISIBLE_DEVICES": str(devices[0]),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONPATH": os.pathsep.join((str(operator / "src"), str(runtime / "python-deps"))),
            "WNN_PORTABLE_DATA_MANIFEST": str(runtime / "inputs" / "portable-data-r1.json"),
            "WNN_PROJECT_ROOT": str(Path(str(request["data_root"])).resolve(strict=True)),
        }
        result = Path(evidence["result"])
        tasks.append(
            {
                "id": f"stage{stage}-trial-{number:06d}",
                "stage": stage,
                "trial": number,
                "gpu": request["gpu_key"],
                "command": command,
                "env": env,
                "cwd": str(operator),
                "output": str(fit_dir / "fixed_final_postprocess_r2"),
                "result": str(result),
                "log": str(runtime / "logs" / f"delivery-stage{stage}-trial-{number:06d}.log"),
                "mode": "successor_fixed_finalization",
                "request_sha256": evidence["request_sha256"],
            }
        )
    pending = [
        f"stage{s}-trial-{n:06d}"
        for s, n in FIT_IDS
        if not fit_evidence(run_dir, runtime, s, n).get("terminal")
    ]
    return tasks, pending


def run_postprocess_task(
    task: dict[str, Any],
    runtime: Path,
    cutoff: float,
    state: dict[str, Any],
    state_path: Path,
    lock: threading.Lock,
    revision: str = "r1",
) -> None:
    """Run one owned postprocessor once, with its fit-specific cutoff and durable receipt."""
    ident, result = task["id"], Path(task["result"])
    prior = state.setdefault("jobs", {}).get(ident, {})
    if result.is_file():
        with lock:
            state["jobs"][ident] = {
                "status": "complete",
                "result": str(result),
                "result_sha256": file_sha256(result),
            }
            atomic_write_json(state_path, state)
        return
    if prior.get("status") in {
        "complete",
        "failed",
        "skipped_cutoff",
        "not_retried_existing_attempt",
    }:
        return
    if revision not in {"r1", "r2"}:
        raise ValueError("delivery retry revision must be r1 or r2")
    receipt_name = (
        f"delivery-{ident}.child.json"
        if revision == "r1"
        else f"delivery-{revision}-{ident}.child.json"
    )
    receipt = runtime / "receipts" / receipt_name
    if receipt.is_file():
        with lock:
            state["jobs"][ident] = {
                "status": "not_retried_existing_attempt",
                "worker_receipt": str(receipt),
            }
            atomic_write_json(state_path, state)
        return
    remaining = int(cutoff - time.time())
    if remaining < 20:
        with lock:
            state["jobs"][ident] = {"status": "skipped_cutoff"}
            atomic_write_json(state_path, state)
        return
    timeout = min(900, remaining - 5)
    command = list(task["command"])
    if task["stage"] in (2, 3) and "--timeout-seconds" in command:
        command[command.index("--timeout-seconds") + 1] = str(max(1, timeout - 15))
    with lock:
        state["jobs"][ident] = {
            "status": "running",
            "gpu_key": task["gpu"],
            "trial_request_sha256": task.get("request_sha256"),
            "started_at_utc": utc_now(),
        }
        atomic_write_json(state_path, state)
    try:
        code, timed_out, record = run_owned_command(
            command,
            cwd=Path(task["cwd"]),
            env=task["env"],
            log_path=(
                Path(task["log"])
                if revision == "r1"
                else runtime / "logs" / f"delivery-{revision}-{ident}.log"
            ),
            receipt_path=receipt,
            owner=current_process_owner() | {"mode": task["mode"]},
            stage=task["stage"],
            trial_number=task["trial"],
            mode=task["mode"],
            gpu_key=task["gpu"],
            output_dir=Path(task["output"]),
            timeout_seconds=timeout,
        )
        status = "complete" if code == 0 and not timed_out and result.is_file() else "failed"
        value = {
            "status": status,
            "exit_code": code,
            "timed_out": timed_out,
            "worker_receipt": str(receipt),
            "worker_receipt_sha256": file_sha256(receipt),
            "result": str(result) if result.is_file() else None,
            "result_sha256": file_sha256(result) if result.is_file() else None,
            "owner_status": record.get("status"),
            "trial_request_sha256": task.get("request_sha256"),
        }
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        value = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    with lock:
        state["jobs"][ident] = value
        atomic_write_json(state_path, state)
