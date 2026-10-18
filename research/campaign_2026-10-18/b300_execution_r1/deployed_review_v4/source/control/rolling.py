"""One admission authority for fixed GPU lanes, export, and paired grading."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fit_stage import _validate_fit_receipt
from journal import Journal, write_json_fsync
from policy import sha256_file
from postprocess_common import _runtime_paths
from postprocess_export_run import _export_one
from postprocess_fit import _fit_binding
from postprocess_grade import _grade_pair
from process_start import Running, start_owned, terminate_owned
from profiles import actor_environment
from runtime_gates import cgroup_snapshot, require_gpus

from runtime import MEMORY_SOFT_LIMIT, _monitor


def run_rolling(
    panel: dict, root: Path, state: Path, journal: Journal, deadline: float
) -> dict:
    """Keep at most one actor per frozen GPU, retaining all source-bound endpoints."""
    jobs = panel["jobs"]
    by_id = {j["id"]: j for j in jobs}
    paths = _runtime_paths(root, jobs)
    # Heterogeneous controller receipts are JSON dictionaries from fixed adapters.
    progress: dict[str, Any] = {
        "pending": list(jobs),
        "handled": set(),
        "records": {},
        "fits": {},
        "exports": {},
        "grades": {},
        "stop": False,
        "first_export_passed": False,
    }
    running: list[Running] = []
    baseline = cgroup_snapshot()["memory_events"]

    def save_progress() -> None:
        write_json_fsync(
            state / "PROGRESS.json",
            {
                "fits": list(progress["records"].values()),
                "pending": len(progress["pending"]),
                "grades": list(progress["grades"].values()),
                "new_admissions_stopped": progress["stop"],
            },
        )

    def finish_one(name: str, result: dict) -> None:
        job = by_id[name]
        record = {"job_id": name, "process": result, "status": result["status"]}
        progress["records"][name] = record
        if result["status"] != "completed" or result["returncode"] != 0:
            progress["stop"] = True
            journal.record("fit_failed", job_id=name, process=result)
            return
        _validate_fit_receipt(job)
        record.update(
            status="fixed_epoch5_receipt_verified",
            fit_receipt_sha256=sha256_file(
                Path(job["run_dir"]) / "producer_receipt.json"
            ),
        )
        fit = _fit_binding(root, job)
        progress["fits"][name] = fit
        journal.record("fit_receipt_verified", job_id=name)
        if progress["stop"] and not progress["first_export_passed"]:
            record["export_status"] = "awaiting_first_export_resource_repair"
            return
        journal.record("export_start", job_id=name)
        limit = time.time() + min(1800, max(0, deadline - time.monotonic()))
        exported = _export_one(root, paths, fit, limit)
        admitted = bool(exported.pop("resource_admitted", False))
        record["export_status"] = exported.get("status")
        if not admitted:
            progress["stop"] = True
            journal.record(
                "export_not_admitted", job_id=name, status=exported.get("status")
            )
            return
        progress["first_export_passed"] = True
        progress["exports"][name] = exported
        journal.record("export_verified", job_id=name)
        key = (job["fold"], job["seed"])
        pair_jobs = [j for j in jobs if (j["fold"], j["seed"]) == key]
        if key in progress["grades"] or not all(
            j["id"] in progress["exports"] for j in pair_jobs
        ):
            return
        fits = {j["arm"]: progress["fits"][j["id"]] for j in pair_jobs}
        exports = {j["arm"]: progress["exports"][j["id"]] for j in pair_jobs}
        journal.record("grade_start", fold=key[0], seed=key[1])
        limit = time.time() + min(1800, max(0, deadline - time.monotonic()))
        grade = _grade_pair(root, paths, key[0], fits, exports, limit)
        progress["grades"][key] = grade
        if grade["status"] != "graded":
            progress["stop"] = True
        journal.record(
            "grade_finished", fold=key[0], seed=key[1], status=grade["status"]
        )

    def advance(active: list[Running], results: dict[str, dict]) -> None:
        for name, result in results.items():
            if name in progress["handled"]:
                continue
            progress["handled"].add(name)
            try:
                finish_one(name, result)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                progress["stop"] = True
                progress["records"].setdefault(name, {"job_id": name}).update(
                    delivery_error_type=type(error).__name__, delivery_error=str(error)
                )
                journal.record(
                    "delivery_failed",
                    job_id=name,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            save_progress()
        if progress["stop"]:
            return
        busy = {task.gpu_uuid for task in active if task.status == "running"}
        for job in list(progress["pending"]):
            if job["gpu_uuid"] in busy:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Campaign admission deadline expired")
            if cgroup_snapshot()["memory_current"] >= MEMORY_SOFT_LIMIT:
                raise RuntimeError("Unit memory admission ceiling reached")
            require_gpus([job], job["gpu_uuid"])
            run = Path(job["run_dir"])
            if run.exists() or run.is_symlink():
                raise FileExistsError(run)
            run.mkdir(parents=True)
            env = actor_environment(root, run, job["gpu_uuid"])
            active.append(
                start_owned(
                    name=job["id"],
                    argv=job["argv"],
                    environment={**env, "WNN_CWD": str(root)},
                    cpus=tuple(job["cpu_ids_initial"]),
                    log_path=state / "logs" / f"{job['id']}.log",
                    timeout_seconds=min(
                        panel["max_fit_seconds"], max(0, deadline - time.monotonic())
                    ),
                )
            )
            progress["pending"].remove(job)
            busy.add(job["gpu_uuid"])
            journal.record(
                "fit_started",
                job_id=job["id"],
                gpu_uuid=job["gpu_uuid"],
                fold=job["fold"],
                seed=job["seed"],
                arm=job["arm"],
                pid=active[-1].process.pid,
            )
        save_progress()

    try:
        advance(running, {})
        _monitor(
            running,
            journal,
            root,
            deadline,
            baseline,
            [root / "runs", root / "exports", root / "grades", root / "scratch", state],
            advance=advance,
        )
    except BaseException:
        for task in running:
            if task.status == "running":
                terminate_owned(task.process)
            if not task.log.closed:
                task.log.close()
        raise
    records = [
        progress["records"].get(j["id"], {"job_id": j["id"], "status": "not_admitted"})
        for j in jobs
    ]
    grades = list(progress["grades"].values())
    complete = (
        len(grades) == 30
        and all(g["status"] == "graded" for g in grades)
        and all(
            r["status"] == "fixed_epoch5_receipt_verified" and "delivery_error" not in r
            for r in records
        )
    )
    return {
        "status": "complete" if complete else "partial",
        "fits": records,
        "grades": grades,
        "remaining_not_admitted": len(progress["pending"]),
        "no_implicit_refits": True,
    }
