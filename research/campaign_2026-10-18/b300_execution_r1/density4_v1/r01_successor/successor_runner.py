"""Hybrid fit/export/grade supervisor for the frozen 60-cell R01 panel."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adoption import reconcile_state
from authority import resource_baseline, validate_live_resources, validate_projection_receipt
from durability import Journal, write_json_fsync
from frozen_panel import (
    FIT_CAP_SECONDS,
    read_object,
    sha256_file,
    validate_panel,
    verify_frozen_packet,
)
from proc_snapshot import read_process, require_controller_identity
from schedule import active_slots, available_jobs, lane_plan
from worker_watch import group_has_live_members, pidfd_exited, pidfd_for, require_same_process

DISK_FLOOR = 8 * 1024**3
RSS_LIMIT = 24 * 1024**3
POLL_SECONDS = 2.0


@dataclass
class CellProcess:
    """One fit process group, either adopted from v4 or started by this controller."""

    job: dict[str, Any]
    kind: str
    pid: int
    pgid: int
    start_ticks: int
    started_unix: float
    deadline_unix: float
    process: Any = None
    pidfd: int | None = None
    status: str = "fit_running"


def _require_old_main_stopped(authority: dict[str, Any]) -> None:
    """Revalidate the exact frozen R01 controller before successor work proceeds."""
    packet = Path(authority["packet_root"])
    snapshot = read_process(int(authority["old_main_pid"]))
    require_controller_identity(
        snapshot,
        expected_pid=int(authority["old_main_pid"]),
        expected_start_ticks=int(authority["old_main_start_ticks"]),
        expected_unit=str(authority["old_unit"]),
        command_marker=str(packet / "execute.py"),
        must_be_stopped=True,
    )
    if (
        snapshot.cgroup_path != authority["old_main_cgroup_path"]
        or snapshot.cmdline_sha256 != authority["main_cmdline_sha256"]
    ):
        raise ValueError("old R01 controller cgroup or command hash changed after freeze")


def _frozen_modules(packet: Path) -> dict[str, Any]:
    """Import the exact packet modules only after the complete manifest verifies."""
    verify_frozen_packet(packet)
    control = str((packet / "control").resolve(strict=True))
    if control not in sys.path:
        sys.path.insert(0, control)
    modules = {}
    for name in (
        "fit_stage",
        "postprocess_common",
        "postprocess_fit",
        "postprocess_export_run",
        "postprocess_grade",
        "profiles",
        "process_start",
        "runtime",
        "runtime_gates",
    ):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != Path(control):
            raise ValueError(
                f"frozen runtime module resolved outside the authorized packet: {name}"
            )
        modules[name] = module
    return modules


def _same_frozen_cohort(
    freeze: dict[str, Any], current: dict[str, Any], jobs: tuple[dict[str, Any], ...]
) -> None:
    """Allow only original seed-1003 workers, live or receipt-validated terminal."""
    expected = {str(job["id"]) for job in jobs if job["seed"] == 1003}
    freeze_snapshot = freeze.get("snapshot")
    if not isinstance(freeze_snapshot, dict) or set(freeze_snapshot.get("live", {})) != expected:
        raise ValueError("FREEZE receipt does not record the exact eight live seed-1003 workers")
    if set(current.get("claimed", [])) != expected or len(current.get("pending", [])) != 52:
        raise ValueError(
            "R01 takeover no longer has the original eight claims and 52 zero-evidence cells"
        )
    live = current.get("live")
    terminal = current.get("terminal_fits")
    if (
        not isinstance(live, dict)
        or not isinstance(terminal, dict)
        or set(live) | set(terminal) != expected
    ):
        raise ValueError("an initial R01 fit is missing or outside the original cohort")
    if set(live) & set(terminal):
        raise ValueError("an original R01 fit is simultaneously live and terminal")
    for job_id, record in live.items():
        previous = freeze_snapshot["live"].get(job_id)
        if not isinstance(previous, dict) or any(
            record.get(key) != previous.get(key) for key in ("pid", "start_ticks", "pgid")
        ):
            raise ValueError(f"adopted worker identity changed across the freeze: {job_id}")
    if current.get("failure_events") or current.get("progress_stop_admission") is not False:
        raise ValueError("old R01 controller recorded a failure or stop-admission state")
    if current["main"].get("state") not in {"T", "t"}:
        raise ValueError("old R01 main controller is no longer stopped")
    before_main = freeze_snapshot.get("main")
    after_main = current.get("main")
    if (
        not isinstance(before_main, dict)
        or not isinstance(after_main, dict)
        or any(
            before_main.get(key) != after_main.get(key)
            for key in ("pid", "start_ticks", "cgroup_path", "cmdline_sha256")
        )
    ):
        raise ValueError("old R01 main identity changed across the freeze receipt")


def _group_rss(runtime_module: Any, process: CellProcess) -> int:
    """Measure one exact worker process-group RSS from /proc."""
    value = runtime_module.process_group_rss_bytes(process.pgid)
    if value < 0:
        raise ValueError("process-group RSS measurement is negative")
    return value


def _terminate_group(process: CellProcess, *, grace_seconds: float = 10) -> None:
    """Terminate only one recorded, unique-session fit process group."""
    try:
        require_same_process(process.pid, process.start_ticks, process.pgid)
        os.killpg(process.pgid, signal.SIGTERM)
    except (ProcessLookupError, FileNotFoundError):
        pass
    else:
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and group_has_live_members(process.pgid):
            time.sleep(0.1)
        if group_has_live_members(process.pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pgid, signal.SIGKILL)
    if process.process is not None:
        process.process.process.wait()
        process.process.log.flush()
        os.fsync(process.process.log.fileno())
        process.process.log.close()


def _validate_fit(root: Path, job: dict[str, Any], modules: dict[str, Any]) -> dict[str, Any]:
    """Require both producer outputs and full v4 source/split/checkpoint binding."""
    producer = modules["fit_stage"]._validate_fit_receipt(job)
    binding = modules["postprocess_fit"]._fit_binding(root, job)
    return {
        "schema": "tabdata-r01-density4-fit-validation.v1",
        "status": "fixed_epoch5_fit_verified",
        "job_id": job["id"],
        "fit_receipt_sha256": binding["fit_receipt_sha256"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "split_sha256": binding["split_sha256"],
        "provenance_sha256": binding["provenance_sha256"],
        "selected_epoch": producer["selected_epoch"],
        "fit_receipt": producer,
        "binding": binding,
    }


def _export_one(
    root: Path,
    paths: dict[str, Path],
    fit: dict[str, Any],
    authority: dict[str, Any],
    baseline: dict[str, int],
    modules: dict[str, Any],
) -> dict[str, Any]:
    """Run or reconcile the frozen export once while the fit's GPU/arm lane stays reserved."""
    validate_live_resources(Path(authority["live_resource_receipt_path"]), baseline)
    job = fit["job"]
    modules["runtime_gates"].require_gpus([job], job["gpu_uuid"])
    return modules["postprocess_export_run"]._export_one(
        root,
        paths,
        fit,
        float(authority["campaign_deadline_unix"]),
    )


def _grade_one(
    root: Path,
    paths: dict[str, Path],
    fold: int,
    fits: dict[str, dict[str, Any]],
    exports: dict[str, dict[str, Any]],
    deadline_unix: float,
    modules: dict[str, Any],
) -> dict[str, Any]:
    """Run the exact official paired R4 grade from the frozen v4 implementation."""
    return modules["postprocess_grade"]._grade_pair(root, paths, fold, fits, exports, deadline_unix)


def run_successor(authority: dict[str, Any]) -> dict[str, Any]:
    """Adopt the first eight R01 fits, fill measured arm/GPU slots, export and grade."""
    root = Path(authority["root"])
    packet = Path(authority["packet_root"])
    state = Path(authority["successor_state_path"])
    freeze_path = state / "FREEZE.json"
    freeze = read_object(freeze_path)
    if freeze.get("status") != "old_main_stopped_exactly":
        raise ValueError("successor execution requires a durable exact FREEZE receipt")
    verify_frozen_packet(packet)
    panel = read_object(packet / "PANEL.json")
    jobs = validate_panel(panel, root)
    current = reconcile_state(
        root=root,
        packet=packet,
        state=Path(authority["old_state"]),
        unit=authority["old_unit"],
        main_pid=authority["old_main_pid"],
        main_start_ticks=authority["old_main_start_ticks"],
        main_cgroup=authority["old_main_cgroup_path"],
        main_cmdline_sha256=authority["main_cmdline_sha256"],
        main_cmd_marker=str(packet / "execute.py"),
        must_be_stopped=True,
    )
    _same_frozen_cohort(freeze, current, jobs)
    if any(
        record.get("status") != "validated_fixed_epoch5_fit"
        for record in current["terminal_fits"].values()
    ):
        raise ValueError("a terminal original fit lacks the frozen successful fit receipt")

    baseline = resource_baseline(Path(authority["resource_preflight_path"]))
    density = read_object(Path(authority["density_receipt_path"]))
    projection = validate_projection_receipt(
        Path(authority["projection_receipt_path"]),
        float(authority["campaign_deadline_unix"]),
        float(authority["postprocess_reserve_seconds"]),
        density,
    )
    validate_live_resources(Path(authority["live_resource_receipt_path"]), baseline)
    if shutil.disk_usage(root).free < DISK_FLOOR:
        raise OSError("less than 8 GiB of physical free disk remains")

    modules = _frozen_modules(packet)
    paths = modules["postprocess_common"]._runtime_paths(root, jobs)
    plan = lane_plan(jobs)
    jobs_by_id = {str(job["id"]): job for job in jobs}
    journal = Journal(state / "events.jsonl")
    fit_records: dict[str, dict[str, Any]] = {}
    fit_bindings: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[str, Any]] = {}
    grades: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    pending_ids = set(current["pending"])
    slots: dict[str, CellProcess] = {}
    delivery_futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    grade_futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    export_gpu_busy: set[str] = set()
    grade_lanes: dict[tuple[int, str], str] = {}
    campaign_deadline = float(authority["campaign_deadline_unix"])
    forecasts = projection["fit_estimate_seconds_by_cell"]

    for job_id, prior in current["terminal_fits"].items():
        job = jobs_by_id[job_id]
        operational = job
        binding = modules["postprocess_fit"]._fit_binding(root, operational)
        fit_bindings[job_id] = binding
        fit_records[job_id] = prior
        slots[job_id] = CellProcess(
            operational,
            "terminal",
            0,
            0,
            0,
            time.time(),
            campaign_deadline,
            status="export_pending",
        )

    for job_id, record in current["live"].items():
        job = jobs_by_id[job_id]
        # The adopted seed1003 fit ran on its frozen original CPU pair.
        # Supplemental pairs apply only to newly admitted opposite-arm cells.
        operational = job
        pid, ticks, pgid = record["pid"], record["start_ticks"], record["pgid"]
        try:
            pidfd = pidfd_for(pid, ticks)
        except (ProcessLookupError, FileNotFoundError):
            pidfd = None
        slots[job_id] = CellProcess(
            operational,
            "adopted",
            pid,
            pgid,
            ticks,
            float(record["started_unix"]),
            min(float(record["deadline_unix"]), campaign_deadline),
            pidfd=pidfd,
        )
        journal.record(
            "fit_adopted",
            job_id=job_id,
            pid=pid,
            pid_start_ticks=ticks,
            pgid=pgid,
            gpu_uuid=operational["gpu_uuid"],
            arm=operational["arm"],
        )

    # The external resource sample is refreshed by the root-owned host watcher.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="r01-delivery"
    )
    stopped_admission: str | None = None
    admission_stop_journaled = False
    try:
        while (
            slots
            or delivery_futures
            or grade_futures
            or (pending_ids and stopped_admission is None)
        ):
            now_unix = time.time()
            old_main_stopped = False
            try:
                _require_old_main_stopped(authority)
                old_main_stopped = True
            except (OSError, ValueError) as error:
                stopped_admission = f"frozen old R01 controller guard failed: {error}"
                journal.record("old_main_guard_failed", phase="loop", error=str(error))
            if now_unix >= campaign_deadline:
                stopped_admission = "original 72-hour campaign deadline reached"
            try:
                resources = validate_live_resources(
                    Path(authority["live_resource_receipt_path"]), baseline
                )
                if resources.get("observed_unix", 0) > now_unix + 2:
                    raise ValueError("host resource observation is in the future")
            except (OSError, ValueError, TimeoutError) as error:
                stopped_admission = f"live resource gate failed: {error}"
            if shutil.disk_usage(root).free < DISK_FLOOR:
                stopped_admission = "physical disk free space fell below 8 GiB"

            # Start at most one export per GPU; its arm slot remains reserved until it ends.
            for job_id, slot in list(slots.items()):
                if slot.status != "export_pending":
                    continue
                uuid = str(slot.job["gpu_uuid"])
                if uuid in export_gpu_busy or stopped_admission is not None:
                    continue
                try:
                    _require_old_main_stopped(authority)
                except (OSError, ValueError) as error:
                    old_main_stopped = False
                    stopped_admission = (
                        f"frozen old R01 controller guard failed before export: {error}"
                    )
                    journal.record(
                        "old_main_guard_failed", phase="export", job_id=job_id, error=str(error)
                    )
                    continue
                fit = fit_bindings.get(job_id)
                if fit is None:
                    try:
                        fit = modules["postprocess_fit"]._fit_binding(root, slot.job)
                        fit_bindings[job_id] = fit
                    except Exception as error:
                        failures.append(
                            {"job_id": job_id, "phase": "fit_binding", "error": str(error)}
                        )
                        journal.record("fit_validation_failed", job_id=job_id, error=str(error))
                        slot.status = "failed"
                        continue
                fit_records.setdefault(
                    job_id, {"status": "fixed_epoch5_fit_verified", "job_id": job_id}
                )
                try:
                    _require_old_main_stopped(authority)
                except (OSError, ValueError) as error:
                    old_main_stopped = False
                    stopped_admission = (
                        f"frozen old R01 controller guard failed before export: {error}"
                    )
                    journal.record(
                        "old_main_guard_failed",
                        phase="export_submit",
                        job_id=job_id,
                        error=str(error),
                    )
                    continue
                future = executor.submit(
                    _export_one, root, paths, fit, authority, baseline, modules
                )
                delivery_futures[future] = job_id
                export_gpu_busy.add(uuid)
                slot.status = "export_running"
                journal.record("export_started", job_id=job_id, gpu_uuid=uuid)

            for future, job_id in list(delivery_futures.items()):
                if not future.done():
                    continue
                delivery_futures.pop(future)
                job = jobs_by_id[job_id]
                export_gpu_busy.discard(str(job["gpu_uuid"]))
                try:
                    receipt = future.result()
                    exports[job_id] = receipt
                    accepted = (
                        receipt.get("resource_admitted") is True
                        or receipt.get("reconciled") is True
                    )
                    if not accepted or receipt.get("status") in {
                        "attempted_failed",
                        "existing_attempt_preserved",
                    }:
                        raise RuntimeError(f"frozen export did not pass admission: {receipt}")
                    journal.record("export_verified", job_id=job_id, receipt=receipt)
                    write_json_fsync(state / "receipts" / f"export-{job_id}.json", receipt)
                    slot = slots[job_id]
                    slot.status = "delivered"
                except BaseException as error:
                    failures.append({"job_id": job_id, "phase": "export", "error": str(error)})
                    journal.record("export_failed", job_id=job_id, error=str(error))
                    slots[job_id].status = "failed"

            # Grade only complete wide/deep pairs, reserving the deep CPU lane meanwhile.
            for job in jobs:
                key = (int(job["seed"]), int(job["fold"]))
                grade_id = f"seed{key[0]}_fold{key[1]}"
                if grade_id in grades or grade_id in grade_futures.values():
                    continue
                pair = [item for item in jobs if (item["seed"], item["fold"]) == key]
                if len(pair) != 2 or {item["arm"] for item in pair} != {"wide", "deep"}:
                    raise ValueError(f"R01 panel pair is malformed: {grade_id}")
                wide, deep = (
                    next(item for item in pair if item["arm"] == "wide"),
                    next(item for item in pair if item["arm"] == "deep"),
                )
                if not all(item["id"] in fit_bindings and item["id"] in exports for item in pair):
                    continue
                lane = (int(deep["gpu_index"]), "deep")
                if lane in grade_lanes:
                    continue
                paired_fits = {
                    "wide": fit_bindings[wide["id"]],
                    "deep": fit_bindings[deep["id"]],
                }
                paired_exports = {"wide": exports[wide["id"]], "deep": exports[deep["id"]]}
                future = executor.submit(
                    _grade_one,
                    root,
                    paths,
                    int(deep["fold"]),
                    paired_fits,
                    paired_exports,
                    campaign_deadline,
                    modules,
                )
                grade_futures[future] = grade_id
                grade_lanes[lane] = grade_id
                journal.record(
                    "grade_started", grade_id=grade_id, fold=deep["fold"], seed=deep["seed"]
                )

            for future, grade_id in list(grade_futures.items()):
                if not future.done():
                    continue
                grade_futures.pop(future)
                lane = next((key for key, value in grade_lanes.items() if value == grade_id), None)
                if lane is not None:
                    grade_lanes.pop(lane)
                try:
                    receipt = future.result()
                    if receipt.get("status") != "graded":
                        raise RuntimeError(f"official paired grade did not pass: {receipt}")
                    grades[grade_id] = receipt
                    journal.record("grade_verified", grade_id=grade_id, receipt=receipt)
                    write_json_fsync(state / "receipts" / f"grade-{grade_id}.json", receipt)
                except BaseException as error:
                    failures.append({"grade_id": grade_id, "phase": "grade", "error": str(error)})
                    journal.record("grade_failed", grade_id=grade_id, error=str(error))

            # Reconcile active fit groups; a stopped old parent leaves exited leaders as zombies.
            for job_id, slot in list(slots.items()):
                if slot.status != "fit_running":
                    if slot.status in {"delivered", "failed"}:
                        slots.pop(job_id)
                    continue
                if now_unix >= slot.deadline_unix:
                    _terminate_group(slot)
                    failures.append(
                        {
                            "job_id": job_id,
                            "phase": "fit",
                            "error": "original per-fit deadline reached",
                        }
                    )
                    journal.record("fit_deadline", job_id=job_id, pgid=slot.pgid)
                    slot.status = "failed"
                    continue
                if slot.kind == "owned":
                    return_code = slot.process.process.poll()
                    exited = return_code is not None
                else:
                    try:
                        snapshot = require_same_process(slot.pid, slot.start_ticks, slot.pgid)
                        exited = snapshot.identity.state in {"Z", "X"} or (
                            slot.pidfd is not None and pidfd_exited(slot.pidfd)
                        )
                    except (ProcessLookupError, FileNotFoundError):
                        exited = True
                    return_code = None
                if not exited:
                    if _group_rss(modules["runtime"], slot) > RSS_LIMIT:
                        stopped_admission = f"fit process group exceeded 24 GiB RSS: {job_id}"
                    continue
                if group_has_live_members(slot.pgid):
                    continue
                try:
                    if slot.kind == "owned" and return_code != 0:
                        raise RuntimeError(f"fit actor exited with status {return_code}")
                    receipt = _validate_fit(root, slot.job, modules)
                    fit_bindings[job_id] = receipt["binding"]
                    fit_records[job_id] = receipt
                    write_json_fsync(state / "receipts" / f"fit-{job_id}.json", receipt)
                    journal.record(
                        "fit_verified",
                        job_id=job_id,
                        pid=slot.pid,
                        pgid=slot.pgid,
                        pid_start_ticks=slot.start_ticks,
                        exit_code=return_code
                        if slot.kind == "owned"
                        else "nonchild_status_unavailable",
                        fit_receipt_sha256=receipt["fit_receipt_sha256"],
                    )
                    slot.status = "export_pending"
                except BaseException as error:
                    failures.append({"job_id": job_id, "phase": "fit", "error": str(error)})
                    journal.record("fit_failed", job_id=job_id, error=str(error))
                    slot.status = "failed"
                finally:
                    if slot.pidfd is not None:
                        os.close(slot.pidfd)
                        slot.pidfd = None

            if stopped_admission is not None:
                if pending_ids and not admission_stop_journaled:
                    journal.record(
                        "admission_stopped", reason=stopped_admission, pending=len(pending_ids)
                    )
                    admission_stop_journaled = True
                for slot in slots.values():
                    if slot.kind == "owned" and slot.status == "fit_running":
                        _terminate_group(slot)
                        slot.status = "failed"
                    elif slot.status == "export_pending":
                        failures.append(
                            {
                                "job_id": slot.job["id"],
                                "phase": "export",
                                "error": stopped_admission,
                            }
                        )
                        journal.record(
                            "export_admission_stopped",
                            job_id=slot.job["id"],
                            reason=stopped_admission,
                        )
                        slot.status = "failed"

            # Fill each free measured GPU/arm lane with one unique zero-evidence cell.
            if stopped_admission is None:
                occupied = {
                    job_id: {"job": slot.job}
                    for job_id, slot in slots.items()
                    if slot.status not in {"failed", "delivered"}
                }
                for lane, grade_id in grade_lanes.items():
                    occupied[f"grade:{grade_id}"] = {"job": {"gpu_index": lane[0], "arm": lane[1]}}
                occupied_slots = active_slots(occupied, plan)
                candidates = available_jobs(jobs, pending_ids, occupied_slots, plan)
                for job in candidates:
                    job_id = str(job["id"])
                    if time.time() >= campaign_deadline:
                        stopped_admission = "original 72-hour campaign deadline reached"
                        break
                    try:
                        _require_old_main_stopped(authority)
                    except (OSError, ValueError) as error:
                        old_main_stopped = False
                        stopped_admission = (
                            f"frozen old R01 controller guard failed before fit spawn: {error}"
                        )
                        journal.record(
                            "old_main_guard_failed",
                            phase="fit_spawn",
                            job_id=job_id,
                            error=str(error),
                        )
                        if pending_ids and not admission_stop_journaled:
                            journal.record(
                                "admission_stopped",
                                reason=stopped_admission,
                                pending=len(pending_ids),
                            )
                            admission_stop_journaled = True
                        break
                    estimate = forecasts.get(job_id)
                    remaining = campaign_deadline - time.time()
                    if (
                        isinstance(estimate, bool)
                        or not isinstance(estimate, (int, float))
                        or estimate <= 0
                        or estimate > FIT_CAP_SECONDS
                        or remaining < estimate
                    ):
                        journal.record(
                            "fit_admission_failed",
                            job_id=job_id,
                            error="measured fit estimate exceeds remaining campaign time",
                        )
                        pending_ids.remove(job_id)
                        failures.append(
                            {
                                "job_id": job_id,
                                "phase": "admission",
                                "error": "measured estimate exceeds campaign time",
                            }
                        )
                        continue
                    validate_live_resources(Path(authority["live_resource_receipt_path"]), baseline)
                    modules["runtime_gates"].require_gpus([job], job["gpu_uuid"])
                    run_dir = Path(job["run_dir"])
                    if run_dir.exists() or run_dir.is_symlink():
                        raise FileExistsError(
                            f"zero-evidence fit directory is no longer fresh: {run_dir}"
                        )
                    journal.record(
                        "fit_spawn_intent",
                        job_id=job_id,
                        run_dir=str(run_dir),
                        gpu_uuid=job["gpu_uuid"],
                        arm=job["arm"],
                        cpu_ids=job["cpu_ids_initial"],
                        argv_sha256=__import__("hashlib")
                        .sha256(json.dumps(job["argv"], separators=(",", ":")).encode())
                        .hexdigest(),
                    )
                    run_dir.parent.mkdir(parents=True, exist_ok=True)
                    run_dir.mkdir()
                    environment = modules["profiles"].actor_environment(
                        root, run_dir, job["gpu_uuid"]
                    )
                    log_path = state / "logs" / f"{job_id}.log"
                    timeout = min(FIT_CAP_SECONDS, remaining)
                    try:
                        _require_old_main_stopped(authority)
                    except (OSError, ValueError) as error:
                        old_main_stopped = False
                        stopped_admission = (
                            f"frozen old R01 controller guard failed before fit spawn: {error}"
                        )
                        journal.record(
                            "old_main_guard_failed",
                            phase="fit_spawn_final",
                            job_id=job_id,
                            error=str(error),
                        )
                        if pending_ids and not admission_stop_journaled:
                            journal.record(
                                "admission_stopped",
                                reason=stopped_admission,
                                pending=len(pending_ids),
                            )
                            admission_stop_journaled = True
                        with contextlib.suppress(OSError):
                            run_dir.rmdir()
                        temporary_dir = root / "tmp" / run_dir.name
                        with contextlib.suppress(OSError):
                            temporary_dir.rmdir()
                        break
                    running = None
                    try:
                        running = modules["process_start"].start_owned(
                            name=job_id,
                            argv=list(job["argv"]),
                            environment={**environment, "WNN_CWD": str(root)},
                            cpus=tuple(job["cpu_ids_initial"]),
                            log_path=log_path,
                            timeout_seconds=timeout,
                        )
                        identity = __import__("proc_snapshot").read_process(running.process.pid)
                    except BaseException as error:
                        if running is not None:
                            if running.process.poll() is None:
                                modules["process_start"].terminate_owned(running.process)
                            running.log.flush()
                            os.fsync(running.log.fileno())
                            running.log.close()
                        journal.record("fit_admission_failed", job_id=job_id, error=str(error))
                        failures.append({"job_id": job_id, "phase": "spawn", "error": str(error)})
                        pending_ids.remove(job_id)
                        continue
                    if identity.identity.process_group != running.process.pid:
                        _terminate_group(
                            CellProcess(
                                job,
                                "owned",
                                running.process.pid,
                                identity.identity.process_group,
                                identity.identity.start_ticks,
                                time.time(),
                                time.time() + timeout,
                            )
                        )
                        raise ValueError("fresh actor process group is not its own process ID")
                    process = CellProcess(
                        job,
                        "owned",
                        running.process.pid,
                        identity.identity.process_group,
                        identity.identity.start_ticks,
                        time.time(),
                        min(campaign_deadline, time.time() + timeout),
                        process=running,
                    )
                    slots[job_id] = process
                    pending_ids.remove(job_id)
                    journal.record(
                        "fit_started",
                        job_id=job_id,
                        pid=process.pid,
                        pgid=process.pgid,
                        pid_start_ticks=process.start_ticks,
                        gpu_uuid=job["gpu_uuid"],
                        arm=job["arm"],
                        fold=job["fold"],
                        seed=job["seed"],
                        cpu_ids=job["cpu_ids_initial"],
                        argv_sha256=hashlib.sha256(
                            json.dumps(job["argv"], separators=(",", ":")).encode()
                        ).hexdigest(),
                    )

            try:
                _require_old_main_stopped(authority)
            except (OSError, ValueError) as error:
                old_main_stopped = False
                stopped_admission = (
                    f"frozen old R01 controller guard failed before heartbeat: {error}"
                )
                journal.record("old_main_guard_failed", phase="heartbeat", error=str(error))
                if pending_ids and not admission_stop_journaled:
                    journal.record(
                        "admission_stopped", reason=stopped_admission, pending=len(pending_ids)
                    )
                    admission_stop_journaled = True
            journal.heartbeat(
                active_fit_ids=[
                    job_id for job_id, slot in slots.items() if slot.status == "fit_running"
                ],
                active_delivery_ids=sorted(delivery_futures.values()),
                pending_fit_count=len(pending_ids),
                failed_fit_count=sum(
                    item.get("phase") in {"fit", "spawn", "admission"} for item in failures
                ),
                campaign_deadline_unix=campaign_deadline,
                old_unit_main_still_stopped=old_main_stopped,
            )
            if slots or delivery_futures or grade_futures:
                time.sleep(POLL_SECONDS)
            elif pending_ids and stopped_admission is None:
                raise RuntimeError("no occupied slot but no schedulable measured lane is available")

    except BaseException as error:
        journal.record("controller_exception", error_type=type(error).__name__, error=str(error))
        for slot in slots.values():
            if slot.kind == "owned" and slot.status == "fit_running":
                _terminate_group(slot)
            if slot.pidfd is not None:
                os.close(slot.pidfd)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    panel_pairs = {(int(job["seed"]), int(job["fold"])) for job in jobs}
    summary = {
        "schema": "tabdata-r01-density4-successor-summary.v1",
        "status": "complete"
        if len(exports) == len(jobs) and len(grades) == len(panel_pairs) and not failures
        else "partial",
        "fit_count_verified": len(fit_records),
        "export_count_verified": len(exports),
        "grade_pair_count": len(grades),
        "pending_fit_count": len(pending_ids),
        "failures": failures,
        "admission_stopped_reason": stopped_admission,
        "old_unit_main_remains_stopped": True,
        "old_unit_retirement_eligible": not slots and not delivery_futures and not grade_futures,
        "projection_fit_only_tail_hours": projection["fit_only_tail_hours"],
        "fit_projection_is_observed_duration": False,
        "frozen_execution_manifest_sha256": sha256_file(packet / "EXECUTION_MANIFEST.json"),
        "panel_sha256": sha256_file(packet / "PANEL.json"),
        "finished_unix": time.time(),
    }
    write_json_fsync(state / "SUMMARY.json", summary)
    journal.record(
        "controller_finished",
        status=summary["status"],
        summary_sha256=sha256_file(state / "SUMMARY.json"),
    )
    return summary
