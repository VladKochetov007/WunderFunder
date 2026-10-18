"""Read-only process and artifact reconciliation for R01-v4 takeover."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from frozen_panel import (
    FIT_CAP_SECONDS,
    R01_PANEL_SHA256,
    parse_journal,
    progress_state,
    read_object,
    sha256_file,
    validate_panel,
    verify_frozen_packet,
)
from proc_snapshot import ProcSnapshot, grouped_members, require_controller_identity, unit_members


def _process_jobs(
    snapshots: tuple[ProcSnapshot, ...],
    jobs: tuple[dict[str, Any], ...],
    *,
    controller_pid: int,
    expected_unit: str,
    affinity_reader: Callable[[int], set[int]],
    start_records: dict[str, dict[str, Any]],
) -> dict[str, ProcSnapshot]:
    """Map every unit process group to one exact old controller or fit actor."""
    groups = grouped_members(snapshots)
    by_pid = {item.identity.pid: item for item in snapshots}
    output: dict[str, ProcSnapshot] = {}
    for pgid, pids in groups.items():
        if controller_pid in pids:
            if len(pids) != 1:
                raise ValueError("old controller shares its process group with another process")
            continue
        leaders = [
            by_pid[pid]
            for pid in pids
            if by_pid[pid].identity.pid == pgid
            and any(by_pid[pid].argv == tuple(str(token) for token in job["argv"]) for job in jobs)
        ]
        if not leaders:
            started_job = next(
                (
                    job
                    for job_id, start in start_records.items()
                    if start.get("pid") == pgid
                    for job in jobs
                    if str(job["id"]) == job_id
                ),
                None,
            )
            if started_job is not None and pgid in by_pid:
                leaders = [by_pid[pgid]]
        if len(leaders) != 1:
            raise ValueError(f"unexplained process group in frozen unit {expected_unit}: {pgid}")
        worker = leaders[0]
        job = next(
            (job for job in jobs if tuple(str(token) for token in job["argv"]) == worker.argv),
            None,
        )
        if job is None:
            job = next(
                job
                for job_id, start in start_records.items()
                if start.get("pid") == worker.identity.pid
                for job in jobs
                if str(job["id"]) == job_id
            )
        active_members = [
            by_pid[pid] for pid in pids if by_pid[pid].identity.state not in {"Z", "X"}
        ]
        if active_members:
            _validate_worker_binding(
                next(
                    (item for item in active_members if item.identity.pid == pgid),
                    active_members[0],
                ),
                job,
            )
        job_id = str(job["id"])
        if job_id in output:
            raise ValueError(f"duplicate live process group for {job_id}")
        for pid in pids:
            member = by_pid[pid]
            if member.identity.state in {"Z", "X"}:
                continue
            try:
                affinity = affinity_reader(pid)
            except (AttributeError, ProcessLookupError, PermissionError) as error:
                raise ValueError(f"cannot verify old worker CPU affinity: {pid}") from error
            if affinity != set(job["cpu_ids_initial"]):
                raise ValueError(f"old worker process affinity differs from panel: {job_id}")
        output[job_id] = worker
    return output


def _validate_worker_binding(snapshot: ProcSnapshot, job: dict[str, Any]) -> None:
    """Bind an old worker's live environment and working directory to its frozen cell."""
    environment = snapshot.environment
    expected = {
        "CUDA_VISIBLE_DEVICES": str(job["gpu_uuid"]),
        "WNN_RUN_DIR": str(job["run_dir"]),
        "WNN_GPU_MEMORY_BYTES": str(6 * 1024**3),
        "WNN_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError(f"old worker environment differs from the frozen runtime: {job['id']}")
    root = Path(str(job["run_dir"])).parent.parent
    if snapshot.cwd != str(root):
        raise ValueError(f"old worker cwd differs from the project root: {job['id']}")
    argv = tuple(str(token) for token in job["argv"])
    for flag, expected_path in (
        ("--source-root", root / "runtime" / "source-a266-lf-subset-v1"),
        ("--data-root", root / "data"),
    ):
        try:
            observed = Path(argv[argv.index(flag) + 1])
        except (ValueError, IndexError) as error:
            raise ValueError(f"frozen actor argv omits {flag}: {job['id']}") from error
        if observed != expected_path:
            raise ValueError(f"old worker {flag} differs from the frozen project data: {job['id']}")


def _validate_finished_fit(root: Path, packet: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Use the packet's exact fit and ancestry validators for a non-live claimed cell."""
    control = packet / "control"
    resolved = control.resolve(strict=True)
    sys.path.insert(0, str(resolved))
    try:
        fit_stage = importlib.import_module("fit_stage")
        postprocess_fit = importlib.import_module("postprocess_fit")
        fit_receipt = fit_stage._validate_fit_receipt(job)
        binding = postprocess_fit._fit_binding(root, job)
    finally:
        if sys.path and sys.path[0] == str(resolved):
            sys.path.pop(0)
    return {
        "status": "validated_fixed_epoch5_fit",
        "job_id": str(job["id"]),
        "fit_receipt_sha256": binding["fit_receipt_sha256"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "split_sha256": binding["split_sha256"],
        "producer_status": fit_receipt["status"],
        "exit_code": "not_observable_for_nonchild_process",
    }


def proc_start_unix(snapshot: ProcSnapshot, proc_root: Path = Path("/proc")) -> float:
    """Convert a stable Linux start-tick identity to Unix time."""
    boot_time = next(
        (
            line.split()[1]
            for line in (proc_root / "stat").read_text(encoding="ascii").splitlines()
            if line.startswith("btime ")
        ),
        None,
    )
    if boot_time is None:
        raise ValueError("/proc/stat has no boot-time record")
    return float(boot_time) + snapshot.identity.start_ticks / os.sysconf("SC_CLK_TCK")


def reconcile_state(
    *,
    root: Path,
    packet: Path,
    state: Path,
    unit: str,
    main_pid: int,
    main_start_ticks: int,
    main_cgroup: str,
    main_cmdline_sha256: str,
    main_cmd_marker: str,
    proc_root: Path = Path("/proc"),
    affinity_reader: Callable[[int], set[int]] | None = None,
    must_be_stopped: bool = True,
) -> dict[str, Any]:
    """Require a stopped exact main process and recover every claimed old R01 cell."""
    reader = affinity_reader or getattr(os, "sched_getaffinity", None)
    if reader is None:
        raise OSError("Linux CPU affinity inspection is required for takeover")
    manifest = verify_frozen_packet(packet)
    panel = read_object(packet / "PANEL.json")
    jobs = validate_panel(panel, root)
    members = unit_members(unit, proc_root)
    main = next((item for item in members if item.identity.pid == main_pid), None)
    if main is None:
        raise ValueError("frozen R01 MainPID is absent from its exact systemd unit")
    require_controller_identity(
        main,
        expected_pid=main_pid,
        expected_start_ticks=main_start_ticks,
        expected_unit=unit,
        command_marker=main_cmd_marker,
        must_be_stopped=must_be_stopped,
    )
    if main.cgroup_path != main_cgroup:
        raise ValueError("frozen R01 MainPID cgroup path differs from root authority")
    if main.cmdline_sha256 != main_cmdline_sha256:
        raise ValueError("frozen R01 MainPID command bytes differ from root authority")
    if sha256_file(packet / "PANEL.json") != R01_PANEL_SHA256:
        raise ValueError("R01 panel changed after freeze authority was sealed")
    events = parse_journal(state / "events.jsonl", jobs)
    progress = progress_state(state / "PROGRESS.json", jobs)
    live = _process_jobs(
        members,
        jobs,
        controller_pid=main_pid,
        expected_unit=unit,
        affinity_reader=reader,
        start_records=events["starts"],
    )
    live_processes: dict[str, ProcSnapshot] = {}
    zombie_processes: dict[str, ProcSnapshot] = {}
    for job_id, snapshot in live.items():
        group = [
            item
            for item in members
            if item.identity.process_group == snapshot.identity.process_group
        ]
        if group and all(item.identity.state in {"Z", "X"} for item in group):
            zombie_processes[job_id] = snapshot
        else:
            live_processes[job_id] = snapshot
        started = events["starts"].get(job_id)
        if started is not None and started["pid"] != snapshot.identity.pid:
            raise ValueError(f"live worker PID differs from journal for {job_id}")
    claimed = set(events["referenced"]) | set(progress["claimed"])
    evidence: dict[str, list[str]] = {str(job["id"]): [] for job in jobs}
    for job in jobs:
        job_id = str(job["id"])
        paths = {
            "run_dir": Path(str(job["run_dir"])),
            "log": state / "logs" / f"{job_id}.log",
            "producer_receipt": Path(str(job["run_dir"])) / "producer_receipt.json",
            "export": root / "exports" / job_id,
            "grade": root / "grades" / f"seed{job['seed']}" / f"fold{job['fold']}",
        }
        for label, path in paths.items():
            if path.is_symlink():
                raise ValueError(f"R01 cell evidence path is a symlink: {path}")
            if path.exists():
                evidence[job_id].append(label)
                claimed.add(job_id)
        if job_id in live_processes:
            evidence[job_id].append("live_exact_argv")
            claimed.add(job_id)
        if job_id in zombie_processes:
            evidence[job_id].append("zombie_worker_group")
            claimed.add(job_id)
    pending = sorted(str(job["id"]) for job in jobs if str(job["id"]) not in claimed)
    if progress["pending"] is not None and progress["pending"] < len(pending):
        raise ValueError("R01 progress pending count is below reconciled zero-evidence cells")
    job_by_id = {str(job["id"]): job for job in jobs}
    live_details: dict[str, dict[str, Any]] = {}
    for job_id, snapshot in live_processes.items():
        started_unix = proc_start_unix(snapshot, proc_root)
        live_details[job_id] = {
            "pid": snapshot.identity.pid,
            "start_ticks": snapshot.identity.start_ticks,
            "pgid": snapshot.identity.process_group,
            "state": snapshot.identity.state,
            "started_unix": started_unix,
            "deadline_unix": started_unix + FIT_CAP_SECONDS,
            "argv_sha256": snapshot.cmdline_sha256,
            "job": job_by_id[job_id],
        }
    terminal: dict[str, dict[str, Any]] = {}
    finished_claims = claimed - set(live_processes)
    for job_id in sorted(finished_claims):
        job = job_by_id[job_id]
        try:
            binding = _validate_finished_fit(root, packet, job)
        except Exception as error:
            raise ValueError(
                f"claimed non-live R01 cell lacks a validated terminal fit: {job_id}"
            ) from error
        terminal[job_id] = binding
    if set(zombie_processes) - set(terminal):
        raise ValueError("zombie old-worker leader lacks its validated terminal fit receipt")
    return {
        "schema": "tabdata-r01-v4-adoption-snapshot.v2",
        "observed_unix": time.time(),
        "unit": unit,
        "packet_manifest_sha256": sha256_file(packet / "EXECUTION_MANIFEST.json"),
        "panel_sha256": sha256_file(packet / "PANEL.json"),
        "main": {
            "pid": main.identity.pid,
            "start_ticks": main.identity.start_ticks,
            "state": main.identity.state,
            "cgroup_path": main.cgroup_path,
            "cmdline_sha256": main.cmdline_sha256,
        },
        "journal_sha256": events["sha256"],
        "claimed": sorted(claimed),
        "pending": pending,
        "live": live_details,
        "terminal_fits": terminal,
        "evidence": evidence,
        "failure_events": events["failures"],
        "progress_stop_admission": progress["stop"],
        "event_start_ids": sorted(events["starts"]),
        "unlogged_live_ids": sorted(set(live_processes) - set(events["starts"])),
        "zero_evidence_count": len(pending),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
    }


def validate_initial_takeover(
    snapshot: dict[str, Any],
    jobs: tuple[dict[str, Any], ...],
    *,
    require_stopped: bool = True,
) -> None:
    """Fail closed unless the exact eight-fit v4 first wave remains the only claim."""
    expected_live = {str(job["id"]) for job in jobs if job["seed"] == 1003}
    expected_all = {str(job["id"]) for job in jobs}
    live = snapshot.get("live")
    main = snapshot.get("main")
    pending = snapshot.get("pending")
    claimed = snapshot.get("claimed")
    evidence = snapshot.get("evidence")
    if not isinstance(live, dict) or set(live) != expected_live or len(live) != 8:
        raise ValueError("takeover is limited to all eight live seed-1003 R01 workers")
    expected_state = {"T", "t"} if require_stopped else {"R", "S", "D", "I"}
    if not isinstance(main, dict) or main.get("state") not in expected_state:
        expectation = "stopped" if require_stopped else "running"
        raise ValueError(f"the exact old R01 controller must be {expectation} for this phase")
    if not isinstance(claimed, list) or set(claimed) != expected_live:
        raise ValueError("takeover requires exactly the original eight claimed R01 cells")
    if (
        not isinstance(pending, list)
        or set(pending) != expected_all - expected_live
        or len(pending) != 52
    ):
        raise ValueError("takeover requires all 52 remaining cells to have zero prior evidence")
    if snapshot.get("failure_events") or snapshot.get("progress_stop_admission") is not False:
        raise ValueError("frozen R01 controller already recorded a failure or stopped admissions")
    if not isinstance(evidence, dict):
        raise ValueError("adoption snapshot lacks per-cell evidence")
    for job_id in expected_live:
        labels = set(evidence.get(job_id, []))
        if not {"live_exact_argv", "run_dir", "log"} <= labels:
            raise ValueError(f"initial R01 worker lacks its run/log evidence: {job_id}")
        record = live[job_id]
        if not isinstance(record, dict) or record.get("state") in {"Z", "X"}:
            raise ValueError(f"initial R01 worker is not a live non-zombie process: {job_id}")
        if record.get("job", {}).get("seed") != 1003:
            raise ValueError(f"initial R01 worker is outside the original seed-1003 wave: {job_id}")
        if "producer_receipt" in labels or "export" in labels or "grade" in labels:
            raise ValueError(f"initial R01 worker already has terminal result evidence: {job_id}")
    for job_id in expected_all - expected_live:
        if evidence.get(job_id):
            raise ValueError(f"pending R01 cell has prior filesystem/process evidence: {job_id}")
    if snapshot.get("zero_evidence_count") != 52:
        raise ValueError("initial R01 takeover must expose exactly 52 zero-evidence cells")
