"""Bounded supervision for controller-owned preparation, profiles, and fits."""

from __future__ import annotations

import os
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from policy import DISK_FREE_MIN, JOB_RSS_LIMIT
from process_start import Running, start_owned, terminate_owned
from runtime_gates import (
    _gpu_process_memory,
    cgroup_snapshot,
    require_gpus,
)

HEARTBEAT_SECONDS = 30
OWNED_DISK_LIMIT = 128 * 1024**3
MEMORY_SOFT_LIMIT = 180 * 1024**3


def process_group_rss_bytes(pgid: int) -> int:
    """Sum RSS for processes still in one owned process group."""
    total_kib = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2 :].split()
            if int(fields[2]) != pgid:
                continue
            for line in (proc / "status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return total_kib * 1024


def owned_tree_bytes(paths: list[Path]) -> int:
    """Count regular files in controller-owned trees without following symlinks."""
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                try:
                    mode = path.lstat().st_mode
                    if stat.S_ISREG(mode):
                        total += path.lstat().st_size
                except FileNotFoundError:
                    continue
    return total


def run_single(
    *,
    name: str,
    argv: list[str],
    environment: dict[str, str],
    cpus: tuple[int, ...],
    log_path: Path,
    timeout_seconds: float,
    journal: Any,
    root: Path,
    global_deadline: float,
    owned_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Run one bounded preparation/profile process with disk and RSS gates."""
    if time.monotonic() >= global_deadline:
        raise TimeoutError(f"global wall deadline passed before starting {name}")
    baseline_events = cgroup_snapshot()["memory_events"]
    task = start_owned(
        name=name,
        argv=argv,
        environment=environment,
        cpus=cpus,
        log_path=log_path,
        timeout_seconds=min(timeout_seconds, max(0, global_deadline - time.monotonic())),
    )
    try:
        return _monitor([task], journal, root, global_deadline, baseline_events, owned_paths)[name]
    except BaseException:
        terminate_owned(task.process)
        task.log.close()
        raise


def run_cohort(
    jobs: list[dict[str, Any]],
    launch: Callable[[dict[str, Any]], Running],
    *,
    journal: Any,
    root: Path,
    global_deadline: float,
    owned_paths: list[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Launch exactly the supplied cohort and supervise each owned group independently."""
    baseline_events = cgroup_snapshot()["memory_events"]
    running: list[Running] = []
    results: dict[str, dict[str, Any]] = {}
    try:
        for job in jobs:
            try:
                snapshot = cgroup_snapshot()
                if snapshot["memory_current"] >= MEMORY_SOFT_LIMIT:
                    raise RuntimeError(
                        "cgroup memory.current reached the 180-GiB admission ceiling"
                    )
                if time.monotonic() >= global_deadline:
                    raise TimeoutError("global wall deadline passed before fit launch")
                require_gpus(jobs, job["gpu_uuid"])
                running.append(launch(job))
            except Exception as error:
                results[job["id"]] = {
                    "returncode": None,
                    "status": "admission_failed",
                    "error": str(error),
                }
                journal.record("fit_admission_failed", job_id=job["id"], error=str(error))
        results.update(
            _monitor(running, journal, root, global_deadline, baseline_events, owned_paths)
        )
    except BaseException as original_error:
        for task in running:
            try:
                if task.status == "running":
                    terminate_owned(task.process)
                    task.status = "stopped_after_controller_exception"
            except BaseException as cleanup_error:
                original_error.add_note(
                    f"failed to terminate owned process group {task.name}: {cleanup_error}"
                )
            finally:
                if not task.log.closed:
                    try:
                        task.log.flush()
                        os.fsync(task.log.fileno())
                    except OSError as log_error:
                        original_error.add_note(f"failed to fsync log for {task.name}: {log_error}")
                    task.log.close()
        raise
    return results


def _monitor(
    running: list[Running],
    journal: Any,
    root: Path,
    global_deadline: float,
    baseline_events: dict[str, int],
    owned_paths: list[Path] | None,
) -> dict[str, dict[str, Any]]:
    """Poll active owned groups, applying per-job RSS and global safety limits."""
    results: dict[str, dict[str, Any]] = {}
    last_heartbeat = 0.0
    last_gpu_sample = 0.0
    while any(task.status == "running" for task in running):
        now = time.monotonic()
        global_reason = None
        if now >= global_deadline:
            global_reason = "global wall deadline reached"
        if shutil.disk_usage(root).free < DISK_FREE_MIN:
            global_reason = "filesystem free-space floor fell below 8 GiB"
        owned_bytes = owned_tree_bytes(owned_paths or [])
        if owned_bytes > OWNED_DISK_LIMIT:
            global_reason = "controller-owned artifacts exceeded the 45-GiB budget"
        snapshot = cgroup_snapshot()
        if snapshot["memory_current"] >= MEMORY_SOFT_LIMIT:
            global_reason = "container memory.current reached the 180-GiB soft safety ceiling"
        if any(
            snapshot["memory_events"].get(key, 0) > baseline_events.get(key, 0)
            for key in ("oom", "oom_kill")
        ):
            global_reason = "container cgroup reported a new OOM event"
        if global_reason:
            for task in running:
                if task.status == "running":
                    terminate_owned(task.process)
                    task.status = "stopped_by_global_gate"
                    results[task.name] = {
                        "returncode": task.process.returncode,
                        "status": task.status,
                    }
                    task.log.close()
                    journal.record("process_stopped", process=task.name, reason=global_reason)
            break
        if now - last_gpu_sample >= 5 and any(
            task.status == "running" and task.gpu_uuid is not None for task in running
        ):
            gpu_usage = _gpu_process_memory()
            for task in running:
                if task.status == "running" and task.gpu_uuid is not None:
                    observed = gpu_usage.get((task.gpu_uuid, task.process.pid))
                    if observed is not None:
                        task.gpu_peak_mib = max(task.gpu_peak_mib or 0, observed)
            last_gpu_sample = now
        for task in running:
            if task.status != "running":
                continue
            task.rss_peak = max(task.rss_peak, process_group_rss_bytes(task.process.pid))
            reason = None
            if now >= task.deadline:
                reason = "process deadline reached"
            elif task.rss_peak > JOB_RSS_LIMIT:
                reason = "owned process-group RSS exceeded 24 GiB"
            code = task.process.poll()
            if reason:
                terminate_owned(task.process)
                code = task.process.returncode
                task.status = "stopped_by_process_gate"
                journal.record("process_stopped", process=task.name, reason=reason)
            elif code is not None:
                task.status = "completed" if code == 0 else "failed"
            if task.status != "running":
                task.log.flush()
                os.fsync(task.log.fileno())
                task.log.close()
                results[task.name] = {
                    "returncode": code,
                    "status": task.status,
                    "elapsed_seconds": time.monotonic() - task.started,
                    "peak_rss_bytes": task.rss_peak,
                    "peak_gpu_process_memory_mib": task.gpu_peak_mib,
                    "gpu_process_memory_attribution": (
                        "matched_container_pid"
                        if task.gpu_peak_mib is not None
                        else "unknown_host_pid_namespace_reconciliation_required"
                    ),
                }
                journal.record("process_finished", process=task.name, **results[task.name])
        if now - last_heartbeat >= HEARTBEAT_SECONDS or not any(
            task.status == "running" for task in running
        ):
            journal.heartbeat(
                active=[task.name for task in running if task.status == "running"],
                cgroup_memory_current=snapshot["memory_current"],
                cgroup_memory_peak=snapshot["memory_peak"],
                memory_events=snapshot["memory_events"],
                disk_free_bytes=shutil.disk_usage(root).free,
                owned_bytes=owned_bytes,
            )
            last_heartbeat = now
        if any(task.status == "running" for task in running):
            time.sleep(min(5, max(0, global_deadline - time.monotonic())))
    return results
