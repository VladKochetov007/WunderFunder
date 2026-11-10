"""Bounded POSIX process-group ownership and resource checks for B300 stages."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from b300_driver_core_r1 import (
    atomic_json,
    directory_bytes,
    owned_group_members,
    proc_entry,
    process_table,
)


class OwnedRunError(RuntimeError):
    """A child failed, exceeded its own deadline or hit a resource guard."""


@dataclass(frozen=True)
class ChildHandle:
    """Birth-bound identity for a receiver-created process session."""

    pid: int
    birth_ticks: int
    pgid: int
    cell: str
    stage: str


class ResourceGuard:
    """Latch deadline, RSS, disk-size or free-space violations for only owned children."""

    def __init__(self, root: Path, control: Path, caps: dict[str, int]) -> None:
        self.root = root
        self.control = control
        self.caps = caps
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._children: dict[int, ChildHandle] = {}
        self.failure: str | None = None

    def register(self, child: ChildHandle) -> None:
        with self._lock:
            self._children[child.pid] = child

    def unregister(self, child: ChildHandle) -> None:
        with self._lock:
            self._children.pop(child.pid, None)

    def check(self, deadline: float) -> dict[str, int | float]:
        """Check the original global cutoff and the aggregate resources of owned groups."""
        with self._lock:
            if self.failure:
                raise OwnedRunError(self.failure)
            if time.time() >= deadline:
                self.failure = "original phase deadline reached"
                raise OwnedRunError(self.failure)
            if time.monotonic() - self._last_check < 10:
                return {}
            self._last_check = time.monotonic()
            table = process_table()
            rss = sum(
                entry.rss_bytes
                for child in self._children.values()
                for entry in owned_group_members(table, child.pgid, child.birth_ticks)
            )
            output_bytes = directory_bytes(self.root)
            filesystem = os.statvfs(self.root)
            free_bytes = filesystem.f_bavail * filesystem.f_frsize
            metrics: dict[str, int | float] = {
                "at_epoch": time.time(),
                "owned_rss_bytes": rss,
                "output_bytes": output_bytes,
                "free_bytes": free_bytes,
            }
            atomic_json(self.control / "resource_status.json", metrics, replace=True)
            if rss > self.caps["owned_rss_bytes"]:
                self.failure = f"owned RSS cap exceeded: {rss}"
            elif output_bytes > self.caps["output_bytes"]:
                self.failure = f"output cap exceeded: {output_bytes}"
            elif free_bytes < self.caps["minimum_free_bytes"]:
                self.failure = f"free-space reserve breached: {free_bytes}"
            if self.failure:
                raise OwnedRunError(self.failure)
            return metrics


def _signal_owned_group(child: ChildHandle, sig: int) -> bool:
    """Signal only a session whose live members still match the recorded process-group ID."""
    if child.pgid != child.pid:
        return False
    table = process_table()
    members = owned_group_members(table, child.pgid, child.birth_ticks)
    if not members:
        return False
    os.killpg(child.pgid, sig)
    return True


def terminate_owned(child: ChildHandle, grace_seconds: float = 8.0) -> None:
    """Stop only the child session created by this receiver, after rechecking its birth."""
    if not _signal_owned_group(child, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        table = process_table()
        if not owned_group_members(table, child.pgid, child.birth_ticks):
            return
        time.sleep(0.1)
    _signal_owned_group(child, signal.SIGKILL)


def _log_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_owned(
    command: list[str],
    *,
    root: Path,
    control: Path,
    cell: str,
    stage: str,
    gpu_uuids: list[str],
    owner: dict[str, int | str],
    environment: dict[str, str],
    deadline: float,
    stage_timeout: int | None,
    guard: ResourceGuard,
    progress: Callable[[], None],
) -> dict[str, object]:
    """Run one unique command in a new session and persist its birth-bound completion receipt."""
    if os.name != "posix":
        raise RuntimeError(
            "B300 owned-stage runner requires Linux process/session semantics"
        )
    if not command or not command[0] or not isinstance(environment, dict):
        raise ValueError("owned command or sanitized environment is malformed")
    receipt_path = control / "owned" / f"{cell}-{stage}.json"
    log_path = control / "logs" / f"{cell}-{stage}.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError(
            f"{cell}/{stage} already has a receipt or log; retries are disabled"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "schema": "wnn.b300-pr74-owned-stage.v1",
        "cell": cell,
        "stage": stage,
        "controller_owner": owner,
        "command": command,
        "gpu_uuids": gpu_uuids,
        "started_epoch": time.time(),
        "status": "launching",
        "automatic_retry": False,
    }
    atomic_json(receipt_path, receipt)
    started = float(receipt["started_epoch"])
    effective_deadline = deadline
    if stage_timeout is not None:
        effective_deadline = min(deadline, started + stage_timeout)
    child: ChildHandle | None = None
    process: subprocess.Popen[bytes] | None = None
    failure: str | None = None
    exit_code: int | None = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            identity = proc_entry(process.pid)
            if identity is not None:
                child = ChildHandle(
                    process.pid, identity.start_ticks, identity.pgid, cell, stage
                )
                if child.pgid != child.pid or identity.session != child.pid:
                    raise OwnedRunError(
                        "new process did not receive an isolated process session"
                    )
                guard.register(child)
            receipt.update(
                status="running",
                child={
                    "pid": process.pid,
                    "birth_ticks": identity.start_ticks if identity else None,
                },
                process_group_id=process.pid,
                log_path=str(log_path),
            )
            atomic_json(receipt_path, receipt, replace=True)
            while process.poll() is None:
                try:
                    resources = guard.check(effective_deadline)
                    if resources:
                        receipt["resources"] = resources
                    progress()
                except Exception as error:  # noqa: BLE001 - terminate owned child on any callback failure.
                    failure = f"{type(error).__name__}: {error}"
                    if child is not None:
                        terminate_owned(child)
                    else:
                        process.terminate()
                    break
                time.sleep(2.0)
            if process.poll() is None:
                try:
                    exit_code = process.wait(timeout=12.0)
                except subprocess.TimeoutExpired:
                    if child is not None:
                        terminate_owned(child, grace_seconds=0.0)
                    exit_code = process.wait()
            else:
                exit_code = process.returncode
            if child is not None:
                table = process_table()
                leftovers = owned_group_members(table, child.pgid, child.birth_ticks)
                if leftovers:
                    terminate_owned(child, grace_seconds=0.0)
                    failure = (
                        failure
                        or "owned rank processes remained after the launcher exited"
                    )
                guard.unregister(child)
    except Exception as error:  # noqa: BLE001 - persist failure and clean up this exact child.
        failure = failure or f"{type(error).__name__}: {error}"
        if child is not None:
            terminate_owned(child)
            guard.unregister(child)
        elif process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    receipt.update(
        finished_epoch=time.time(),
        exit_code=exit_code,
        timed_out=isinstance(failure, str) and "deadline" in failure.lower(),
        status="completed" if failure is None and exit_code == 0 else "failed",
        error=failure,
    )
    if child is not None:
        receipt["child"] = {"pid": child.pid, "birth_ticks": child.birth_ticks}
        receipt["process_group_id"] = child.pgid
    receipt["log_path"] = str(log_path)
    if log_path.exists():
        receipt["log_sha256"] = _log_sha(log_path)
        receipt["log_bytes"] = log_path.stat().st_size
    atomic_json(receipt_path, receipt, replace=True)
    if receipt["status"] != "completed":
        raise OwnedRunError(f"{cell}/{stage} failed: {failure or f'exit {exit_code}'}")
    return receipt
