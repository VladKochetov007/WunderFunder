"""Linux process-tree watchdog draft for the W384 v4 Kaggle entrypoint.

The entrypoint must call ``start_watchdog`` before bundle discovery, archive
hashing/extraction, test imports, or CUDA initialization, then call
``stop_watchdog`` in a ``finally`` block. Keep this file out of the frozen v3
bundle; root review is required before integrating it into a new v4 identity.
"""

from __future__ import annotations

import json
import os
import select
import signal
import time
from pathlib import Path


def _proc_identity(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], int(fields[19])


def _children(pid: int) -> list[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(value) for value in raw.split()] if raw else []


def _descendants(root_pid: int, monitor_pid: int) -> list[int]:
    ordered: list[int] = []
    seen = {root_pid, monitor_pid}

    def visit(pid: int) -> None:
        for child in _children(pid):
            if child in seen:
                continue
            seen.add(child)
            visit(child)
            ordered.append(child)

    visit(root_pid)
    return ordered


def _write_timeout(path: Path, record: dict[str, object]) -> None:
    payload = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(payload)
    os.replace(temp, path)


def _watch(
    root_pid: int,
    root_start_ticks: int,
    timeout_seconds: int,
    report: str,
    ready_fd: int,
) -> None:
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    os.write(ready_fd, b"R")
    os.close(ready_fd)
    monitor_pid = os.getpid()
    started = time.monotonic()
    deadline = started + timeout_seconds
    while not stopping:
        identity = _proc_identity(root_pid)
        if identity is None or identity[1] != root_start_ticks or identity[0] == "Z":
            return
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(1.0, deadline - now))
    if stopping:
        return

    victims = _descendants(root_pid, monitor_pid)
    captured = []
    for pid in victims:
        identity = _proc_identity(pid)
        if identity is not None:
            captured.append({"pid": pid, "start_ticks": identity[1]})
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    grace_deadline = time.monotonic() + 10.0
    while time.monotonic() < grace_deadline:
        if not any(
            identity is not None and identity[0] != "Z"
            for item in captured
            if (identity := _proc_identity(int(item["pid"]))) is not None
        ):
            break
        time.sleep(0.2)

    killed = []
    for item in captured:
        pid = int(item["pid"])
        identity = _proc_identity(pid)
        if (
            identity is not None
            and identity[0] != "Z"
            and identity[1] == item["start_ticks"]
        ):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass

    record = {
        "status": "whole_job_deadline_exceeded",
        "root_pid": root_pid,
        "root_start_ticks": root_start_ticks,
        "timeout_seconds": timeout_seconds,
        "watchdog_elapsed_seconds": time.monotonic() - started,
        "term_sent_pids": [item["pid"] for item in captured],
        "kill_sent_pids": killed,
        "kill_order": "descendants first; main kernel process after timeout record is flushed",
    }
    try:
        _write_timeout(Path(report), record)
    finally:
        identity = _proc_identity(root_pid)
        if identity is not None and identity[1] == root_start_ticks:
            try:
                os.kill(root_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def start_watchdog(
    timeout_seconds: int = 32_400,
    report_path: str = "/kaggle/working/whole_job_watchdog.json",
) -> int:
    """Fork a monitor before any worker setup and return its PID to the parent."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    root_pid = os.getpid()
    identity = _proc_identity(root_pid)
    if identity is None:
        raise RuntimeError("Cannot bind watchdog to the current process identity")
    ready_read_fd, ready_write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read_fd)
        try:
            _watch(root_pid, identity[1], timeout_seconds, report_path, ready_write_fd)
        except BaseException as exc:
            record = {
                "status": "watchdog_internal_error_fail_closed",
                "root_pid": root_pid,
                "root_start_ticks": identity[1],
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            try:
                _write_timeout(Path(report_path), record)
            except BaseException:
                pass
            current = _proc_identity(root_pid)
            if current is not None and current[1] == identity[1]:
                try:
                    os.kill(root_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            os._exit(0)
    os.close(ready_write_fd)
    ready, _, _ = select.select([ready_read_fd], [], [], 5.0)
    if not ready or os.read(ready_read_fd, 1) != b"R":
        os.close(ready_read_fd)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        raise RuntimeError("Whole-job watchdog failed to start; refusing kernel work")
    os.close(ready_read_fd)
    return pid


def stop_watchdog(pid: int) -> None:
    """Stop and reap the monitor after the kernel's ordinary work finishes."""
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    if waited == pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    while True:
        try:
            os.waitpid(pid, 0)
            return
        except InterruptedError:
            continue
