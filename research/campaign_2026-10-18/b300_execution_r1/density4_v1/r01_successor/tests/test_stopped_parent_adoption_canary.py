"""Linux-only canary for a stopped owner and terminal adopted process group."""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))

from proc_snapshot import read_process  # noqa: E402
from worker_watch import group_has_live_members, pidfd_exited, pidfd_for  # noqa: E402


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "pidfd_open"),
    reason="requires Linux procfs and pidfd support",
)
def test_stopped_parent_can_adopt_completed_session_worker(tmp_path: Path) -> None:
    """A child fit exits to zombie while its stopped controller cannot reap it."""
    marker = tmp_path / "allow-worker-exit"
    worker_code = """
import pathlib, sys, time
marker = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 30
while not marker.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(11)
    time.sleep(0.02)
"""
    controller_code = """
import subprocess, sys
worker = subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], sys.argv[2]], start_new_session=True
)
print(worker.pid, flush=True)
if worker.wait() != 0:
    raise SystemExit(12)
"""
    controller = subprocess.Popen(
        [sys.executable, "-c", controller_code, worker_code, str(marker)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker_pid = None
    worker_pidfd = None
    try:
        assert controller.stdout is not None
        ready, _, _ = select.select([controller.stdout], [], [], 10)
        assert ready, "fake controller did not publish its worker PID"
        worker_pid = int(controller.stdout.readline().strip())
        worker = read_process(worker_pid)
        assert worker.identity.process_group == worker_pid
        worker_pidfd = pidfd_for(worker_pid, worker.identity.start_ticks)
        assert worker_pidfd is not None

        os.kill(controller.pid, signal.SIGSTOP)
        stop_deadline = time.monotonic() + 5
        while time.monotonic() < stop_deadline:
            if read_process(controller.pid).identity.state in {"T", "t"}:
                break
            time.sleep(0.01)
        assert read_process(controller.pid).identity.state in {"T", "t"}

        marker.touch()
        exit_deadline = time.monotonic() + 10
        while time.monotonic() < exit_deadline and not pidfd_exited(worker_pidfd):
            time.sleep(0.02)

        assert pidfd_exited(worker_pidfd), "worker did not exit while its parent was stopped"
        assert read_process(worker_pid).identity.state == "Z"
        assert not group_has_live_members(worker_pid)
        assert controller.poll() is None, "stopped controller unexpectedly completed"
    finally:
        marker.touch(exist_ok=True)
        try:
            if read_process(controller.pid).identity.state in {"T", "t"}:
                os.kill(controller.pid, signal.SIGCONT)
        except (FileNotFoundError, ProcessLookupError):
            pass
        try:
            controller.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            for pgid in (worker_pid, controller.pid):
                if pgid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
            controller.communicate(timeout=5)
        assert controller.returncode == 0
        if worker_pidfd is not None:
            os.close(worker_pidfd)
