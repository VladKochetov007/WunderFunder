"""R2 birth-bound descendant ownership for torchrun's per-rank sessions."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from b300_driver_core_r1 import (
    ProcIdentity,
    atomic_json,
    directory_bytes,
    proc_entry,
    process_table,
)


class OwnedRunError(RuntimeError):
    """A child failed, exceeded its cutoff, or violated a resource guard."""


@dataclass
class ChildHandle:
    """Leader birth plus every observed descendant PID/birth, across sessions."""

    pid: int
    birth_ticks: int
    cell: str
    stage: str
    descendants: dict[int, int] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.descendants[self.pid] = self.birth_ticks

    def capture(self, table: dict[int, ProcIdentity]) -> dict[int, int]:
        """Update and return a stable closure snapshot under its per-launcher lock."""
        with self.lock:
            self.descendants = descendant_closure(table, self.descendants)
            return dict(self.descendants)


def descendant_closure(
    table: dict[int, ProcIdentity], known: dict[int, int]
) -> dict[int, int]:
    """Extend exact birth-bound ownership through parent links, ignoring foreign sessions."""
    result = dict(known)
    changed = True
    while changed:
        changed = False
        for entry in table.values():
            parent_birth = result.get(entry.ppid)
            parent = table.get(entry.ppid)
            if (
                parent_birth is not None
                and parent is not None
                and parent.start_ticks == parent_birth
                and result.get(entry.pid) != entry.start_ticks
            ):
                result[entry.pid] = entry.start_ticks
                changed = True
    return result


def live_owned(
    table: dict[int, ProcIdentity], known: dict[int, int]
) -> list[ProcIdentity]:
    """Return only currently present PID/birth pairs from a persistent ownership closure."""
    return [
        entry
        for pid, birth in known.items()
        if (entry := table.get(pid)) is not None and entry.start_ticks == birth
    ]


class ResourceGuard:
    """Bound only recorded launcher descendants, including ranks in new sessions."""

    def __init__(self, root: Path, control: Path, caps: dict[str, int]) -> None:
        self.root = root
        self.control = control
        self.caps = caps
        self._lock = threading.RLock()
        self._children: dict[int, ChildHandle] = {}
        self.failure: str | None = None
        self._last_metrics = 0.0

    def register(self, child: ChildHandle) -> None:
        with self._lock:
            self._children[child.pid] = child

    def unregister(self, child: ChildHandle) -> None:
        with self._lock:
            self._children.pop(child.pid, None)

    def check(
        self, deadline: float, table: dict[int, ProcIdentity] | None = None
    ) -> dict[str, int | float]:
        """Update descendant closure, aggregate owned RSS, and enforce fixed resource bounds."""
        with self._lock:
            if self.failure:
                raise OwnedRunError(self.failure)
            if time.time() >= deadline:
                self.failure = "original fit deadline reached"
                raise OwnedRunError(self.failure)
            snapshot = process_table() if table is None else table
            for child in self._children.values():
                child.capture(snapshot)
            if time.monotonic() - self._last_metrics < 2.0:
                return {}
            self._last_metrics = time.monotonic()
            all_live = {
                (entry.pid, entry.start_ticks): entry
                for child in self._children.values()
                for entry in live_owned(snapshot, child.descendants)
            }
            rss = sum(entry.rss_bytes for entry in all_live.values())
            output_bytes = directory_bytes(self.root)
            filesystem = os.statvfs(self.root)
            free_bytes = filesystem.f_bavail * filesystem.f_frsize
            metrics: dict[str, int | float] = {
                "at_epoch": time.time(),
                "owned_descendant_count": len(all_live),
                "owned_rss_bytes": rss,
                "output_bytes": output_bytes,
                "free_bytes": free_bytes,
            }
            atomic_json(self.control / "resource_status_r2.json", metrics, replace=True)
            if rss > self.caps["owned_rss_bytes"]:
                self.failure = f"owned descendant RSS cap exceeded: {rss}"
            elif output_bytes > self.caps["output_bytes"]:
                self.failure = f"output cap exceeded: {output_bytes}"
            elif free_bytes < self.caps["minimum_free_bytes"]:
                self.failure = f"free-space reserve breached: {free_bytes}"
            if self.failure:
                raise OwnedRunError(self.failure)
            return metrics


def _capture(child: ChildHandle) -> list[ProcIdentity]:
    table = process_table()
    return live_owned(table, child.capture(table))


def _signal_exact(child: ChildHandle, sig: int) -> int:
    """Signal only still-live recorded PID/birth pairs; never use broad process groups."""
    signalled = 0
    with child.lock:
        owned = dict(child.descendants)
    for pid, birth in sorted(owned.items(), reverse=True):
        current = proc_entry(pid)
        if current is None or current.start_ticks != birth or current.state == "Z":
            continue
        try:
            os.kill(pid, sig)
            signalled += 1
        except ProcessLookupError:
            continue
    return signalled


def terminate_owned(child: ChildHandle, grace_seconds: float = 8.0) -> list[int]:
    """Stop all captured descendants, following further descendants before force-kill."""
    _capture(child)
    _signal_exact(child, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _capture(child):
            return []
        time.sleep(0.1)
    remaining = _capture(child)
    if remaining:
        _signal_exact(child, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _capture(child):
            time.sleep(0.1)
    return [entry.pid for entry in _capture(child)]


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
    """Run once, account/stop exact descendant births across session boundaries."""
    if os.name != "posix":
        raise RuntimeError(
            "B300 R2 ownership requires Linux /proc and PID birth identities"
        )
    receipt_path = control / "owned" / f"{cell}-{stage}-r2.json"
    log_path = control / "logs" / f"{cell}-{stage}-r2.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError(f"{cell}/{stage} R2 already has a receipt or log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipt: dict[str, object] = {
        "schema": "wnn.b300-pr74-owned-stage.r2",
        "cell": cell,
        "stage": stage,
        "controller_owner": owner,
        "command": command,
        "gpu_uuids": gpu_uuids,
        "started_epoch": started,
        "status": "launching",
        "automatic_retry": False,
    }
    atomic_json(receipt_path, receipt)
    effective_deadline = (
        min(deadline, started + stage_timeout) if stage_timeout else deadline
    )
    process: subprocess.Popen[bytes] | None = None
    child: ChildHandle | None = None
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
            identity = None
            for _ in range(100):
                identity = proc_entry(process.pid)
                if identity is not None or process.poll() is not None:
                    break
                time.sleep(0.02)
            if identity is None:
                raise OwnedRunError(
                    "launcher exited before its PID birth could be recorded"
                )
            if identity.pgid != process.pid or identity.session != process.pid:
                raise OwnedRunError(
                    "launcher did not receive its isolated start_new_session"
                )
            child = ChildHandle(process.pid, identity.start_ticks, cell, stage)
            guard.register(child)
            receipt.update(
                status="running",
                launcher={"pid": child.pid, "birth_ticks": child.birth_ticks},
                log_path=str(log_path),
            )
            atomic_json(receipt_path, receipt, replace=True)
            while process.poll() is None:
                snapshot = process_table()
                child.capture(snapshot)
                try:
                    resources = guard.check(effective_deadline, snapshot)
                    if resources:
                        receipt["resources"] = resources
                    progress()
                except Exception as error:  # noqa: BLE001 - always stop this exact owned tree.
                    failure = f"{type(error).__name__}: {error}"
                    leftovers = terminate_owned(child)
                    if leftovers:
                        failure += f"; owned pids survived cleanup: {leftovers}"
                    break
                time.sleep(0.5)
            exit_code = process.wait()
            if child is not None:
                remaining_deadline = time.monotonic() + 8.0
                while time.monotonic() < remaining_deadline and _capture(child):
                    time.sleep(0.1)
                leftovers = [entry.pid for entry in _capture(child)]
                if leftovers:
                    terminate_owned(child, grace_seconds=0.0)
                    failure = (
                        failure
                        or f"owned worker processes remained after launcher exit: {leftovers}"
                    )
                guard.unregister(child)
    except Exception as error:  # noqa: BLE001 - preserve failure and clean up recorded identities.
        failure = failure or f"{type(error).__name__}: {error}"
        if child is not None:
            leftovers = terminate_owned(child)
            if leftovers:
                failure += f"; owned pids survived cleanup: {leftovers}"
            guard.unregister(child)
        elif process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    receipt.update(
        finished_epoch=time.time(),
        exit_code=exit_code,
        timed_out=isinstance(failure, str) and "deadline" in failure.lower(),
        status="completed" if failure is None and exit_code == 0 else "failed",
        error=failure,
        owned_pid_births=(
            [
                {"pid": pid, "birth_ticks": birth}
                for pid, birth in sorted(child.capture(process_table()).items())
            ]
            if child is not None
            else []
        ),
    )
    receipt["log_path"] = str(log_path)
    if log_path.exists():
        receipt["log_sha256"] = _log_sha(log_path)
        receipt["log_bytes"] = log_path.stat().st_size
    atomic_json(receipt_path, receipt, replace=True)
    if receipt["status"] != "completed":
        raise OwnedRunError(f"{cell}/{stage} failed: {failure or f'exit {exit_code}'}")
    return receipt
