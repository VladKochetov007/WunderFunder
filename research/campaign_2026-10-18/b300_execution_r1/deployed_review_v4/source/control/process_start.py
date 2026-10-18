"""Owned process-group creation and cleanup for the finite controller."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def terminate_owned(process: subprocess.Popen[bytes], grace_seconds: float = 10) -> None:
    """Stop only the process group created by this controller."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@dataclass
class Running:
    name: str
    process: subprocess.Popen[bytes]
    log: Any
    started: float
    deadline: float
    rss_peak: int = 0
    gpu_uuid: str | None = None
    gpu_peak_mib: int | None = None
    status: str = "running"


def _set_affinity(cpus: tuple[int, ...]) -> Callable[[], None]:
    def apply() -> None:
        os.sched_setaffinity(0, set(cpus))

    return apply


def start_owned(
    *,
    name: str,
    argv: list[str],
    environment: dict[str, str],
    cpus: tuple[int, ...],
    log_path: Path,
    timeout_seconds: float,
) -> Running:
    """Start one actor in a fresh session with only its assigned CPU affinity."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(log_path)
    stream = log_path.open("xb")
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=environment.get("WNN_CWD") or None,
            env={key: value for key, value in environment.items() if key != "WNN_CWD"},
            start_new_session=True,
            preexec_fn=_set_affinity(cpus),
        )
    except BaseException:
        stream.close()
        raise
    return Running(
        name,
        process,
        stream,
        time.monotonic(),
        time.monotonic() + timeout_seconds,
        gpu_uuid=environment.get("CUDA_VISIBLE_DEVICES"),
    )
