"""External time/call limits and subprocess ownership for Harness² probes."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Budget:
    """One wall-clock deadline shared by probes, editors and selection."""

    seconds: int
    max_rollouts: int
    max_calls: int
    started: float = field(default_factory=time.monotonic)
    rollouts: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.seconds <= 14400:
            raise ValueError("Wall budget must be between 1 and 14400 seconds")

    def remaining(self) -> int:
        """Return whole remaining seconds; never turn an expired limit into a default."""
        remaining = math.floor(self.seconds - (time.monotonic() - self.started))
        if remaining < 1:
            raise TimeoutError("Harness² campaign wall budget exhausted")
        return remaining

    def admit_rollout(self) -> None:
        """Charge each attempted execution before starting it."""
        self.remaining()
        if self.rollouts >= self.max_rollouts:
            raise RuntimeError("Harness² rollout budget exhausted")
        self.rollouts += 1

    def admit_call(self) -> None:
        """Charge an editor/selector/reflection attempt before calling a model."""
        self.remaining()
        if self.calls >= self.max_calls:
            raise RuntimeError("Harness² editor call budget exhausted")
        self.calls += 1


def write_json(path: Path, value: object) -> None:
    """Write durable metadata without serializing credentials or process environments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_worker(argv: list[str], workspace: Path, timeout: int) -> None:
    """Run one explicitly configured worker and terminate its owned tree on timeout."""
    if not argv or not all(isinstance(part, str) and part for part in argv):
        raise ValueError("Worker argv must be a nonempty array of strings")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    flags = (
        {"start_new_session": True}
        if os.name != "nt"
        else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
    )
    with (
        (workspace / "worker.stdout.log").open("wb") as stdout,
        (workspace / "worker.stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            **flags,
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"Worker exited with status {code}; see preserved worker logs")
