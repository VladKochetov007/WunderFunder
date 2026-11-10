from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

PROGRESS_FILENAME = "monitor_progress.json"
LOGGER = logging.getLogger(__name__)


class MonitorProgress:
    """Publish phase and successful-update progress independently of the CUDA loop.

    The external observer remains the process-liveness authority. This optional
    thread only snapshots Python progress; a fresh snapshot is not proof that an
    optimizer update or evaluation has completed.
    """

    def __init__(self, output: Path) -> None:
        """Enable periodic snapshots when the launch wrapper supplies a cadence."""
        self.path = output / PROGRESS_FILENAME
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._state: dict[str, object] = {
            "schema": "wnn.monitor-progress.v1",
            "pid": os.getpid(),
            "phase": "initializing",
            "step": 0,
            "epoch": 0,
            "phase_started_at": time.time(),
        }
        cadence = float(os.environ.get("WNN_MONITOR_HEARTBEAT_SECONDS", "0"))
        if cadence and not 1 <= cadence <= 60:
            raise ValueError("monitor heartbeat cadence must be zero or between 1 and 60 seconds")
        self._thread: threading.Thread | None = None
        if cadence:
            self.flush()
            self._thread = threading.Thread(
                target=self._heartbeat, args=(cadence,), name="monitor-progress", daemon=True
            )
            self._thread.start()

    def phase(self, phase: str, step: int, epoch: int | float) -> None:
        """Publish a phase boundary without changing any training state."""
        with self._lock:
            if self._state["phase"] != phase:
                self._state["phase_started_at"] = time.time()
            self._state.update(phase=phase, step=step, epoch=epoch)
        self.flush()

    def optimizer_update(self, step: int, epoch: float, forwarded_rows_per_second: float) -> None:
        """Record only an optimizer update that passed the finite-update checks."""
        with self._lock:
            if self._state["phase"] != "training":
                self._state["phase_started_at"] = time.time()
            self._state.update(
                phase="training",
                step=step,
                epoch=epoch,
                last_successful_update=step,
                last_successful_update_at=time.time(),
                forwarded_rows_per_second=forwarded_rows_per_second,
            )

    def pretraining_epoch(self, epoch: int, step: int, loss: float) -> None:
        """Record training-only future-input loss without creating validation metrics."""
        if epoch < 1 or step < 0 or not 0 <= loss < float("inf"):
            raise ValueError("pretraining progress requires a finite loss and completed epoch")
        with self._lock:
            self._state.update(
                phase="pretraining_epoch_completed",
                step=step,
                epoch=epoch,
                pretext_loss=loss,
                pretext_task="future_input_innovations_v1",
                metric_scope="training_only",
                last_successful_epoch=epoch,
                last_successful_epoch_at=time.time(),
            )
        self.flush()

    def flush(self) -> None:
        """Atomically replace a small observational snapshot, separate from fit logs."""
        with self._lock:
            state = dict(self._state)
            state["heartbeat_at"] = datetime.now(UTC).isoformat()
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            payload = json.dumps(state, sort_keys=True, allow_nan=False).encode()
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, self.path)

    def close(self) -> None:
        """Stop the optional snapshot thread after a terminal event."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.flush()

    def _heartbeat(self, cadence: float) -> None:
        """Keep reporting the same phase even while evaluation or saves block."""
        while not self._stopped.wait(cadence):
            try:
                self.flush()
            except (OSError, ValueError):
                LOGGER.exception(
                    "monitor progress snapshot failed; external observer remains active"
                )
                return


def record_successful_update(
    monitor: object, step: int, epoch: float, forwarded_rows_per_second: float
) -> None:
    """Publish progress for canonical monitors while preserving legacy monitor seams."""
    progress = getattr(monitor, "progress", None)
    if isinstance(progress, MonitorProgress):
        progress.optimizer_update(step, epoch, forwarded_rows_per_second)


def close_monitor_progress(monitor: object) -> None:
    """Close observational snapshots without bypassing distributed worker cleanup."""
    progress = getattr(monitor, "progress", None)
    if isinstance(progress, MonitorProgress):
        try:
            progress.close()
        except (OSError, ValueError):
            LOGGER.exception("final monitor snapshot failed")
