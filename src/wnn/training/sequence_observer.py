from __future__ import annotations

import json
import math
import os
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

from wnn.training.sequence_observer_resources import (
    ProcessReading,
    gpu_resources,
    host_resources,
    process_table,
    read_process,
    tracked_processes,
)

OBSERVER_FILENAME = "monitor_observer.jsonl"
MAX_LOG_BYTES = 256 * 1_048_576
MAX_TAIL_BYTES = 1_048_576


def _small_json(path: Path) -> dict[str, object]:
    """Read an optional bounded local observation, without following links."""
    try:
        if path.is_symlink() or path.stat().st_size > 65_536:
            return {}
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _tail_event(path: Path) -> dict[str, object]:
    """Read the last complete producer event; a partial write is not an event."""
    try:
        if path.is_symlink():
            return {}
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - MAX_TAIL_BYTES))
            lines = stream.read(MAX_TAIL_BYTES).split(b"\n")
        for line in reversed(lines[:-1]):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("schema") == "wnn.monitor.v1":
                return value
    except OSError:
        return {}
    return {}


def _age(timestamp: object, now: float) -> float | None:
    """Decode optional seconds or aware ISO time, leaving missing ages unknown."""
    try:
        value = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            if isinstance(timestamp, str)
            else float(timestamp)
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, now - value) if math.isfinite(value) else None


class SequenceObserver:
    """Observe a bound worker family and every GPU independently of training I/O."""

    def __init__(self, output: Path, pid: int, interval: float = 10) -> None:
        """Bind the root PID's actual Linux start time before the first sample.

        No process is signalled. Descendant membership uses PPID and start ticks;
        a later process reusing a bound PID is excluded. Output may precede the
        trainer's directory creation, so the observer creates no run directory.
        """
        if not 1 <= interval <= 60:
            raise ValueError("observer interval must be between 1 and 60 seconds")
        root = read_process(pid)
        if root is None or not root.is_alive:
            raise ValueError("observer requires a live readable root PID")
        self.output = output
        self.root = root
        self.interval = interval
        self.identities = {pid: root.start_ticks}
        self._previous: tuple[float, dict[int, ProcessReading]] | None = None

    def sample(self, returncode: int | None = None) -> dict[str, object]:
        """Append one complete observation, including missing/stalled progress ages."""
        now = time.time()
        processes = tracked_processes(process_table(), self.identities)
        alive = {pid: reading for pid, reading in processes.items() if reading.is_alive}
        metrics, gpu_context = gpu_resources(set(alive))
        metrics.update(self._process_metrics(alive))
        metrics.update(host_resources(self.output))
        progress = _small_json(self.output / "monitor_progress.json")
        progress_pid = progress.get("pid")
        if progress_pid not in alive:
            progress = {}
        latest = _tail_event(self.output / "monitor_events.jsonl")
        context = latest.get("context", {})
        context = context if isinstance(context, dict) else {}
        phase = progress.get("phase") or context.get("phase") or context.get("status") or "startup"
        terminal = _small_json(self.output / "terminal_status.json")
        if not alive:
            phase = terminal.get("status") or ("completed" if returncode == 0 else "exited")
        for name, timestamp in (
            ("event", latest.get("wall_time")),
            ("progress_heartbeat", progress.get("heartbeat_at")),
            ("successful_update", progress.get("last_successful_update_at")),
            ("phase", progress.get("phase_started_at")),
        ):
            age = _age(timestamp, now)
            if age is not None:
                metrics[f"observer/{name}_age_seconds"] = age
        checkpoint_times: list[float] = []
        for name in ("training_state.pt", "epoch_index.json", "model.pt"):
            path = self.output / name
            try:
                if not path.is_symlink() and path.is_file():
                    checkpoint_times.append(path.stat().st_mtime)
            except OSError:
                continue
        if checkpoint_times:
            metrics["observer/checkpoint_age_seconds"] = max(0, now - max(checkpoint_times))
        rate = progress.get("forwarded_rows_per_second")
        if isinstance(rate, (int, float)) and math.isfinite(rate):
            metrics["throughput/last_successful_update_forwarded_rows_per_second"] = rate
        successful_update = progress.get("last_successful_update")
        if type(successful_update) is int:
            metrics["observer/last_successful_update"] = successful_update
        metrics["observer/interval_seconds"] = self.interval
        event = {
            "schema": "wnn.monitor.v1",
            "kind": "resources",
            "step": int(progress.get("step", latest.get("step", 0))),
            "epoch": float(progress.get("epoch", latest.get("epoch", 0))),
            "wall_time": datetime.now(UTC).isoformat(),
            "metrics": metrics,
            "context": {
                "phase": phase,
                "observer": "independent Linux process",
                "worker_pid": self.root.pid,
                "worker_start_ticks": self.root.start_ticks,
                "worker_returncode": returncode,
                "processes": [
                    {
                        "pid": item.pid,
                        "start_ticks": item.start_ticks,
                        "state": item.state,
                        "parent_pid": item.parent_pid,
                    }
                    for item in processes.values()
                ],
                "resource_scope": "all visible GPUs; bound worker and descendant process family",
                "sampling": f"every {self.interval:g} seconds; observations are not optimizer progress",
                "last_successful_update_source": "finite-update hook; absent until observed",
                **gpu_context,
            },
        }
        self._append(event)
        return event

    def _process_metrics(self, alive: dict[int, ProcessReading]) -> dict[str, int | float]:
        """Keep CPU utilization sampled from ticks, separately from process liveness."""
        now = time.monotonic()
        metrics: dict[str, int | float] = {
            "observer/process_alive_count": len(alive),
            "observer/root_process_alive": int(self.root.pid in alive),
            "cpu/worker_family_resident_bytes": sum(item.resident_bytes for item in alive.values()),
        }
        if self._previous is not None and hasattr(os, "sysconf"):
            previous_time, previous = self._previous
            ticks = sum(
                max(0, item.cpu_ticks - previous[pid].cpu_ticks)
                for pid, item in alive.items()
                if pid in previous
            )
            metrics["cpu/worker_family_utilization_pct"] = (
                100 * ticks / max(now - previous_time, 1e-9) / os.sysconf("SC_CLK_TCK")
            )
        self._previous = now, dict(alive)
        return metrics

    def _append(self, event: dict[str, object]) -> None:
        """Append a separate bounded journal; never mutate resumable fit streams."""
        if not self.output.exists():
            return
        if self.output.is_symlink() or not self.output.is_dir():
            raise ValueError("observer output must be a direct run directory")
        payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(payload) > 131_072:
            raise ValueError("observer event exceeds its line-size bound")
        descriptor = os.open(
            self.output / OBSERVER_FILENAME,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size + len(payload) > MAX_LOG_BYTES
            ):
                raise ValueError("observer journal is not regular or reached its 256 MiB bound")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("observer append made no progress")
                offset += written
        finally:
            os.close(descriptor)
