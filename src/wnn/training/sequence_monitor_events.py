from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from wnn.training.sequence_fit_utils import truncate_with_evidence

EVENT_SCHEMA = "wnn.monitor.v1"
MAX_EVALUATION_EVENTS = 10_000
MAX_EVALUATION_LINE_BYTES = 32_768
MAX_EVENT_LINE_BYTES = 131_072
MAX_EVENT_LOG_BYTES = 1 << 30
MONITOR_EVENT_FILENAME = "monitor_events.jsonl"
MONITOR_EVALUATION_FILENAME = "monitor_evaluations.jsonl"
MonitorKind = Literal["status", "train_step", "evaluation", "diagnostics", "resources"]
_EVENT_KINDS = {"status", "train_step", "evaluation", "diagnostics", "resources"}


class MonitorEventWriter:
    """Append monitor events and retain one bounded, replay-safe evaluation per epoch."""

    def __init__(self, path: Path, *, resume: bool) -> None:
        """Open event streams for a new run or validate their resumed append-only prefixes."""
        self.path = path
        self.evaluation_path = path.with_name(MONITOR_EVALUATION_FILENAME)
        _prepare_append_file(path, resume)
        _prepare_append_file(self.evaluation_path, resume)
        self._evaluation_events: dict[
            int, tuple[int, dict[str, int | float], dict[str, object]]
        ] = {}
        if resume:
            self._evaluation_events = {}
        self._restored = not resume

    def append(
        self,
        kind: MonitorKind,
        step: int,
        epoch: int | float,
        metrics: dict[str, int | float],
        context: dict[str, object],
    ) -> None:
        """Append one validated event, mirroring each unique evaluation to the archive."""
        if not self._restored:
            raise RuntimeError("restore monitor offsets from the safe checkpoint before appending")
        event = _event(kind, step, epoch, metrics, context)
        line = _event_line(event)
        if kind == "evaluation":
            if len(line) > MAX_EVALUATION_LINE_BYTES:
                raise ValueError("monitor evaluation event exceeds its bounded line size")
            previous = self._evaluation_events.get(epoch)
            if previous is not None:
                if previous != _evaluation_signature(step, metrics, context):
                    raise ValueError("resumed evaluation conflicts with its saved epoch report")
                return
            else:
                if len(self._evaluation_events) >= MAX_EVALUATION_EVENTS:
                    raise ValueError("monitor evaluation journal reached its 10,000-epoch bound")
                _append_line(self.evaluation_path, line)
                self._evaluation_events[epoch] = _evaluation_signature(step, metrics, context)
        if len(line) > MAX_EVENT_LINE_BYTES:
            raise ValueError("monitor event exceeds its bounded line size")
        if self.path.stat().st_size + len(line) > MAX_EVENT_LOG_BYTES:
            raise ValueError("monitor event log reached its 1 GiB bound")
        _append_line(self.path, line)

    def offsets(self) -> dict[str, int]:
        """Return flushed byte offsets for both monitor streams at a safe boundary."""
        return monitor_event_offsets(self.path.parent)

    def restore_offsets(self, offsets: Mapping[str, object], global_step: int) -> None:
        """Preserve post-checkpoint bytes, truncate both streams, and reload eval history."""
        if set(offsets) != {"events", "evaluations"} or any(
            type(value) is not int or value < 0 for value in offsets.values()
        ):
            raise ValueError("checkpoint monitor offsets are malformed")
        truncate_with_evidence(
            self.path,
            int(offsets["events"]),
            rank=0,
            global_step=global_step,
        )
        truncate_with_evidence(
            self.evaluation_path,
            int(offsets["evaluations"]),
            rank=0,
            global_step=global_step,
        )
        self._evaluation_events = _load_evaluation_events(self.evaluation_path)
        self._restored = True


def _prepare_append_file(path: Path, resume: bool) -> None:
    """Create only fresh logs and reject links, unexpected files, or missing resume logs."""
    if path.is_symlink():
        raise ValueError("monitor event path cannot be a symlink")
    if path.exists():
        if not resume or not stat.S_ISREG(path.lstat().st_mode):
            raise FileExistsError("monitor event path already exists or is not regular")
        if path.stat().st_size > (
            MAX_EVALUATION_EVENTS * MAX_EVALUATION_LINE_BYTES
            if path.name == MONITOR_EVALUATION_FILENAME
            else MAX_EVENT_LOG_BYTES
        ):
            raise ValueError(f"monitor log exceeds its configured byte bound: {path.name}")
    elif resume:
        raise FileNotFoundError(f"resumed monitored fit is missing {path.name}")
    else:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)


def monitor_event_offsets(output: Path) -> dict[str, int]:
    """Read both shared stream positions and require regular non-link files."""
    paths = {
        "events": output / MONITOR_EVENT_FILENAME,
        "evaluations": output / MONITOR_EVALUATION_FILENAME,
    }
    result: dict[str, int] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"monitor log is not a regular file: {path.name}")
        result[name] = path.stat().st_size
        limit = (
            MAX_EVENT_LOG_BYTES
            if name == "events"
            else MAX_EVALUATION_EVENTS * MAX_EVALUATION_LINE_BYTES
        )
        if result[name] > limit:
            raise ValueError(f"monitor log exceeds its configured byte bound: {path.name}")
    return result


def _load_evaluation_events(
    path: Path,
) -> dict[int, tuple[int, dict[str, int | float], dict[str, object]]]:
    """Validate the bounded epoch journal before permitting an idempotent resume."""
    maximum_size = MAX_EVALUATION_EVENTS * MAX_EVALUATION_LINE_BYTES
    if path.stat().st_size > maximum_size:
        raise ValueError("monitor evaluation journal exceeds its bounded size")
    records: dict[int, tuple[int, dict[str, int | float], dict[str, object]]] = {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("monitor evaluation journal contains a partial line")
            if len(line) > MAX_EVALUATION_LINE_BYTES:
                raise ValueError("monitor evaluation journal contains an oversized line")
            _load_evaluation_event(line, records)
    return records


def _load_evaluation_event(
    line: bytes,
    records: dict[int, tuple[int, dict[str, int | float], dict[str, object]]],
) -> None:
    """Validate and retain one complete epoch report line."""
    event = json.loads(line)
    if not isinstance(event, dict) or set(event) != {
        "schema",
        "kind",
        "step",
        "epoch",
        "wall_time",
        "metrics",
        "context",
    }:
        raise ValueError("monitor evaluation journal has an invalid event shape")
    if event["schema"] != EVENT_SCHEMA or event["kind"] != "evaluation":
        raise ValueError("monitor evaluation journal contains a non-evaluation event")
    epoch, step = event["epoch"], event["step"]
    if type(epoch) is not int or not 1 <= epoch <= MAX_EVALUATION_EVENTS:
        raise ValueError("monitor evaluation epoch is outside its configured bound")
    if type(step) is not int or step < 0:
        raise ValueError("monitor evaluation step is invalid")
    if epoch in records:
        raise ValueError("monitor evaluation journal contains duplicate epochs")
    metrics, context = event["metrics"], event["context"]
    _validate_metrics(metrics)
    if not isinstance(context, dict):
        raise ValueError("monitor evaluation context must be a mapping")
    records[epoch] = _evaluation_signature(step, metrics, context)


def _evaluation_signature(
    step: int, metrics: dict[str, int | float], context: dict[str, object]
) -> tuple[int, dict[str, int | float], dict[str, object]]:
    """Compare replayed official scores and cohort identity, excluding volatile diagnostics."""
    stable_metrics = {
        key: value
        for key, value in metrics.items()
        if not key.startswith("timing/") and "/prediction_" not in key
    }
    return step, stable_metrics, context.copy()


def _event(
    kind: MonitorKind,
    step: int,
    epoch: int,
    metrics: dict[str, int | float],
    context: dict[str, object],
) -> dict[str, object]:
    """Validate event fields and create the shared on-disk schema record."""
    if kind not in _EVENT_KINDS or type(step) is not int or step < 0:
        raise ValueError("monitor event kind or step is invalid")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
        raise ValueError("monitor event epoch is invalid")
    if not math.isfinite(float(epoch)) or epoch < 0:
        raise ValueError("monitor event epoch is invalid")
    if kind == "evaluation" and type(epoch) is not int:
        raise ValueError("monitor evaluation epoch must be an integer completed-epoch counter")
    _validate_metrics(metrics)
    return {
        "schema": EVENT_SCHEMA,
        "kind": kind,
        "step": step,
        "epoch": epoch,
        "wall_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
        "context": context,
    }


def _validate_metrics(metrics: object) -> None:
    """Require a flat slash-key map with only finite numeric values."""
    if not isinstance(metrics, dict):
        raise TypeError("monitor metrics must be a flat mapping")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key or "/" not in key:
            raise ValueError("monitor metric names must be nonempty slash keys")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"monitor metric {key} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"monitor metric {key} must be finite")


def _event_line(event: Mapping[str, object]) -> bytes:
    return (json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _append_line(path: Path, payload: bytes) -> None:
    """Append one complete JSONL record without replacing prior bytes."""
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("monitor event stream must remain a regular file")
        _write_all(descriptor, payload)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one complete JSONL line despite operating-system short writes."""
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("monitor event append made no progress")
        offset += written
