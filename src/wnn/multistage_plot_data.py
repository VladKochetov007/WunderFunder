from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TextIO

from wnn.multistage_search import file_sha256

TRAIN_SAMPLE_STRIDE = 10
RESOURCE_SAMPLE_STRIDE = 5
MAX_EVENTS_PER_LINE = 131_072


def read_jsonl(path: Path, *, kinds: set[str] | None = None) -> Iterator[dict[str, object]]:
    """Stream bounded JSONL event objects and optionally retain selected kinds."""
    with path.open("r", encoding="utf-8") as stream:
        yield from _read_lines(stream, kinds)


def _read_lines(stream: TextIO, kinds: set[str] | None) -> Iterator[dict[str, object]]:
    for line_number, line in enumerate(stream, start=1):
        if len(line.encode("utf-8")) > MAX_EVENTS_PER_LINE:
            raise ValueError(f"monitor event line {line_number} exceeds the parser bound")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"monitor event line {line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"monitor event line {line_number} is not an object")
        if kinds is not None and value.get("kind") not in kinds:
            continue
        yield value


def read_training_events(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Read source events with a declared stride for dense telemetry and all evaluations."""
    samples: list[dict[str, object]] = []
    counts: dict[str, int] = {"train_step": 0, "resources": 0, "diagnostics": 0, "status": 0}
    for event in read_jsonl(path):
        kind = event.get("kind")
        if not isinstance(kind, str):
            continue
        if kind in counts:
            counts[kind] += 1
        if kind == "train_step":
            step = event.get("step")
            if isinstance(step, int) and step % TRAIN_SAMPLE_STRIDE == 0:
                samples.append(event)
        elif kind == "resources":
            if counts["resources"] % RESOURCE_SAMPLE_STRIDE == 1:
                samples.append(event)
        elif kind in ("diagnostics", "status"):
            samples.append(event)
    policy = {
        "train_step_stride": TRAIN_SAMPLE_STRIDE,
        "resources_stride": RESOURCE_SAMPLE_STRIDE,
        "train_step_events_seen": counts["train_step"],
        "resource_events_seen": counts["resources"],
        "diagnostic_events_seen": counts["diagnostics"],
        "status_events_seen": counts["status"],
        "retained_dense_events": len(samples),
        "raw_events_sha256": file_sha256(path),
    }
    return samples, policy


def read_evaluation_events(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Retain every producer-native evaluation event without downsampling."""
    rows = list(read_jsonl(path, kinds={"evaluation"}))
    policy = {"evaluation_events_seen": len(rows), "raw_evaluations_sha256": file_sha256(path)}
    return rows, policy


def elapsed_seconds(event: dict[str, object], first_wall_time: str | None) -> float | None:
    """Return elapsed event wall time when both source timestamps are timezone-aware."""
    wall_time = event.get("wall_time")
    if not isinstance(wall_time, str) or first_wall_time is None:
        return None
    try:
        start = datetime.fromisoformat(first_wall_time.replace("Z", "+00:00"))
        current = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = (current - start).total_seconds()
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def metric(event: dict[str, object], key: str) -> float | None:
    """Return one finite metric value from a native monitor event."""
    values = event.get("metrics")
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None
