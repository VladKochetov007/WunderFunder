from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from wnn.dashboard.schema import MAX_METRIC_KEYS_PER_RUN, parse_event
from wnn.dashboard.storage import MissingFileError, open_run_file

MAX_RETAINED_EVENTS = 4_000
MAX_RETAINED_BYTES = 1_048_576
INITIAL_TAIL_BYTES = 2_097_152
MAX_READ_BYTES = 1_048_576
MAX_LINE_BYTES = 131_072
STALE_AFTER_SECONDS = 30


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for reader health reporting."""
    return datetime.now(UTC)


def _downsample(events: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """Select evenly spaced retained events while keeping both endpoints."""
    if len(events) <= limit:
        return events
    if limit < 2:
        return events[-1:]
    count = len(events)
    indices = {round(index * (count - 1) / (limit - 1)) for index in range(limit)}
    return [event for index, event in enumerate(events) if index in indices]


@dataclass
class EventStream:
    """Incremental bounded state for one append-only monitor event log."""

    events: deque[dict[str, object]] = field(default_factory=deque)
    retained_bytes: int = 0
    signature: tuple[int, int] | None = None
    offset: int = 0
    marker: bytes = b""
    partial: bytearray = field(default_factory=bytearray)
    discarding_long_line: bool = False
    invalid_lines: int = 0
    total_events: int = 0
    dropped_events: int = 0
    initial_history_limited: bool = False
    read_error: str | None = None
    seen_file: bool = False
    last_event_at: datetime | None = None

    def refresh(self, root: Path, run_id: str) -> None:
        """Read a bounded append or reset after rotation, truncation, or error."""
        try:
            with open_run_file(root, run_id, "monitor_events.jsonl") as handle:
                file_stat = os.fstat(handle.fileno())
                self.read_error = None
                self.seen_file = True
                identity = (file_stat.st_dev, file_stat.st_ino)
                marker_changed = self._marker_changed(handle, file_stat.st_size)
                if self.signature != identity or file_stat.st_size < self.offset or marker_changed:
                    self._reset(identity)
                    self._read_initial(handle, file_stat.st_size)
                else:
                    self._read_append(handle, file_stat.st_size)
                self._update_marker(handle)
        except MissingFileError:
            if self.seen_file:
                self.read_error = "monitor log is unavailable"
        except (OSError, ValueError):
            self.read_error = "monitor log could not be read safely"

    def status(self) -> str:
        """Summarize whether the log is live, stale, empty, or has errors."""
        if self.read_error is not None or self.invalid_lines:
            return "error"
        if not self.events:
            return "no_data"
        if self.last_event_at is None:
            return "error"
        age = (_utc_now() - self.last_event_at).total_seconds()
        return "stale" if age > STALE_AFTER_SECONDS else "live"

    def snapshot(self, limit: int) -> dict[str, object]:
        """Return a bounded, downsampled API representation of retained events."""
        retained = list(self.events)
        keys: set[str] = set()
        metric_keys_truncated = False
        for event in retained:
            for key in event["metrics"]:
                if not isinstance(key, str) or key in keys:
                    continue
                if len(keys) >= MAX_METRIC_KEYS_PER_RUN:
                    metric_keys_truncated = True
                    continue
                keys.add(key)
        metric_keys = sorted(keys)
        last_event_at = self.last_event_at.isoformat() if self.last_event_at else None
        return {
            "status": self.status(),
            "events": _downsample(retained, limit),
            "metric_keys": metric_keys,
            "metric_keys_truncated": metric_keys_truncated,
            "event_count": self.total_events,
            "retained_count": len(retained),
            "dropped_count": self.dropped_events,
            "history_limited": self.initial_history_limited,
            "invalid_lines": self.invalid_lines,
            "last_event_at": last_event_at,
            "error": self.read_error
            or (f"{self.invalid_lines} invalid monitor lines" if self.invalid_lines else None),
        }

    def _reset(self, identity: tuple[int, int]) -> None:
        self.events.clear()
        self.retained_bytes = 0
        self.signature = identity
        self.offset = 0
        self.marker = b""
        self.partial.clear()
        self.discarding_long_line = False
        self.invalid_lines = 0
        self.total_events = 0
        self.dropped_events = 0
        self.initial_history_limited = False
        self.last_event_at = None

    def _read_initial(self, handle: BinaryIO, size: int) -> None:
        start = max(0, size - INITIAL_TAIL_BYTES)
        self.initial_history_limited = start > 0
        handle.seek(start)
        chunk = handle.read(min(INITIAL_TAIL_BYTES, size - start))
        self.offset = start + len(chunk)
        if start > 0:
            newline = chunk.find(b"\n")
            if newline < 0:
                self.discarding_long_line = True
                return
            chunk = chunk[newline + 1 :]
        self._consume(chunk)

    def _read_append(self, handle: BinaryIO, size: int) -> None:
        available = max(0, min(MAX_READ_BYTES, size - self.offset))
        if not available:
            return
        handle.seek(self.offset)
        chunk = handle.read(available)
        self.offset += len(chunk)
        self._consume(chunk)

    def _marker_changed(self, handle: BinaryIO, size: int) -> bool:
        if not self.marker or size < self.offset:
            return False
        marker_start = self.offset - len(self.marker)
        handle.seek(marker_start)
        return handle.read(len(self.marker)) != self.marker

    def _update_marker(self, handle: BinaryIO) -> None:
        marker_size = min(64, self.offset)
        if not marker_size:
            self.marker = b""
            return
        handle.seek(self.offset - marker_size)
        self.marker = handle.read(marker_size)

    def _consume(self, chunk: bytes) -> None:
        data = bytes(self.partial) + chunk
        self.partial.clear()
        if self.discarding_long_line:
            newline = data.find(b"\n")
            if newline < 0:
                return
            data = data[newline + 1 :]
            self.discarding_long_line = False
        lines = data.split(b"\n")
        for line in lines[:-1]:
            self._consume_line(line)
        trailing = lines[-1]
        if len(trailing) > MAX_LINE_BYTES:
            self.invalid_lines += 1
            self.discarding_long_line = True
        elif trailing:
            self.partial.extend(trailing)

    def _consume_line(self, line: bytes) -> None:
        line = line.removesuffix(b"\r")
        if len(line) > MAX_LINE_BYTES:
            self.invalid_lines += 1
            return
        event = parse_event(line)
        if event is None:
            self.invalid_lines += 1
            return
        self.total_events += 1
        event_size = len(
            json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        if event_size > MAX_RETAINED_BYTES:
            self.dropped_events += 1
            return
        self.events.append(event)
        self.retained_bytes += event_size
        while len(self.events) > MAX_RETAINED_EVENTS or self.retained_bytes > MAX_RETAINED_BYTES:
            dropped = self.events.popleft()
            self.retained_bytes -= len(
                json.dumps(dropped, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )
            self.dropped_events += 1
        event_time = event["wall_time"]
        if isinstance(event_time, str):
            self.last_event_at = datetime.fromisoformat(event_time)
