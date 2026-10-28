from __future__ import annotations

import json
import os
import stat
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from wnn.dashboard.storage import (
    MAX_LISTED_RUNS,
    MissingFileError,
    UnsafePathError,
    discover_run_ids,
    read_run_metadata,
    validate_run_id,
)
from wnn.dashboard.stream import EventStream

MAX_ACTIVE_STREAMS = 20
MAX_SELECTED_RUNS = 8
DEFAULT_EVENT_LIMIT = 1_200
MAX_EVENT_LIMIT = 1_500
RUNS_REFRESH_COUNT = 12


def _empty_metadata() -> dict[str, object]:
    """Return explicit placeholders for unavailable run metadata."""
    return {"source": "unknown", "config": {}, "provenance": {}, "synthetic": False}


class DashboardReader:
    """Read bounded metadata and monitor events below one local run root."""

    def __init__(self, run_root: Path | str) -> None:
        """Validate and retain the selected run root without following a root symlink."""
        requested = Path(run_root).expanduser().absolute()
        if requested.is_symlink():
            raise ValueError("the run root cannot be a symlink")
        self.root = requested.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("the run root must be a directory")
        self._lock = RLock()
        self._streams: OrderedDict[str, EventStream] = OrderedDict()
        self._metadata: dict[str, tuple[tuple[int, ...], dict[str, object]]] = {}
        self._metadata_errors: dict[str, str] = {}

    def list_runs(self, source_filter: str = "", config_filter: str = "") -> dict[str, object]:
        """List recent runs, with case-insensitive source and config filtering."""
        with self._lock:
            candidates = discover_run_ids(self.root)
            results: list[dict[str, object]] = []
            for index, (run_id, _) in enumerate(candidates):
                metadata, metadata_error = self._get_metadata(run_id)
                stream = self._streams.get(run_id)
                if index < RUNS_REFRESH_COUNT or stream is not None:
                    stream = self._get_stream(run_id)
                    stream.refresh(self.root, run_id)
                summary = self._summary(run_id, metadata, metadata_error, stream)
                if not self._matches(summary, source_filter, config_filter):
                    continue
                results.append(summary)
            return {"runs": results, "truncated": len(candidates) >= MAX_LISTED_RUNS}

    def get_runs(self, run_ids: list[str], limit: int = DEFAULT_EVENT_LIMIT) -> dict[str, object]:
        """Refresh and return a bounded event window for up to eight selected runs."""
        if not run_ids or len(run_ids) > MAX_SELECTED_RUNS:
            raise ValueError("select between one and eight runs")
        if any(not validate_run_id(run_id) for run_id in run_ids):
            raise ValueError("invalid run identifier")
        limit = max(2, min(int(limit), MAX_EVENT_LIMIT))
        snapshots: list[dict[str, object]] = []
        with self._lock:
            for run_id in dict.fromkeys(run_ids):
                self._ensure_run_directory(run_id)
                metadata, metadata_error = self._get_metadata(run_id)
                stream = self._get_stream(run_id)
                stream.refresh(self.root, run_id)
                snapshot = self._summary(run_id, metadata, metadata_error, stream)
                snapshot.update(stream.snapshot(limit))
                snapshot["synthetic"] = self._is_synthetic(metadata, stream)
                snapshots.append(snapshot)
        return {"runs": snapshots, "max_points": limit}

    def _ensure_run_directory(self, run_id: str) -> None:
        run_path = self.root / run_id
        try:
            run_stat = os.lstat(run_path)
        except FileNotFoundError as exc:
            raise ValueError("run is unavailable") from exc
        if stat.S_ISLNK(run_stat.st_mode) or not stat.S_ISDIR(run_stat.st_mode):
            raise UnsafePathError("run is not a direct directory")

    def _get_metadata(self, run_id: str) -> tuple[dict[str, object], str | None]:
        cached = self._metadata.get(run_id)
        cached_signature = cached[0] if cached else None
        try:
            metadata, signature = read_run_metadata(self.root, run_id, cached_signature)
        except MissingFileError:
            self._metadata_errors[run_id] = "run metadata is missing"
            return _empty_metadata(), self._metadata_errors[run_id]
        except (OSError, ValueError, json.JSONDecodeError):
            self._metadata_errors[run_id] = "run metadata is unavailable or invalid"
            return _empty_metadata(), self._metadata_errors[run_id]
        if metadata is None and cached is not None:
            return cached[1], self._metadata_errors.get(run_id)
        if metadata is None:
            return _empty_metadata(), "run metadata is unavailable"
        self._metadata[run_id] = (signature, metadata)
        self._metadata_errors.pop(run_id, None)
        return metadata, None

    def _get_stream(self, run_id: str) -> EventStream:
        stream = self._streams.pop(run_id, None)
        if stream is None:
            stream = EventStream()
        self._streams[run_id] = stream
        while len(self._streams) > MAX_ACTIVE_STREAMS:
            self._streams.popitem(last=False)
        return stream

    def _summary(
        self,
        run_id: str,
        metadata: dict[str, object],
        metadata_error: str | None,
        stream: EventStream | None,
    ) -> dict[str, object]:
        events = list(stream.events) if stream else []
        latest = events[-1] if events else None
        context = latest.get("context", {}) if latest else {}
        return {
            "run_id": run_id,
            "source": metadata["source"],
            "config": metadata["config"],
            "provenance": metadata["provenance"],
            "metadata_status": "incomplete" if metadata_error else "ok",
            "metadata_error": metadata_error,
            "synthetic": self._is_synthetic(metadata, stream),
            "status": stream.status() if stream else "unread",
            "latest_step": latest.get("step") if latest else None,
            "last_event_at": stream.last_event_at.isoformat()
            if stream and stream.last_event_at
            else None,
            "event_count": stream.total_events if stream else 0,
            "retained_count": len(stream.events) if stream else 0,
            "dropped_count": stream.dropped_events if stream else 0,
            "history_limited": stream.initial_history_limited if stream else False,
            "invalid_lines": stream.invalid_lines if stream else 0,
            "error": (
                stream.read_error
                or (
                    f"{stream.invalid_lines} invalid monitor lines"
                    if stream.invalid_lines
                    else None
                )
                if stream
                else None
            ),
            "context_synthetic": bool(context.get("synthetic"))
            if isinstance(context, dict)
            else False,
        }

    @staticmethod
    def _matches(summary: dict[str, object], source_filter: str, config_filter: str) -> bool:
        source_text = json.dumps(summary["source"], ensure_ascii=False).casefold()
        config_text = json.dumps(summary["config"], ensure_ascii=False).casefold()
        return source_filter.casefold() in source_text and config_filter.casefold() in config_text

    @staticmethod
    def _is_synthetic(metadata: dict[str, object], stream: EventStream | None) -> bool:
        if metadata.get("synthetic") is True:
            return True
        source_text = json.dumps(metadata.get("source", ""), ensure_ascii=False).casefold()
        if "synthetic" in source_text:
            return True
        if stream:
            for event in reversed(stream.events):
                context = event.get("context", {})
                if isinstance(context, dict) and context.get("synthetic") is True:
                    return True
        return False
