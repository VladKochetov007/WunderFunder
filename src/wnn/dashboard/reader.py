from __future__ import annotations

import json
import os
import stat
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from wnn.dashboard.history import RunStreams, new_run_streams, snapshot_history
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
MAX_METADATA_CACHE_ENTRIES = MAX_LISTED_RUNS
MAX_SELECTED_RUNS = 8
DEFAULT_EVENT_LIMIT = 1_200
MAX_EVENT_LIMIT = 1_500
RUNS_REFRESH_COUNT = 12


def _empty_metadata() -> dict[str, object]:
    """Return explicit placeholders for unavailable run metadata."""
    return {"source": "unknown", "config": {}, "provenance": {}, "synthetic": False}


@dataclass
class MetadataCacheEntry:
    """Keep each run's last good metadata and any bounded transient error."""

    signature: tuple[int, ...] | None
    metadata: dict[str, object]
    error: str | None = None


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
        self._streams: OrderedDict[str, RunStreams] = OrderedDict()
        self._metadata: OrderedDict[str, MetadataCacheEntry] = OrderedDict()

    def list_runs(self, source_filter: str = "", config_filter: str = "") -> dict[str, object]:
        """List recent runs, with case-insensitive source and config filtering."""
        with self._lock:
            candidates = discover_run_ids(self.root)
            results: list[dict[str, object]] = []
            for index, (run_id, _) in enumerate(candidates[:MAX_LISTED_RUNS]):
                metadata, metadata_error = self._get_metadata(run_id)
                streams = self._streams.get(run_id)
                if index < RUNS_REFRESH_COUNT or streams is not None:
                    streams = self._get_streams(run_id)
                    streams.training.refresh(self.root, run_id)
                    streams.evaluations.refresh(self.root, run_id)
                    streams.observer.refresh(self.root, run_id)
                    streams.rank_zero.refresh(self.root, run_id)
                summary = self._summary(run_id, metadata, metadata_error, streams)
                if not self._matches(summary, source_filter, config_filter):
                    continue
                results.append(summary)
            return {"runs": results, "truncated": len(candidates) > MAX_LISTED_RUNS}

    def get_runs(self, run_ids: list[str], limit: int = DEFAULT_EVENT_LIMIT) -> dict[str, object]:
        """Refresh and return a bounded event window for up to eight selected runs."""
        if not run_ids or len(run_ids) > MAX_SELECTED_RUNS:
            raise ValueError("select between one and eight runs")
        if any(not validate_run_id(run_id) for run_id in run_ids):
            raise ValueError("invalid run identifier")
        limit = max(2, min(int(limit), MAX_EVENT_LIMIT))
        snapshots: list[dict[str, object]] = []
        max_points = limit
        with self._lock:
            for run_id in dict.fromkeys(run_ids):
                self._ensure_run_directory(run_id)
                metadata, metadata_error = self._get_metadata(run_id)
                streams = self._get_streams(run_id)
                streams.training.refresh(self.root, run_id)
                streams.evaluations.refresh(self.root, run_id)
                streams.observer.refresh(self.root, run_id)
                streams.rank_zero.refresh(self.root, run_id)
                snapshot = self._summary(run_id, metadata, metadata_error, streams)
                snapshot.update(snapshot_history(streams, limit))
                run_points = snapshot.get("max_points")
                if isinstance(run_points, int):
                    max_points = max(max_points, run_points)
                snapshot["synthetic"] = self._is_synthetic(metadata, streams)
                snapshots.append(snapshot)
        return {"runs": snapshots, "max_points": max_points, "requested_points": limit}

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
        if cached is not None:
            self._metadata.move_to_end(run_id)
        cached_signature = cached.signature if cached else None
        try:
            metadata, signature = read_run_metadata(self.root, run_id, cached_signature)
        except MissingFileError:
            return self._metadata_failure(run_id, cached, "run metadata is missing")
        except (OSError, ValueError, json.JSONDecodeError):
            return self._metadata_failure(run_id, cached, "run metadata is unavailable or invalid")
        if metadata is None and cached is not None:
            cached.error = None
            self._metadata.move_to_end(run_id)
            return cached.metadata, None
        if metadata is None:
            return self._metadata_failure(run_id, cached, "run metadata is unavailable")
        self._remember_metadata(run_id, MetadataCacheEntry(signature, metadata))
        return metadata, None

    def _metadata_failure(
        self, run_id: str, cached: MetadataCacheEntry | None, error: str
    ) -> tuple[dict[str, object], str]:
        if cached is None:
            cached = MetadataCacheEntry(None, _empty_metadata(), error)
        else:
            cached.error = error
        self._remember_metadata(run_id, cached)
        return _empty_metadata(), error

    def _remember_metadata(self, run_id: str, entry: MetadataCacheEntry) -> None:
        self._metadata[run_id] = entry
        self._metadata.move_to_end(run_id)
        while len(self._metadata) > MAX_METADATA_CACHE_ENTRIES:
            self._metadata.popitem(last=False)

    def _get_streams(self, run_id: str) -> RunStreams:
        streams = self._streams.pop(run_id, None)
        if streams is None:
            streams = new_run_streams()
        self._streams[run_id] = streams
        while len(self._streams) > MAX_ACTIVE_STREAMS:
            self._streams.popitem(last=False)
        return streams

    def _summary(
        self,
        run_id: str,
        metadata: dict[str, object],
        metadata_error: str | None,
        streams: RunStreams | None,
    ) -> dict[str, object]:
        training = streams.training if streams else None
        evaluations = streams.evaluations if streams else None
        observer = streams.observer if streams else None
        rank_zero = streams.rank_zero if streams else None
        events = list(training.events) if training else []
        if not events and evaluations:
            events = list(evaluations.events)
        latest_stream = max(
            (
                stream
                for stream in (training, evaluations, observer)
                if stream is not None and stream.last_event_at is not None
            ),
            key=lambda stream: stream.last_event_at,
            default=None,
        )
        if latest_stream is not None:
            events = list(latest_stream.events)
        latest = events[-1] if events else None
        context = latest.get("context", {}) if latest else {}
        return {
            "run_id": run_id,
            "source": metadata["source"],
            "config": metadata["config"],
            "provenance": metadata["provenance"],
            "metadata_status": "incomplete" if metadata_error else "ok",
            "metadata_error": metadata_error,
            "synthetic": self._is_synthetic(metadata, streams),
            "status": self._overall_status(training, evaluations, observer),
            "phase": context.get("phase") if isinstance(context, dict) else None,
            "observer_status": observer.status() if observer else "unread",
            "observer_context": observer.events[-1]["context"]
            if observer and observer.events
            else {},
            "observer_metrics": observer.events[-1]["metrics"]
            if observer and observer.events
            else {},
            "latest_step": latest.get("step") if latest else None,
            "last_event_at": latest_stream.last_event_at.isoformat()
            if latest_stream and latest_stream.last_event_at
            else None,
            "event_count": (training.total_events if training else 0)
            + (rank_zero.total_events if rank_zero else 0),
            "retained_count": (len(training.events) if training else 0)
            + (len(rank_zero.events) if rank_zero else 0),
            "dropped_count": (training.dropped_events if training else 0)
            + (rank_zero.dropped_events if rank_zero else 0),
            "history_limited": (training.history_is_limited() if training else False)
            or (rank_zero.history_is_limited() if rank_zero else False),
            "invalid_lines": training.invalid_lines if training else 0,
            "evaluation_status": evaluations.status() if evaluations else "unread",
            "evaluation_event_count": evaluations.total_events if evaluations else 0,
            "evaluation_retained_count": len(evaluations.events) if evaluations else 0,
            "evaluation_dropped_count": evaluations.dropped_events if evaluations else 0,
            "evaluation_history_limited": (
                evaluations.history_is_limited() if evaluations else False
            ),
            "evaluation_error": self._stream_error(evaluations),
            "error": (
                self._stream_error(training)
                or self._stream_error(evaluations)
                or self._stream_error(observer)
            ),
            "context_synthetic": bool(context.get("synthetic"))
            if isinstance(context, dict)
            else False,
        }

    @staticmethod
    def _stream_error(stream: EventStream | None) -> str | None:
        if stream is None:
            return None
        if stream.read_error:
            return stream.read_error
        if stream.invalid_lines:
            return f"{stream.invalid_lines} invalid monitor lines"
        return None

    @staticmethod
    def _overall_status(
        training: EventStream | None,
        evaluations: EventStream | None,
        observer: EventStream | None = None,
    ) -> str:
        if training is None:
            return "unread"
        statuses = [training.status()]
        if evaluations:
            statuses.append(evaluations.status())
        if "error" in statuses:
            return "error"
        if observer is not None and observer.events:
            if observer.status() == "error":
                return "error"
            latest = observer.events[-1]
            count = latest["metrics"].get("observer/process_alive_count")
            if count == 0:
                terminal = latest["context"].get("phase") in (
                    "completed",
                    "early_stopped",
                    "profile_complete",
                )
                return "completed" if terminal else "exited"
            if observer.status() == "live" and isinstance(count, (int, float)) and count > 0:
                return "live"
            # A training event is progress evidence; an absent heartbeat is still visible.
        if "live" in statuses:
            return "live"
        if "stale" in statuses:
            return "stale"
        if "no_data" in statuses:
            return "no_data"
        return "unread"

    @staticmethod
    def _matches(summary: dict[str, object], source_filter: str, config_filter: str) -> bool:
        source_text = json.dumps(summary["source"], ensure_ascii=False).casefold()
        config_text = json.dumps(summary["config"], ensure_ascii=False).casefold()
        return source_filter.casefold() in source_text and config_filter.casefold() in config_text

    @staticmethod
    def _is_synthetic(metadata: dict[str, object], streams: RunStreams | None) -> bool:
        if metadata.get("synthetic") is True:
            return True
        source_text = json.dumps(metadata.get("source", ""), ensure_ascii=False).casefold()
        if "synthetic" in source_text:
            return True
        if streams:
            events = (
                *streams.training.events,
                *streams.evaluations.events,
                *streams.observer.events,
            )
            for event in reversed(events):
                context = event.get("context", {})
                if isinstance(context, dict) and context.get("synthetic") is True:
                    return True
        return False
