from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any  # Runtime cgroup snapshots expose dynamic counter mappings.

MAX_ADVICE_BYTES = 768_000_000
MAX_RELIEF_EVENTS = 320
MAX_TOTAL_ADVICE_BYTES = 245_760_000_000
MIN_RELIEF_INTERVAL_SECONDS = 20.0
MAX_TRACE_ROWS = 2_048
MAX_TRACE_BYTES = 2_097_152
MIN_INACTIVE_FILE_BYTES = 768_000_000
MAX_ANON_GROWTH_BYTES = 128_000_000
MAX_ACTIVE_FILE_GROWTH_BYTES = 256_000_000
MAX_REFAULT_BYTES = 64_000_000


class ReliefBudget:
    """Bound event size, cumulative advice, and event rate for one worker."""

    def __init__(
        self,
        *,
        maximum_events: int,
        maximum_episode_bytes: int,
        maximum_total_bytes: int,
        minimum_interval_seconds: float,
    ) -> None:
        if maximum_events < 0:
            raise ValueError("maximum relief events cannot be negative")
        if maximum_episode_bytes <= 0 or maximum_total_bytes < 0:
            raise ValueError("relief byte budgets must be positive")
        if maximum_total_bytes > maximum_events * maximum_episode_bytes:
            raise ValueError("total relief budget exceeds the event-by-event byte ceiling")
        if minimum_interval_seconds < 0:
            raise ValueError("minimum relief interval cannot be negative")
        self.maximum_events = maximum_events
        self.maximum_episode_bytes = maximum_episode_bytes
        self.maximum_total_bytes = maximum_total_bytes
        self.minimum_interval_seconds = minimum_interval_seconds
        self.events = 0
        self.bytes_reserved = 0
        self.last_reservation_time: float | None = None

    def snapshot(self) -> dict[str, int | float | None]:
        """Return a small serializable budget receipt."""
        return {
            "events": self.events,
            "maximum_events": self.maximum_events,
            "bytes_reserved": self.bytes_reserved,
            "maximum_total_bytes": self.maximum_total_bytes,
            "maximum_episode_bytes": self.maximum_episode_bytes,
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "last_reservation_monotonic_seconds": self.last_reservation_time,
        }

    def reserve(self, *, now: float, requested_bytes: int) -> tuple[bool, str | None]:
        """Reserve one bounded pressure event, failing closed on rate or size caps."""
        if not isfinite(now) or now < 0 or requested_bytes <= 0:
            return False, "invalid_event_request"
        if requested_bytes > self.maximum_episode_bytes:
            return False, "episode_byte_limit"
        if self.events >= self.maximum_events:
            return False, "event_limit"
        if self.bytes_reserved + requested_bytes > self.maximum_total_bytes:
            return False, "total_byte_limit"
        if self.last_reservation_time is not None:
            elapsed = now - self.last_reservation_time
            if elapsed < 0:
                return False, "monotonic_clock_regressed"
            if elapsed < self.minimum_interval_seconds:
                return False, "minimum_interval"
        self.events += 1
        self.bytes_reserved += requested_bytes
        self.last_reservation_time = now
        return True, None


def has_reclaimable_file_pressure(
    current: dict[str, int],
    previous: dict[str, int] | None,
    *,
    page_size: int,
) -> tuple[bool, list[str]]:
    """Require inactive-file headroom and stable anonymous/working-set signals."""
    reasons: list[str] = []
    required = {
        "active_file",
        "anon",
        "file",
        "inactive_file",
        "shmem",
        "workingset_refault_file",
    }
    if not required.issubset(current):
        return False, ["required memory.stat counters are absent"]
    if previous is None or not required.issubset(previous):
        return False, ["no comparable prior memory.stat sample"]
    if page_size <= 0:
        return False, ["invalid system page size"]
    if current["inactive_file"] < max(MIN_INACTIVE_FILE_BYTES, current["anon"] // 4):
        reasons.append("inactive_file below conservative reclaim minimum")
    if current["active_file"] > 4 * max(current["inactive_file"], 1):
        reasons.append("active_file dominates inactive_file working set")
    if current["file"] - current["shmem"] < MIN_INACTIVE_FILE_BYTES:
        reasons.append("file-backed cache excluding shmem is too small")
    if current["anon"] - previous["anon"] > MAX_ANON_GROWTH_BYTES:
        reasons.append("anonymous memory grew too quickly")
    if current["active_file"] - previous["active_file"] > MAX_ACTIVE_FILE_GROWTH_BYTES:
        reasons.append("active_file grew too quickly")
    refault_pages = current["workingset_refault_file"] - previous["workingset_refault_file"]
    if refault_pages < 0:
        reasons.append("working-set refault counter reset")
    elif refault_pages * page_size > MAX_REFAULT_BYTES:
        reasons.append("file working-set refault rate is high")
    return not reasons, reasons


def memory_snapshot_matches(snapshot: dict[str, Any]) -> bool:
    """Check that the exported cgroup counters came from one memory.stat read."""
    if snapshot.get("cgroup_memory_stat_status") != "available":
        return False
    return bool(
        snapshot.get("cgroup_memory_stat_path")
        and snapshot.get("cgroup_available_bytes") == snapshot.get("cgroup_memory_available_bytes")
        and snapshot.get("cgroup_limit_bytes") == snapshot.get("cgroup_memory_limit_bytes")
    )


def host_headroom_passes(
    available_bytes: int,
    total_bytes: int,
    *,
    minimum_bytes: int,
    minimum_fraction: float,
) -> bool:
    """Apply the unchanged host byte and fraction gates to one resource sample."""
    return (
        available_bytes >= minimum_bytes
        and total_bytes > 0
        and available_bytes / total_bytes >= minimum_fraction
    )


def relief_target_bytes(limit_bytes: int, floor_bytes: int, fraction: float) -> int:
    """Add a fixed recovery margin above the existing hard cgroup gates."""
    return max(floor_bytes, int(limit_bytes * fraction)) + 512_000_000


def _read_recent_groups(trace_path: Path, *, max_rows: int) -> list[tuple[str, int, int, int]]:
    size = trace_path.stat().st_size
    with trace_path.open("rb") as stream:
        stream.seek(max(0, size - MAX_TRACE_BYTES))
        if size > MAX_TRACE_BYTES:
            stream.readline()
        raw = stream.read(MAX_TRACE_BYTES + 1)
    if len(raw) > MAX_TRACE_BYTES:
        raise ValueError("rowgroup trace exceeds bounded recovery read")
    recent: deque[str] = deque(raw.decode("ascii").splitlines(), maxlen=max_rows)
    groups: list[tuple[str, int, int, int]] = []
    seen: dict[tuple[str, int], tuple[int, int]] = {}
    for line in reversed(recent):
        fields = line.split("\t")
        if len(fields) != 4 or fields[0] not in {"train", "valid"}:
            raise ValueError("rowgroup trace has an invalid record")
        split, raw_index, raw_start, raw_end = fields
        index, start, end = int(raw_index), int(raw_start), int(raw_end)
        if index < 0 or start < 4 or end <= start:
            raise ValueError("rowgroup trace has invalid index or byte range")
        record = (split, index)
        byte_range = (start, end)
        if record in seen and seen[record] != byte_range:
            raise ValueError("rowgroup trace has conflicting byte ranges for one rowgroup")
        if record not in seen:
            seen[record] = byte_range
            groups.append((split, index, start, end))
    return groups


def _bound_identity(path: Path, binding: dict[str, Any]) -> bool:
    observed = path.stat()
    return (
        str(path.resolve(strict=True)) == binding.get("canonical_path")
        and observed.st_dev == int(binding["device"])
        and observed.st_ino == int(binding["inode"])
        and observed.st_size == int(binding["size"])
        and observed.st_mtime_ns == int(binding["mtime_ns"])
        and observed.st_ctime_ns == int(binding["ctime_ns"])
        and observed.st_mode == int(binding["mode"])
    )


def advise_recent_rowgroups(
    raw_paths: dict[str, Path],
    bindings: dict[str, dict[str, Any]],
    trace_path: Path,
    *,
    page_size: int,
    fadvise: Callable[[int, int, int, int], Any] | None = None,
    max_bytes: int = MAX_ADVICE_BYTES,
) -> dict[str, Any]:
    """Hint eviction only for traced rowgroups of verified train/valid inputs."""
    result: dict[str, Any] = {
        "status": "not_started",
        "advice": "POSIX_FADV_DONTNEED",
        "semantics": "kernel hint only; memory.stat after-call is the recovery check",
        "max_bytes": max_bytes,
        "requested_bytes": 0,
        "calls": [],
    }
    fadvise = fadvise or getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(fadvise) or dontneed is None or page_size <= 0 or max_bytes <= 0:
        result.update({"status": "unsupported", "diagnostic": "POSIX_FADV_DONTNEED unavailable"})
        return result
    descriptors: dict[str, int] = {}
    try:
        groups = _read_recent_groups(trace_path, max_rows=MAX_TRACE_ROWS)
        if not groups:
            result.update({"status": "no_recent_rowgroups"})
            return result
        if set(raw_paths) != {"train", "valid"} or set(bindings) != {"train", "valid"}:
            raise ValueError("cache advice requires both verified raw split bindings")
        file_bytes: dict[str, int] = {}
        descriptors: dict[str, int] = {}
        for split in {record[0] for record in groups}:
            path = raw_paths[split]
            if not _bound_identity(path, bindings[split]):
                raise ValueError(f"{split} input identity changed before cache advice")
            file_bytes[split] = path.stat().st_size
            descriptors[split] = os.open(path, os.O_RDONLY)
        advised = 0
        for split, index, rowgroup_start, rowgroup_end in groups:
            if advised >= max_bytes:
                break
            start, end = rowgroup_start, rowgroup_end
            if start < 4 or end <= start or end > file_bytes[split]:
                raise ValueError(
                    "traced rowgroup byte range is outside the identity-bound parquet file"
                )
            start = start - (start % page_size)
            end = min(file_bytes[split], ((end + page_size - 1) // page_size) * page_size)
            length = min(end - start, max_bytes - advised)
            length -= length % page_size
            if length <= 0:
                continue
            if not _bound_identity(raw_paths[split], bindings[split]):
                raise ValueError(f"{split} input identity changed during cache advice")
            returned = fadvise(descriptors[split], start, length, dontneed)
            if returned not in (None, 0):
                raise OSError(int(returned), os.strerror(int(returned)))
            advised += length
            result["calls"].append(
                {"split": split, "rowgroup": index, "offset_bytes": start, "length_bytes": length}
            )
        result.update(
            {"status": "advised" if advised else "no_advisable_ranges", "requested_bytes": advised}
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        result.update({"status": "failed_closed", "diagnostic": f"{type(error).__name__}: {error}"})
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    return result
