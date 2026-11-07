from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wnn.dashboard.schema import MAX_METRIC_KEYS_PER_RUN
from wnn.dashboard.stream import EventStream, downsample_events

MAX_EVALUATION_EVENTS = 512
MAX_EVALUATION_BYTES = 2_097_152


@dataclass
class RunStreams:
    """Hold bounded training, evaluation, and independent-observer journals."""

    training: EventStream
    evaluations: EventStream
    observer: EventStream


def new_run_streams() -> RunStreams:
    """Construct the fixed-name evaluation journal with its bounded record cap."""
    return RunStreams(
        training=EventStream(),
        evaluations=EventStream(
            filename="monitor_evaluations.jsonl",
            max_retained_events=MAX_EVALUATION_EVENTS,
            max_retained_bytes=MAX_EVALUATION_BYTES,
            max_line_bytes=32_768,
            expected_kind="evaluation",
        ),
        observer=EventStream(filename="monitor_observer.jsonl", expected_kind="resources"),
    )


def _event_identity(event: dict[str, object]) -> tuple[object, object, object]:
    """Identify an evaluation event by kind, epoch, and step."""
    return event.get("kind"), event.get("epoch"), event.get("step")


def _sort_key(event: dict[str, object]) -> tuple[int, float, float]:
    """Keep training progress ordered even if the host wall clock changes on resume."""
    try:
        timestamp = datetime.fromisoformat(str(event.get("wall_time"))).timestamp()
    except (ValueError, OverflowError):
        timestamp = 0.0
    step = event.get("step")
    epoch = event.get("epoch")
    return (
        step if isinstance(step, int) else 0,
        float(epoch) if isinstance(epoch, (int, float)) else 0.0,
        timestamp,
    )


def _merge_events(
    train_events: list[dict[str, object]],
    journal_events: list[dict[str, object]],
    limit: int,
) -> tuple[list[dict[str, object]], int, int, int]:
    """Keep journal evaluations, remove exact repeats, and sample the train tail."""
    retained_evaluations: list[dict[str, object]] = []
    evaluation_by_identity: dict[tuple[object, object, object], dict[str, object]] = {}
    duplicates = 0
    conflicts = 0
    for event in journal_events:
        identity = _event_identity(event)
        existing = evaluation_by_identity.get(identity)
        if existing == event:
            duplicates += 1
            continue
        if existing is not None:
            conflicts += 1
        else:
            evaluation_by_identity[identity] = event
        retained_evaluations.append(event)

    train_pool: list[dict[str, object]] = []
    for event in train_events:
        if event.get("kind") == "evaluation":
            existing = evaluation_by_identity.get(_event_identity(event))
            if existing == event:
                duplicates += 1
                continue
            if existing is not None:
                conflicts += 1
        train_pool.append(event)

    effective_limit = max(limit, len(retained_evaluations))
    train_capacity = max(0, effective_limit - len(retained_evaluations))
    merged = downsample_events(train_pool, train_capacity) + retained_evaluations
    merged.sort(key=_sort_key)
    return merged, duplicates, conflicts, effective_limit


def snapshot_history(streams: RunStreams, limit: int) -> dict[str, object]:
    """Merge retained logs while reserving points for every retained evaluation."""
    training = streams.training.snapshot(limit)
    evaluations = streams.evaluations.snapshot(MAX_EVALUATION_EVENTS)
    observer = streams.observer.snapshot(limit)
    train_events = list(streams.training.events) + list(streams.observer.events)
    journal_events = list(streams.evaluations.events)
    merged, duplicates, conflicts, effective_limit = _merge_events(
        train_events, journal_events, limit
    )
    keys = (
        set(training["metric_keys"])
        | set(evaluations["metric_keys"])
        | set(observer["metric_keys"])
    )
    metric_keys = sorted(keys)[:MAX_METRIC_KEYS_PER_RUN]
    evaluation_status = "error" if conflicts else evaluations["status"]
    evaluation_error = evaluations["error"]
    if conflicts and not evaluation_error:
        evaluation_error = f"{conflicts} conflicting evaluation identities"
    return {
        "events": merged,
        "metric_keys": metric_keys,
        "metric_keys_truncated": (
            training["metric_keys_truncated"]
            or evaluations["metric_keys_truncated"]
            or observer["metric_keys_truncated"]
            or len(keys) > MAX_METRIC_KEYS_PER_RUN
        ),
        "evaluation_status": evaluation_status,
        "evaluation_event_count": evaluations["event_count"],
        "evaluation_retained_count": evaluations["retained_count"],
        "evaluation_dropped_count": evaluations["dropped_count"],
        "evaluation_history_limited": (evaluations["history_limited"]),
        "evaluation_invalid_lines": evaluations["invalid_lines"],
        "evaluation_error": evaluation_error,
        "evaluation_duplicates": duplicates,
        "evaluation_conflicts": conflicts,
        "max_points": effective_limit,
        "observer_status": observer["status"],
        "observer_error": observer["error"],
    }
