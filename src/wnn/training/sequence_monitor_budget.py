from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from wnn.training.sequence_monitor_events import EXTENDED_EVENT_LINE_CAPS, MAX_EVENT_LOG_BYTES


@dataclass(frozen=True)
class MonitorEventBudget:
    """Estimated JSONL line and byte bounds for extended monitor events."""

    event_counts: dict[str, int]
    line_caps_bytes: dict[str, int]
    event_bytes: dict[str, int]
    estimated_total_bytes: int
    journal_limit_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return the byte-bound receipt written with the run's monitoring plan."""
        return {
            "schema": "wnn.monitor-event-budget.v1",
            "event_counts": dict(self.event_counts),
            "line_caps_bytes": dict(self.line_caps_bytes),
            "estimated_bytes_by_kind": dict(self.event_bytes),
            "estimated_total_bytes": self.estimated_total_bytes,
            "journal_limit_bytes": self.journal_limit_bytes,
            "remaining_bytes": self.journal_limit_bytes - self.estimated_total_bytes,
        }


def validate_monitor_event_budget(
    total_updates: int,
    epochs: int,
    training_every_steps: int,
    diagnostics_every_steps: int,
) -> MonitorEventBudget:
    """Preflight all-epoch event bytes, including evaluation and restart statuses."""
    if min(total_updates, epochs, training_every_steps, diagnostics_every_steps) < 1:
        raise ValueError("monitor event budget inputs must be positive")
    training_events = math.ceil(total_updates / training_every_steps) + epochs
    diagnostic_events = math.ceil(total_updates / diagnostics_every_steps)
    event_counts = {
        "status": 2 * epochs + 4,
        "train_step": training_events,
        "evaluation": epochs,
        "diagnostics": diagnostic_events,
        "resources": diagnostic_events,
    }
    return _validate_budget(event_counts)


def validate_profile_monitor_event_budget(
    profile_updates: int,
    updates_per_epoch: int,
    training_every_steps: int,
    diagnostics_every_steps: int,
) -> MonitorEventBudget:
    """Preflight only the sampled telemetry and status lines emitted by a profile."""
    if min(profile_updates, updates_per_epoch, training_every_steps, diagnostics_every_steps) < 1:
        raise ValueError("profile monitor event budget inputs must be positive")
    complete_epochs = profile_updates // updates_per_epoch
    training_events = math.ceil(profile_updates / training_every_steps) + complete_epochs
    diagnostic_events = math.ceil(profile_updates / diagnostics_every_steps)
    event_counts = {
        "status": 2,
        "train_step": training_events,
        "evaluation": 0,
        "diagnostics": diagnostic_events,
        "resources": diagnostic_events,
    }
    return _validate_budget(event_counts)


def _validate_budget(event_counts: Mapping[str, int]) -> MonitorEventBudget:
    """Multiply each event-kind count by its enforced line cap and enforce 1 GiB."""
    if set(event_counts) != set(EXTENDED_EVENT_LINE_CAPS):
        raise ValueError("monitor event counts must cover each extended journal event kind")
    if any(type(count) is not int or count < 0 for count in event_counts.values()):
        raise ValueError("monitor event counts must be nonnegative integers")
    counts = {kind: event_counts[kind] for kind in EXTENDED_EVENT_LINE_CAPS}
    line_caps = dict(EXTENDED_EVENT_LINE_CAPS)
    event_bytes = {kind: counts[kind] * line_caps[kind] for kind in line_caps}
    total = sum(event_bytes.values())
    if total > MAX_EVENT_LOG_BYTES:
        raise ValueError(
            f"estimated monitor event journal is {total} bytes, exceeding the 1 GiB bound; "
            "increase training_telemetry_every_steps or diagnostics_every_steps"
        )
    return MonitorEventBudget(counts, line_caps, event_bytes, total, MAX_EVENT_LOG_BYTES)
