from __future__ import annotations

import math

from wnn.training.sequence_monitor_events import MAX_EVENT_LINE_BYTES, MAX_EVENT_LOG_BYTES

MAX_MONITOR_EVENT_COUNT = MAX_EVENT_LOG_BYTES // MAX_EVENT_LINE_BYTES


def validate_monitor_event_budget(
    total_updates: int,
    epochs: int,
    training_every_steps: int,
    diagnostics_every_steps: int,
) -> int:
    """Bound worst-case journal line count before an opt-in monitored fit."""
    if min(total_updates, epochs, training_every_steps, diagnostics_every_steps) < 1:
        raise ValueError("monitor event budget inputs must be positive")
    training_events = math.ceil(total_updates / training_every_steps) + epochs
    diagnostic_steps = math.ceil(total_updates / diagnostics_every_steps)
    event_count = training_events + 2 * diagnostic_steps + 2 * epochs + 4
    if event_count > MAX_MONITOR_EVENT_COUNT:
        raise ValueError(
            "monitoring cadence can exceed the 1 GiB event journal bound; "
            "increase training_telemetry_every_steps or diagnostics_every_steps"
        )
    return event_count
