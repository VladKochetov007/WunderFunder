"""Small stdlib contract checks shared by the E427 controller and scorer."""

from __future__ import annotations

from typing import Any

ARMS = ["aligned", "stale", "noop"]
COMPARISONS = [["aligned", "stale"], ["aligned", "noop"]]
PANELS = {
    "fold0_primary_historically_exposed": (468, 374, 94),
    "stats_excluded_368_historically_exposed": (368, 0, 368),
    "fold0_nonfit_94_diagnostic": (94, 0, 94),
}


def validate_screen_design(design: dict[str, Any], metric_id: str) -> None:
    """Validate the actual preregistration schema; never substitute defaults."""
    if (design.get("metric_id") != metric_id
            or type(design.get("bootstrap_replicates")) is not int
            or design.get("bootstrap_replicates") != 2000
            or type(design.get("bootstrap_seed")) is not int
            or design.get("bootstrap_seed") != 271828
            or design.get("arms") != ARMS
            or design.get("comparisons") != ["aligned_minus_stale", "aligned_minus_noop"]
            or design.get("primary_panel") != "fold0_primary_historically_exposed"
            or design.get("secondary_panel") != "stats_excluded_368_historically_exposed"
            or design.get("diagnostic_panel") != "fold0_nonfit_94_diagnostic"):
        raise ValueError("E427 score inputs differ from the frozen preregistration schema")
    panel_values = design.get("panels")
    if not isinstance(panel_values, dict) or set(panel_values) != set(PANELS):
        raise ValueError("E427 preregistration panel set changed")
    for name, (count, overlap, not_fit) in PANELS.items():
        value = panel_values[name]
        if (value.get("expected_n") != count
                or value.get("expected_fit_overlap") != overlap
                or value.get("expected_not_fit") != not_fit):
            raise ValueError(f"E427 preregistration membership changed for {name}")
    qualification = design.get("qualification")
    if not isinstance(qualification, str) or "no independent generalization claim" not in qualification.lower():
        raise ValueError("E427 preregistration must disclaim untouched generalization")


def remaining_export_timeout(index: int, export_count: int = 3, timeout_seconds: int = 1800) -> int:
    """Return the remaining fixed timeout reserve after a zero-based export index."""
    if (type(index) is not int or type(export_count) is not int or type(timeout_seconds) is not int
            or export_count != 3 or timeout_seconds != 1800 or not 0 <= index < export_count):
        raise ValueError("E427 export reservation index or fixed timeout is invalid")
    return (export_count - index - 1) * timeout_seconds


def validate_final_memory_events(record: dict[str, Any]) -> None:
    """Require authoritative, explicit zero-valued final OOM counters."""
    memory = record.get("final_memory_events")
    fields = ("oom", "oom_kill", "oom_group_kill")
    if (not isinstance(memory, dict)
            or any(key not in memory or type(memory[key]) is not int or memory[key] != 0
                   for key in fields)):
        raise ValueError("runner record lacks explicit zero final OOM-event counters")
