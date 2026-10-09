"""Validate the D02 producer schedule across optimizer group layouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

Record = dict[str, Any]
STEP_FIELDS = ("attempt_index", "epoch_index", "batch_index", "chunk_index", "scheduler_step_pre")
RATE_FIELDS = ("lr", "beta1")
SCHEDULE_SCHEMA = "wnn.loss-pair.onecycle-trace.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _groups(row: Mapping[str, Any], label: str, index: int) -> list[tuple[float, float]]:
    groups = row.get("optimizer_groups")
    _require(isinstance(groups, list) and groups, f"{label} step {index}: optimizer groups missing")
    normalized: list[tuple[float, float]] = []
    for expected_index, group in enumerate(groups):
        _require(isinstance(group, Mapping), f"{label} step {index}: malformed optimizer group")
        _require(
            group.get("group_index") == expected_index,
            f"{label} step {index}: noncanonical group index",
        )
        values: list[float] = []
        for field in RATE_FIELDS:
            value = group.get(field)
            _require(
                isinstance(value, (int, float)) and math.isfinite(float(value)),
                f"{label} step {index}: invalid {field}",
            )
            values.append(float(value))
        _require(
            values[0] >= 0.0 and 0.0 <= values[1] < 1.0,
            f"{label} step {index}: out-of-range LR/beta1",
        )
        normalized.append((values[0], values[1]))
    return normalized


def _validate_arm(
    name: str, payload: Mapping[str, Any]
) -> tuple[list[Record], list[list[tuple[float, float]]]]:
    optimizer = payload.get("optimizer")
    expected_groups = {"adamw": 1, "muon": 2}.get(str(optimizer))
    _require(expected_groups is not None, f"{name}: unsupported optimizer in schedule audit")
    schedule = payload.get("schedule_trace")
    updates = payload.get("update_rows")
    _require(isinstance(schedule, Mapping), f"{name}: schedule trace missing")
    _require(schedule.get("schema") == SCHEDULE_SCHEMA, f"{name}: wrong schedule schema")
    rows = schedule.get("rows")
    _require(isinstance(rows, list) and rows, f"{name}: schedule rows missing")
    _require(
        isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)),
        f"{name}: update trace missing",
    )
    _require(len(rows) == len(updates), f"{name}: schedule/update trace lengths differ")
    _require(schedule.get("total_steps") == len(rows), f"{name}: total_steps differs from rows")

    normalized_groups: list[list[tuple[float, float]]] = []
    for index, (scheduled, updated) in enumerate(zip(rows, updates, strict=True)):
        _require(
            isinstance(scheduled, Mapping) and isinstance(updated, Mapping),
            f"{name} step {index}: malformed trace row",
        )
        for field in STEP_FIELDS:
            _require(isinstance(scheduled.get(field), int), f"{name} step {index}: missing {field}")
            _require(
                scheduled.get(field) == updated.get(field),
                f"{name} step {index}: schedule/update {field} differ",
            )
        _require(scheduled["attempt_index"] == index, f"{name}: attempt indices are not contiguous")
        _require(
            scheduled["scheduler_step_pre"] == index, f"{name}: scheduler steps are not contiguous"
        )
        scheduled_groups = _groups(scheduled, name, index)
        update_groups = _groups(updated, name, index)
        _require(
            len(scheduled_groups) == expected_groups,
            f"{name} step {index}: expected {expected_groups} {optimizer} groups, saw {len(scheduled_groups)}",
        )
        _require(
            scheduled_groups == update_groups,
            f"{name} step {index}: update trace differs from schedule trace",
        )
        normalized_groups.append(scheduled_groups)
    return list(rows), normalized_groups


def audit_quartet_schedules(arms: Mapping[str, Mapping[str, Any]]) -> Record:
    """Compare real optimizer schedules semantically, retaining each raw layout.

    AdamW has one optimizer group; Muon has matrix and vector groups. For each
    attempt the common clocks must align, while every Muon group must use the
    AdamW control's LR and beta1. The two raw group lists remain distinct.
    """
    required = {"adamw_wmse", "adamw_huber_d1", "muon_wmse", "muon_huber_d1"}
    _require(set(arms) == required, "schedule audit requires the exact frozen quartet")
    normalized: dict[str, tuple[list[Record], list[list[tuple[float, float]]]]] = {}
    for name, payload in arms.items():
        expected_optimizer = "muon" if name.startswith("muon_") else "adamw"
        _require(
            payload.get("optimizer") == expected_optimizer,
            f"{name}: optimizer label differs from frozen cell",
        )
        normalized[name] = _validate_arm(name, payload)

    anchor_name = "adamw_wmse"
    anchor_rows, anchor_groups = normalized[anchor_name]
    _require(len(anchor_groups[0]) == 1, "AdamW anchor must retain its one-group schedule")
    for name, (rows, group_rows) in normalized.items():
        _require(len(rows) == len(anchor_rows), f"{name}: attempt count differs from AdamW")
        for index, (row, anchor_row, rates, anchor_rates) in enumerate(
            zip(rows, anchor_rows, group_rows, anchor_groups, strict=True)
        ):
            for field in STEP_FIELDS:
                _require(
                    row[field] == anchor_row[field],
                    f"{name} step {index}: {field} differs from AdamW",
                )
            for lr, beta1 in rates:
                anchor_lr, anchor_beta1 = anchor_rates[0]
                _require(
                    math.isclose(lr, anchor_lr, rel_tol=1e-12, abs_tol=1e-15),
                    f"{name} step {index}: LR differs from AdamW",
                )
                _require(
                    math.isclose(beta1, anchor_beta1, rel_tol=1e-12, abs_tol=1e-15),
                    f"{name} step {index}: beta1 differs from AdamW",
                )

    return {
        "status": "pass",
        "attempt_count": len(anchor_rows),
        "step_fields": list(STEP_FIELDS),
        "rate_fields": list(RATE_FIELDS),
        "raw_group_cardinality": {
            name: len(group_rows[0]) for name, (_, group_rows) in normalized.items()
        },
        "interpretation": (
            "Per-attempt epoch/batch/chunk/scheduler clocks and each LR/beta1 value match the AdamW control. "
            "AdamW retains one group; each Muon arm retains its actual two groups."
        ),
        "numeric_tolerance": {"relative": 1e-12, "absolute": 1e-15},
    }
