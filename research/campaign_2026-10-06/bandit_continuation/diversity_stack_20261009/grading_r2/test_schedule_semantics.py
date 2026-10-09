"""Producer-shaped regression tests for optimizer schedule equivalence."""

from __future__ import annotations

from copy import deepcopy

import pytest
from schedule_semantics import audit_quartet_schedules


def _trace_row(index: int, n_groups: int, lr: float = 0.001, beta1: float = 0.95) -> dict:
    return {
        "attempt_index": index,
        "epoch_index": index // 2,
        "batch_index": index // 2,
        "chunk_index": index % 2,
        "scheduler_step_pre": index,
        "optimizer_groups": [
            {"group_index": group, "lr": lr, "beta1": beta1} for group in range(n_groups)
        ],
    }


def _arm(optimizer: str, steps: int = 4) -> dict:
    n_groups = 1 if optimizer == "adamw" else 2
    rows = [_trace_row(index, n_groups) for index in range(steps)]
    return {
        "optimizer": optimizer,
        "schedule_trace": {
            "schema": "wnn.loss-pair.onecycle-trace.v1",
            "total_steps": steps,
            "rows": deepcopy(rows),
        },
        "update_rows": deepcopy(rows),
    }


def _quartet() -> dict:
    return {
        "adamw_wmse": _arm("adamw"),
        "adamw_huber_d1": _arm("adamw"),
        "muon_wmse": _arm("muon"),
        "muon_huber_d1": _arm("muon"),
    }


def test_one_adamw_group_matches_both_actual_muon_groups() -> None:
    result = audit_quartet_schedules(_quartet())
    assert result["status"] == "pass"
    assert result["attempt_count"] == 4
    assert result["raw_group_cardinality"] == {
        "adamw_wmse": 1,
        "adamw_huber_d1": 1,
        "muon_wmse": 2,
        "muon_huber_d1": 2,
    }


@pytest.mark.parametrize("field,value", [("lr", 0.0011), ("beta1", 0.94)])
def test_mismatch_in_either_muon_group_is_rejected(field: str, value: float) -> None:
    arms = _quartet()
    arms["muon_wmse"]["schedule_trace"]["rows"][2]["optimizer_groups"][1][field] = value
    arms["muon_wmse"]["update_rows"][2]["optimizer_groups"][1][field] = value
    with pytest.raises(ValueError, match="differs from AdamW"):
        audit_quartet_schedules(arms)


def test_clock_drift_is_rejected_even_when_rates_match() -> None:
    arms = _quartet()
    for trace in (
        arms["muon_huber_d1"]["schedule_trace"]["rows"],
        arms["muon_huber_d1"]["update_rows"],
    ):
        trace[1]["batch_index"] = 9
    with pytest.raises(ValueError, match="batch_index differs from AdamW"):
        audit_quartet_schedules(arms)


def test_schedule_must_match_the_arm_update_trace() -> None:
    arms = _quartet()
    arms["muon_huber_d1"]["update_rows"][0]["optimizer_groups"][1]["lr"] = 0.002
    with pytest.raises(ValueError, match="update trace differs from schedule trace"):
        audit_quartet_schedules(arms)


def test_schedule_group_arity_must_match_optimizer() -> None:
    arms = _quartet()
    for trace in (arms["muon_wmse"]["schedule_trace"]["rows"], arms["muon_wmse"]["update_rows"]):
        trace[0]["optimizer_groups"] = trace[0]["optimizer_groups"][:1]
    with pytest.raises(ValueError, match="expected 2 muon groups"):
        audit_quartet_schedules(arms)
