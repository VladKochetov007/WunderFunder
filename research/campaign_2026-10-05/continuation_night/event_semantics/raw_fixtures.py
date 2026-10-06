"""Independent hand-fixture oracle for the registered E492 raw event counts."""

from __future__ import annotations

import numpy as np

ROWS = 20_000
RAW_COLUMNS = ("i0_p0", "i0_p11", "a6", "t0", "t1")
MASK_COLUMNS = ("need_prediction", "is_scored")
EVENT_NAMES = ("bid", "ask", "or", "and")


def validate_raw_group(table: dict[str, np.ndarray], expected_seq_ix: int) -> None:
    """Reject malformed sequence metadata before making any event counts."""
    required = {"seq_ix", "step_in_seq", *MASK_COLUMNS, *RAW_COLUMNS}
    if required - table.keys():
        raise ValueError(f"missing fixture columns: {sorted(required - table.keys())}")
    if any(np.asarray(table[name]).shape != (ROWS,) for name in required):
        raise ValueError("fixture columns must each contain exactly 20,000 rows")
    if not np.all(np.asarray(table["seq_ix"]) == expected_seq_ix):
        raise ValueError("fixture sequence identity mismatch")
    if not np.array_equal(np.asarray(table["step_in_seq"]), np.arange(ROWS)):
        raise ValueError("fixture steps must equal arange(20,000)")
    for name in MASK_COLUMNS:
        values = np.asarray(table[name])
        if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
            raise ValueError(f"fixture mask must be numeric or boolean: {name}")
        if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError(f"fixture mask must be finite and binary: {name}")
    for name in RAW_COLUMNS:
        if not np.isfinite(np.asarray(table[name])).all():
            raise ValueError(f"fixture raw field must be finite: {name}")


def _transitions(values: np.ndarray) -> np.ndarray:
    """Calculate raw adjacent-row transitions, independently of any mask."""
    values = np.asarray(values)
    result = np.zeros(values.shape, dtype=bool)
    result[1:] = values[1:] != values[:-1]
    return result


def count_rows(table: dict[str, np.ndarray], node: dict[str, object]) -> list[dict[str, object]]:
    """Reproduce E492 counts from full raw groups before selecting current rows."""
    validate_raw_group(table, int(node["seq_ix"]))
    bid = _transitions(table["i0_p0"])
    ask = _transitions(table["i0_p11"])
    auxiliary = _transitions(table["a6"])
    clocks = {"bid": bid, "ask": ask, "or": bid | ask, "and": bid & ask}
    eligible = (np.asarray(table["step_in_seq"]) >= 99) & np.asarray(
        table["need_prediction"], dtype=bool
    )
    populations = {
        "eligible": eligible,
        "scored": eligible & np.asarray(table["is_scored"], dtype=bool),
    }
    result = []
    for population, selected in populations.items():
        for event_name in EVENT_NAMES:
            event = clocks[event_name]
            for target_name in ("t0", "t1"):
                target = np.asarray(table[target_name])
                target_events = _transitions(target)
                clipped = np.clip(target[selected], -2, 2)
                result.append(
                    {
                        "row_group": node["row_group"],
                        "seq_ix": node["seq_ix"],
                        "component": node["component"],
                        "role": node["role"],
                        "class_diagnostic": node["label_class_diagnostic_only"],
                        "population": population,
                        "clock": event_name,
                        "target": target_name,
                        "rows": int(np.count_nonzero(selected)),
                        "events": int(np.count_nonzero(selected & event)),
                        "target_changes": int(np.count_nonzero(selected & target_events)),
                        "changes_without_event": int(
                            np.count_nonzero(selected & target_events & ~event)
                        ),
                        "events_without_change": int(
                            np.count_nonzero(selected & event & ~target_events)
                        ),
                        "a6_clock_disagreements": int(
                            np.count_nonzero(selected & (auxiliary != event))
                        ),
                        "absolute_target_mass": float(np.abs(clipped).sum(dtype=np.float64)),
                    }
                )
    return result


def make_hand_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create a full-shape toy group with isolated and simultaneous transitions."""
    table = {
        "seq_ix": np.full(ROWS, 73, dtype=np.int64),
        "step_in_seq": np.arange(ROWS, dtype=np.int64),
        "need_prediction": np.ones(ROWS, dtype=np.uint8),
        "is_scored": np.ones(ROWS, dtype=np.uint8),
        "i0_p0": np.zeros(ROWS, dtype=np.float32),
        "i0_p11": np.zeros(ROWS, dtype=np.float32),
        "a6": np.zeros(ROWS, dtype=np.float32),
        "t0": np.zeros(ROWS, dtype=np.float32),
        "t1": np.zeros(ROWS, dtype=np.float32),
    }
    table["need_prediction"][0] = 0
    table["is_scored"][100] = 0
    table["i0_p0"][99:101] = 1.0  # bid-only at 99
    table["i0_p0"][101:] = 2.0  # both at 101
    table["i0_p11"][100:101] = 1.0  # ask-only at 100
    table["i0_p11"][101:] = 2.0  # both at 101
    table["a6"][101:] = 1.0
    table["t0"][99] = 0.25
    table["t0"][100] = 0.5
    table["t0"][101] = 0.75
    table["t1"][100] = -0.5
    table["t1"][101] = -1.0
    node: dict[str, object] = {
        "row_group": 17,
        "seq_ix": 73,
        "component": "fixture-component",
        "role": "fit",
        "label_class_diagnostic_only": 2,
    }
    return table, node
