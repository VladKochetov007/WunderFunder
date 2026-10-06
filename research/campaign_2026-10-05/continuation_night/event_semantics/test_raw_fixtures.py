from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

EVENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_DIR))

import analyze  # noqa: E402
import reproduce_counts  # noqa: E402
from raw_fixtures import count_rows, make_hand_fixture, validate_raw_group  # noqa: E402


def _index(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {(str(row["population"]), str(row["clock"]), str(row["target"])): row for row in rows}


def test_hand_fixture_preserves_raw_transition_before_current_row_selection() -> None:
    table, node = make_hand_fixture()
    rows = _index(count_rows(table, node))
    first = rows[("eligible", "bid", "t0")]
    assert first["rows"] == 19_901
    assert first["target_changes"] == 4
    assert first["changes_without_event"] == 2
    assert first["absolute_target_mass"] == pytest.approx(1.5)


def test_bid_and_ask_clocks_each_include_simultaneous_changes() -> None:
    table, node = make_hand_fixture()
    rows = _index(count_rows(table, node))
    assert rows[("eligible", "bid", "t0")]["events"] == 2
    assert rows[("eligible", "ask", "t0")]["events"] == 2
    assert rows[("eligible", "or", "t0")]["events"] == 3
    assert rows[("eligible", "and", "t0")]["events"] == 1
    assert rows[("eligible", "bid", "t1")]["a6_clock_disagreements"] == 1
    assert rows[("eligible", "ask", "t1")]["a6_clock_disagreements"] == 1


def test_frozen_analyzer_matches_independent_hand_fixture_oracle() -> None:
    table, node = make_hand_fixture()
    expected = count_rows(table, node)
    observed = analyze.summarize_group(table, node)
    assert observed == expected


def test_reproducer_normalizes_csv_integer_keys_and_uses_aggregate_schema() -> None:
    table, node = make_hand_fixture()
    expected = count_rows(table, node)
    csv_rows = [{name: str(value) for name, value in row.items()} for row in expected]

    component_check = reproduce_counts._assert_csv_matches(
        "fixture-per-component", csv_rows, expected, reproduce_counts.KEY_FIELDS
    )
    expected_aggregate = reproduce_counts._aggregate(expected)
    aggregate_csv_rows = [
        {name: str(value) for name, value in row.items()} for row in expected_aggregate
    ]
    aggregate_check = reproduce_counts._assert_csv_matches(
        "fixture-aggregate",
        aggregate_csv_rows,
        expected_aggregate,
        reproduce_counts.AGGREGATE_KEY_FIELDS,
    )

    assert component_check["rows"] == 16
    assert aggregate_check["rows"] == 16
    assert len(reproduce_counts.AGGREGATE_KEY_FIELDS) == 4


def test_independent_oracle_rejects_nonconsecutive_steps() -> None:
    table, node = make_hand_fixture()
    table["step_in_seq"][101] = 102
    with pytest.raises(ValueError, match="arange"):
        validate_raw_group(table, int(node["seq_ix"]))
    with pytest.raises(ValueError, match="ordered, consecutive"):
        analyze.summarize_group(table, node)


@pytest.mark.parametrize("column,bad_value", [("need_prediction", 2), ("is_scored", float("nan"))])
def test_frozen_analyzer_rejects_invalid_binary_masks(column: str, bad_value: float) -> None:
    table, node = make_hand_fixture()
    if isinstance(bad_value, float) and math.isnan(bad_value):
        table[column] = table[column].astype(float)
    table[column][50] = bad_value
    with pytest.raises(ValueError, match="finite and binary"):
        validate_raw_group(table, int(node["seq_ix"]))
    with pytest.raises(ValueError, match="binary row mask"):
        analyze.summarize_group(table, node)
