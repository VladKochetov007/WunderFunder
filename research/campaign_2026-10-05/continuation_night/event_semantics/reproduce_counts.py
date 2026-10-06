"""Independently recount E492 fit-only raw events and compare sealed CSV tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from raw_fixtures import count_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COLUMNS = (
    "seq_ix",
    "step_in_seq",
    "need_prediction",
    "is_scored",
    "i0_p0",
    "i0_p11",
    "a6",
    "t0",
    "t1",
)
KEY_FIELDS = (
    "row_group",
    "seq_ix",
    "component",
    "role",
    "class_diagnostic",
    "population",
    "clock",
    "target",
)
AGGREGATE_KEY_FIELDS = ("class_diagnostic", "population", "clock", "target")
COUNT_FIELDS = (
    "rows",
    "events",
    "target_changes",
    "changes_without_event",
    "events_without_change",
    "a6_clock_disagreements",
)
FLOAT_FIELDS = ("absolute_target_mass",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _key(row: dict[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize CSV strings and source integers under an explicit table schema."""
    return tuple(str(row[name]) for name in fields)


def _count_value(value: object) -> Decimal:
    """Parse a serialized count exactly and reject non-integral values."""
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid E492 count value: {value!r}") from error
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"E492 count must be a finite integer: {value!r}")
    return parsed


def _assert_csv_matches(
    label: str,
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, object]],
    key_fields: tuple[str, ...],
) -> dict[str, int]:
    actual = {_key(row, key_fields): row for row in actual_rows}
    expected = {_key(row, key_fields): row for row in expected_rows}
    if len(actual) != len(actual_rows) or len(expected) != len(expected_rows):
        raise ValueError(f"duplicate E492 {label} table row")
    if actual.keys() != expected.keys():
        missing = len(expected.keys() - actual.keys())
        extra = len(actual.keys() - expected.keys())
        raise ValueError(f"E492 {label} row identities differ (missing={missing}, extra={extra})")
    compared = 0
    for key, expected_row in expected.items():
        actual_row = actual[key]
        for field in COUNT_FIELDS:
            if _count_value(actual_row[field]) != _count_value(expected_row[field]):
                raise ValueError(
                    f"E492 {label} count mismatch {key}/{field}: "
                    f"{actual_row[field]} != {expected_row[field]}"
                )
        for field in FLOAT_FIELDS:
            if not np.isclose(
                float(actual_row[field]), float(expected_row[field]), rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    f"E492 {label} float mismatch {key}/{field}: "
                    f"{actual_row[field]} != {expected_row[field]}"
                )
        compared += 1
    return {"rows": compared, "metrics_per_row": len(COUNT_FIELDS) + len(FLOAT_FIELDS)}


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sums: dict[tuple[int, str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        key = (
            int(row["class_diagnostic"]),
            str(row["population"]),
            str(row["clock"]),
            str(row["target"]),
        )
        for field in (*COUNT_FIELDS, *FLOAT_FIELDS):
            sums[key][field] += float(row[field])
    return [
        {
            "class_diagnostic": key[0],
            "population": key[1],
            "clock": key[2],
            "target": key[3],
            **values,
        }
        for key, values in sorted(sums.items())
    ]


def run_reproduction(artifact_dir: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"choose a fresh E492 reproduction path: {output}")
    report_path = artifact_dir / "report.json"
    per_component_path = artifact_dir / "per_component.csv"
    aggregate_path = artifact_dir / "aggregate.csv"
    cohort_path = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
    data_path = ROOT / "datasets/valid.parquet"
    report = json.loads(report_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    nodes = [node for node in cohort["cohort"] if node["role"] == "fit"]
    if len(nodes) != 24 or len({node["component"] for node in nodes}) != 24:
        raise ValueError("independent E492 reproduction requires 24 distinct fit components")
    if sha256_file(cohort_path) != report["cohort_sha256"]:
        raise ValueError("E492 cohort identity differs from the original report")
    if data_path.stat().st_size != report["dataset_size_bytes"]:
        raise ValueError("E492 raw dataset size differs from the original report")
    analyzer_path = HERE / "analyze.py"
    if sha256_file(analyzer_path) != report["source_sha256"]:
        raise ValueError("E492 analyzer source changed after original artifact creation")

    parquet = pq.ParquetFile(data_path)
    expected_rows: list[dict[str, object]] = []
    for node in nodes:
        table_arrow = parquet.read_row_group(
            int(node["row_group"]), columns=list(COLUMNS), use_threads=False
        )
        table = {name: table_arrow[name].to_numpy() for name in COLUMNS}
        expected_rows.extend(count_rows(table, node))
        del table, table_arrow

    component_check = _assert_csv_matches(
        "per-component", _read_rows(per_component_path), expected_rows, KEY_FIELDS
    )
    expected_aggregate = _aggregate(expected_rows)
    aggregate_check = _assert_csv_matches(
        "aggregate",
        _read_rows(aggregate_path),
        expected_aggregate,
        AGGREGATE_KEY_FIELDS,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    receipt = {
        "study": "E492",
        "hypothesis": "H120",
        "status": "independent_fit_only_raw_count_reproduction_passed",
        "source_sha256": sha256_file(Path(__file__)),
        "independent_oracle_sha256": sha256_file(HERE / "raw_fixtures.py"),
        "analyzer_source_sha256": sha256_file(analyzer_path),
        "cohort_sha256": sha256_file(cohort_path),
        "dataset_sha256": sha256_file(data_path),
        "per_component_csv_sha256": sha256_file(per_component_path),
        "aggregate_csv_sha256": sha256_file(aggregate_path),
        "fit_components_recounted": len(nodes),
        "evaluation_components_loaded": 0,
        "model_or_fit_calls": 0,
        "raw_group_count": component_check,
        "aggregate_count": aggregate_check,
        "tolerance": {
            "counts": "exact integers",
            "absolute_target_mass": "absolute tolerance 1e-9",
        },
        "claim_boundary": "reproduces descriptive fit-panel counts only; no predictivity or gain claim",
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(run_reproduction(arguments.artifact_dir, arguments.output), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
