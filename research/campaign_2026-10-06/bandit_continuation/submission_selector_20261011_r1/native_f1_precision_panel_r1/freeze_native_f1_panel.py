"""Freeze the preregistered F1 complete-component feature-only panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_PANEL_SHA256 = "4d68036453ca83545efc35be66936ab4f9f821502d2036b8d3b41f28fd194e4f"
EXPECTED_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
EXPECTED_GROUP_COUNT = 41
EXPECTED_COMPONENT_COUNT = 14
EXPECTED_EVALUATION_GROUP_COUNT = 457
SEQUENCE_LENGTH = 20_000
LABEL_COLUMNS = {"t0", "t1", "is_scored"}


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array's contiguous C-order bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def cgroup_snapshot() -> dict[str, str | None]:
    """Capture the current process scope's bounded cgroup counters."""
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line.split("::", 1)[1] for line in membership if "::" in line), None)
    result: dict[str, str | None] = {"path": relative}
    if relative is None:
        return result
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for name in ("cpu.max", "cpu.stat", "memory.current", "memory.peak", "memory.events"):
        path = directory / name
        result[name] = path.read_text().strip() if path.is_file() else None
    return result


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def validate_selection(panel: dict[str, object], split: dict[str, object]) -> list[dict[str, int]]:
    """Check that selected valid-row groups equal whole selected eval components."""
    if panel.get("schema") != "wnn.complete_component_native_precision_panel.r1":
        raise ValueError("unexpected selection-panel schema")
    if panel.get("selection_seed") != 572911:
        raise ValueError("selection seed changed")
    if split.get("fold") != 1:
        raise ValueError("the panel must use fold 1")
    sequences = _object(split.get("sequences"), "split.sequences")
    valid_rows = sequences.get("valid")
    if not isinstance(valid_rows, list) or len(valid_rows) != 1873:
        raise ValueError("fold1 valid sequence mapping must contain 1,873 row groups")
    mapping: list[dict[str, int]] = []
    for index, raw_row in enumerate(valid_rows):
        row = _object(raw_row, f"split.sequences.valid[{index}]")
        if row.get("row_group") != index:
            raise ValueError(f"fold1 row-group order mismatch at index {index}")
        sequence_id = row.get("seq_ix")
        component_id = row.get("component")
        if not isinstance(sequence_id, int) or not isinstance(component_id, int):
            raise ValueError(f"invalid valid sequence mapping at row group {index}")
        mapping.append({"row_group": index, "sequence_id": sequence_id, "component": component_id})

    groups = panel.get("groups")
    components = panel.get("components")
    if not isinstance(groups, list) or len(groups) != EXPECTED_GROUP_COUNT:
        raise ValueError("the frozen panel must contain exactly 41 row groups")
    if not isinstance(components, list) or len(set(components)) != EXPECTED_COMPONENT_COUNT:
        raise ValueError("the frozen panel must name exactly 14 complete components")
    if any(not isinstance(group, int) or group < 0 or group >= len(mapping) for group in groups):
        raise ValueError("panel contains an invalid valid.parquet row-group index")
    if len(set(groups)) != len(groups):
        raise ValueError("panel contains duplicate row groups")

    evaluation = _object(split.get("evaluation"), "split.evaluation")
    evaluation_groups = evaluation.get("valid")
    if (
        not isinstance(evaluation_groups, list)
        or len(evaluation_groups) != EXPECTED_EVALUATION_GROUP_COUNT
    ):
        raise ValueError("fold1 split must identify 457 evaluation row groups")
    evaluation_set = set(evaluation_groups)
    if not set(groups) <= evaluation_set:
        raise ValueError("panel includes a group outside the fold1 evaluation partition")

    selected_components = {mapping[group]["component"] for group in groups}
    if selected_components != set(components):
        raise ValueError("selected row groups do not map to the frozen component list")
    complete_component_groups = {
        row["row_group"]
        for row in mapping
        if row["row_group"] in evaluation_set and row["component"] in selected_components
    }
    if set(groups) != complete_component_groups:
        missing = sorted(complete_component_groups - set(groups))
        extra = sorted(set(groups) - complete_component_groups)
        raise ValueError(f"component selection is incomplete: missing={missing}, extra={extra}")
    return [mapping[group] for group in groups]


def freeze_panel(
    project_root: Path,
    panel_path: Path,
    split_path: Path,
    valid_path: Path,
    fixture_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Copy only features and causal sequence metadata for the fixed panel."""
    for path in (project_root, panel_path, split_path, valid_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if fixture_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to replace a frozen feature fixture or receipt")
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("component-panel digest differs from the root-owned frozen panel")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("fold1 split manifest digest mismatch")

    panel = json.loads(panel_path.read_text())
    split = json.loads(split_path.read_text())
    selected = validate_selection(panel, split)
    sys.path.insert(0, str(project_root.resolve(strict=True)))
    from utils import FEATURE_COLUMNS  # noqa: PLC0415
    from utils import SEQUENCE_LENGTH as PROJECT_SEQUENCE_LENGTH  # noqa: PLC0415

    if PROJECT_SEQUENCE_LENGTH != SEQUENCE_LENGTH:
        raise ValueError("project sequence length differs from the frozen panel contract")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
    if LABEL_COLUMNS.intersection(columns):
        raise ValueError("feature-only fixture schema accidentally includes labels or score mask")

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = fixture_path.with_name(f".{fixture_path.name}.{os.getpid()}.partial")
    if temporary_path.exists():
        raise FileExistsError(temporary_path)
    temporary_path.touch(mode=0o600, exist_ok=False)
    started = time.perf_counter()
    before = cgroup_snapshot()
    group_records: list[dict[str, object]] = []
    source_stat = valid_path.stat()

    try:
        source = pq.ParquetFile(valid_path)
        if source.metadata.num_row_groups != 1873:
            raise ValueError("valid.parquet row-group count differs from the fold1 split mapping")
        with pq.ParquetWriter(
            temporary_path,
            pa.schema([source.schema_arrow.field(name) for name in columns]),
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            write_statistics=False,
        ) as writer:
            for selected_row in selected:
                group_index = selected_row["row_group"]
                table = source.read_row_group(group_index, columns=columns, use_threads=False)
                if table.num_rows != SEQUENCE_LENGTH:
                    raise ValueError(f"row group {group_index} is not one complete sequence")
                sequence_values = table["seq_ix"].to_numpy(zero_copy_only=False)
                step_values = table["step_in_seq"].to_numpy(zero_copy_only=False)
                need_values = table["need_prediction"].to_numpy(zero_copy_only=False)
                if not np.all(sequence_values == selected_row["sequence_id"]):
                    raise ValueError(f"sequence identity mismatch at row group {group_index}")
                if not np.array_equal(step_values, np.arange(SEQUENCE_LENGTH)):
                    raise ValueError(f"step order mismatch at row group {group_index}")
                if need_values[:99].any():
                    raise ValueError(f"need_prediction includes warmup rows at group {group_index}")
                features = np.column_stack(
                    [table[name].to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
                ).astype(np.float32, copy=False)
                if features.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
                    raise ValueError(f"unexpected feature shape at row group {group_index}")
                if not np.isfinite(features).all():
                    raise ValueError(f"non-finite feature at row group {group_index}")
                group_records.append(
                    {
                        **selected_row,
                        "rows": table.num_rows,
                        "feature_columns": len(FEATURE_COLUMNS),
                        "features_sha256": sha256_array(features),
                        "need_prediction_sha256": sha256_array(need_values),
                        "need_prediction_count": int(need_values.sum()),
                    }
                )
                writer.write_table(table, row_group_size=SEQUENCE_LENGTH)

        os.link(temporary_path, fixture_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    output = pq.ParquetFile(fixture_path)
    if output.metadata.num_row_groups != EXPECTED_GROUP_COUNT:
        raise ValueError("written feature fixture does not contain exactly 41 row groups")
    if set(output.schema_arrow.names) & LABEL_COLUMNS:
        raise ValueError("written feature fixture contains label or scoring-mask columns")
    after = cgroup_snapshot()
    receipt: dict[str, object] = {
        "schema": "wnn.native_f1_feature_panel_receipt.r1",
        "purpose": "feature-only native inference input; official labels are streamed separately",
        "selection": {
            "path": str(panel_path.resolve()),
            "sha256": sha256_file(panel_path),
            "seed": panel["selection_seed"],
            "groups": panel["groups"],
            "components": panel["components"],
            "evaluation_exposure": panel["evaluation_exposure"],
            "split_path": str(split_path.resolve()),
            "split_sha256": sha256_file(split_path),
        },
        "source_data": {
            "path": str(valid_path.resolve()),
            "bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "full_file_sha256": None,
            "identity_basis": "fold1 split SHA, selected row-group IDs, sequence IDs, feature and need hashes",
        },
        "fixture": {
            "path": str(fixture_path.resolve()),
            "bytes": fixture_path.stat().st_size,
            "sha256": sha256_file(fixture_path),
            "row_groups": EXPECTED_GROUP_COUNT,
            "rows_per_group": SEQUENCE_LENGTH,
            "columns": columns,
            "contains_labels": False,
            "contains_is_scored": False,
            "group_records": group_records,
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "project_utils_sha256": sha256_file(project_root / "utils.py"),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cgroup_before": before,
            "cgroup_after": after,
        },
    }
    with receipt_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Create the frozen 41-group feature fixture and source-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze_panel(
        args.project_root,
        args.panel,
        args.split,
        args.valid_parquet,
        args.fixture,
        args.receipt,
    )
    print(
        json.dumps(
            {"fixture": receipt["fixture"], "execution": receipt["execution"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
