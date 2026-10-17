#!/usr/bin/env python3
"""Verify target-cache row alignment against the original validation parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def stat_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def validate(args: argparse.Namespace) -> dict[str, object]:
    inventory_path = args.output_dir / "INVENTORY.json"
    result_path = args.output_dir / "RESULTS.json"
    output_path = args.output_dir / "ALIGNMENT.json"
    if not inventory_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            "inventory and completed grade must exist before alignment validation"
        )
    if output_path.exists():
        raise FileExistsError(output_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    cache_path = args.cache / "valid_targets.npz"
    parquet_path = args.data / "valid.parquet"
    for label, path, expected in (
        ("target cache", cache_path, inventory["target_cache"]),
        ("validation parquet", parquet_path, inventory["data"]),
    ):
        actual = stat_record(path)
        stable = {key: expected[key] for key in actual}
        if actual != stable:
            raise ValueError(f"{label} differs from pre-grade inventory stat")

    with np.load(cache_path, allow_pickle=False) as archive:
        cached_y = archive["y"]
        cached_mask = archive["mask"].astype(bool, copy=False)
        cached_seq = archive["seq_ix"].astype(np.int64, copy=False)
        row_groups = archive["row_group"].astype(np.int64, copy=False)
    positions = {int(group): position for position, group in enumerate(row_groups)}
    groups = [int(group) for group in inventory["shared_complete_coverage"]["row_groups"]]
    parquet = pq.ParquetFile(parquet_path)
    if parquet.metadata.num_row_groups != len(row_groups):
        raise ValueError("parquet row-group count differs from target-cache order")
    expected_steps = np.arange(cached_y.shape[1], dtype=np.int64)
    expected_need = expected_steps >= 99
    selected_rows = 0
    checked_rows = 0
    checks = {
        "seq_ix_mismatched_rows": 0,
        "step_mismatched_rows": 0,
        "target_mismatched_values": 0,
        "scored_mask_mismatched_rows": 0,
        "need_prediction_mismatched_rows": 0,
        "official_mask_mismatched_rows": 0,
        "nonfinite_cached_targets": 0,
    }
    for group in groups:
        position = positions[group]
        table = parquet.read_row_group(
            group,
            columns=["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"],
            use_threads=False,
        )
        if table.num_rows != cached_y.shape[1]:
            raise ValueError(f"row group {group} has {table.num_rows} rows")
        seq = table["seq_ix"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        step = table["step_in_seq"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        targets = np.column_stack(
            [table[name].to_numpy(zero_copy_only=False) for name in ("t0", "t1")]
        ).astype(np.float32, copy=False)
        row_mask = cached_mask[position]
        checks["seq_ix_mismatched_rows"] += int(np.count_nonzero(seq != cached_seq[position]))
        checks["step_mismatched_rows"] += int(np.count_nonzero(step != expected_steps))
        checks["target_mismatched_values"] += int(np.count_nonzero(targets != cached_y[position]))
        checks["scored_mask_mismatched_rows"] += int(np.count_nonzero(scored != row_mask))
        checks["need_prediction_mismatched_rows"] += int(np.count_nonzero(need != expected_need))
        official = need & scored
        cached_official = row_mask & expected_need
        checks["official_mask_mismatched_rows"] += int(
            np.count_nonzero(official != cached_official)
        )
        checks["nonfinite_cached_targets"] += int(
            np.count_nonzero(~np.isfinite(cached_y[position]))
        )
        selected_rows += int(official.sum())
        checked_rows += table.num_rows
    if any(checks.values()):
        raise AssertionError(f"target/parquet alignment mismatches found: {checks}")
    if selected_rows != inventory["target_schema"]["selected_rows_common_complete"]:
        raise AssertionError("official selected-row count differs from inventory")

    lookahead_presence = {
        candidate["cache_name"]: {
            "key_present": "lookahead" in candidate["checkpoint_settings"],
            "value": candidate["checkpoint_settings"].get("lookahead"),
        }
        for candidate in inventory["candidates"]
    }
    if any(
        not item["key_present"] or item["value"] not in ("", None)
        for item in lookahead_presence.values()
    ):
        raise ValueError(f"checkpoint lookahead was not explicitly empty: {lookahead_presence}")
    report = {
        "schema": "wunder.prediction-audit.alignment.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "validation_parquet_sha256": inventory["data"]["sha256"],
        "target_cache_sha256": inventory["target_cache"]["sha256"],
        "scope": {
            "row_groups": groups,
            "row_group_count": len(groups),
            "rows_checked": checked_rows,
            "official_selected_rows": selected_rows,
            "fold_counts": inventory["shared_complete_coverage"]["fold_counts"],
            "membership_sha256": inventory["shared_complete_coverage"]["membership_sha256"],
        },
        "checks": checks,
        "checkpoint_lookahead_key_presence_and_value": lookahead_presence,
        "prediction_order": "historical prediction arrays cannot be aligned to sequence IDs from current cache metadata: no producer receipt exists; their producer row order remains unverified",
        "result": "exact validation-cache target, scoring mask, sequence ID, and within-sequence step alignment on every shared-complete row group",
    }
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"alignment": str(output_path), **report["scope"], "checks": checks}, sort_keys=True
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
