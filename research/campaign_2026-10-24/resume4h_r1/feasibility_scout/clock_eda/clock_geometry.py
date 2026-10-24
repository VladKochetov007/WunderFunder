"""Check all-row event identity and age loss from fixed clock grouping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from clock_eda import ROOT, feature_layout


def group_age_error(ages: np.ndarray, groups: list[list[int]]) -> tuple[int, float]:
    """Count unequal ages and total normalized log-age gap over field-row pairs."""
    unequal = 0
    log_gap = 0.0
    denominator = np.log1p(10_000)
    for members in groups:
        field_age = np.minimum(ages[:, members], 10_000)
        group_age = field_age.min(axis=1, keepdims=True)
        unequal += int((field_age != group_age).sum())
        log_gap += float((np.log1p(field_age) - np.log1p(group_age)).sum()) / denominator
    return unequal, log_gap


def run(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Stream the fixed sample without labels and preserve exact event digests."""
    plan = json.loads(plan_path.read_text())
    basis_path = ROOT / plan["sample_plan"]
    if hashlib.sha256(basis_path.read_bytes()).hexdigest() != plan["sample_plan_sha256"]:
        raise ValueError("frozen sample changed")
    basis = json.loads(basis_path.read_text())
    names, _, family_map = feature_layout()
    if names != plan["feature_order"]:
        raise ValueError("feature order changed")
    source = ROOT / "datasets/valid.parquet"
    initial_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parquet = pq.ParquetFile(source, memory_map=True)
    if parquet.num_row_groups != 1873:
        raise ValueError("unexpected source footer")
    digests = [hashlib.sha256() for _ in names]
    change_counts = np.zeros(112, dtype=np.int64)
    seven = [
        list(range(a, b))
        for a, b in [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112)]
    ]
    groups = {"seven_existing_groups": seven, "nine_named_families": list(family_map.values())}
    totals = {name: {"unequal_pairs": 0, "normalized_log_age_gap_sum": 0.0} for name in groups}
    selected_count = 0
    members = []
    columns = names + ["seq_ix", "need_prediction", "is_scored"]
    for step, row_group in enumerate(basis["sample"]["valid"]):
        table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
        values = np.column_stack([table[name].to_numpy() for name in names]).astype(np.float32)
        seq = table["seq_ix"].to_numpy()
        if values.shape != (20_000, 112) or len(np.unique(seq)) != 1:
            raise ValueError("sample member is not one complete sequence")
        if not np.isfinite(values).all():
            raise ValueError("non-finite source feature")
        selected = table["need_prediction"].to_numpy() & table["is_scored"].to_numpy()
        changes = np.zeros_like(values, dtype=bool)
        changes[1:] = values[1:] != values[:-1]
        change_counts += changes.sum(axis=0, dtype=np.int64)
        packed = np.packbits(changes, axis=0, bitorder="little")
        for index, digest in enumerate(digests):
            digest.update(packed[:, index].tobytes())
        positions = np.arange(20_000, dtype=np.int32)[:, None]
        ages = positions - np.maximum.accumulate(np.where(changes, positions, 0), axis=0)
        selected_ages = ages[selected]
        for name, grouping in groups.items():
            count, gap = group_age_error(selected_ages, grouping)
            totals[name]["unequal_pairs"] += count
            totals[name]["normalized_log_age_gap_sum"] += gap
        count = int(selected.sum())
        selected_count += count
        members.append({"row_group": row_group, "seq_ix": int(seq[0]), "selected_rows": count})
        del table, values, seq, selected, changes, packed, ages, selected_ages
        if (step + 1) % 16 == 0:
            print(f"all-row clock geometry: {step + 1}/96", flush=True)
    if (source.stat().st_size, source.stat().st_mtime_ns) != initial_stat:
        raise ValueError("source stat changed during scan")
    classes: dict[str, list[str]] = {}
    for name, digest in zip(names, digests, strict=True):
        classes.setdefault(digest.hexdigest(), []).append(name)
    for statistics in totals.values():
        statistics["unequal_pair_rate"] = statistics["unequal_pairs"] / (selected_count * 112)
        statistics["mean_normalized_log_age_gap"] = statistics["normalized_log_age_gap_sum"] / (
            selected_count * 112
        )
    result = {
        "schema": "wnn.clock-geometry.v1",
        "status": "complete_descriptive_no_target",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "all_rows": 96 * 20_000,
        "selected_rows": selected_count,
        "members": members,
        "field_event_sha256": {
            name: digest.hexdigest() for name, digest in zip(names, digests, strict=True)
        },
        "all_row_change_counts": dict(zip(names, change_counts.tolist(), strict=True)),
        "exact_event_classes": list(classes.values()),
        "grouping_age_comparison": totals,
        "columns_read": columns,
        "source_stat": list(initial_stat),
        "new_fits": 0,
        "target_reads": False,
        "model_scores": 0,
        "limits": [
            "Exact pattern equality applies only to this fixed sample, not all data.",
            "Capped ages use10,000 and log1p normalization as the current age recipe.",
            "All-row event identity is stronger than scored-row Jaccard; both are descriptive.",
            "Grouping differences do not establish a predictive benefit.",
        ],
    }
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    """Run a frozen no-target clock geometry scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(
        json.dumps(
            {"status": result["status"], "exact_clock_classes": len(result["exact_event_classes"])}
        )
    )


if __name__ == "__main__":
    main()
