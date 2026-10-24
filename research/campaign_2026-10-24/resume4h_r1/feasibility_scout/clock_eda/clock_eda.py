"""Stream no-target update-clock statistics for frozen validation sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
POSITION_EDGES = (0, 99, 500, 2_000, 10_000, 20_000)
AGE_THRESHOLDS = (1, 16, 64, 512, 4_096, 10_000)
FAMILY_WIDTHS = (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))


def feature_layout() -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Return the live notebook's 112 fields and its nine known families."""
    names: list[str] = []
    groups: dict[str, list[int]] = {}
    for stream in range(2):
        for family, width in FAMILY_WIDTHS:
            group = f"i{stream}_{family}"
            groups[group] = list(range(len(names), len(names) + width))
            names.extend(f"{group}{index}" for index in range(width))
    groups["a"] = list(range(len(names), len(names) + 8))
    names.extend(f"a{index}" for index in range(8))
    return names, list(groups), groups


def _divide(counts: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide counts safely for empty bins and fields."""
    return np.divide(
        counts,
        denominator,
        out=np.zeros_like(counts, dtype=np.float64),
        where=denominator != 0,
    )


def summarize_sequence(
    features: np.ndarray,
    seq_ix: np.ndarray,
    need_prediction: np.ndarray,
    is_scored: np.ndarray,
    groups: dict[str, list[int]],
    position_edges: tuple[int, ...] = POSITION_EDGES,
    age_thresholds: tuple[int, ...] = AGE_THRESHOLDS,
) -> dict[str, Any]:
    """Count exact change events, event overlaps, censored ages and position."""
    values = np.asarray(features, dtype=np.float32)
    ids = np.asarray(seq_ix)
    selected = np.asarray(need_prediction, dtype=bool) & np.asarray(is_scored, dtype=bool)
    rows = len(values)
    if values.shape != (rows, 112) or rows == 0 or ids.shape != (rows,):
        raise ValueError("expected one nonempty 112-feature sequence")
    if len(np.unique(ids)) != 1 or selected.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError("sequence ID, row masks, or feature values are invalid")
    if position_edges[0] != 0 or position_edges[-1] != rows:
        raise ValueError("position bands must span the complete sequence")

    changed = np.zeros(values.shape, dtype=bool)
    changed[1:] = values[1:] != values[:-1]  # Row zero is not an observed event.
    seen = np.maximum.accumulate(changed, axis=0)
    index = np.arange(rows, dtype=np.int32)
    last = np.maximum.accumulate(np.where(changed, index[:, None], 0), axis=0)
    ages = index[:, None] - last
    censored = ~seen
    band_count = len(position_edges) - 1
    threshold_count = len(age_thresholds)
    eligible_by_band = np.zeros(band_count, dtype=np.int64)
    changes_by_band = np.zeros((band_count, 112), dtype=np.int64)
    censored_by_band = np.zeros((band_count, 112), dtype=np.int64)
    ages_by_band = np.zeros((band_count, 112, threshold_count), dtype=np.int64)
    for band, (start, stop) in enumerate(zip(position_edges[:-1], position_edges[1:], strict=True)):
        selected_rows = np.flatnonzero(selected[start:stop]) + start
        eligible_by_band[band] = len(selected_rows)
        if not len(selected_rows):
            continue
        events = changed[selected_rows]
        changes_by_band[band] = events.sum(axis=0, dtype=np.int64)
        censored_by_band[band] = censored[selected_rows].sum(axis=0, dtype=np.int64)
        age_rows = ages[selected_rows]
        for j, threshold in enumerate(age_thresholds):
            ages_by_band[band, :, j] = (age_rows >= threshold).sum(axis=0, dtype=np.int64)

    selected_events = changed[selected].astype(np.int32, copy=False)
    return {
        "seq_ix": int(ids[0]),
        "rows": rows,
        "eligible_rows": int(selected.sum()),
        "selected_change_events": int(selected_events.sum()),
        "selected_change_events_by_family": {
            name: int(selected_events[:, indices].sum()) for name, indices in groups.items()
        },
        "eligible_rows_by_position": eligible_by_band,
        "change_counts_by_position": changes_by_band,
        "censored_counts_by_position": censored_by_band,
        "age_ge_counts_by_position": ages_by_band,
        "cochange_counts": selected_events.T @ selected_events,
    }


def _summarize_groups(
    jaccard: np.ndarray, group_names: list[str], groups: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """Return the mean feature-pair Jaccard for each family-pair cell."""
    summaries = []
    for left_index, left_name in enumerate(group_names):
        left = np.asarray(groups[left_name], dtype=np.int64)
        for right_name in group_names[left_index:]:
            right = np.asarray(groups[right_name], dtype=np.int64)
            block = jaccard[np.ix_(left, right)]
            if left_name == right_name:
                block = block[np.triu_indices(len(left), k=1)]
            summaries.append(
                {
                    "group_a": left_name,
                    "group_b": right_name,
                    "feature_pair_count": int(block.size),
                    "mean_jaccard": float(block.mean()) if block.size else 0.0,
                }
            )
    return summaries


def finalize(
    totals: dict[str, Any],
    sequence_counts: list[dict[str, Any]],
    names: list[str],
    group_names: list[str],
    groups: dict[str, list[int]],
    position_edges: tuple[int, ...],
    age_thresholds: tuple[int, ...],
) -> dict[str, Any]:
    """Build compact, plot-ready cochange and position/age summaries."""
    intersections = totals["cochange_counts"]
    changes = totals["change_counts_by_position"].sum(axis=0)
    unions = changes[:, None] + changes[None, :] - intersections
    jaccard = _divide(intersections, unions)
    eligible = totals["eligible_rows_by_position"]
    change_counts = totals["change_counts_by_position"]
    censored_counts = totals["censored_counts_by_position"]
    age_counts = totals["age_ge_counts_by_position"]
    family_change_counts = np.stack(
        [change_counts[:, indices].sum(axis=1) for indices in groups.values()], axis=1
    )
    family_age_rates = np.stack(
        [
            _divide(
                age_counts[:, indices].sum(axis=1),
                (eligible[:, None] * len(indices)),
            )
            for indices in groups.values()
        ],
        axis=1,
    )
    return {
        "sampled_sequence_count": len(sequence_counts),
        "eligible_rows": int(totals["eligible_rows"]),
        "sequence_counts": sequence_counts,
        "pairwise_update_events": {
            "feature_order": names,
            "feature_group_by_feature": [
                group for group, indices in groups.items() for _ in indices
            ],
            "cochange_event_counts": intersections.tolist(),
            "jaccard": jaccard.tolist(),
            "group_pair_mean_jaccard": _summarize_groups(jaccard, group_names, groups),
        },
        "position_activity_age": {
            "position_edges": list(position_edges),
            "position_labels": [
                f"{start}-{stop - 1}"
                for start, stop in zip(position_edges[:-1], position_edges[1:], strict=True)
            ],
            "age_thresholds": list(age_thresholds),
            "eligible_rows_by_position": eligible.tolist(),
            "change_counts_by_feature": change_counts.tolist(),
            "change_rates_by_feature": _divide(change_counts, eligible[:, None]).tolist(),
            "censored_counts_by_feature": censored_counts.tolist(),
            "censored_rates_by_feature": _divide(censored_counts, eligible[:, None]).tolist(),
            "age_ge_counts_by_feature": age_counts.tolist(),
            "age_ge_rates_by_feature": _divide(age_counts, eligible[:, None, None]).tolist(),
            "feature_group_order": group_names,
            "change_counts_by_group": family_change_counts.tolist(),
            "age_ge_rates_by_group": family_age_rates.tolist(),
        },
        "limits": [
            "Fixed, previously exposed validation sample; no fresh holdout claim.",
            "Exact adjacent-row float32 inequality defines an event; row zero is excluded.",
            "No labels, targets, predictions, model scores, or fitted coefficients are read or computed.",
            "Associations are descriptive and do not establish model value or causality.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def run(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Read 96 frozen valid row groups one sequence at a time, no target fields."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    parent_path = ROOT / plan["parent_sample_plan"]["path"]
    if _sha256(parent_path) != plan["parent_sample_plan"]["sha256"]:
        raise ValueError("parent sample plan hash differs")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    sample = plan["sample"]["valid"]
    if parent.get("id") != "EDA-CENSORING-20261024-R2" or sample != parent["sample"]["valid"]:
        raise ValueError("clock plan does not preserve the R2 validation sample")
    if len(sample) != 96 or len(set(sample)) != 96:
        raise ValueError("expected exactly 96 unique frozen validation row groups")
    names, group_names, groups = feature_layout()
    if names != plan["feature_names"] or group_names != plan["feature_groups"]:
        raise ValueError("feature order differs from the frozen live notebook contract")

    source = ROOT / plan["source_path"]
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parquet = pq.ParquetFile(source, memory_map=True)
    if parquet.num_row_groups != int(plan["source_row_group_count"]):
        raise ValueError("validation Parquet row-group count differs")
    columns = [*names, "seq_ix", "need_prediction", "is_scored"]
    if columns != plan["input_columns"]:
        raise ValueError("input columns differ from the frozen no-target contract")
    if not set(columns).issubset(set(parquet.schema_arrow.names)):
        raise ValueError("validation source lacks a required no-target input column")
    edges = tuple(plan["position_edges"])
    thresholds = tuple(plan["age_thresholds"])
    totals = {
        "eligible_rows": 0,
        "eligible_rows_by_position": np.zeros(len(edges) - 1, dtype=np.int64),
        "change_counts_by_position": np.zeros((len(edges) - 1, 112), dtype=np.int64),
        "censored_counts_by_position": np.zeros((len(edges) - 1, 112), dtype=np.int64),
        "age_ge_counts_by_position": np.zeros(
            (len(edges) - 1, 112, len(thresholds)), dtype=np.int64
        ),
        "cochange_counts": np.zeros((112, 112), dtype=np.int64),
    }
    sequence_counts = []
    for row_group in sample:
        table = parquet.read_row_group(int(row_group), columns=columns, use_threads=False)
        features = np.column_stack([table[name].to_numpy() for name in names])
        stats = summarize_sequence(
            features,
            table["seq_ix"].to_numpy(),
            table["need_prediction"].to_numpy(),
            table["is_scored"].to_numpy(),
            groups,
            edges,
            thresholds,
        )
        if len(features) != int(plan["rows_per_sequence"]):
            raise ValueError(f"row group {row_group} is not one complete sequence")
        sequence_counts.append(
            {
                "row_group": int(row_group),
                **{
                    key: stats[key]
                    for key in (
                        "seq_ix",
                        "rows",
                        "eligible_rows",
                        "selected_change_events",
                        "selected_change_events_by_family",
                    )
                },
            }
        )
        totals["eligible_rows"] += stats["eligible_rows"]
        for key in (
            "eligible_rows_by_position",
            "change_counts_by_position",
            "censored_counts_by_position",
            "age_ge_counts_by_position",
            "cochange_counts",
        ):
            totals[key] += stats[key]
        del table, features, stats

    result = {
        "schema": "wnn.clock-update-eda.v1",
        "status": "complete_descriptive_no_target_eda",
        "plan_sha256": _sha256(plan_path.resolve()),
        "parent_sample_plan_sha256": plan["parent_sample_plan"]["sha256"],
        "reviewed_live_source_sha256": plan["change_definition_source"][
            "reviewed_live_source_sha256"
        ],
        "source": {
            "path": str(source),
            "file_bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "row_group_count": parquet.num_row_groups,
            "row_groups_read": len(sequence_counts),
            "columns_read": columns,
        },
        "denominator": "need_prediction AND is_scored",
        "new_fits": 0,
        "model_evaluations": 0,
        **finalize(totals, sequence_counts, names, group_names, groups, edges, thresholds),
    }
    output_path = output_path.resolve()
    scratch = Path("/home/vlad/development/wunder/research_tmp").resolve()
    if not output_path.is_relative_to(scratch):
        raise ValueError("EDA output must use approved physical scratch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    """Run the frozen streaming EDA command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"status": result["status"], "sequences": 96, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
