"""One-off six-column, whole-row-group E535 diagnostic for the frozen H230 fold0 panel."""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _age_since(flags: np.ndarray) -> np.ndarray:
    row = np.arange(len(flags), dtype=np.int32)
    last = np.maximum.accumulate(np.where(flags, row, -1))
    return np.where(last >= 0, row - last, row + 1)


def _write_progress(path: Path, *, status: str, records: list[dict], groups: list[int],
                    age_rows: np.ndarray, age_weight: np.ndarray, value_counts: list[dict],
                    near_counts: dict[str, int], near_denominator: int, started: float,
                    split_path: Path, crosswalk_path: Path, input_record: dict) -> None:
    payload = {
        "id": "E535_FIXED468_STREAM_PROGRESS_V1_20261008",
        "status": status,
        "completed_groups": len(records),
        "requested_groups": len(groups),
        "completed_row_groups": [r["row_group"] for r in records],
        "records": records,
        "event_age_scored_rows": age_rows.tolist(),
        "event_age_target_weight": age_weight.tolist(),
        "target_value_counts_scored": value_counts,
        "near_event_target_counts": near_counts,
        "near_event_denominator": near_denominator,
        "elapsed_seconds": time.monotonic() - started,
        "input": input_record,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "crosswalk_sha256": hashlib.sha256(crosswalk_path.read_bytes()).hexdigest(),
        "interpretation": "Descriptive fixed-panel data scan only; no predictions, fit, or score.",
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, separators=(",", ":")))
    temp.replace(path)


def main():
    root = Path.cwd()
    lineage = root / "research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535"
    plots = lineage / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    crosswalk_path = lineage / "E535_REFERENCE_PANEL_RECONSTRUCTION_V1.json"
    split = json.loads(split_path.read_text())
    crosswalk = json.loads(crosswalk_path.read_text())
    groups = sorted(int(v) for v in split["evaluation"]["valid"])
    reference_groups = set(int(v) for v in crosswalk["current_reconstruction"]["selected_row_groups_sorted"])
    assert len(groups) == len(set(groups)) == 468
    assert len(reference_groups) == 432 and reference_groups.issubset(set(groups))
    columns = ["seq_ix", "i0_p0", "i0_p11", "t0", "t1", "is_scored"]
    data_path = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/datasets/valid.parquet")
    parquet = pq.ParquetFile(data_path)
    assert parquet.metadata.num_row_groups == 1873
    assert all(c in parquet.schema_arrow.names for c in columns)
    stat = data_path.stat()
    input_record = {
        "path": str(data_path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "row_groups": parquet.metadata.num_row_groups, "rows": parquet.metadata.num_rows,
        "projected_schema": {c: str(parquet.schema_arrow.field(c).type) for c in columns},
        "whole_file_sha256": None,
        "whole_file_hash_note": "Not computed; only six projected columns from 468 row groups were read.",
    }
    started = time.monotonic()
    wall_start = time.time()
    scan_budget_seconds = 780.0
    records: list[dict] = []
    age_rows = np.zeros((6, 6), dtype=np.int64)
    age_weight = np.zeros((2, 6, 6), dtype=np.float64)
    value_counts: list[dict] = [dict(), dict()]
    near_counts = {str(r): 0 for r in (0, 1, 2, 5)}
    near_denominator = 0
    progress_path = lineage / "E535_FIXED468_STREAM_PROGRESS_V1.json"
    final_path = lineage / "E535_FIXED468_VALID_ROW_GROUP_STREAM_V1.json"
    warmup = 99
    scored_window = 20000 - warmup
    age_edges = np.asarray([1, 3, 6, 11, 21], dtype=np.int32)
    age_labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    complete = True
    for ordinal, group in enumerate(groups, start=1):
        if time.monotonic() - started >= scan_budget_seconds:
            complete = False
            break
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        if table.num_rows != 20000:
            raise ValueError(f"row group {group} has {table.num_rows} rows, expected 20000")
        seq = table.column("seq_ix").combine_chunks().to_numpy(zero_copy_only=False)
        seq_values = np.unique(seq)
        if len(seq_values) != 1:
            raise ValueError(f"row group {group} crosses sequence ids: {seq_values[:5]}")
        seq_ix = int(seq_values[0])
        p0 = table.column("i0_p0").combine_chunks().to_numpy(zero_copy_only=False)
        p11 = table.column("i0_p11").combine_chunks().to_numpy(zero_copy_only=False)
        t0 = table.column("t0").combine_chunks().to_numpy(zero_copy_only=False)
        t1 = table.column("t1").combine_chunks().to_numpy(zero_copy_only=False)
        scored = table.column("is_scored").combine_chunks().to_numpy(zero_copy_only=False).astype(bool, copy=False)
        if scored[:warmup].any():
            raise ValueError(f"row group {group} has scored warmup rows")
        if not (np.isfinite(t0).all() and np.isfinite(t1).all()):
            raise ValueError(f"nonfinite target found in row group {group}")
        n = table.num_rows
        row = np.arange(n)
        px_change = np.zeros(n, dtype=bool)
        px_change[1:] = (p0[1:] != p0[:-1]) & (p11[1:] != p11[:-1])
        t0_change = np.zeros(n, dtype=bool)
        t1_change = np.zeros(n, dtype=bool)
        t0_change[1:] = t0[1:] != t0[:-1]
        t1_change[1:] = t1[1:] != t1[:-1]
        target_change = t0_change | t1_change
        post = row >= warmup
        scored_mask = post & scored
        score_rows = np.flatnonzero(scored_mask)
        w0 = np.abs(np.clip(t0, -2.0, 2.0).astype(np.float64, copy=False))
        w1 = np.abs(np.clip(t1, -2.0, 2.0).astype(np.float64, copy=False))
        px_age = _age_since(px_change)
        label_age = _age_since(target_change)
        px_bin = np.searchsorted(age_edges, px_age[score_rows], side="right")
        label_bin = np.searchsorted(age_edges, label_age[score_rows], side="right")
        flat_bins = px_bin * 6 + label_bin
        age_rows += np.bincount(flat_bins, minlength=36).reshape(6, 6)
        age_weight[0] += np.bincount(flat_bins, weights=w0[score_rows], minlength=36).reshape(6, 6)
        age_weight[1] += np.bincount(flat_bins, weights=w1[score_rows], minlength=36).reshape(6, 6)
        for head, vals in ((0, t0[score_rows]), (1, t1[score_rows])):
            unique, counts = np.unique(vals, return_counts=True)
            for value, count in zip(unique, counts, strict=True):
                key = format(float(value), ".9g")
                value_counts[head][key] = value_counts[head].get(key, 0) + int(count)
        target_rows = np.flatnonzero(target_change & post)
        near_denominator += len(target_rows)
        prefix_px = np.r_[0, np.cumsum(px_change, dtype=np.int64)]
        for radius in (0, 1, 2, 5):
            lo = np.maximum(target_rows - radius, 0)
            hi = np.minimum(target_rows + radius + 1, n)
            near_counts[str(radius)] += int(np.count_nonzero(prefix_px[hi] - prefix_px[lo]))
        t0_post, t1_post = t0[warmup:], t1[warmup:]
        post_change = np.zeros(scored_window, dtype=bool)
        post_change[1:] = (t0_post[1:] != t0_post[:-1]) | (t1_post[1:] != t1_post[:-1])
        bounds = np.r_[0, np.flatnonzero(post_change) + 1, scored_window]
        runs = np.diff(bounds)
        mass0 = float(w0[score_rows].sum(dtype=np.float64))
        mass1 = float(w1[score_rows].sum(dtype=np.float64))
        target_count = int(np.count_nonzero(target_change & post))
        same_row = int(np.count_nonzero(target_change & post & px_change))
        record = {
            "row_group": group, "seq_ix": seq_ix, "current_reference_432_panel": group in reference_groups,
            "rows": n, "scored_rows": int(len(score_rows)), "postwarm_rows": scored_window,
            "paired_i0_p0_i0_p11_price_code_changes_postwarm": int(np.count_nonzero(px_change & post)),
            "target_pair_changes_postwarm": target_count,
            "t0_changes_postwarm": int(np.count_nonzero(t0_change & post)),
            "t1_changes_postwarm": int(np.count_nonzero(t1_change & post)),
            "target_pair_changes_on_scored_rows": int(np.count_nonzero(target_change & scored_mask)),
            "same_row_target_price_event_cooccurrences": same_row,
            "target_run_rows_median_postwarm": float(np.median(runs)),
            "target_run_rows_p90_postwarm": float(np.quantile(runs, .9)),
            "target_abs_clipped_weight_t0": mass0, "target_abs_clipped_weight_t1": mass1,
            "paired_price_code_event_rate_per_1000_postwarm": float(np.count_nonzero(px_change & post) * 1000 / scored_window),
            "target_pair_change_rate_per_1000_postwarm": float(target_count * 1000 / scored_window),
            "target_change_near_price_event_counts": {}, "target_change_near_price_event_fractions": {},
        }
        for radius in (0, 1, 2, 5):
            lo = np.maximum(target_rows - radius, 0)
            hi = np.minimum(target_rows + radius + 1, n)
            c = int(np.count_nonzero(prefix_px[hi] - prefix_px[lo]))
            record["target_change_near_price_event_counts"][str(radius)] = c
            record["target_change_near_price_event_fractions"][str(radius)] = c / len(target_rows) if len(target_rows) else None
        records.append(record)
        del table, seq, seq_values, p0, p11, t0, t1, scored, w0, w1, row, px_change
        del t0_change, t1_change, target_change, post, scored_mask, score_rows, px_age, label_age
        del px_bin, label_bin, flat_bins, target_rows, prefix_px, t0_post, t1_post, post_change, bounds, runs
        if ordinal % 25 == 0:
            _write_progress(progress_path, status="in_progress", records=records, groups=groups,
                age_rows=age_rows, age_weight=age_weight, value_counts=value_counts,
                near_counts=near_counts, near_denominator=near_denominator, started=started,
                split_path=split_path, crosswalk_path=crosswalk_path, input_record=input_record)
            gc.collect()
    if len(records) != len(groups):
        complete = False
    _write_progress(progress_path, status="complete" if complete else "incomplete_runtime_bound",
        records=records, groups=groups, age_rows=age_rows, age_weight=age_weight,
        value_counts=value_counts, near_counts=near_counts, near_denominator=near_denominator,
        started=started, split_path=split_path, crosswalk_path=crosswalk_path, input_record=input_record)
    if not complete:
        raise TimeoutError(f"bounded scan stopped after {len(records)}/{len(groups)} groups; progress receipt preserved")
    assert len(records) == 468 and len({r["row_group"] for r in records}) == 468
    assert len({r["seq_ix"] for r in records}) == 468

    def assign_tertiles(key: str) -> list[int]:
        order = sorted(range(len(records)), key=lambda i: (records[i][key], records[i]["row_group"]))
        labels = [0] * len(records)
        for rank, ix in enumerate(order):
            labels[ix] = min(2, rank * 3 // len(records))
        return labels

    activity_tertile = assign_tertiles("paired_price_code_event_rate_per_1000_postwarm")
    label_tertile = assign_tertiles("target_pair_change_rate_per_1000_postwarm")
    for ix, rec in enumerate(records):
        rec["paired_price_code_activity_tertile"] = int(activity_tertile[ix])
        rec["target_pair_change_clock_tertile"] = int(label_tertile[ix])
    tertile_grid = []
    for activity in range(3):
        for clock in range(3):
            subset = [r for ix, r in enumerate(records) if activity_tertile[ix] == activity and label_tertile[ix] == clock]
            m0 = float(sum(r["target_abs_clipped_weight_t0"] for r in subset))
            m1 = float(sum(r["target_abs_clipped_weight_t1"] for r in subset))
            tertile_grid.append({"activity_tertile": activity, "target_change_clock_tertile": clock,
                "groups": len(subset), "scored_rows": int(sum(r["scored_rows"] for r in subset)),
                "t0_abs_clipped_target_weight": m0, "t1_abs_clipped_target_weight": m1,
                "t0_weight_share": None, "t1_weight_share": None})
    total_mass = [float(sum(r[f"target_abs_clipped_weight_t{k}"] for r in records)) for k in (0, 1)]
    for rec in tertile_grid:
        rec["t0_weight_share"] = rec["t0_abs_clipped_target_weight"] / total_mass[0] if total_mass[0] else None
        rec["t1_weight_share"] = rec["t1_abs_clipped_target_weight"] / total_mass[1] if total_mass[1] else None
    ref_records = [r for r in records if r["current_reference_432_panel"]]
    extra_records = [r for r in records if not r["current_reference_432_panel"]]
    panel_summary = {}
    for name, subset in (("current_reconstructed_reference_432", ref_records), ("H230_only_extra_36", extra_records)):
        panel_summary[name] = {
            "groups": len(subset), "scored_rows": int(sum(r["scored_rows"] for r in subset)),
            "target_pair_changes_postwarm": int(sum(r["target_pair_changes_postwarm"] for r in subset)),
            "paired_price_code_changes_postwarm": int(sum(r["paired_i0_p0_i0_p11_price_code_changes_postwarm"] for r in subset)),
            "t0_abs_clipped_target_weight": float(sum(r["target_abs_clipped_weight_t0"] for r in subset)),
            "t1_abs_clipped_target_weight": float(sum(r["target_abs_clipped_weight_t1"] for r in subset)),
            "mean_scored_rows_per_group": float(np.mean([r["scored_rows"] for r in subset])) if subset else None,
            "median_target_run_rows_per_group": float(np.median([r["target_run_rows_median_postwarm"] for r in subset])) if subset else None,
        }
    age_table = []
    for a, a_name in enumerate(age_labels):
        for l, l_name in enumerate(age_labels):
            age_table.append({"paired_price_event_age_bin": a_name, "target_pair_change_age_bin": l_name,
                "scored_rows": int(age_rows[a, l]), "t0_abs_clipped_target_weight": float(age_weight[0, a, l]),
                "t1_abs_clipped_target_weight": float(age_weight[1, a, l]),
                "t0_weight_share": float(age_weight[0, a, l] / total_mass[0]) if total_mass[0] else None,
                "t1_weight_share": float(age_weight[1, a, l] / total_mass[1]) if total_mass[1] else None})
    near_summary = {
        "target_pair_change_events_postwarm": near_denominator,
        "counts_with_at_least_one_paired_price_code_change_within_rows": near_counts,
        "fractions_with_at_least_one_paired_price_code_change_within_rows": {
            r: v / near_denominator if near_denominator else None for r, v in near_counts.items()},
        "window_definition": "For radius r at target event row t, count if a paired code-change event exists in inclusive [t-r,t+r], with sequence-boundary truncation; descriptive co-occurrence only.",
    }
    summary = {
        "groups": len(records), "rows": int(sum(r["rows"] for r in records)),
        "scored_rows": int(sum(r["scored_rows"] for r in records)),
        "target_pair_changes_postwarm": int(sum(r["target_pair_changes_postwarm"] for r in records)),
        "paired_price_code_changes_postwarm": int(sum(r["paired_i0_p0_i0_p11_price_code_changes_postwarm"] for r in records)),
        "same_row_target_price_event_cooccurrences": int(sum(r["same_row_target_price_event_cooccurrences"] for r in records)),
        "t0_abs_clipped_target_weight": total_mass[0], "t1_abs_clipped_target_weight": total_mass[1],
        "target_run_rows_median_sequence_median": float(np.median([r["target_run_rows_median_postwarm"] for r in records])),
        "target_run_rows_p90_sequence_median": float(np.median([r["target_run_rows_p90_postwarm"] for r in records])),
        "target_value_counts_scored": value_counts, "near_event_summary": near_summary,
        "reference_vs_extra_panel": panel_summary, "sequence_activity_by_target_clock_tertile": tertile_grid,
        "row_event_age_by_target_age": age_table,
        "activity_tertile_definition": "Stable equal-count rank tertiles over per-sequence paired code-event rate per 1,000 postwarm rows; row_group breaks ties.",
        "target_clock_tertile_definition": "Stable equal-count rank tertiles over per-sequence pair target-change rate per 1,000 postwarm rows; row_group breaks ties.",
        "age_bin_labels": age_labels,
        "target_weights": "Per selected row, absolute value after official target clip to [-2,2]; no predictions and no WP score computed.",
    }
    for rec in records:
        rec.pop("target_change_near_price_event_counts", None)
        rec.pop("target_change_near_price_event_fractions", None)
    payload = {
        "id": "E535_FIXED468_VALID_ROW_GROUP_STREAM_V1_20261008", "status": "complete_descriptive_scan_no_score",
        "split_manifest": str(split_path.relative_to(root)), "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "current_reference_crosswalk": str(crosswalk_path.relative_to(root)), "current_reference_crosswalk_sha256": hashlib.sha256(crosswalk_path.read_bytes()).hexdigest(),
        "current_reference_groups_sha256": hashlib.sha256(json.dumps(sorted(reference_groups), separators=(",", ":")).encode()).hexdigest(),
        "H230_evaluation_groups_sha256": hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest(),
        "declared_evaluation_groups": len(groups), "reconstructed_current_reference_groups": len(reference_groups),
        "H230_only_extra_groups": len(set(groups) - reference_groups), "selected_columns": columns,
        "read_mode": "pyarrow ParquetFile.read_row_group, use_threads=False, one complete row group at a time",
        "official_need_prediction_mask": "row index >= 99; steps 0..98 are warmup per docs/data_overview.md",
        "data_input": input_record, "elapsed_seconds": time.monotonic() - started, "started_unix_seconds": wall_start,
        "resource_limit": {"cpu_quota": "100% / one CPU", "memory_max_bytes": 1500000000, "runtime_max_seconds": 900, "threads": 1},
        "summary": summary, "records": records,
        "interpretation_limit": "Previously exposed fold0 validation; descriptive target/activity diagnostics only. No heldout score, public forecast, causal result, or target-law claim. p0/p11 are column identifiers; paired changes are an event proxy, not verified best-quote updates.",
    }
    final_path.write_text(json.dumps(payload, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for subset, legend_label, color in ((ref_records, "current reference reconstruction (432)", "#2878B5"), (extra_records, "H230-only extra (36)", "#D95F02")):
        sizes = [max(8, min(90, 12 + .015 * r["scored_rows"])) for r in subset]
        ax[0, 0].scatter([r["paired_price_code_event_rate_per_1000_postwarm"] for r in subset], [r["target_pair_change_rate_per_1000_postwarm"] for r in subset], s=sizes, alpha=.72, label=legend_label, color=color, edgecolors="none")
    ax[0, 0].set(xlabel="paired i0_p0/i0_p11 code changes / 1,000 postwarm rows", ylabel="t0/t1 pair changes / 1,000 postwarm rows", title="Sequence activity vs target clock")
    ax[0, 0].legend(frameon=False, fontsize=8); ax[0, 0].grid(alpha=.2)
    for head, panel in ((0, ax[0, 1]), (1, ax[0, 2])):
        matrix = np.zeros((3, 3), dtype=float)
        for item in tertile_grid:
            matrix[item["activity_tertile"], item["target_change_clock_tertile"]] = item[f"t{head}_weight_share"] or 0.0
        image = panel.imshow(matrix, cmap="Blues", vmin=0, vmax=max(.01, matrix.max()))
        for a in range(3):
            for l in range(3): panel.text(l, a, f"{matrix[a,l]*100:.1f}%", ha="center", va="center", fontsize=9)
        panel.set(xticks=[0,1,2], xticklabels=["low","mid","high"], yticks=[0,1,2], yticklabels=["low","mid","high"], xlabel="target-change-rate tertile", ylabel="price-code activity tertile", title=f"t{head} official target-weight share")
        fig.colorbar(image, ax=panel, fraction=.046, pad=.04)
    for head, panel in ((0, ax[1, 0]), (1, ax[1, 1])):
        matrix = age_weight[head] / total_mass[head] if total_mass[head] else age_weight[head]
        image = panel.imshow(matrix, cmap="magma", aspect="auto")
        for a in range(6):
            for l in range(6):
                if matrix[a,l] >= .012: panel.text(l, a, f"{matrix[a,l]*100:.1f}%", ha="center", va="center", color="white", fontsize=7)
        panel.set(xticks=range(6), xticklabels=age_labels, yticks=range(6), yticklabels=age_labels, xlabel="rows since last t0/t1 pair change", ylabel="rows since paired p-code event", title=f"t{head} target-weight share by row clock")
        fig.colorbar(image, ax=panel, fraction=.046, pad=.04)
    ax[1, 2].axis("off")
    summary_text = (f"Fixed H230 fold0 panel: {len(records)} groups\nCurrent reference reconstruction: {len(ref_records)}; H230-only extra: {len(extra_records)}\nScored rows: {summary['scored_rows']:,}\nTarget-pair changes: {summary['target_pair_changes_postwarm']:,}\nPaired p0/p11 code events: {summary['paired_price_code_changes_postwarm']:,}\nSame-row co-occurrences: {summary['same_row_target_price_event_cooccurrences']:,}\nClipped-|target| mass: t0 {total_mass[0]:,.1f} / t1 {total_mass[1]:,.1f}\nDescriptive exposed-data scan; no predictions or score.")
    ax[1, 2].text(.02, .98, summary_text, va="top", family="monospace", fontsize=10, transform=ax[1, 2].transAxes)
    ax[1, 2].set_title("Membership and scan summary")
    fig.suptitle("E535 fixed 468-group fold0 scan — paired price-code proxy, target clocks, and official target-weight mass")
    fig.savefig(plots / "E535_FIXED468_event_target_weight_diagnostic_V1.png", dpi=170)
    fig.savefig(plots / "E535_FIXED468_event_target_weight_diagnostic_V1.svg")
    print(json.dumps({"id": payload["id"], "status": payload["status"], "summary": summary,
        "near_event": near_summary, "panel": panel_summary, "elapsed_seconds": payload["elapsed_seconds"],
        "json": str(final_path), "png": str(plots / "E535_FIXED468_event_target_weight_diagnostic_V1.png")}, separators=(",", ":")))
    return fig


if __name__ == "__main__":
    FIGURE = main()
