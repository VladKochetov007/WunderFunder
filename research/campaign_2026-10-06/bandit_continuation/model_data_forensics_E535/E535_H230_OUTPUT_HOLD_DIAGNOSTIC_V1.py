"""One-shot group-streamed output-hold screen for the frozen H230 candidate."""

from __future__ import annotations

import gc
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
WORKTREE_ROOT = Path("/home/vlad/development/wunder/worktrees/wnn-model-data-forensics-E535")
LINEAGE = (
    WORKTREE_ROOT / "research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535"
)
GRADE_PATH = (
    MAIN_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_MATCHED_CONTROL_GRADE_V1_20261008.json"
)
EXPORT_PATH = (
    MAIN_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SEED3_EXPORT_TERMINAL_RECEIPT_V1_SOURCEV2_20261008.json"
)
PREDICTION_PATH = MAIN_ROOT / "cache/pred_h230_hetero_s3_ep1_20261008_v1_valid.npy"
PREDICTION_SIDECAR_PATH = PREDICTION_PATH.with_suffix(".json")
SPLIT_PATH = MAIN_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PARQUET_PATH = MAIN_ROOT / "datasets/valid.parquet"
METRIC_PATH = MAIN_ROOT / "src/wnn/gmetric.py"
RESULT_PATH = LINEAGE / "E535_H230_OUTPUT_HOLD_RESULT_V1_20261008.json"
PROGRESS_PATH = LINEAGE / "E535_H230_OUTPUT_HOLD_PROGRESS_V1_20261008.json"
PNG_PATH = LINEAGE / "plots/E535_H230_seed3_ep1_event_output_hold_V1.png"
SVG_PATH = LINEAGE / "plots/E535_H230_seed3_ep1_event_output_hold_V1.svg"
EXPECTED_METRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
EXPECTED_BASELINE_WP = 0.6817195016815873
EXPECTED_SELECTED_ROWS = 808_863
EXPECTED_PREDICTION_SHA256 = "8f0f53f7990739847cb4ecd785b508ef32a59c982a702b5e05cf04bbd1d78e78"
EXPECTED_SIDECAR_SHA256 = "71feb03d21843c396ded01664b6f160198910c86935cf3fb9d77e8bf5717b4f4"
WARMUP = 99
SEQUENCE_LENGTH = 20_000
MAX_SCAN_SECONDS = 420.0


def sha256_file(path: Path) -> str:
    """Hash a small provenance file without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def event_hold(baseline: np.ndarray, paired_event: np.ndarray) -> np.ndarray:
    """Hold baseline emissions from each paired event, reset at each sequence."""
    indices = np.maximum.accumulate(
        np.where(paired_event, np.arange(len(paired_event), dtype=np.int32), 0)
    )
    return np.asarray(baseline[indices], dtype=np.float32).copy()


def _write_progress(
    records: list[dict[str, object]],
    requested: int,
    started: float,
    status: str = "in_progress",
) -> None:
    payload = {
        "id": "E535_H230_OUTPUT_HOLD_PROGRESS_V1_20261008",
        "status": status,
        "completed_groups": len(records),
        "requested_groups": requested,
        "elapsed_seconds": time.monotonic() - started,
        "group_hash_records": records,
        "interpretation": "Execution progress only; no partial score is a result.",
    }
    temporary = PROGRESS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(PROGRESS_PATH)


def _result_or_none(accumulator: object) -> dict[str, float | int | str] | None:
    if accumulator.selected_rows == 0:
        return None
    return accumulator.result()


def main() -> plt.Figure:
    """Run one fixed output transform and stream its official score comparison."""
    if RESULT_PATH.exists():
        raise FileExistsError(RESULT_PATH)
    split_bytes = SPLIT_PATH.read_bytes()
    grade_bytes = GRADE_PATH.read_bytes()
    export_bytes = EXPORT_PATH.read_bytes()
    split = json.loads(split_bytes)
    grade = json.loads(grade_bytes)
    export = json.loads(export_bytes)
    groups = sorted(int(group) for group in split["evaluation"]["valid"])
    if groups != grade["evaluation_groups"] or len(groups) != 468:
        raise ValueError("frozen fold0 group membership differs from the bound grade")
    if sha256_file(METRIC_PATH) != EXPECTED_METRIC_SHA256:
        raise ValueError("official metric source differs from the root grade freeze")
    if grade["hashes"]["metric"] != EXPECTED_METRIC_SHA256:
        raise ValueError("root grade does not bind the expected official metric")
    if grade["hashes"]["bd26_h230_hetero_s3_f0_prediction"] != EXPECTED_PREDICTION_SHA256:
        raise ValueError("root grade prediction hash differs from the requested H230 candidate")
    if (
        grade["result"]["candidate"]["wp"] != EXPECTED_BASELINE_WP
        or grade["result"]["candidate"]["selected_rows"] != EXPECTED_SELECTED_ROWS
    ):
        raise ValueError("root grade score or selected-row count differs from the frozen baseline")

    prediction_binding = export["prediction"]
    if (
        prediction_binding["path"] != "cache/pred_h230_hetero_s3_ep1_20261008_v1_valid.npy"
        or prediction_binding["sha256"] != EXPECTED_PREDICTION_SHA256
        or prediction_binding["sidecar_sha256"] != EXPECTED_SIDECAR_SHA256
    ):
        raise ValueError("H230 export receipt does not bind the requested prediction and sidecar")
    sidecar_bytes = PREDICTION_SIDECAR_PATH.read_bytes()
    if hashlib.sha256(sidecar_bytes).hexdigest() != EXPECTED_SIDECAR_SHA256:
        raise ValueError("H230 prediction sidecar hash mismatch")
    sidecar = json.loads(sidecar_bytes)
    if (
        sidecar["run"] != "bd26_h230_hetero_s3_f0"
        or sidecar["split"] != "valid"
        or sidecar["epoch"] != 1
        or sidecar["inference_precision"] != "fp16"
        or sidecar["prediction_sha256"] != EXPECTED_PREDICTION_SHA256
        or sidecar["row_groups"] != list(range(1873))
        or prediction_binding["row_group_order_sha256"]
        != "73e879bb572abbab3d4b1af9bb6ab179c54c8b2c2ee8485242bbb153fe5ef84e"
    ):
        raise ValueError("H230 prediction sidecar row order or identity mismatch")
    pred_stat = PREDICTION_PATH.stat()
    if pred_stat.st_size != prediction_binding["bytes"]:
        raise ValueError("H230 prediction file size differs from the sealed export receipt")
    prediction = np.load(PREDICTION_PATH, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (1873, SEQUENCE_LENGTH, 2) or prediction.dtype != np.float32:
        raise ValueError("H230 prediction mmap shape or stored dtype mismatch")

    sys.path.insert(0, str(MAIN_ROOT / "src"))
    from wnn.gmetric import METRIC_ID, GlobalAccumulator

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parquet = pq.ParquetFile(PARQUET_PATH)
    columns = ["i0_p0", "i0_p11", "t0", "t1", "is_scored"]
    if parquet.metadata.num_row_groups != 1873 or parquet.metadata.num_rows != 37_460_000:
        raise ValueError("validation Parquet row-group geometry differs from frozen source")
    if any(column not in parquet.schema_arrow.names for column in columns):
        raise ValueError("required output-hold diagnostic columns are missing")

    baseline_global = GlobalAccumulator()
    hold_global = GlobalAccumulator()
    baseline_event = GlobalAccumulator()
    hold_event = GlobalAccumulator()
    baseline_stale = GlobalAccumulator()
    hold_stale = GlobalAccumulator()
    selected_rows = 0
    event_rows = 0
    stale_rows = 0
    event_weight = np.zeros(2, dtype=np.float64)
    stale_weight = np.zeros(2, dtype=np.float64)
    abs_delta_sum = np.zeros((3, 2), dtype=np.float64)
    changed_rows = np.zeros(3, dtype=np.int64)
    selected_by_partition = np.zeros(3, dtype=np.int64)
    group_records: list[dict[str, object]] = []
    output_stream_hash = hashlib.sha256()
    started = time.monotonic()
    input_stat = PARQUET_PATH.stat()

    for ordinal, group in enumerate(groups, start=1):
        if time.monotonic() - started >= MAX_SCAN_SECONDS:
            _write_progress(group_records, len(groups), started)
            raise TimeoutError(
                f"output-hold screen stopped at {len(group_records)}/{len(groups)} groups"
            )
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {group} has unexpected sequence length")
        p0 = table.column("i0_p0").combine_chunks().to_numpy(zero_copy_only=False)
        p11 = table.column("i0_p11").combine_chunks().to_numpy(zero_copy_only=False)
        t0 = table.column("t0").combine_chunks().to_numpy(zero_copy_only=False)
        t1 = table.column("t1").combine_chunks().to_numpy(zero_copy_only=False)
        scored = table.column("is_scored").combine_chunks().to_numpy(zero_copy_only=False)
        if scored[:WARMUP].any():
            raise ValueError(f"scored warmup row found in group {group}")
        if not (np.isfinite(t0).all() and np.isfinite(t1).all()):
            raise ValueError(f"nonfinite target in evaluation group {group}")

        # Exact ev_i0px event definition: simultaneous changes in i0_p0 and i0_p11.
        paired_event = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        paired_event[1:] = (p0[1:] != p0[:-1]) & (p11[1:] != p11[:-1])
        # Only this group-indexed mmap slice feeds the transform; targets are not inputs.
        baseline = prediction[group]
        if not np.isfinite(baseline).all():
            raise ValueError(f"nonfinite H230 baseline output in group {group}")
        held = event_hold(baseline, paired_event)
        if not np.array_equal(held[paired_event], baseline[paired_event]):
            raise AssertionError("event-row output must equal the baseline output")
        if not np.array_equal(held[0], baseline[0]):
            raise AssertionError("sequence-first causal output was not preserved")

        target = np.stack((t0, t1), axis=-1)
        mask = scored.astype(bool, copy=False) & (np.arange(SEQUENCE_LENGTH) >= WARMUP)
        event_mask = mask & paired_event
        stale_mask = mask & ~paired_event
        base_acc_added = baseline_global.add(target, baseline, mask)
        hold_acc_added = hold_global.add(target, held, mask)
        baseline_event.add(target, baseline, event_mask)
        hold_event.add(target, held, event_mask)
        baseline_stale.add(target, baseline, stale_mask)
        hold_stale.add(target, held, stale_mask)
        if base_acc_added.shape != (2, 6) or hold_acc_added.shape != (2, 6):
            raise AssertionError("official metric accumulator returned an unexpected shape")

        clipped_target = np.clip(target, -2.0, 2.0).astype(np.float64, copy=False)
        weights = np.abs(clipped_target)
        event_weight += weights[event_mask].sum(axis=0, dtype=np.float64)
        stale_weight += weights[stale_mask].sum(axis=0, dtype=np.float64)
        selected_rows += int(mask.sum())
        n_event = int(event_mask.sum())
        n_stale = int(stale_mask.sum())
        event_rows += n_event
        stale_rows += n_stale
        selected_by_partition += (int(mask.sum()), n_event, n_stale)

        changed = np.any(held != baseline, axis=-1)
        delta = np.abs(held.astype(np.float64) - baseline.astype(np.float64))
        for index, part in enumerate((mask, event_mask, stale_mask)):
            selected = int(part.sum())
            if selected:
                abs_delta_sum[index] += delta[part].sum(axis=0, dtype=np.float64)
                changed_rows[index] += int(np.count_nonzero(changed & part))

        baseline_hash = hashlib.sha256()
        baseline_hash.update(memoryview(baseline).cast("B"))
        held_hash = hashlib.sha256()
        held_hash.update(memoryview(held).cast("B"))
        output_stream_hash.update(struct.pack("<I", group))
        output_stream_hash.update(held_hash.digest())
        group_records.append(
            {
                "row_group": group,
                "rows": table.num_rows,
                "scored_rows": int(mask.sum()),
                "scored_event_rows": n_event,
                "scored_stale_rows": n_stale,
                "baseline_group_slice_sha256": baseline_hash.hexdigest(),
                "held_group_output_sha256": held_hash.hexdigest(),
                "event_rows_preserved_exactly": True,
                "first_output_preserved": True,
            }
        )
        del table, p0, p11, t0, t1, scored, paired_event, baseline, held
        del target, mask, event_mask, stale_mask, clipped_target, weights, changed, delta
        if ordinal % 25 == 0:
            _write_progress(group_records, len(groups), started)
            gc.collect()

    if len(group_records) != 468 or selected_rows != EXPECTED_SELECTED_ROWS:
        raise ValueError("completed stream does not match the frozen evaluation/score mask")
    baseline_score = baseline_global.result()
    hold_score = hold_global.result()
    if abs(float(baseline_score["wp"]) - EXPECTED_BASELINE_WP) > 1e-9:
        raise ValueError(
            "streamed baseline does not reproduce the bound root official score: "
            f"{baseline_score['wp']}"
        )
    if event_rows + stale_rows != selected_rows:
        raise AssertionError("event and stale scored partitions do not cover the metric mask")

    slice_scores = {
        "event": {
            "baseline": _result_or_none(baseline_event),
            "held": _result_or_none(hold_event),
            "selected_rows": event_rows,
            "clipped_abs_target_mass": event_weight.tolist(),
        },
        "stale": {
            "baseline": _result_or_none(baseline_stale),
            "held": _result_or_none(hold_stale),
            "selected_rows": stale_rows,
            "clipped_abs_target_mass": stale_weight.tolist(),
        },
    }
    changed_fraction = [
        float(changed_rows[i] / selected_by_partition[i]) if selected_by_partition[i] else None
        for i in range(3)
    ]
    mean_abs_delta = [
        (abs_delta_sum[i] / selected_by_partition[i]).tolist() if selected_by_partition[i] else None
        for i in range(3)
    ]
    payload: dict[str, object] = {
        "id": "E535_H230_OUTPUT_HOLD_RESULT_V1_20261008",
        "status": "complete_single_predeclared_output_transform_exposed_data",
        "observed_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "claim_scope": "One deterministic output-cache transform on an exposed fold0 model and exposed 468-group panel; no public forecast.",
        "transform": {
            "definition": "At each row, emit the baseline prediction at the most recent row where both i0_p0 and i0_p11 changed; initialize each sequence with its baseline row-0 output. Refresh at event rows. No target enters the transform.",
            "event_definition": "For t>0: (i0_p0[t] != i0_p0[t-1]) AND (i0_p11[t] != i0_p11[t-1]); row 0 has no preceding event.",
            "hidden_model_state_changed": False,
            "parameters_searched": 0,
            "future_features_or_targets_used": False,
            "sequence_reset_groups": len(groups),
            "warmup_rows_transformed_causally_from_row0": True,
        },
        "baseline_identity": {
            "run": sidecar["run"],
            "epoch": sidecar["epoch"],
            "prediction_path": str(PREDICTION_PATH.relative_to(MAIN_ROOT)),
            "prediction_sha256_bound_by_root_grade_and_export": EXPECTED_PREDICTION_SHA256,
            "prediction_sidecar_path": str(PREDICTION_SIDECAR_PATH.relative_to(MAIN_ROOT)),
            "prediction_sidecar_sha256": EXPECTED_SIDECAR_SHA256,
            "prediction_storage_dtype": str(prediction.dtype),
            "sidecar_inference_precision": sidecar["inference_precision"],
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "root_grade_path": str(GRADE_PATH.relative_to(MAIN_ROOT)),
            "root_grade_sha256": hashlib.sha256(grade_bytes).hexdigest(),
            "root_grade_source_sha256": grade["source_sha256"],
            "root_baseline_wp": EXPECTED_BASELINE_WP,
            "root_selected_rows": EXPECTED_SELECTED_ROWS,
            "row_order": "Canonical valid row-group order 0..1872; only each of the 468 frozen group slices was accessed.",
            "row_group_order_sha256": prediction_binding["row_group_order_sha256"],
        },
        "metric": {
            "id": METRIC_ID,
            "source_path": str(METRIC_PATH.relative_to(MAIN_ROOT)),
            "source_sha256": sha256_file(METRIC_PATH),
            "clip": 2.0,
            "selected_mask": "is_scored AND row_index>=99; unchanged from global_wp_clip2_v1.",
            "subset_slices_are_descriptive_only": True,
        },
        "membership": {
            "split_path": str(SPLIT_PATH.relative_to(MAIN_ROOT)),
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "groups": len(groups),
            "sorted_groups_sha256": hashlib.sha256(
                json.dumps(groups, separators=(",", ":")).encode()
            ).hexdigest(),
            "groups_sha256_in_root_grade_source": grade["hashes"]["split"],
            "evaluation_groups_match_root_grade": True,
        },
        "data": {
            "path": str(PARQUET_PATH),
            "bytes": input_stat.st_size,
            "mtime_ns": input_stat.st_mtime_ns,
            "whole_file_sha256": None,
            "whole_file_hash_note": "No whole-Parquet hash or full-table read was performed; only the five projected columns in selected groups were read.",
            "row_groups_read": len(group_records),
            "selected_columns": ["i0_p0", "i0_p11", "t0", "t1", "is_scored"],
            "read_mode": "PyArrow ParquetFile.read_row_group(use_threads=False), one complete selected group at a time.",
        },
        "scores": {
            "baseline": baseline_score,
            "held": hold_score,
            "delta_held_minus_baseline": {
                "t0": float(hold_score["t0"]) - float(baseline_score["t0"]),
                "t1": float(hold_score["t1"]) - float(baseline_score["t1"]),
                "wp": float(hold_score["wp"]) - float(baseline_score["wp"]),
            },
            "event_and_stale_subsets": slice_scores,
            "selected_rows": selected_rows,
            "event_rows": event_rows,
            "stale_rows": stale_rows,
            "changed_prediction_row_fraction": {
                "all_scored": changed_fraction[0],
                "event_scored": changed_fraction[1],
                "stale_scored": changed_fraction[2],
            },
            "mean_absolute_prediction_delta": {
                "all_scored_by_target": mean_abs_delta[0],
                "event_scored_by_target": mean_abs_delta[1],
                "stale_scored_by_target": mean_abs_delta[2],
            },
        },
        "output_hashing": {
            "hash_algorithm": "sha256",
            "hash_order": "ascending frozen row-group ID; each transformed group is C-order float32 bytes",
            "aggregate_group_hash_sha256": output_stream_hash.hexdigest(),
            "records": group_records,
            "full_prediction_array_materialized": False,
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "declared_cpu_quota": "100% / 1 CPU",
            "declared_memory_high_bytes": 800_000_000,
            "declared_memory_max_bytes": 1_000_000_000,
            "declared_wall_limit_seconds": 480,
            "script_internal_wall_limit_seconds": MAX_SCAN_SECONDS,
            "arrow_threads": 1,
            "full_npz_decompressed": False,
            "full_prediction_copy_created": False,
            "fit_or_gpu_work": False,
        },
        "known_prior_evidence": {
            "artifact": "research/campaign_2026-10-06/ONLINE.md:31-35 and research/bandit_tree/generated/INDEX.md:176",
            "scope": "Earlier output holds on the incumbent family were reported screen_failed; this exact H230 endpoint and predeclared paired-p0/p11 transform are a separate one-off exposed-data diagnostic.",
            "a_negative_does_not_reject_state_memory_or_architecture_changes": True,
        },
        "interpretation_limit": "This is a deterministic ablation of one already-scored H230 candidate on previously exposed fold0 groups. It is not a fresh forecast, causal proof, model-state intervention, training comparison, or general rejection of output/state-memory methods.",
    }

    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _write_progress(group_records, len(groups), started, status="complete")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {"baseline": "#4C78A8", "held": "#F58518"}
    for target_index, target_name in enumerate(("t0", "t1")):
        axes[0].bar(
            target_index - 0.18,
            baseline_score[target_name],
            width=0.36,
            color=colors["baseline"],
            label="baseline" if target_index == 0 else None,
        )
        axes[0].bar(
            target_index + 0.18,
            hold_score[target_name],
            width=0.36,
            color=colors["held"],
            label="held" if target_index == 0 else None,
        )
    axes[0].set(
        title="Official global WP",
        xticks=[0, 1],
        xticklabels=["t0", "t1"],
        ylabel="global weighted Pearson",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    labels = ["event t0", "event t1", "stale t0", "stale t1"]
    base_values = [
        slice_scores["event"]["baseline"]["t0"],
        slice_scores["event"]["baseline"]["t1"],
        slice_scores["stale"]["baseline"]["t0"],
        slice_scores["stale"]["baseline"]["t1"],
    ]
    hold_values = [
        slice_scores["event"]["held"]["t0"],
        slice_scores["event"]["held"]["t1"],
        slice_scores["stale"]["held"]["t0"],
        slice_scores["stale"]["held"]["t1"],
    ]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, base_values, width=0.36, color=colors["baseline"])
    axes[1].bar(x + 0.18, hold_values, width=0.36, color=colors["held"])
    axes[1].set(
        title="Descriptive subset WP",
        xticks=x,
        xticklabels=labels,
        ylabel="subset weighted Pearson",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].tick_params(axis="x", labelrotation=20)
    exposure = np.asarray([event_rows, stale_rows], dtype=np.float64) / selected_rows
    changed = np.asarray([changed_fraction[1], changed_fraction[2]], dtype=np.float64)
    x = np.arange(2)
    axes[2].bar(x - 0.18, exposure, width=0.36, label="scored-row share", color="#72B7B2")
    axes[2].bar(x + 0.18, changed, width=0.36, label="rows whose output changes", color="#E45756")
    axes[2].set(
        title="Rows by event status",
        xticks=x,
        xticklabels=["event", "stale"],
        ylabel="fraction of selected rows",
        ylim=(0, 1),
    )
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "H230 seed3 epoch1 output hold at paired i0_p0/i0_p11 events\n"
        "float32 stored emissions; sidecar records fp16 inference — exposed fold0 diagnostic"
    )
    fig.savefig(PNG_PATH, dpi=180)
    fig.savefig(SVG_PATH)
    print(
        json.dumps(
            {
                "id": payload["id"],
                "baseline": baseline_score,
                "held": hold_score,
                "delta": payload["scores"]["delta_held_minus_baseline"],
                "selected_rows": selected_rows,
                "event_rows": event_rows,
                "stale_rows": stale_rows,
                "elapsed_seconds": payload["execution"]["elapsed_seconds"],
                "aggregate_group_hash_sha256": output_stream_hash.hexdigest(),
            },
            separators=(",", ":"),
        )
    )
    return fig


if __name__ == "__main__":
    FIGURE = main()
