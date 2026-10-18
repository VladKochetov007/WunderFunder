"""Project only four raw feature columns from the first eight F0 groups."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SCRATCH = Path("/home/vlad/development/wunder/research_tmp/kalman-online-20261018")
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
DATA_PATH = ROOT / "datasets/valid.parquet"

sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(ROOT / "src"))

from kalman_features import RAW_COLUMNS, OnlineKalmanFeatures, identity_unscaled_prior  # noqa: E402

from wnn.splits import load_campaign_split  # noqa: E402


def main() -> None:
    """Validate split identity and causally filter eight feature-only groups."""
    started = time.perf_counter()
    cpu_started = time.process_time()
    split = load_campaign_split(SPLIT_PATH, ROOT)
    if split.fold != 0:
        raise ValueError("expected fold zero")
    split_raw = SPLIT_PATH.read_bytes()
    split_manifest = json.loads(split_raw)
    groups = tuple(map(int, split.evaluation_groups[:8]))
    if len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError("expected eight unique source-qualified F0 evaluation groups")
    source = split_manifest["sources"]["datasets/valid.parquet"]
    stat = DATA_PATH.stat()
    if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
        raise ValueError("validation parquet identity differs from frozen split")
    parquet = pq.ParquetFile(DATA_PATH)
    schema_names = parquet.schema_arrow.names
    raw_names = schema_names[4:116]
    if len(raw_names) != 112 or parquet.num_row_groups <= max(groups):
        raise ValueError("producer feature schema or row-group bounds differ")
    selected_names = [raw_names[index] for index in RAW_COLUMNS]
    if any(name in {"t0", "t1", "is_scored", "need_prediction"} for name in selected_names):
        raise ValueError("selected feature projection includes a forbidden field")
    sequence_by_group = {
        int(item["row_group"]): int(item["seq_ix"]) for item in split_manifest["sequences"]["valid"]
    }
    model = OnlineKalmanFeatures(identity_unscaled_prior())
    group_receipts: list[dict[str, int | bool]] = []
    first_raw: np.ndarray | None = None
    first_features: np.ndarray | None = None
    first_two_prefixes: list[np.ndarray] = []
    for index, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=selected_names, use_threads=False)
        values = np.column_stack([table.column(name).to_numpy() for name in selected_names])
        if values.shape != (20_000, 4) or not np.isfinite(values).all():
            raise ValueError(f"invalid feature-only projection for row group {group}")
        features = model.transform_selected_sequence(values)
        if not np.isfinite(features).all():
            raise FloatingPointError(f"nonfinite Kalman features in row group {group}")
        group_receipts.append(
            {
                "row_group": group,
                "sequence_id": sequence_by_group[group],
                "rows": len(values),
                "finite": True,
            }
        )
        if index < 2:
            first_two_prefixes.append(values[:1_000].copy())
        if index == 0:
            first_raw = values[:1_000].copy()
            first_features = features[:1_000].copy()
        del table, values, features

    if first_raw is None or first_features is None or len(first_two_prefixes) != 2:
        raise RuntimeError("first-group diagnostics were not retained")
    resetter = OnlineKalmanFeatures(identity_unscaled_prior())
    a = resetter.transform_selected_sequence(first_two_prefixes[0])
    resetter.transform_selected_sequence(first_two_prefixes[1])
    a_after_b = resetter.transform_selected_sequence(first_two_prefixes[0])
    if not np.array_equal(a, a_after_b):
        raise AssertionError("actual-group prefix A-B-A reset check failed")

    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    rows = np.arange(1_000)
    for channel, (column, label) in enumerate(zip(RAW_COLUMNS, selected_names, strict=True)):
        scalar_block = channel * 5
        trend_block = 20 + channel * 5
        raw = first_raw[:, channel]
        rw_level = raw + first_features[:, scalar_block]
        trend_level = raw + first_features[:, trend_block] - first_features[:, trend_block + 4]
        axes[channel, 0].plot(rows, raw, linewidth=0.7, alpha=0.7, label="raw")
        axes[channel, 0].plot(rows, rw_level, linewidth=0.9, label="RW filtered level")
        axes[channel, 0].plot(rows, trend_level, linewidth=0.9, label="local trend level")
        axes[channel, 0].set_title(f"raw column {column} ({label})")
        axes[channel, 1].plot(
            rows, first_features[:, scalar_block + 1], linewidth=0.7, label="RW prior innovation"
        )
        axes[channel, 1].plot(
            rows, first_features[:, trend_block + 1], linewidth=0.7, label="trend prior innovation"
        )
        axes[channel, 1].set_title("pre-update innovations")
        axes[channel, 0].legend(loc="upper right", fontsize=7)
        axes[channel, 1].legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("sequence row")
    axes[-1, 1].set_xlabel("sequence row")
    fig.suptitle("Causal fixed-Q/R filters; first 1,000 rows of F0 evaluation group 1")
    fig.tight_layout()
    plot_path = SCRATCH / "f0_first_group_raw_filtered_innovations.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    synthetic_receipt = json.loads((SCRATCH / "synthetic_receipt.json").read_text(encoding="utf-8"))
    synthetic_causal = synthetic_receipt["checks"]["future_perturbation_prefix_equal"]
    if not synthetic_causal:
        raise AssertionError("the fixed synthetic causality check did not pass")
    receipt = {
        "schema": "kalman_online_f0_input_projection.v1",
        "status": "bounded input-only engineering probe; no score or quality claim",
        "split_path": str(SPLIT_PATH),
        "split_sha256": hashlib.sha256(split_raw).hexdigest(),
        "valid_parquet": {
            "path": str(DATA_PATH),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "row_group_count": parquet.num_row_groups,
        },
        "groups": group_receipts,
        "selected_raw_columns": list(RAW_COLUMNS),
        "projected_column_names": selected_names,
        "projected_fields_only": selected_names,
        "labels_and_masks_read": False,
        "normalization_mode": "identity-unscaled-reference; no evaluation-derived fitting",
        "causal_reset_checks": {
            "synthetic_future_perturbation_prefix_equal": synthetic_causal,
            "actual_group_prefix_a_b_a_equal": True,
            "reset_scope": "each call processes a new independent sequence from a fresh filter state",
        },
        "all_projected_inputs_finite": True,
        "all_features_finite": True,
        "minimum_localtrend_covariance_eigenvalue": model.covariance_min_eigenvalue(),
        "plot_path": str(plot_path),
        "plot_sha256": hashlib.sha256(plot_path.read_bytes()).hexdigest(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "elapsed_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_imported": "torch" in sys.modules,
        "gpu_used": False,
    }
    receipt_path = SCRATCH / "f0_input_projection_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
