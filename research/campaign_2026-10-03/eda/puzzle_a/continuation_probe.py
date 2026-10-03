from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRICE_COLUMNS = [f"i0_p{i}" for i in range(22)]
LOG = logging.getLogger(__name__)


def feature_rows(prices: np.ndarray) -> np.ndarray:
    """Use only current inputs; labels and later rows are unavailable here."""
    return prices.astype(np.float64)


def weighted_corr(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute pooled clipped target-weighted Pearson correlation."""
    p = np.clip(prediction, -2, 2)
    y = np.clip(target, -2, 2)
    w = np.abs(y)
    p = p - np.average(p, weights=w)
    y = y - np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w * p * p) * np.sum(w * y * y))
    return float(np.sum(w * p * y) / denominator) if denominator else 0.0


def r_squared(prediction: np.ndarray, target: np.ndarray) -> float:
    """Evaluate error against the held-out target mean."""
    return float(1 - np.mean((target - prediction) ** 2) / np.var(target))


def ambiguity(prices: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Measure empirical conflicts in exact repeated current price states."""
    _, states = np.unique(prices, axis=0, return_inverse=True)
    n_states = int(states.max() + 1)
    counts = np.bincount(states, minlength=n_states)
    means = np.bincount(states, weights=target) / counts
    within_variance = float(np.mean((target - means[states]) ** 2))
    pairs, frequencies = np.unique(
        np.column_stack([states, np.round(target, 5)]),
        axis=0,
        return_counts=True,
    )
    best = np.zeros(n_states, dtype=int)
    np.maximum.at(best, pairs[:, 0].astype(int), frequencies)
    return {
        "states": n_states,
        "repeat_row_fraction": float(np.mean(counts[states] > 1)),
        "modal_disagreement": float(1 - best.sum() / len(target)),
        "within_state_variance": within_variance,
        "within_variance_fraction": within_variance / float(np.var(target)),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit fixed alpha ridge with an unpenalized intercept."""
    penalty = np.eye(x.shape[1]) * 10
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def main() -> None:
    """Execute the frozen bounded test and preserve measured provenance."""
    started = time.monotonic()
    frozen_bytes = (HERE / "HYPOTHESES.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "running_frozen_probe"
    state["commands"].append(
        "run_cpu.sh eda-puzzle-a .venv/bin/python research/campaign_2026-10-03/eda/puzzle_a/continuation_probe.py"
    )
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    records = []
    provenance = []
    training_x, training_y, control_y = [], [], []
    validation = []
    fingerprints: dict[str, list[str]] = {}
    prefix_checks = []
    for split in ["train", "valid"]:
        file_path = ROOT / "datasets" / f"{split}.parquet"
        parquet = pq.ParquetFile(file_path)
        stat = file_path.stat()
        provenance.append(
            {
                "path": str(file_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "row_groups": parquet.num_row_groups,
            }
        )
        columns = PRICE_COLUMNS + ["seq_ix", "step_in_seq", "t0", "t1"]
        if split == "valid":
            columns.append("is_scored")
        for group in frozen["cohorts"][f"{split}_row_groups"]:
            table = parquet.read_row_group(group, columns=columns)
            p = np.column_stack([table[c].to_numpy() for c in PRICE_COLUMNS])
            targets = np.column_stack([table[c].to_numpy() for c in ["t0", "t1"]]).astype(
                np.float64
            )
            seq_id = int(table["seq_ix"][0].as_py())
            steps = table["step_in_seq"].to_numpy()
            assert np.array_equal(steps, np.arange(len(steps)))
            assert np.all(table["seq_ix"].to_numpy() == seq_id)
            assert np.isfinite(p).all() and np.isfinite(targets).all()
            prefix_checks.append(bool(np.array_equal(feature_rows(p)[:256], feature_rows(p[:256]))))
            # A limited selected-cohort exact-window screen, not an overlap census.
            for start in range(0, len(p) - 63, 64):
                digest = hashlib.sha256(p[start : start + 64].tobytes()).hexdigest()
                fingerprints.setdefault(digest, []).append(f"{split}:{seq_id}:{start}")
            sample_hash = hashlib.sha256(p.tobytes() + targets.tobytes()).hexdigest()
            summed = targets.sum(axis=1)
            difference = targets[:, 0] - targets[:, 1]
            keep = steps >= 99
            mask = table["is_scored"].to_numpy() & keep if split == "valid" else keep
            nonzero = np.abs(targets[targets != 0])
            q = float(nonzero.min())
            record = {
                "split": split,
                "row_group": group,
                "seq_ix": seq_id,
                "prior_exposure": "scan1 target summaries; valid also scan2 full input summaries; detailed use not proven",
                "sample_sha256": sample_hash,
                "rows": len(p),
                "scored_rows": int(mask.sum()),
                "diagnostic_quantum": q,
                "integer_residual_max": float(np.max(np.abs(targets / q - np.round(targets / q)))),
                "sum_state": ambiguity(p[keep], summed[keep]),
                "difference_state": ambiguity(p[keep], difference[keep]),
                "shifted_sum_state": ambiguity(p[keep], np.roll(summed, 997)[keep]),
            }
            records.append(record)
            xy = np.column_stack([summed, difference])
            if split == "train":
                selected = np.flatnonzero(keep)[::10]
                training_x.append(feature_rows(p[selected]))
                training_y.append(xy[selected])
                control_y.append(np.roll(xy, 997, axis=0)[selected])
            else:
                validation.append((feature_rows(p[mask]), targets[mask], xy[mask], record))
            LOG.info("Completed %s group=%d id=%d", split, group, seq_id)
    x = np.concatenate(training_x)
    y = np.concatenate(training_y)
    shifted_y = np.concatenate(control_y)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    aligned_coef = fit_ridge(design, y)
    shifted_coef = fit_ridge(design, shifted_y)
    target_rows, predictions, control_predictions = [], [], []
    for x_valid, targets, xy, record in validation:
        xv = np.column_stack([np.ones(len(x_valid)), (x_valid - mean) / scale])
        prediction, control_prediction = xv @ aligned_coef, xv @ shifted_coef
        record["aligned_sum_r2"] = r_squared(prediction[:, 0], xy[:, 0])
        record["shifted_sum_r2"] = r_squared(control_prediction[:, 0], xy[:, 0])
        record["aligned_difference_r2"] = r_squared(prediction[:, 1], xy[:, 1])
        target_rows.append(targets)
        predictions.append(prediction)
        control_predictions.append(control_prediction)
    targets = np.concatenate(target_rows)
    pred = np.concatenate(predictions)
    control = np.concatenate(control_predictions)
    xy = np.column_stack([targets.sum(axis=1), targets[:, 0] - targets[:, 1]])
    metrics = {}
    for name, values in [("aligned", pred), ("shifted_control", control)]:
        metrics[name] = {
            "sum_r2": r_squared(values[:, 0], xy[:, 0]),
            "difference_r2": r_squared(values[:, 1], xy[:, 1]),
            "sum_only_wp": [weighted_corr(values[:, 0] / 2, targets[:, j]) for j in range(2)],
            "sum_difference_wp": [
                weighted_corr(
                    (values[:, 0] + (1 if j == 0 else -1) * values[:, 1]) / 2, targets[:, j]
                )
                for j in range(2)
            ],
        }
    cross_windows = [v for v in fingerprints.values() if len({s.rsplit(":", 1)[0] for s in v}) > 1]
    result = {
        "hypotheses_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.monotonic() - started,
        "provenance": provenance,
        "records": records,
        "metrics": metrics,
        "train_fit_rows": len(x),
        "valid_scored_rows": len(targets),
        "prefix_checks_pass": all(prefix_checks),
        "cross_sequence_equal_64row_price_windows": cross_windows[:30],
        "equal_window_group_count": len(cross_windows),
        "overlap_limit": "Only 22 i0 prices in aligned nonoverlapping windows; equality can arise from static prices and is not proof of shared source trajectory. No connected components established.",
    }
    (HERE / "continuation_results.json").write_text(json.dumps(result, indent=2))
    np.savez(
        HERE / "continuation_ridge.npz",
        mean=mean,
        scale=scale,
        aligned_coef=aligned_coef,
        shifted_coef=shifted_coef,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    valid_records = [r for r in records if r["split"] == "valid"]
    positions = np.arange(len(valid_records))
    for field, label in [
        ("sum_state", "S, current state"),
        ("difference_state", "D, current state"),
        ("shifted_sum_state", "S shifted 997"),
    ]:
        axes[0].plot(
            positions, [r[field]["modal_disagreement"] for r in valid_records], ".-", label=label
        )
    axes[0].set_ylabel("Fraction disagreeing with state mode")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        positions, [r["aligned_sum_r2"] for r in valid_records], ".-", label="Aligned S ridge"
    )
    axes[1].plot(
        positions, [r["shifted_sum_r2"] for r in valid_records], ".-", label="Shifted-label S ridge"
    )
    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_ylabel("R² on scored rows")
    axes[1].legend(fontsize=8)
    for j in range(2):
        axes[2].bar(
            j - 0.15,
            metrics["aligned"]["sum_only_wp"][j],
            width=0.3,
            label="Aligned" if j == 0 else None,
        )
        axes[2].bar(
            j + 0.15,
            metrics["shifted_control"]["sum_only_wp"][j],
            width=0.3,
            label="Shifted control" if j == 0 else None,
        )
    axes[2].set_xticks([0, 1], ["t0", "t1"])
    axes[2].set_ylabel("Pooled clipped WP; sum-only prediction")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("Fixed valid group index (900 + x)")
    fig.tight_layout()
    fig.savefig(HERE / "continuation_falsifier.png", dpi=140)
    plt.close(fig)
    state["phase"] = "probe_complete"
    state["inspected_sequence_ids"] = [
        {"split": r["split"], "row_group": r["row_group"], "seq_ix": r["seq_ix"]} for r in records
    ]
    state["artifacts"] += [
        "HYPOTHESES.json",
        "continuation_probe.py",
        "continuation_results.json",
        "continuation_ridge.npz",
        "continuation_falsifier.png",
    ]
    state["pending_work"] = ["inspect figure", "headless runnable notebook", "final report"]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    LOG.info("Metrics: %s", json.dumps(metrics))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
