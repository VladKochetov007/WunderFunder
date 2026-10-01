"""Independent, CPU-only event geometry audit on one chain-excluded fold.

The selected row groups are frozen from a feature-derived split before targets are
read. Label-derived market classes and regressions are diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRICE = [f"i0_p{i}" for i in range(22)]
VOL = [f"i0_v{i}" for i in range(22)]
OTHER_PRICE = [f"i1_p{i}" for i in range(22)]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])
AGE_BINS = [0, 1, 3, 10, 50, 20_001]


def class_from_labels(y: np.ndarray) -> int:
    values = np.unique(np.abs(y.astype(np.float64)))
    gaps = np.diff(values)
    gaps = gaps[gaps > 1e-6]
    if not len(gaps):
        return -1
    k = int(np.argmin(np.abs(QUANTA - gaps.min())))
    return k if abs(QUANTA[k] - gaps.min()) < 1e-3 else -1


def select_indices(split: dict, n_fit: int, n_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "fit_train": sorted(rng.choice(split["fit"]["train"], n_fit, replace=False).tolist()),
        "eval_valid": sorted(rng.choice(split["evaluation"]["valid"], n_eval, replace=False).tolist()),
    }


def columns_numpy(table: object, names: list[str]) -> np.ndarray:
    return np.column_stack([table.column(name).to_numpy() for name in names])


def last_event_age(event: np.ndarray) -> np.ndarray:
    index = np.arange(len(event), dtype=np.int32)
    last = np.maximum.accumulate(np.where(event, index, -20_001))
    return index - last


def load_stage(path: Path, groups: list[int], *, evaluation: bool) -> dict:
    pf = pq.ParquetFile(path)
    matrices: list[np.ndarray] = []
    jumps: list[np.ndarray] = []
    labels: list[int] = []
    event_totals = Counter()
    moment = np.zeros((4, 2, len(AGE_BINS) - 1, 5), np.float64)
    sample_counts = Counter()
    for group in groups:
        names = PRICE + VOL + OTHER_PRICE + ["t0", "t1"]
        if evaluation:
            names.append("is_scored")
        tab = pf.read_row_group(group, columns=names, use_threads=False)
        p = columns_numpy(tab, PRICE).astype(np.float32, copy=False)
        v = columns_numpy(tab, VOL).astype(np.float32, copy=False)
        p1 = columns_numpy(tab, OTHER_PRICE).astype(np.float32, copy=False)
        y = columns_numpy(tab, ["t0", "t1"]).astype(np.float32, copy=False)
        label_class = class_from_labels(y)
        sample_counts[str(label_class)] += 1
        dp = np.diff(p, axis=0, prepend=p[:1])
        dv = np.diff(v, axis=0, prepend=v[:1])
        dp1 = np.diff(p1, axis=0, prepend=p1[:1])
        dy = np.diff(y, axis=0, prepend=y[:1])
        event = np.any(dp != 0, axis=1)
        volume_event = np.any(dv != 0, axis=1)
        i1_event = np.any(dp1 != 0, axis=1)
        target_event = np.any(dy != 0, axis=1)
        event_totals.update({
            "rows": len(y),
            "i0_price": int(event.sum()),
            "i0_volume": int(volume_event.sum()),
            "i1_price": int(i1_event.sum()),
            "target": int(target_event.sum()),
            "target_on_i0_price": int(np.logical_and(target_event, event).sum()),
            "target_on_i1_only": int(np.logical_and(target_event, i1_event & ~event).sum()),
            "i0_price_without_target": int(np.logical_and(event, ~target_event).sum()),
        })
        # Equally spaced per sequence so a very active sequence cannot dominate.
        loc = np.flatnonzero(event & (np.arange(len(event)) >= 99))
        if len(loc) > 256:
            loc = loc[np.linspace(0, len(loc) - 1, 256, dtype=np.int64)]
        matrices.append(dp[loc].astype(np.float64))
        jumps.append(dy[loc].astype(np.float64))
        labels.extend([label_class] * len(loc))
        if evaluation:
            scored = tab.column("is_scored").to_numpy().astype(bool)
            scored[:99] = False
            age = last_event_age(event)
            age_bin = np.clip(np.searchsorted(AGE_BINS, age, side="right") - 1, 0, len(AGE_BINS) - 2)
            clipped_y = np.clip(y.astype(np.float64), -2, 2)
            for target in range(2):
                yy = clipped_y[:, target]
                ww = np.abs(yy)
                for b in range(len(AGE_BINS) - 1):
                    keep = scored & (age_bin == b)
                    z = yy[keep]
                    w = ww[keep]
                    if label_class >= 0:
                        moment[label_class, target, b] += [len(z), w.sum(), np.dot(w, z), np.dot(w, z * z), np.count_nonzero(z)]
    return {
        "event_deltas": np.concatenate(matrices),
        "target_jumps": np.concatenate(jumps),
        "event_classes": np.array(labels, dtype=np.int8),
        "counts": dict(event_totals),
        "class_sequence_counts": dict(sample_counts),
        "age_moments": moment,
    }


def r2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.square(a - a.mean()).sum()
    return float(1 - np.square(a - b).sum() / denominator) if denominator else float("nan")


def geometry(fit: dict, evaluation: dict) -> dict:
    x = fit["event_deltas"]
    xe = evaluation["event_deltas"]
    y = fit["target_jumps"]
    ye = evaluation["target_jumps"]
    mu = x.mean(axis=0)
    _, singular, v_t = np.linalg.svd(x - mu, full_matrices=False)
    variance_share = (singular * singular) / np.square(singular).sum()
    out: dict = {
        "fit_event_samples": len(x),
        "eval_event_samples": len(xe),
        "pooled_delta_p_svd_explained_cumulative": np.cumsum(variance_share)[[0, 1, 3, 7, 15, 21]].tolist(),
        "linear_event_jump_r2": {},
        "linear_event_jump_mae": {},
        "linear_sum_contrast_r2": {},
        "by_class_rank2": {},
        "label_jump_nonzero_rate_fit": np.mean(np.any(y != 0, axis=1)).item(),
        "label_jump_nonzero_rate_eval": np.mean(np.any(ye != 0, axis=1)).item(),
    }
    # Fixed ridge regularizer and ranks, fit only on training row groups.
    for rank in (0, 1, 2, 4, 8, 22):
        xfit = np.ones((len(x), 1)) if rank == 0 else np.column_stack([np.ones(len(x)), (x - mu) @ v_t[:rank].T])
        xval = np.ones((len(xe), 1)) if rank == 0 else np.column_stack([np.ones(len(xe)), (xe - mu) @ v_t[:rank].T])
        reg = np.eye(xfit.shape[1]) * 1e-3
        reg[0, 0] = 0
        beta = np.linalg.solve(xfit.T @ xfit + reg, xfit.T @ y)
        pred = xval @ beta
        out["linear_event_jump_r2"][str(rank)] = [r2(ye[:, k], pred[:, k]) for k in range(2)]
        out["linear_event_jump_mae"][str(rank)] = np.mean(np.abs(ye - pred), axis=0).tolist()
        summed = np.column_stack([y[:, 0] + y[:, 1], y[:, 0] - y[:, 1]])
        summed_eval = np.column_stack([ye[:, 0] + ye[:, 1], ye[:, 0] - ye[:, 1]])
        beta_sum = np.linalg.solve(xfit.T @ xfit + reg, xfit.T @ summed)
        pred_sum = xval @ beta_sum
        out["linear_sum_contrast_r2"][str(rank)] = [
            r2(summed_eval[:, k], pred_sum[:, k]) for k in range(2)
        ]
    for label_class in range(4):
        fi = fit["event_classes"] == label_class
        ei = evaluation["event_classes"] == label_class
        train = x[fi]
        valid = xe[ei]
        train_target = np.column_stack([y[fi, 0] + y[fi, 1], y[fi, 0] - y[fi, 1]])
        valid_target = np.column_stack([ye[ei, 0] + ye[ei, 1], ye[ei, 0] - ye[ei, 1]])
        center = train.mean(axis=0)
        _, s, axes = np.linalg.svd(train - center, full_matrices=False)
        xx = np.column_stack([np.ones(len(train)), (train - center) @ axes[:2].T])
        xxv = np.column_stack([np.ones(len(valid)), (valid - center) @ axes[:2].T])
        beta = np.linalg.solve(xx.T @ xx + np.diag([0, 1e-3, 1e-3]), xx.T @ train_target)
        prediction = xxv @ beta
        out["by_class_rank2"][str(label_class)] = {
            "train_events": int(fi.sum()),
            "eval_events": int(ei.sum()),
            "delta_p_rank2_variance_share": float(np.square(s[:2]).sum() / np.square(s).sum()),
            "target_jump_sum_contrast_r2": [r2(valid_target[:, k], prediction[:, k]) for k in range(2)],
            "fraction_target_jump_sum_zero": float(np.mean(np.abs(valid_target[:, 0]) < 1e-6)),
        }
    return out


def score_mass(stage: dict) -> dict:
    moments = stage["age_moments"]
    report = {}
    for label_class in [-1, 0, 1, 2, 3]:
        subset = moments.sum(axis=0) if label_class < 0 else moments[label_class]
        class_report = {}
        for t in range(2):
            part = subset[t]
            sums = part.sum(axis=0)
            center = sums[2] / sums[1]
            center_mass = part[:, 3] - 2 * center * part[:, 2] + center * center * part[:, 1]
            class_report[f"t{t}"] = {
                "global_weighted_mean": center,
                "age_bins": ["0", "1-2", "3-9", "10-49", "50+"],
                "row_share": (part[:, 0] / sums[0]).tolist(),
                "absolute_target_weight_share": (part[:, 1] / sums[1]).tolist(),
                "centered_weighted_target_variance_share": (center_mass / center_mass.sum()).tolist(),
                "nonzero_target_share": (part[:, 4] / sums[4]).tolist(),
                "selected_rows": int(sums[0]),
            }
        report["all" if label_class < 0 else f"class{label_class}"] = class_report
    mass = {}
    for t in range(2):
        all_parts = moments[:, t]
        totals = all_parts.sum(axis=(0, 1))
        center = totals[2] / totals[1]
        centered = (
            all_parts[:, :, 3]
            - 2 * center * all_parts[:, :, 2]
            + center * center * all_parts[:, :, 1]
        )
        mass[f"t{t}"] = {
            "selected_row_share_by_class": (all_parts[:, :, 0].sum(axis=1) / totals[0]).tolist(),
            "global_centered_target_variance_share_by_class": (
                centered.sum(axis=1) / centered.sum()
            ).tolist(),
            "global_centered_variance_share_class3_age10plus": float(
                centered[3, 3:].sum() / centered.sum()
            ),
        }
    report["global_mass_by_label_derived_class_diagnostic"] = mass
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fit", type=int, default=48)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_raw = split_path.read_bytes()
    split = json.loads(split_raw)
    picked = select_indices(split, args.n_fit, args.n_eval, args.seed)
    HERE.mkdir(parents=True, exist_ok=True)
    selection = {
        "split": str(split_path.relative_to(ROOT)),
        "split_sha256": hashlib.sha256(split_raw).hexdigest(),
        "seed": args.seed,
        "row_groups": picked,
        "selection_before_target_access": True,
    }
    selection_path = HERE / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("Frozen row-group selection differs from existing selection.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    started = time.monotonic()
    fit = load_stage(ROOT / "datasets/train.parquet", picked["fit_train"], evaluation=False)
    evaluation = load_stage(ROOT / "datasets/valid.parquet", picked["eval_valid"], evaluation=True)
    result = {
        "scope": "independent fold0 CPU exploratory sample; not an untouched confirmation set",
        "method": "pyarrow ParquetFile.read_row_group; 256 equally spaced price events per selected sequence for geometry",
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "fit_counts": fit["counts"],
        "eval_counts": evaluation["counts"],
        "fit_class_sequences_label_derived_diagnostic": fit["class_sequence_counts"],
        "eval_class_sequences_label_derived_diagnostic": evaluation["class_sequence_counts"],
        "price_event_geometry": geometry(fit, evaluation),
        "global_wp_target_weight_by_i0_price_event_age": score_mass(evaluation),
        "wall_seconds": time.monotonic() - started,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
