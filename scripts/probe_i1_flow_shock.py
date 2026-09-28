"""Screen causal volume-code shocks at class-3 quiet-interval midpoints."""

from __future__ import annotations

import argparse

import decide
import numpy as np
from sklearn.metrics import roc_auc_score

from wnn.data import SequenceFile


def observations(
    source: SequenceFile,
    row_group: int,
    index: int,
    anchor: np.ndarray,
    partner: np.ndarray,
    volume_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one causal snapshot per scored long i0 price-code interval."""
    sequence = source.read(row_group)
    x = sequence.features
    volume = x[:, volume_start : volume_start + 22]
    price_change = np.flatnonzero(np.any(x[1:, :22] != x[:-1, :22], axis=1)) + 1
    volume_change = np.flatnonzero(np.any(volume[1:] != volume[:-1], axis=1)) + 1
    starts = np.r_[0, price_change]
    ends = np.r_[price_change, len(x)]
    records = []
    labels = []
    for start, end in zip(starts, ends, strict=True):
        if end == len(x) or end - start < 64:
            continue
        midpoint = (start + end) // 2
        if midpoint < decide.WARMUP or not sequence.scored[midpoint]:
            continue
        direction = np.sign(sequence.targets[midpoint, 0] - sequence.targets[midpoint, 1])
        if direction == 0:
            continue
        left = np.searchsorted(volume_change, start, side="right")
        right = np.searchsorted(volume_change, midpoint, side="right")
        event_rows = volume_change[left:right]
        shocks = []
        for horizon in (4, 16):
            before = (
                int(event_rows[-min(horizon, len(event_rows))] - 1) if len(event_rows) else start
            )
            shocks.append(volume[midpoint] - volume[before])
        a = anchor[index, midpoint]
        b = partner[index, midpoint]
        pair_direction = 0.5 * (a[0] - a[1] + b[0] - b[1])
        records.append(np.r_[pair_direction, shocks[0], shocks[1]])
        labels.append(int(direction))
    return np.asarray(records, dtype=np.float64).reshape(-1, 45), np.asarray(labels, dtype=np.int8)


def fit_ridge(
    features: np.ndarray, labels: np.ndarray, full: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fixed 10% diagonal ridge after fold-local standardization."""
    used = features if full else features[:, :1]
    mean = used.mean(axis=0)
    scale = np.maximum(used.std(axis=0), 1e-5)
    design = np.column_stack((np.ones(len(used)), (used - mean) / scale))
    gram = design.T @ design
    ridge = 0.1 * np.diag(gram)
    ridge[0] = 0.0
    gram[np.diag_indices_from(gram)] += ridge + 1e-8 * max(gram.trace(), 1.0)
    coefficient = np.linalg.solve(gram, design.T @ labels)
    return mean, scale, coefficient


def predict(features: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply a training-fold-only normalization and ridge coefficients."""
    mean, scale, coefficient = fit
    used = features[:, : len(mean)]
    return np.column_stack((np.ones(len(used)), (used - mean) / scale)) @ coefficient


def main() -> None:
    """Evaluate incremental volume-code signal beyond the frozen mixed pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-fold", type=int, default=128)
    parser.add_argument("--volume-start", type=int, choices=(22, 74), default=74)
    args = parser.parse_args()

    data = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    row_groups = data["row_group"]
    market_class = data["market_class"]
    chain, fold = decide._chains()
    chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    train_chains = chain_data["chain"][chain_data["split"] == "train"]
    pure = ~np.isin(chain, train_chains)
    source = SequenceFile(decide.ROOT / "datasets" / "valid.parquet")
    values = {}
    for k in (0, 1):
        anchor = np.load(
            decide.ROOT / "cache" / f"pred_ab68_ema6_since_ep5_f{k}_valid.npy", mmap_mode="r"
        )
        partner = np.load(
            decide.ROOT / "cache" / f"pred_ab62_base_s4_f{k}_valid.npy", mmap_mode="r"
        )
        candidates = np.flatnonzero(pure & (fold == k) & (market_class == 3))[: args.per_fold]
        records = [
            observations(
                source, int(row_groups[index]), int(index), anchor, partner, args.volume_start
            )
            for index in candidates
        ]
        features = np.concatenate([item[0] for item in records])
        labels = np.concatenate([item[1] for item in records])
        values[k] = features, labels
        print(f"fold={k} sequences={len(candidates)} midpoints={len(labels)}", flush=True)

    for eval_fold in (0, 1):
        fit_features, fit_labels = values[1 - eval_fold]
        eval_features, eval_labels = values[eval_fold]
        base = predict(eval_features, fit_ridge(fit_features, fit_labels, full=False))
        full = predict(eval_features, fit_ridge(fit_features, fit_labels, full=True))
        base_auc = roc_auc_score(eval_labels > 0, base)
        full_auc = roc_auc_score(eval_labels > 0, full)
        partial = np.corrcoef(eval_labels - base, full - base)[0, 1]
        print(
            f"eval_fold={eval_fold} base_auc={base_auc:.6f} full_auc={full_auc:.6f} "
            f"auc_gain={full_auc - base_auc:+.6f} partial_corr={partial:+.6f}"
        )


if __name__ == "__main__":
    main()
