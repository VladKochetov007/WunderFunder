"""Screen whether anonymized feature means have stable train/valid geometry.

This is a cheap sequence-level screen for representation-learning ideas. It
does not estimate whether a denoising encoder improves the competition score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def principal_subspace_overlap(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    """Compare the top covariance subspaces after common feature scaling."""
    center_left = left - np.mean(left, axis=0)
    center_right = right - np.mean(right, axis=0)
    pooled_scale = np.sqrt((np.var(center_left, axis=0) + np.var(center_right, axis=0)) / 2)
    pooled_scale = np.maximum(pooled_scale, 1e-8)
    left_cov = np.cov((center_left / pooled_scale).T)
    right_cov = np.cov((center_right / pooled_scale).T)
    left_vectors = np.linalg.eigh(left_cov).eigenvectors[:, -rank:]
    right_vectors = np.linalg.eigh(right_cov).eigenvectors[:, -rank:]
    return float(np.sum((left_vectors.T @ right_vectors) ** 2) / rank)


def main() -> None:
    """Record class-conditional source shifts and covariance overlap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "artifacts" / "domain_geometry.json",
    )
    args = parser.parse_args()
    if not 1 <= args.rank <= len(FEATURE_COLUMNS):
        parser.error("--rank must be between 1 and the number of features")

    columns = [f"{name}_mean" for name in FEATURE_COLUMNS]
    arrays = {}
    for source in ("train", "valid"):
        table = pq.read_table(
            ROOT / "cache" / f"seqstats_{source}.parquet",
            columns=["cls", *columns],
        )
        cls = table.column("cls").to_numpy()
        matrix = np.column_stack([table.column(name).to_numpy() for name in columns])
        arrays[source] = (cls, matrix)

    results = {}
    rng = np.random.default_rng(41)
    for market_class in range(4):
        train_class, train_matrix = arrays["train"]
        valid_class, valid_matrix = arrays["valid"]
        train = train_matrix[train_class == market_class]
        valid = valid_matrix[valid_class == market_class]
        if not np.isfinite(train).all() or not np.isfinite(valid).all():
            raise ValueError(f"Nonfinite sequence feature means in class {market_class}")
        scale = np.sqrt((train.var(axis=0) + valid.var(axis=0)) / 2)
        shift = np.abs(train.mean(axis=0) - valid.mean(axis=0)) / np.maximum(scale, 1e-8)
        train_halves = np.array_split(rng.permutation(len(train)), 2)
        valid_halves = np.array_split(rng.permutation(len(valid)), 2)
        results[str(market_class)] = {
            "train_sequences": int(len(train)),
            "valid_sequences": int(len(valid)),
            "median_standardized_mean_shift": float(np.median(shift)),
            "p90_standardized_mean_shift": float(np.quantile(shift, 0.9)),
            "features_with_shift_over_1sd": int(np.sum(shift > 1)),
            "top_subspace_overlap": principal_subspace_overlap(train, valid, args.rank),
            "train_split_overlap": principal_subspace_overlap(
                train[train_halves[0]], train[train_halves[1]], args.rank
            ),
            "valid_split_overlap": principal_subspace_overlap(
                valid[valid_halves[0]], valid[valid_halves[1]], args.rank
            ),
        }
    output = {"feature_count": len(columns), "rank": args.rank, "classes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
