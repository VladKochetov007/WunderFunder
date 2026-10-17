"""Bounded raw-feature moment scan; full-sequence statistics are diagnostic only."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)
PREFIXES = (32, 100, 512, 2048, 8192, 20000)


def scan_file(parquet: pq.ParquetFile, path: Path, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Read one complete row group at a time and retain only tiny moment tables."""
    shape = (len(groups), len(PREFIXES), len(FEATURE_COLUMNS))
    moments = {name: np.empty(shape, dtype=np.float64) for name in ("mean", "var", "min", "max")}
    sequence_ids = []
    for index, group in enumerate(groups):
        table = parquet.read_row_group(
            int(group), columns=["seq_ix", *FEATURE_COLUMNS], use_threads=False
        )
        if table.num_rows != 20000:
            raise ValueError(f"incomplete sequence: {path}:{group}")
        sequence_ids.append(table.column("seq_ix")[0].as_py())
        values = np.column_stack([table.column(name).to_numpy() for name in FEATURE_COLUMNS])
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite input: {path}:{group}")
        for prefix_index, length in enumerate(PREFIXES):
            prefix = values[:length].astype(np.float64)
            moments["mean"][index, prefix_index] = prefix.mean(axis=0)
            moments["var"][index, prefix_index] = prefix.var(axis=0)
            moments["min"][index, prefix_index] = prefix.min(axis=0)
            moments["max"][index, prefix_index] = prefix.max(axis=0)
        if index % 32 == 0:
            LOGGER.info("%s group %d/%d", path.name, index + 1, len(groups))
    return {**moments, "groups": groups, "sequence_ids": np.asarray(sequence_ids)}


def fit_circles(
    train: dict[str, np.ndarray], valid: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    """Fit E[X²]=b E[X]+c on train and inspect fixed-circle residuals on valid."""
    rows = []
    for index, name in enumerate(FEATURE_COLUMNS):
        mean = train["mean"][:, -1, index]
        variance = train["var"][:, -1, index]
        design = np.column_stack([mean, np.ones_like(mean)])
        beta, intercept = np.linalg.lstsq(design, variance + mean**2, rcond=None)[0]
        vm = valid["mean"][:, -1, index]
        vv = valid["var"][:, -1, index]
        prediction = beta * vm + intercept - vm**2
        residual = vv - prediction
        denominator = max(float(np.mean((vv - vv.mean()) ** 2)), 1e-16)
        low = train["min"][:, -1, index]
        high = train["max"][:, -1, index]
        envelope = (high - mean) * (mean - low)
        active = envelope > 1e-12
        slack = envelope - variance
        radius2 = float(intercept + beta**2 / 4)
        rows.append(
            {
                "column": name,
                "center": float(beta / 2),
                "radius2": radius2,
                "valid_r2": float(1 - np.mean(residual**2) / denominator),
                "valid_rmse_variance": float(np.sqrt(np.mean(residual**2))),
                "train_mean_span": float(np.ptp(mean)),
                "train_variance_span": float(np.ptp(variance)),
                "two_support_fraction": float(np.mean(np.abs(slack[active]) < 1e-7))
                if active.any()
                else 0.0,
                "active_sequences": int(active.sum()),
                "median_relative_slack": float(np.median(slack[active] / envelope[active]))
                if active.any()
                else None,
            }
        )
    return sorted(rows, key=lambda row: float(row["valid_r2"]), reverse=True)


def main() -> None:
    """Save reproducible scan arrays and descriptive rankings without loading a full file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rng = np.random.default_rng(20261017)
    scans = {}
    sources = {}
    for split in ("train", "valid"):
        path = args.root / "datasets" / f"{split}.parquet"
        parquet = pq.ParquetFile(path)
        eligible = np.arange(parquet.num_row_groups)
        if split == "train":
            eligible = eligible[~np.isin(eligible, list(CORRUPT_TRAIN_ROW_GROUPS))]
        groups = np.sort(rng.choice(eligible, args.groups, replace=False))
        scans[split] = scan_file(parquet, path, groups)
        stat = path.stat()
        sources[split] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "row_groups": parquet.num_row_groups,
        }
        parquet.close()
        del parquet
    np.savez_compressed(
        args.output / "moments.npz",
        prefixes=PREFIXES,
        columns=FEATURE_COLUMNS,
        **{f"{split}_{key}": value for split, scan in scans.items() for key, value in scan.items()},
    )
    ranking = fit_circles(scans["train"], scans["valid"])
    (args.output / "scan.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "seed": 20261017,
                "prefixes": PREFIXES,
                "ddof": 0,
                "ranking": ranking,
                "wall_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Leading descriptive fits: %s", json.dumps(ranking[:8]))


if __name__ == "__main__":
    main()
