from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from clock_probe import DATA, features, read_inputs, save_json, scores

HERE = Path(__file__).resolve().parent


def direct_wp(pred: np.ndarray, target: np.ndarray) -> float:
    """Check moment-based scoring against the centered weighted calculation."""
    pred = np.clip(pred.astype(np.float64), -2, 2)
    target = np.clip(target.astype(np.float64), -2, 2)
    weights = np.abs(target)
    p = pred - np.average(pred, weights=weights)
    y = target - np.average(target, weights=weights)
    return float(
        np.sum(weights * p * y)
        / np.sqrt(np.sum(weights * p * p) * np.sum(weights * y * y))
    )


def main() -> None:
    """Inspect existing moments; verify one sequence without fitting new models."""
    manifest = json.loads((HERE / "clock_probe_manifest.json").read_text())
    saved = np.load(HERE / "clock_probe_stats.npz")
    stats, activity = saved["stats"], saved["activity"]
    quantum = np.array([s["prior_target_spacing"] for s in manifest["valid"]])
    diagnostic = {
        "status": "post-score exploratory diagnostics; not a new frozen test",
        "by_prior_target_spacing": {},
    }
    for value in np.unique(quantum):
        mask = quantum == value
        pooled = scores(stats[mask].sum(0))
        diagnostic["by_prior_target_spacing"][str(value)] = {
            "count": int(mask.sum()),
            "half_counts": [int(mask[:32].sum()), int(mask[32:].sum())],
            "mean_wp": dict(zip(saved["configs"].tolist(), pooled.mean(1).tolist())),
            "own_minus_shared": float(pooled[1].mean() - pooled[0].mean()),
        }
    diagnostic["mean_group_rates"] = dict(
        zip(saved["groups"].tolist(), activity[:, 0].mean(0).tolist())
    )
    diagnostic["mean_group_off_shared_rates"] = dict(
        zip(saved["groups"].tolist(), activity[:, 1].mean(0).tolist())
    )
    diagnostic["mean_abs_prefix_rate_error"] = dict(
        zip(
            saved["groups"].tolist(),
            np.abs(activity[:, 0] - activity[:, 2]).mean(0).tolist(),
        )
    )
    weights = np.load(HERE / "clock_probe_weights.npz")
    columns = weights["columns"].tolist()
    pf = pq.ParquetFile(DATA / "valid.parquet")
    sample = manifest["valid"][0]
    x = read_inputs(pf, sample["rg"], columns)
    table = pf.read_row_group(
        sample["rg"], columns=["seq_ix", "t0", "t1", "is_scored"], use_threads=False
    )
    actual_id = int(table["seq_ix"][0].as_py())
    if actual_id != sample["seq_ix"]:
        raise AssertionError("Manifest sequence ID differs from parquet")
    y = np.column_stack([table[t].to_numpy() for t in ("t0", "t1")])
    mask = table["is_scored"].to_numpy().astype(bool) & (np.arange(len(x)) >= 99)
    errors = []
    for k, config in enumerate(saved["configs"]):
        a = features(x, columns, str(config))[mask].astype(np.float64)
        prediction = a @ weights["coefficients"][k]
        direct = np.array([direct_wp(prediction[:, j], y[mask, j]) for j in range(2)])
        errors.append(float(np.max(np.abs(direct - scores(stats[0, k])))))
    if max(errors) > 1e-10:
        raise AssertionError(f"Direct score check failed: {errors}")
    diagnostic["direct_score_check"] = {
        "rg": sample["rg"],
        "seq_ix": actual_id,
        "n_scored": int(mask.sum()),
        "max_abs_error": max(errors),
    }
    diagnostic["executed_script_hash_matches_manifest"] = (
        hashlib.sha256((HERE / "clock_probe_v1_executed.py").read_bytes()).hexdigest()
        == manifest["script_sha256"]
    )
    if not diagnostic["executed_script_hash_matches_manifest"]:
        raise AssertionError("Archived executed script hash differs from manifest")
    save_json("clock_probe_diagnostics.json", diagnostic)


if __name__ == "__main__":
    main()
