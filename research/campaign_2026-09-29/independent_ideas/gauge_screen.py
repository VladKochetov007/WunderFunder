"""Feature-only falsifier for whole-sequence price-code translation augmentation.

The sample is frozen by ``select`` before ``run`` reads book states. Target values,
scoring masks, and validation examples never enter the test statistic. Training
market classes only stratify the sample and define separate fit pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRICE_COLUMNS = [f"i0_p{i}" for i in range(22)]


def select() -> None:
    table = pq.read_table(ROOT / "cache/seqstats_train.parquet", columns=["row_group", "cls"])
    rg = table.column("row_group").to_numpy().astype(int)
    cls = table.column("cls").to_numpy().astype(int)
    rng = np.random.default_rng(20411)
    groups: dict[str, dict[str, list[int]]] = {}
    for market in range(4):
        pool = np.sort(rg[(cls == market) & (rg != 528)])
        chosen = rng.choice(pool, size=36, replace=False).astype(int)
        groups[str(market)] = {"fit": chosen[:24].tolist(), "eval": chosen[24:].tolist()}
    payload = {
        "seed": 20411,
        "source": "datasets/train.parquet",
        "class_source": "cache/seqstats_train.parquet train-label-derived cls for strata only",
        "feature_columns": PRICE_COLUMNS,
        "groups": groups,
        "state_rows_per_sequence": 128,
        "shift_scale": "0.5 * SD of train sequence means projected on top between-sequence PC",
        "predeclared_pass": {
            "shifted_mean_nn_distance_over_baseline_max": 1.25,
            "shifted_mean_nn_distance_over_random_max": 0.75,
            "gauge_event_energy_over_random_max": 0.5,
        },
    }
    path = HERE / "gauge_selection.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


def price_states(file: pq.ParquetFile, row_group: int, *, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    table = file.read_row_group(row_group, columns=PRICE_COLUMNS, use_threads=False)
    x = np.column_stack([table.column(name).to_numpy(zero_copy_only=False) for name in PRICE_COLUMNS])
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite price code, row group {row_group}")
    rng = np.random.default_rng(seed + row_group)
    keep = np.sort(rng.choice(len(x), size=n, replace=False))
    event = np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1
    selected_events = event[:: max(1, len(event) // n)][:n]
    dx = x[selected_events] - x[selected_events - 1]
    return np.asarray(x[keep], dtype=np.float64), np.asarray(dx, dtype=np.float64)


def class_result(file: pq.ParquetFile, fit: list[int], evaluation: list[int], seed: int, n: int) -> dict:
    fitted = [price_states(file, i, seed=seed, n=n) for i in fit]
    tested = [price_states(file, i, seed=seed, n=n) for i in evaluation]
    xfit = np.concatenate([pair[0] for pair in fitted])
    xeval = np.concatenate([pair[0] for pair in tested])
    deltas = np.concatenate([pair[1] for pair in fitted])
    center = xfit.mean(0)
    scale = np.maximum(xfit.std(0), 1e-5)
    f = (xfit - center) / scale
    e = (xeval - center) / scale
    d = deltas / scale
    mean = np.stack([((pair[0] - center) / scale).mean(0) for pair in fitted])
    centered = mean - mean.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    gauge = vt[0]
    if np.dot(gauge, np.ones_like(gauge)) < 0:
        gauge = -gauge
    rng = np.random.default_rng(seed + 90_000 + fit[0])
    random = rng.standard_normal(len(gauge))
    random -= random.dot(gauge) * gauge
    random /= np.linalg.norm(random)
    amplitude = 0.5 * np.std(centered @ gauge)
    tree = cKDTree(f)

    def distances(z: np.ndarray) -> dict[str, float]:
        dist, _ = tree.query(z, k=1, workers=1)
        return {
            "mean": float(dist.mean()),
            "median": float(np.median(dist)),
            "p90": float(np.quantile(dist, 0.9)),
            "zero_fraction": float(np.mean(dist < 1e-8)),
        }

    baseline = distances(e)
    coherent = [distances(e + sign * amplitude * gauge) for sign in (-1, 1)]
    arbitrary = [distances(e + sign * amplitude * random) for sign in (-1, 1)]
    coherent_mean = float(np.mean([item["mean"] for item in coherent]))
    arbitrary_mean = float(np.mean([item["mean"] for item in arbitrary]))
    delta_cov = d.T @ d / len(d)
    g_energy = float(gauge @ delta_cov @ gauge)
    r_energy = float(random @ delta_cov @ random)
    ratios = {
        "shifted_over_baseline": coherent_mean / max(baseline["mean"], 1e-12),
        "shifted_over_random": coherent_mean / max(arbitrary_mean, 1e-12),
        "gauge_event_energy_over_random": g_energy / max(r_energy, 1e-12),
    }
    return {
        "fit_sequences": len(fit),
        "eval_sequences": len(evaluation),
        "fit_states": len(f),
        "eval_states": len(e),
        "shift_amplitude_standardized_l2": float(amplitude),
        "between_pc1_variance_share": float(np.linalg.svd(centered, compute_uv=False)[0] ** 2 / np.sum(centered**2)),
        "baseline": baseline,
        "coherent": coherent,
        "random_orthogonal": arbitrary,
        "ratios": ratios,
    }


def run() -> None:
    selection_path = HERE / "gauge_selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    file = pq.ParquetFile(ROOT / selection["source"])
    results = {}
    for market, groups in selection["groups"].items():
        results[market] = class_result(
            file, groups["fit"], groups["eval"], selection["seed"], selection["state_rows_per_sequence"]
        )
    gate = selection["predeclared_pass"]
    limit_to_ratio = {
        "shifted_mean_nn_distance_over_baseline_max": "shifted_over_baseline",
        "shifted_mean_nn_distance_over_random_max": "shifted_over_random",
        "gauge_event_energy_over_random_max": "gauge_event_energy_over_random",
    }
    pass_by_class = {
        key: all(value["ratios"][limit_to_ratio[metric]] <= limit for metric, limit in gate.items())
        for key, value in results.items()
    }
    payload = {
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "controls": "same shift amplitude; random direction orthogonal to coherent direction; same heldout states",
        "classes": results,
        "predeclared_pass_by_class": pass_by_class,
        "predeclared_pass_all_classes": all(pass_by_class.values()),
    }
    path = HERE / "gauge_result.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)
    print(json.dumps({key: value["ratios"] for key, value in results.items()}, indent=2))
    print("PASS", all(pass_by_class.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "run"))
    args = parser.parse_args()
    {"select": select, "run": run}[args.stage]()
