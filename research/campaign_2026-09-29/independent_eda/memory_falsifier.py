"""Test causal content retrieval on the next i0 price event, with no target fitting.

At event i, a key is the current projected price state. A stored value is the
observed move *after* an earlier key; the current key enters memory only when
the next event arrives. The 32-entry content and position controls see the
same completed transitions. The last-only control sees the most recent one.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pyarrow.parquet as pq
from study import HERE, PRICE, ROOT, class_from_labels, columns_numpy


def read_price(pf: pq.ParquetFile, row_group: int, *, with_class: bool) -> tuple[np.ndarray, int]:
    names = PRICE + (["t0", "t1"] if with_class else [])
    table = pf.read_row_group(row_group, columns=names, use_threads=False)
    p = columns_numpy(table, PRICE).astype(np.float64, copy=False)
    label_class = class_from_labels(columns_numpy(table, ["t0", "t1"])) if with_class else -1
    return p, label_class


def price_states(p: np.ndarray) -> np.ndarray:
    event = np.any(np.diff(p, axis=0) != 0, axis=1)
    positions = np.r_[0, np.flatnonzero(event) + 1]
    positions = positions[positions >= 99]
    return p[positions]


def fit_projection(groups: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    diffs = []
    for group in groups:
        p, _ = read_price(pf, group, with_class=False)
        states = price_states(p)
        delta = np.diff(states, axis=0)
        loc = np.linspace(0, len(delta) - 1, min(256, len(delta)), dtype=np.int64)
        diffs.append(delta[loc])
    d = np.concatenate(diffs)
    center = d.mean(axis=0)
    _, singular, axes = np.linalg.svd(d - center, full_matrices=False)
    projection = axes[:2].T / np.maximum(singular[:2] / np.sqrt(len(d)), 1e-10)
    d2 = d @ projection
    scale = float(np.median(np.linalg.norm(d2, axis=1)))
    return projection, center, scale


def forecast(states: np.ndarray, projection: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    z = states @ projection
    delta = np.diff(z, axis=0)
    length = len(z)
    out = {name: np.zeros((length, 2), np.float64) for name in (
        "zero", "last", "position", "content", "reversion", "anti_last"
    )}
    for i in range(2, length):
        count = min(32, i)
        idx = np.arange(i - count, i)
        values = delta[idx]
        out["last"][i] = values[-1]
        out["anti_last"][i] = -values[-1]
        out["reversion"][i] = np.mean(z[idx] - z[i], axis=0)
        age = np.arange(count - 1, -1, -1)
        position_weight = np.exp(-age / 8.0)
        out["position"][i] = position_weight @ values / position_weight.sum()
        key_distance2 = np.square(z[idx] - z[i]).sum(axis=1)
        log_weight = -key_distance2 / max(scale * scale, 1e-12)
        content_weight = np.exp(log_weight - log_weight.max())
        out["content"][i] = content_weight @ values / content_weight.sum()
    return {"target_next_delta": delta, **out, "first_eligible": 2}


def fit_control_coefficients(groups: list[int], projection: np.ndarray, scale: float) -> dict[str, float]:
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    num = {name: 0.0 for name in ("position", "content", "reversion", "anti_last")}
    den = {name: 0.0 for name in num}
    for group in groups:
        p, _ = read_price(pf, group, with_class=False)
        prediction = forecast(price_states(p), projection, scale)
        target_length = len(prediction["target_next_delta"])
        actual = prediction["target_next_delta"][2:]
        for name in num:
            feature = prediction[name][2:target_length]
            num[name] += float(np.sum(feature * actual))
            den[name] += float(np.sum(feature * feature))
    return {name: num[name] / max(den[name], 1e-12) for name in num}


def summarize(rows: list[dict]) -> dict:
    result = {}
    for label_class in [-1, 0, 1, 2, 3]:
        subset = rows if label_class < 0 else [r for r in rows if r["class"] == label_class]
        target = np.concatenate([r["target"] for r in subset])
        zero_error = np.square(target).sum()
        scores = {}
        for name in ("zero", "last", "position", "content", "content_fit", "position_fit", "reversion_fit", "anti_last_fit"):
            predicted = np.concatenate([r[name] for r in subset])
            mse = float(np.mean(np.square(target - predicted)))
            scores[name] = {
                "mse": mse,
                "mse_ratio_to_zero": float(np.square(target - predicted).sum() / zero_error),
                "mae": float(np.mean(np.abs(target - predicted))),
            }
        result["all" if label_class < 0 else str(label_class)] = {
            "sequences": len(subset),
            "events": len(target),
            "scores": scores,
        }
    return result


def main() -> None:
    selection_path = HERE / "selection.json"
    selection_raw = selection_path.read_bytes()
    selection = json.loads(selection_raw)
    start = time.monotonic()
    projection, center, scale = fit_projection(selection["row_groups"]["fit_train"])
    coefficients = fit_control_coefficients(selection["row_groups"]["fit_train"], projection, scale)
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    rows = []
    for group in selection["row_groups"]["eval_valid"]:
        p, label_class = read_price(pf, group, with_class=True)
        result = forecast(price_states(p), projection, scale)
        start_index = result.pop("first_eligible")
        target_length = len(result["target_next_delta"])
        rows.append({
            "group": group,
            "class": label_class,
            "target": result["target_next_delta"][start_index:],
            **{name: result[name][start_index:target_length] for name in ("zero", "last", "position", "content")},
            **{f"{name}_fit": coefficients[name] * result[name][start_index:target_length]
               for name in ("position", "content", "reversion", "anti_last")},
        })
    report = {
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "fit_price_events_sampled_per_sequence": 256,
        "projection_rank": 2,
        "projection_fit": "train fit rowgroups, feature-only SVD of event deltas",
        "projection_center_norm": float(np.linalg.norm(center)),
        "content_scale": scale,
        "feature_only_train_fitted_scalar_coefficients": coefficients,
        "memory_length": 32,
        "position_half_life_events": 8 * np.log(2),
        "targets_used_in_fit": False,
        "class_labels_used_only_to_stratify_report": True,
        "result": summarize(rows),
        "wall_seconds": time.monotonic() - start,
    }
    path = HERE / "memory_results.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
