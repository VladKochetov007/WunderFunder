"""Feature-only causal memory forecast of the next four price-event average.

Only histories whose four successor events are already observed enter memory.
The evaluation outcome is future *features*, never a candidate input.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pyarrow.parquet as pq
from memory_falsifier import fit_projection, price_states, read_price
from study import HERE, ROOT


def forecast(states: np.ndarray, projection: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    z = states @ projection
    n = len(z)
    sums = np.cumsum(z, axis=0)
    # Value at j: average of event states j+1..j+4, measured from state j.
    value = (sums[4:] - sums[:-4]) / 4 - z[:-4]
    result = {name: np.zeros((n, 2), np.float64) for name in (
        "zero", "last", "position", "content", "reversion"
    )}
    for i in range(5, n):
        newest = i - 4
        idx = np.arange(max(0, newest - 31), newest + 1)
        val = value[idx]
        result["last"][i] = val[-1]
        age = newest - idx
        pos_w = np.exp(-age / 8.0)
        result["position"][i] = pos_w @ val / pos_w.sum()
        distance = np.square(z[idx] - z[i]).sum(axis=1)
        weight = np.exp(-(distance - distance.min()) / max(scale * scale, 1e-12))
        result["content"][i] = weight @ val / weight.sum()
        result["reversion"][i] = np.mean(z[idx] - z[i], axis=0)
    result["future_feature_average"] = value
    return result


def fit_coefficients(groups: list[int], projection: np.ndarray, scale: float) -> dict[str, float]:
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    num = {name: 0.0 for name in ("last", "position", "content", "reversion")}
    den = {name: 0.0 for name in num}
    for group in groups:
        p, _ = read_price(pf, group, with_class=False)
        result = forecast(price_states(p), projection, scale)
        outcome = result["future_feature_average"][5:]
        length = len(result["future_feature_average"])
        for name in num:
            feature = result[name][5:length]
            num[name] += float(np.sum(feature * outcome))
            den[name] += float(np.sum(feature * feature))
    return {name: num[name] / max(den[name], 1e-12) for name in num}


def main() -> None:
    start = time.monotonic()
    raw = (HERE / "selection.json").read_bytes()
    selection = json.loads(raw)["row_groups"]
    projection, _, scale = fit_projection(selection["fit_train"])
    coefficients = fit_coefficients(selection["fit_train"], projection, scale)
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    all_rows = []
    for group in selection["eval_valid"]:
        p, label_class = read_price(pf, group, with_class=True)
        result = forecast(price_states(p), projection, scale)
        length = len(result["future_feature_average"])
        all_rows.append({
            "class": label_class,
            "outcome": result["future_feature_average"][5:],
            **{name: result[name][5:length] for name in coefficients},
        })
    summaries = {}
    for label_class in [-1, 0, 1, 2, 3]:
        rows = all_rows if label_class < 0 else [r for r in all_rows if r["class"] == label_class]
        outcome = np.concatenate([r["outcome"] for r in rows])
        zero_error = np.square(outcome).sum()
        scores = {"zero": 1.0}
        for name, coefficient in coefficients.items():
            feature = np.concatenate([r[name] for r in rows])
            scores[name] = float(np.square(outcome - coefficient * feature).sum() / zero_error)
        summaries["all" if label_class < 0 else str(label_class)] = {
            "sequences": len(rows), "events": len(outcome), "mse_ratio_to_zero": scores
        }
    report = {
        "selection_sha256": hashlib.sha256(raw).hexdigest(),
        "target": "future mean of four i0 price-event states minus current state, in two train-fitted feature PCA coordinates",
        "target_labels_used_in_fit": False,
        "label_derived_class_used_only_for_report": True,
        "memory_length": 32,
        "eligible_history_requires_four_successor_events_already_observed": True,
        "feature_only_train_coefficients": coefficients,
        "result": summaries,
        "wall_seconds": time.monotonic() - start,
    }
    (HERE / "horizon4_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
