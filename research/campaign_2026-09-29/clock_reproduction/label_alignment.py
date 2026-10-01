"""Exploratory equal-row auxiliary label alignment on the frozen clock panel."""

from __future__ import annotations

import json
import resource
import time

import numpy as np
import pyarrow.parquet as pq
from replay import (
    HERE,
    LENGTH,
    Q3,
    ROOT,
    class_and_ticks,
    fit_decoder,
    midpoint_mean,
    path,
    read_group,
)


def weighted_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    total = weight.sum()
    xm = np.dot(weight, x) / total
    ym = np.dot(weight, y) / total
    cross = np.dot(weight, (x - xm) * (y - ym)) / total
    xx = np.dot(weight, (x - xm) ** 2) / total
    yy = np.dot(weight, (y - ym) ** 2) / total
    return float(cross / np.sqrt(xx * yy))


def main() -> None:
    started = time.monotonic()
    manifest = json.loads((HERE / "preregistration.json").read_text())
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    spreads, moves, _ = fit_decoder(train, manifest["fit_train_groups"])
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {name: [] for name in ("code", "quote", "contrast", "weight")}
        for horizon in (8, 24)
    }
    for group in manifest["evaluation_valid_groups"]:
        prices, y, scored = read_group(valid, group, labels=True)
        assert y is not None and scored is not None
        regime, _ = class_and_ticks(y)
        if regime != 3:
            continue
        state = path(prices, spreads, moves)
        origin = np.searchsorted(state["events"], np.arange(LENGTH), side="right") - 1
        for horizon in (8, 24):
            code_mean, code_known = midpoint_mean(state, "code", horizon)
            quote_mean, quote_known = midpoint_mean(state, "quote", horizon)
            selected = scored & code_known[origin] & quote_known[origin]
            vectors[horizon]["code"].append(np.clip(Q3 * code_mean[origin[selected]], -2, 2))
            vectors[horizon]["quote"].append(np.clip(Q3 * quote_mean[origin[selected]], -2, 2))
            clipped = np.clip(y[selected], -2, 2)
            vectors[horizon]["contrast"].append(0.5 * (clipped[:, 0] - clipped[:, 1]))
            vectors[horizon]["weight"].append(np.abs(clipped).sum(axis=1))
    output = {}
    for horizon, blocks in vectors.items():
        values = {key: np.concatenate(part) for key, part in blocks.items()}
        output[str(horizon)] = {
            "matched_scored_rows": int(len(values["weight"])),
            "matched_scored_target_mass": float(values["weight"].sum()),
            "code_to_target_contrast_weighted_corr": weighted_corr(values["code"], values["contrast"], values["weight"]),
            "quote_to_target_contrast_weighted_corr": weighted_corr(values["quote"], values["contrast"], values["weight"]),
            "code_to_target_contrast_pearson": float(np.corrcoef(values["code"], values["contrast"])[0, 1]),
            "quote_to_target_contrast_pearson": float(np.corrcoef(values["quote"], values["contrast"])[0, 1]),
        }
    destination = HERE / "label_alignment.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps({
        "status": "post-replay exploratory diagnostic; no GPU promotion uses this selection",
        "comparison": "same class3 scored rows, TRAIN-fitted decoder, target contrast=0.5*(clip t0 - clip t1), weight=sum |clip targets|",
        "horizons": output,
        "resources": {"wall_seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
