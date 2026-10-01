"""Exploratory post-replay check of historical E-079 outside/touch claims."""

from __future__ import annotations

import json
import resource
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from replay import (
    HERE,
    LENGTH,
    ROOT,
    class_and_ticks,
    fit_decoder,
    implied_spread,
    midpoint_mean,
    path,
    read_group,
)


def main() -> None:
    started = time.monotonic()
    manifest = json.loads((HERE / "preregistration.json").read_text())
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    spreads, moves, _ = fit_decoder(train, manifest["fit_train_groups"])
    totals: dict[str, Counter] = {name: Counter() for name in ("all", "outside", "spread1_touch", "quiet", "event")}
    records = []
    for group in manifest["evaluation_valid_groups"]:
        prices, y, scored = read_group(valid, group, labels=True)
        assert y is not None and scored is not None
        regime, ticks = class_and_ticks(y)
        if regime != 3:
            continue
        state = path(prices, spreads, moves)
        origin = np.searchsorted(state["events"], np.arange(LENGTH), side="right") - 1
        mean, known = midpoint_mean(state, "quote", 4)
        spread = state["spread"][origin]
        eligible = scored & known[origin]
        estimated_future = np.rint(mean[origin] - spread / 2)
        true_future = np.where(ticks[:, 0] >= 1, ticks[:, 0], -spread - ticks[:, 1])
        true_spread = implied_spread(ticks)
        target_mass = np.abs(np.clip(y, -2, 2)).sum(axis=1)
        outside = (ticks[:, 0] >= 1) | (ticks[:, 1] >= 1)
        touch = (spread == 1) & (((ticks[:, 0] == 0) & (ticks[:, 1] == -1)) |
                                  ((ticks[:, 0] == -1) & (ticks[:, 1] == 0)))
        event = np.zeros(LENGTH, dtype=bool)
        event[state["events"][state["quotes"]]] = True
        partitions = {"all": np.ones(LENGTH, dtype=bool), "outside": outside,
                      "spread1_touch": touch, "quiet": ~event, "event": event}
        item = {"valid_group": group}
        for name, partition in partitions.items():
            rows = eligible & partition
            selected = scored & partition
            result = {
                "full_scored_rows": int(selected.sum()),
                "eligible_rows": int(rows.sum()),
                "future_exact": int(np.sum(rows & (estimated_future == true_future))),
                "spread_exact": int(np.sum(rows & (spread == true_spread))),
                "full_target_mass": float(target_mass[selected].sum()),
                "eligible_target_mass": float(target_mass[rows].sum()),
                "future_failure_mass": float(target_mass[rows & (estimated_future != true_future)].sum()),
            }
            item[name] = result
            totals[name].update(result)
        records.append(item)
    destination = HERE / "e079_diagnostic.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps({
        "status": "post-replay exploratory diagnostic, not a frozen promotion test",
        "origin": "TRAIN label-derived spread and move maps; physical quote identity unknown",
        "records": records,
        "totals": {key: dict(value) for key, value in totals.items()},
        "resources": {"wall_seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"totals": {k: dict(v) for k, v in totals.items()},
                      "output": str(destination)}))


if __name__ == "__main__":
    main()
