"""Quiet-stretch counterfactuals on the frozen seed-4 blend.

No fit. Horizon 8 and age 32 are the recipe and the wave-1 cell, not tuned
here. One validation sequence is read at a time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HORIZON = 8
SINCE_SCALE = 32.0
SINCE_FLOOR = 0.25
QUIET_AGE = 32


def _next_change(y: np.ndarray) -> np.ndarray:
    """First row strictly after t where the target pair changes, else T."""
    length = y.shape[0]
    changed = np.zeros(length, dtype=bool)
    changed[1:] = np.any(y[1:] != y[:-1], axis=1)
    index = np.arange(length, dtype=np.int32)
    marker = np.where(changed, index, np.int32(length))
    at_or_after = np.minimum.accumulate(marker[::-1])[::-1]
    nxt = np.empty(length, dtype=np.int32)
    nxt[:-1] = at_or_after[1:]
    nxt[-1] = length
    return nxt


def _and_age(p0: np.ndarray, p11: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Age since the last joint i0 touch, and the index of that touch."""
    length = p0.shape[0]
    event = np.zeros(length, dtype=bool)
    event[1:] = (p0[1:] != p0[:-1]) & (p11[1:] != p11[:-1])
    index = np.arange(length, dtype=np.int32)
    last = np.maximum.accumulate(np.where(event, index, np.int32(0)))
    return index - last, last


def _add(acc: GlobalAccumulator, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> None:
    if np.any(mask):
        acc.add(y, p, mask)


def _wp(acc: GlobalAccumulator) -> dict[str, float] | None:
    if acc.selected_rows == 0:
        return None
    result = acc.result()
    return {"t0": float(result["t0"]), "t1": float(result["t1"]), "wp": float(result["wp"]), "rows": int(result["selected_rows"])}


def main() -> None:
    started = time.time()
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as blob:
        y_all = np.asarray(blob["y"])
        mask_all = np.asarray(blob["mask"], dtype=bool)
        row_group = np.asarray(blob["row_group"])
        seq_ix = np.asarray(blob["seq_ix"])
    parent = np.load(cache / "pred_c20261004_combo_s4_final_fp32_valid.npy", mmap_mode="r")
    head = np.load(cache / "pred_c20261004_combo_head_s4_final_fp32_valid.npy", mmap_mode="r")
    if y_all.shape != parent.shape or parent.shape != head.shape:
        raise SystemExit(f"shape mismatch y={y_all.shape} parent={parent.shape} head={head.shape}")
    if y_all.nbytes > 2_000_000_000:
        raise SystemExit(f"targets too large to load under the 10 GB cap: {y_all.nbytes}")

    if y_all.ndim != 3 or y_all.shape[1:] != (20000, 2):
        raise SystemExit(f"expected one 20000-row sequence per row group, got {y_all.shape}")
    valid_pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    if valid_pf.num_row_groups != y_all.shape[0] or row_group.shape[0] != y_all.shape[0]:
        raise SystemExit("row-group count does not match the target cache")

    names = (
        "blend",
        "quiet_blend",
        "quiet_held",
        "quiet_survives_blend",
        "quiet_ends_blend",
        "long_event_blend",
        "global_hold_quiet",
        "global_hard_hold",
    )
    full = {name: GlobalAccumulator() for name in names}
    dev = {name: GlobalAccumulator() for name in names}
    weight = {
        key: np.zeros(2, dtype=np.float64)
        for key in (
            "scored",
            "quiet",
            "quiet_survives",
            "quiet_ends",
            "survives",
            "since_floor",
            "train_reweight_mass",
        )
    }

    for group in range(y_all.shape[0]):
        parquet_group = int(row_group[group])
        table = valid_pf.read_row_group(parquet_group, columns=["i0_p0", "i0_p11"], use_threads=False)
        p0 = table.column("i0_p0").to_numpy()
        p11 = table.column("i0_p11").to_numpy()
        if p0.shape[0] != y_all.shape[1]:
            raise SystemExit(f"row group {parquet_group} length does not match the cache")
        y = y_all[group]
        mask = mask_all[group]
        pred = (0.5 * (parent[group].astype(np.float32) + head[group].astype(np.float32))).astype(np.float32)
        age, last = _and_age(p0, p11)
        nxt = _next_change(y)
        index = np.arange(y.shape[0], dtype=np.int32)
        quiet = age >= QUIET_AGE
        survives = (nxt > (index + HORIZON)) & ((index + HORIZON) < y.shape[0])
        held = pred[last]
        long_event = np.zeros(y.shape[0], dtype=bool)
        long_event[np.unique(last[quiet])] = True
        clipped = np.clip(y, -2.0, 2.0)
        row_weight = np.abs(clipped)
        since_weight = np.maximum(SINCE_FLOOR, 1.0 / (1.0 + age.astype(np.float64) / SINCE_SCALE))
        scored = mask
        weight["scored"] += row_weight[scored].sum(axis=0)
        weight["quiet"] += row_weight[scored & quiet].sum(axis=0)
        weight["quiet_survives"] += row_weight[scored & quiet & survives].sum(axis=0)
        weight["quiet_ends"] += row_weight[scored & quiet & ~survives].sum(axis=0)
        weight["survives"] += row_weight[scored & survives].sum(axis=0)
        weight["since_floor"] += row_weight[scored & (since_weight <= SINCE_FLOOR + 1e-12)].sum(axis=0)
        weight["train_reweight_mass"] += (row_weight[scored] * since_weight[scored][:, None]).sum(axis=0)

        bags = [full]
        if not is_holdout(int(seq_ix[group])):
            bags.append(dev)
        quiet_replaced = pred.copy()
        quiet_replaced[quiet] = held[quiet]
        for bag in bags:
            _add(bag["blend"], y, pred, scored)
            _add(bag["quiet_blend"], y, pred, scored & quiet)
            _add(bag["quiet_held"], y, held, scored & quiet)
            _add(bag["quiet_survives_blend"], y, pred, scored & quiet & survives)
            _add(bag["quiet_ends_blend"], y, pred, scored & quiet & ~survives)
            _add(bag["long_event_blend"], y, pred, scored & long_event)
            _add(bag["global_hold_quiet"], y, quiet_replaced, scored)
            _add(bag["global_hard_hold"], y, held, scored)

    scored_w = weight["scored"]
    shares = {
        key: (weight[key] / scored_w).tolist()
        for key in weight
        if key != "scored"
    }
    payload = {
        "metric": "global_wp_clip2_v1",
        "horizon": HORIZON,
        "quiet_age": QUIET_AGE,
        "since_scale": SINCE_SCALE,
        "since_floor": SINCE_FLOOR,
        "seconds": round(time.time() - started, 3),
        "weight_shares_t0_t1": shares,
        "full_valid": {name: _wp(acc) for name, acc in full.items()},
        "non_holdout": {name: _wp(acc) for name, acc in dev.items()},
    }
    path = OUT / "wave1_stretch.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    full_scores = payload["full_valid"]
    quiet_delta = full_scores["quiet_held"]["wp"] - full_scores["quiet_blend"]["wp"]
    global_delta = full_scores["global_hold_quiet"]["wp"] - full_scores["blend"]["wp"]
    print(
        "STRETCH_DONE",
        f"blend={full_scores['blend']['wp']:.6f}",
        f"quiet_hold_delta={quiet_delta:.6f}",
        f"global_quiet_hold_delta={global_delta:.6f}",
        f"hard_hold={full_scores['global_hard_hold']['wp']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
