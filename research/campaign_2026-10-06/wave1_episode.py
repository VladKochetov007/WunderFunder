"""Matched row-loss vs episode-loss linear maps on far no-trade rows.

Alpha 1 is frozen. Fit is split bit 0 of non-holdout validation. Eval is bit 1.
This is a protocol comparison, not a public forecast.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq

from wave1_anatomy import (
    COLS,
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    _clock_and_features,
    _solve_ridge,
    _split_bit,
)
from wave1_online import _changed, _recent
from wave1_stretch import HORIZON, QUIET_AGE, ROOT, _add, _and_age, _next_change, _wp
from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

DECAY = float(np.exp(-1.0 / 4.0406))
CLIP = 2.0


def _far_notrade(cols: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    age, last = _and_age(cols["i0_p0"], cols["i0_p11"])
    nxt = _next_change(y)
    index = np.arange(y.shape[0], dtype=np.int32)
    quiet = age >= QUIET_AGE
    far = quiet & (nxt > (index + HORIZON)) & ((index + HORIZON) < y.shape[0])
    trade = _changed(cols["i0_dp0"]) | _changed(cols["i0_dv0"])
    return far & ~_recent(trade, 8), last


def _accumulate_row(
    xtx: np.ndarray, xty: np.ndarray, x: np.ndarray, y: np.ndarray
) -> None:
    clipped = np.clip(y, -CLIP, CLIP)
    weight = np.abs(clipped)
    design = np.column_stack([np.ones(x.shape[0]), x])
    for target in range(2):
        scale = np.sqrt(np.maximum(weight[:, target], 0.0))
        weighted = design * scale[:, None]
        xtx[target] += weighted.T @ weighted
        xty[target] += weighted.T @ (clipped[:, target] * scale)


def _accumulate_episode(
    xtx: np.ndarray, xty: np.ndarray, x: np.ndarray, y: np.ndarray, last: np.ndarray
) -> None:
    clipped = np.clip(y, -CLIP, CLIP)
    for sid in np.unique(last):
        take = last == sid
        mass = np.abs(clipped[take]).sum(axis=0)
        xbar = x[take].mean(axis=0)
        ybar = clipped[take].mean(axis=0)
        design = np.concatenate(([1.0], xbar))
        for target in range(2):
            if mass[target] <= 0:
                continue
            row = design * np.sqrt(mass[target])
            xtx[target] += np.outer(row, row)
            xty[target] += row * (ybar[target] * np.sqrt(mass[target]))


def _apply(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    return np.stack([design @ beta[0], design @ beta[1]], axis=1).astype(np.float32)


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
    valid_pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    dim = 1 + len(FEATURE_NAMES) + 2
    row_xtx = np.zeros((2, dim, dim))
    row_xty = np.zeros((2, dim))
    ep_xtx = np.zeros((2, dim, dim))
    ep_xty = np.zeros((2, dim))
    affine_xtx = np.zeros((2, 3, 3))
    affine_xty = np.zeros((2, 3))
    eval_pack: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for group in range(y_all.shape[0]):
        seq = int(seq_ix[group])
        if is_holdout(seq):
            continue
        table = valid_pf.read_row_group(int(row_group[group]), columns=list(COLS), use_threads=False)
        cols = {name: table.column(name).to_numpy() for name in COLS}
        y = y_all[group]
        mask = mask_all[group]
        pred = (0.5 * (parent[group].astype(np.float32) + head[group].astype(np.float32))).astype(
            np.float32
        )
        far, last = _far_notrade(cols, y)
        take = mask & far
        if not np.any(take):
            continue
        feats = _clock_and_features(cols, DECAY)["features"][take]
        x = np.column_stack([feats, pred[take]])
        y_take = y[take]
        p_take = pred[take]
        last_take = last[take]
        if _split_bit(seq, "wave1b") == 0:
            _accumulate_row(row_xtx, row_xty, x, y_take)
            _accumulate_episode(ep_xtx, ep_xty, x, y_take, last_take)
            _accumulate_row(affine_xtx, affine_xty, p_take, y_take)
        else:
            eval_pack.append((y_take, p_take, x, np.ones(x.shape[0], dtype=bool)))

    row_beta = np.stack([_solve_ridge(row_xtx[i], row_xty[i], PRIMARY_ALPHA) for i in range(2)])
    ep_beta = np.stack([_solve_ridge(ep_xtx[i], ep_xty[i], PRIMARY_ALPHA) for i in range(2)])
    aff_beta = np.stack([_solve_ridge(affine_xtx[i], affine_xty[i], PRIMARY_ALPHA) for i in range(2)])
    acc = {name: GlobalAccumulator() for name in ("blend", "affine", "row", "episode")}
    for y_take, p_take, x, mask in eval_pack:
        _add(acc["blend"], y_take, p_take, mask)
        _add(acc["affine"], y_take, _apply(aff_beta, p_take), mask)
        _add(acc["row"], y_take, _apply(row_beta, x), mask)
        _add(acc["episode"], y_take, _apply(ep_beta, x), mask)

    scores = {name: _wp(acc[name]) for name in acc}
    payload = {
        "alpha": PRIMARY_ALPHA,
        "seconds": round(time.time() - started, 3),
        "eval_scores": scores,
        "delta_episode_vs_row": scores["episode"]["wp"] - scores["row"]["wp"],
        "delta_episode_vs_blend": scores["episode"]["wp"] - scores["blend"]["wp"],
        "delta_row_vs_blend": scores["row"]["wp"] - scores["blend"]["wp"],
        "feature_names": list(FEATURE_NAMES) + ["pred0", "pred1"],
    }
    path = ROOT / "research" / "campaign_2026-10-06" / "wave1_episode.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "EPISODE_DONE",
        f"blend={scores['blend']['wp']:.6f}",
        f"row={scores['row']['wp']:.6f}",
        f"episode={scores['episode']['wp']:.6f}",
        f"delta_ep_row={payload['delta_episode_vs_row']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
