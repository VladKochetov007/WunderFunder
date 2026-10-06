"""Hold the last-trade prediction across far rows that have had no recent trade."""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq

from wave1_online import COLS, _changed, _recent
from wave1_anatomy import _since_last as _age_since
from wave1_stretch import HORIZON, QUIET_AGE, ROOT, _add, _and_age, _next_change, _wp
from wnn.gmetric import GlobalAccumulator


def main() -> None:
    started = time.time()
    cache = ROOT / "cache"
    with np.load(cache / "valid_targets.npz") as blob:
        y_all = np.asarray(blob["y"])
        mask_all = np.asarray(blob["mask"], dtype=bool)
        row_group = np.asarray(blob["row_group"])
    parent = np.load(cache / "pred_c20261004_combo_s4_final_fp32_valid.npy", mmap_mode="r")
    head = np.load(cache / "pred_c20261004_combo_head_s4_final_fp32_valid.npy", mmap_mode="r")
    valid_pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    names = (
        "blend",
        "far_notrade_blend",
        "far_notrade_last_trade",
        "far_notrade_last_and",
        "global_last_trade",
    )
    acc = {name: GlobalAccumulator() for name in names}

    for group in range(y_all.shape[0]):
        table = valid_pf.read_row_group(int(row_group[group]), columns=list(COLS), use_threads=False)
        cols = {name: table.column(name).to_numpy() for name in COLS}
        y = y_all[group]
        mask = mask_all[group]
        pred = (0.5 * (parent[group].astype(np.float32) + head[group].astype(np.float32))).astype(
            np.float32
        )
        age, last_and = _and_age(cols["i0_p0"], cols["i0_p11"])
        nxt = _next_change(y)
        index = np.arange(y.shape[0], dtype=np.int32)
        quiet = age >= QUIET_AGE
        far = quiet & (nxt > (index + HORIZON)) & ((index + HORIZON) < y.shape[0])
        trade = _changed(cols["i0_dp0"]) | _changed(cols["i0_dv0"])
        notrade = far & ~_recent(trade, 8)
        _age, last_trade = _age_since(trade)
        held_trade = pred[last_trade]
        held_and = pred[last_and]
        replaced = pred.copy()
        replaced[notrade] = held_trade[notrade]
        _add(acc["blend"], y, pred, mask)
        _add(acc["far_notrade_blend"], y, pred, mask & notrade)
        _add(acc["far_notrade_last_trade"], y, held_trade, mask & notrade)
        _add(acc["far_notrade_last_and"], y, held_and, mask & notrade)
        _add(acc["global_last_trade"], y, replaced, mask)

    payload = {
        "seconds": round(time.time() - started, 3),
        "scores": {name: _wp(acc[name]) for name in names},
    }
    cell = payload["scores"]
    payload["delta_far_notrade_last_trade"] = (
        cell["far_notrade_last_trade"]["wp"] - cell["far_notrade_blend"]["wp"]
    )
    payload["delta_global"] = cell["global_last_trade"]["wp"] - cell["blend"]["wp"]
    path = ROOT / "research" / "campaign_2026-10-06" / "wave1_tradehold.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "TRADEHOLD_DONE",
        f"cell={cell['far_notrade_blend']['wp']:.6f}",
        f"last_trade={cell['far_notrade_last_trade']['wp']:.6f}",
        f"delta={payload['delta_far_notrade_last_trade']:.6f}",
        f"global_delta={payload['delta_global']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
