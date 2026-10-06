"""WP of the frozen blend on far/near rows split by recent trade, volume, i1, OR."""

from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq

from wave1_online import COLS, _changed, _recent
from wave1_stretch import HORIZON, QUIET_AGE, ROOT, _add, _and_age, _next_change, _wp
from wnn.gmetric import GlobalAccumulator

SLICES = ("far", "far_trade8", "far_notrade8", "far_or8", "far_i18", "near", "near_trade8")


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
    acc = {name: GlobalAccumulator() for name in SLICES}

    for group in range(y_all.shape[0]):
        table = valid_pf.read_row_group(int(row_group[group]), columns=list(COLS), use_threads=False)
        cols = {name: table.column(name).to_numpy() for name in COLS}
        y = y_all[group]
        mask = mask_all[group]
        pred = (0.5 * (parent[group].astype(np.float32) + head[group].astype(np.float32))).astype(
            np.float32
        )
        age, _last = _and_age(cols["i0_p0"], cols["i0_p11"])
        nxt = _next_change(y)
        index = np.arange(y.shape[0], dtype=np.int32)
        quiet = age >= QUIET_AGE
        far = quiet & (nxt > (index + HORIZON)) & ((index + HORIZON) < y.shape[0])
        near = quiet & ~far
        or8 = _recent(_changed(cols["i0_p0"]) | _changed(cols["i0_p11"]), 8)
        trade8 = _recent(_changed(cols["i0_dp0"]) | _changed(cols["i0_dv0"]), 8)
        i18 = _recent(_changed(cols["i1_p0"]) | _changed(cols["i1_p11"]), 8)
        _add(acc["far"], y, pred, mask & far)
        _add(acc["far_trade8"], y, pred, mask & far & trade8)
        _add(acc["far_notrade8"], y, pred, mask & far & ~trade8)
        _add(acc["far_or8"], y, pred, mask & far & or8)
        _add(acc["far_i18"], y, pred, mask & far & i18)
        _add(acc["near"], y, pred, mask & near)
        _add(acc["near_trade8"], y, pred, mask & near & trade8)

    payload = {
        "seconds": round(time.time() - started, 3),
        "slices": {name: _wp(acc[name]) for name in SLICES},
    }
    path = ROOT / "research" / "campaign_2026-10-06" / "wave1_slices.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    s = payload["slices"]
    print(
        "SLICES_DONE",
        f"far={s['far']['wp']:.6f}",
        f"far_trade={s['far_trade8']['wp']:.6f}",
        f"far_notrade={s['far_notrade8']['wp']:.6f}",
        f"near={s['near']['wp']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
