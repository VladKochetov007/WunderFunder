"""Causal output-space trackers and a precursor check on the frozen blend.

No coefficient search. The two EMA decays and the age-1 lock are predeclared.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wave1_stretch import HORIZON, QUIET_AGE, ROOT, _add, _and_age, _next_change, _wp
from wnn.gmetric import GlobalAccumulator
from wnn.splits import is_holdout

OUT = Path(__file__).resolve().parent
COLS = ("i0_p0", "i0_p11", "i0_v0", "i0_v11", "i0_dp0", "i0_dv0", "i1_p0", "i1_p11")
EMA_DECAYS = (("ema8", float(np.exp(-1.0 / 8.0))), ("ema32", float(np.exp(-1.0 / 32.0))))


def _changed(column: np.ndarray) -> np.ndarray:
    out = np.zeros(column.shape[0], dtype=bool)
    out[1:] = column[1:] != column[:-1]
    return out


def _recent(flag: np.ndarray, window: int) -> np.ndarray:
    """True if flag was true on this row or any of the previous window-1 rows."""
    count = np.cumsum(flag.astype(np.int32))
    lagged = np.zeros_like(count)
    lagged[window:] = count[:-window]
    return (count - lagged) > 0


def _ema(pred: np.ndarray, last: np.ndarray, decay: float) -> np.ndarray:
    """Causal EMA of predictions, reset to the current prediction on each new stretch start."""
    out = np.empty_like(pred, dtype=np.float32)
    state = pred[0].astype(np.float64)
    out[0] = pred[0]
    for t in range(1, pred.shape[0]):
        if last[t] == t:
            state = pred[t].astype(np.float64)
        else:
            state = decay * state + (1.0 - decay) * pred[t].astype(np.float64)
        out[t] = state
    return out


def _age1_hold(pred: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Carry the first post-touch prediction; row 0 if the stretch starts at the origin."""
    source = np.minimum(last + 1, pred.shape[0] - 1)
    source = np.where(last == 0, 0, source)
    return pred[source]


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

    names = ("blend", "age1_hold", "ema8", "ema32", "global_age1", "global_ema8", "global_ema32")
    full = {name: GlobalAccumulator() for name in names}
    far_acc = {name: GlobalAccumulator() for name in ("blend", "age1_hold", "ema8", "ema32")}
    near_acc = {name: GlobalAccumulator() for name in ("blend", "age1_hold", "ema8", "ema32")}
    motion = {
        key: {"rows": 0, "or8": 0, "vol8": 0, "trade8": 0, "i18": 0, "any8": 0}
        for key in ("far", "near", "quiet")
    }

    for group in range(y_all.shape[0]):
        parquet_group = int(row_group[group])
        table = valid_pf.read_row_group(parquet_group, columns=list(COLS), use_threads=False)
        cols = {name: table.column(name).to_numpy() for name in COLS}
        y = y_all[group]
        mask = mask_all[group]
        pred = (0.5 * (parent[group].astype(np.float32) + head[group].astype(np.float32))).astype(
            np.float32
        )
        age, last = _and_age(cols["i0_p0"], cols["i0_p11"])
        nxt = _next_change(y)
        index = np.arange(y.shape[0], dtype=np.int32)
        quiet = age >= QUIET_AGE
        far = quiet & (nxt > (index + HORIZON)) & ((index + HORIZON) < y.shape[0])
        near = quiet & ~far
        or_ev = _changed(cols["i0_p0"]) | _changed(cols["i0_p11"])
        vol = _changed(cols["i0_v0"]) | _changed(cols["i0_v11"])
        trade = _changed(cols["i0_dp0"]) | _changed(cols["i0_dv0"])
        i1 = _changed(cols["i1_p0"]) | _changed(cols["i1_p11"])
        any_move = or_ev | vol | trade | i1
        held = _age1_hold(pred, last)
        emas = {name: _ema(pred, last, decay) for name, decay in EMA_DECAYS}
        for key, take in (("far", far), ("near", near), ("quiet", quiet)):
            rows = mask & take
            n = int(rows.sum())
            if n == 0:
                continue
            motion[key]["rows"] += n
            motion[key]["or8"] += int(_recent(or_ev, 8)[rows].sum())
            motion[key]["vol8"] += int(_recent(vol, 8)[rows].sum())
            motion[key]["trade8"] += int(_recent(trade, 8)[rows].sum())
            motion[key]["i18"] += int(_recent(i1, 8)[rows].sum())
            motion[key]["any8"] += int(_recent(any_move, 8)[rows].sum())
        _add(full["blend"], y, pred, mask)
        _add(full["age1_hold"], y, held, mask)
        _add(full["ema8"], y, emas["ema8"], mask)
        _add(full["ema32"], y, emas["ema32"], mask)
        quiet_age1 = pred.copy()
        quiet_age1[quiet] = held[quiet]
        quiet_ema8 = pred.copy()
        quiet_ema8[quiet] = emas["ema8"][quiet]
        quiet_ema32 = pred.copy()
        quiet_ema32[quiet] = emas["ema32"][quiet]
        _add(full["global_age1"], y, quiet_age1, mask)
        _add(full["global_ema8"], y, quiet_ema8, mask)
        _add(full["global_ema32"], y, quiet_ema32, mask)
        _add(far_acc["blend"], y, pred, mask & far)
        _add(far_acc["age1_hold"], y, held, mask & far)
        _add(far_acc["ema8"], y, emas["ema8"], mask & far)
        _add(far_acc["ema32"], y, emas["ema32"], mask & far)
        _add(near_acc["blend"], y, pred, mask & near)
        _add(near_acc["age1_hold"], y, held, mask & near)
        _add(near_acc["ema8"], y, emas["ema8"], mask & near)
        _add(near_acc["ema32"], y, emas["ema32"], mask & near)
        if is_holdout(int(seq_ix[group])):
            continue

    rates = {
        key: {name: (vals[name] / vals["rows"] if vals["rows"] else 0.0) for name in vals if name != "rows"}
        | {"rows": vals["rows"]}
        for key, vals in motion.items()
    }
    payload = {
        "metric": "global_wp_clip2_v1",
        "seconds": round(time.time() - started, 3),
        "ema_decays": {name: decay for name, decay in EMA_DECAYS},
        "full_valid": {name: _wp(acc) for name, acc in full.items()},
        "far_cell": {name: _wp(acc) for name, acc in far_acc.items()},
        "near_cell": {name: _wp(acc) for name, acc in near_acc.items()},
        "motion_in_last_8": rates,
    }
    path = OUT / "wave1_online.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    far = payload["far_cell"]
    print(
        "ONLINE_DONE",
        f"far_blend={far['blend']['wp']:.6f}",
        f"far_age1={far['age1_hold']['wp']:.6f}",
        f"far_ema8={far['ema8']['wp']:.6f}",
        f"far_any8={rates['far']['any8']:.4f}",
        f"near_any8={rates['near']['any8']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
