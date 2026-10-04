"""Q7: how in-sample is the real distillation teacher on its TRAIN rows? (input to the lambda-return card)

Teacher = mean of c20260929_since_s{3,4}_f0 train predictions (the D256 fold-0 student's teacher). On 200 random
fold-0 fit-train sequences: |y|-weighted slope of y on M and weighted corr per class, vs the same teacher family's
out-of-fold valid numbers. Also the class-3 lambda-return (l = 0.5) weighted corr with y vs the teacher's.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq

from cr_dtw_readout_core import OUT, ROOT

ENDPOINT = np.float32(-5.199337482452393)


def wstats(y, m):
    w = np.abs(y)
    n = w.sum()
    my, mm = (w * y).sum() / n, (w * m).sum() / n
    cov = (w * y * m).sum() / n - my * mm
    vm = (w * m * m).sum() / n - mm**2
    vy = (w * y * y).sum() / n - my**2
    return cov / vm, cov / np.sqrt(vm * vy)


def lam_return(M, y, chg_next, lam):
    G = np.empty_like(M)
    T = len(M)
    for t in range(T - 1, -1, -1):
        G[t] = y[t] if (t == T - 1 or chg_next[t]) else (1 - lam) * M[t + 1] + lam * G[t + 1]
    return G


def main():
    split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
    fit = np.array(sorted(split["fit"]["train"]), np.int64)
    with np.load(ROOT / "cache/train_targets.npz") as z:
        rg, cls_all = z["row_group"], z["market_class"]
    pos = {int(g): i for i, g in enumerate(rg)}
    T3 = np.load(ROOT / "cache/pred_c20260929_since_s3_f0_train.npy", mmap_mode="r")
    T4 = np.load(ROOT / "cache/pred_c20260929_since_s4_f0_train.npy", mmap_mode="r")
    f = pq.ParquetFile(ROOT / "datasets/train.parquet")
    pick = np.random.default_rng(3).choice(fit, 200, replace=False)
    acc = {c: {"y": [], "m": [], "g": []} for c in range(4)}
    for g in pick:
        j = pos[int(g)]
        c = int(cls_all[j])
        t = f.read_row_group(int(g), columns=["t0", "t1", "a6"], use_threads=False)
        y = np.clip(np.column_stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()]), -2, 2)[99:]
        M = np.clip(0.5 * (np.asarray(T3[j], np.float32) + np.asarray(T4[j], np.float32)), -2, 2)[99:]
        chg = np.zeros(len(y), bool)
        chg[:-1] = (y[1:] != y[:-1]).any(1)  # row t is the last of its run
        for k in range(2):
            acc[c]["y"].append(y[:, k])
            acc[c]["m"].append(M[:, k])
            if c == 3:
                acc[c]["g"].append(lam_return(M[:, k].astype(np.float64), y[:, k].astype(np.float64), chg, 0.5))
    res = {}
    for c in range(4):
        yy, mm = np.concatenate(acc[c]["y"]), np.concatenate(acc[c]["m"])
        sl, co = wstats(yy, mm)
        res[f"c{c}"] = {"slope_y_on_M_train_insample": round(float(sl), 3), "wcorr_train_insample": round(float(co), 3)}
        if c == 3:
            gg = np.concatenate(acc[c]["g"])
            res["c3"]["wcorr_lambda0.5_return_with_y"] = round(float(wstats(yy, gg)[1]), 3)
    print(json.dumps(res, indent=1))
    (OUT / "q7_teacher_insample.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
