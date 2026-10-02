"""DTW round / objective agent: is the stopped bootstrap target unbiased inside causal strata (class x rows since the
last a6 event)? Reports slope_rel = Cov_w(T, TE) / Cov_w(y, TE) and mean_w(T - y) / RMS_w(y) per stratum (scored rows,
|y| weights, OOF teacher TE). Unbiased targets give slope_rel = 1 and zero mean shift in every causal stratum."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
T = 20000
SINCE_EDGES = np.array([0, 2, 8, 32, 128, 512, 1 << 30])
z = np.load(ROOT / "cache/valid_targets.npz")
Y, MASK = z["y"], z["mask"].copy()
MASK[:, :99] = False
CLS = z["market_class"].astype(int)
rg = z["row_group"].astype(int)
pos = {int(g): i for i, g in enumerate(rg)}
FOLD = np.full(len(rg), -1)
for f in range(4):
    for g in load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT).evaluation_groups:
        FOLD[pos[g]] = f
a6 = pq.ParquetFile(ROOT / "datasets/valid.parquet").read(columns=["a6"]).column(0).to_numpy().reshape(-1, T)
EVENT = np.isclose(a6, float(a6.min()), atol=1e-6)
del a6
ts = [[np.load(ROOT / f"cache/pred_c20260929_since_s{s}_f{f}_valid.npy", mmap_mode="r") for f in range(4)] for s in (3, 4)]
names = ["y", "a.7", "T_h8", "T_h32", ".3T_h8+.7TE", "H-059", "TD run end"]
S = np.zeros((4, 6, len(names), 6))  # w, wT, wTE, wTE2, wT.TE, wT2
rows = np.arange(T)
for i in range(len(Y)):
    f, c, m = FOLD[i], CLS[i], MASK[i]
    y = np.clip(np.asarray(Y[i], np.float64), -2, 2)
    TE = np.clip(0.5 * (np.asarray(ts[0][f][i], np.float64) + np.asarray(ts[1][f][i], np.float64)), -2, 2)
    start = np.zeros(T, bool)
    start[0] = True
    start[1:] = (Y[i][1:] != Y[i][:-1]).any(-1)
    ids = np.cumsum(start) - 1
    tau = np.append(np.flatnonzero(start)[1:], T)[ids]
    ev = EVENT[i].copy()
    ev[0] = True
    sb = (np.searchsorted(SINCE_EDGES, rows - np.flatnonzero(ev)[np.cumsum(ev) - 1], side="right") - 1)[m]
    g = np.exp(-(tau - rows) / 32.0)
    for k in range(2):
        yk, Tk = y[:, k], TE[:, k]
        th = {h: np.where((tau > rows + h) & (rows + h < T), Tk[np.minimum(rows + h, T - 1)], yk) for h in (8, 32)}
        tg = [yk, .3 * yk + .7 * Tk, th[8], th[32], .3 * th[8] + .7 * Tk, (.1 + .4 * g) * yk + (.9 - .4 * g) * Tk, Tk[tau - 1]]
        w, te = np.abs(yk[m]), Tk[m]
        for j, t in enumerate(tg):
            t = t[m]
            st = np.stack([w, w * t, w * te, w * te * te, w * t * te, w * t * t], 1)
            S[c, :, j] += np.stack([np.bincount(sb, st[:, q], minlength=6) for q in range(6)], 1)
res = {}
labels = ["<2", "2-8", "8-32", "32-128", "128-512", ">=512"]
for c in range(4):
    res[c] = {}
    for b in range(6):
        w, wt, wte, wte2, wtte, wtt = np.moveaxis(S[c, b], -1, 0)
        cov = wtte / w - wt * wte / w ** 2
        res[c][labels[b]] = {nm: [round(float(cov[j] / cov[0]), 3), round(float((wt[j] - wt[0]) / w[j] / np.sqrt(wtt[0] / w[0])), 4)]
                             for j, nm in enumerate(names)}
    print(f"class {c}: slope_rel by since-bin", {lb: {nm: v[0] for nm, v in res[c][lb].items() if nm != 'y'} for lb in labels})
(OUT / "strata.json").write_text(json.dumps(res, indent=1))
