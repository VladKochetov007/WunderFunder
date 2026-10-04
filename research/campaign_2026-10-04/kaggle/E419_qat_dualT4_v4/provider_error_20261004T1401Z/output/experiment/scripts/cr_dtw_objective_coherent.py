"""DTW round / objective agent, fourth pass: run-coherent target noise under TRAINING weights.

Label noise is shared by every row of a label run, so its gradient contribution adds coherently over the run. For a
target T the per-run noise sum is N_s = sum_{t in s} w_t (T_t - TE_t) with the student's training row weight
w_t = |y| (propensity^2 + 0.03) on rows >= 99 (all rows, not only scored ones). Reported: sum_s N_s^2 relative to the
same for T = y (coherent noise), next to the per-row ratio sum_t w_t (T - TE)^2 / sum_t w_t (y - TE)^2, by class and by
run length. Targets read future rows: training-side only.
Writes research/campaign_2026-10-02/dtw/objective/coherent.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
T = 20000
LEN_EDGES = np.array([0, 8, 64, 1024, 1 << 30])
LEN_LABELS = ["<8", "8-64", "64-1024", ">=1024"]


def main():
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y = z["y"]
    CLS = z["market_class"].astype(int)
    rg = z["row_group"].astype(int)
    pos = {int(g): i for i, g in enumerate(rg)}
    FOLD = np.full(len(rg), -1)
    for f in range(4):
        for g in load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT).evaluation_groups:
            FOLD[pos[g]] = f
    ts = [[np.load(ROOT / f"cache/pred_c20260929_since_s{s}_f{f}_valid.npy", mmap_mode="r") for f in range(4)] for s in (3, 4)]
    prop = [np.load(ROOT / f"cache/propcache_c20260929_prop_f{f}_valid.npy", mmap_mode="r") for f in range(4)]
    names = ["y", "a.5", "a.7", "a.85", "T_h8", "T_h32", ".5T_h8+.5TE", ".3T_h8+.7TE", ".3T_h32+.7TE",
             "H-059", "TD run end"]
    rows = np.arange(T)
    coh = np.zeros((4, len(LEN_LABELS), len(names)))
    per = np.zeros((4, len(LEN_LABELS), len(names)))
    for i in range(len(Y)):
        f, c = FOLD[i], CLS[i]
        y = np.clip(np.asarray(Y[i], np.float64), -2, 2)
        TE = np.clip(0.5 * (np.asarray(ts[0][f][i], np.float64) + np.asarray(ts[1][f][i], np.float64)), -2, 2)
        wrow = np.asarray(prop[f][rg[i]], np.float64) ** 2 + 0.03
        wrow[:99] = 0.0
        start = np.zeros(T, bool)
        start[0] = True
        start[1:] = (Y[i][1:] != Y[i][:-1]).any(-1)
        ids = np.cumsum(start) - 1
        firsts = np.flatnonzero(start)
        tau = np.append(firsts[1:], T)[ids]
        length = np.diff(np.append(firsts, T))
        lb = np.searchsorted(LEN_EDGES, length, side="right") - 1
        g = np.exp(-(tau - rows) / 32.0)
        for k in range(2):
            yk, Tk = y[:, k], TE[:, k]
            th = {h: np.where((tau > rows + h) & (rows + h < T), Tk[np.minimum(rows + h, T - 1)], yk) for h in (8, 32)}
            tg = [yk, .5 * yk + .5 * Tk, .3 * yk + .7 * Tk, .15 * yk + .85 * Tk, th[8], th[32], .5 * th[8] + .5 * Tk,
                  .3 * th[8] + .7 * Tk, .3 * th[32] + .7 * Tk, (.1 + .4 * g) * yk + (.9 - .4 * g) * Tk, Tk[tau - 1]]
            w = np.abs(yk) * wrow
            for j, t in enumerate(tg):
                e = w * (t - Tk)
                ns = np.bincount(ids, e, minlength=len(firsts))
                coh[c, :, j] += np.bincount(lb, ns ** 2, minlength=len(LEN_LABELS))
                per[c, :, j] += np.bincount(lb, np.bincount(ids, w * (t - Tk) ** 2, minlength=len(firsts)), minlength=len(LEN_LABELS))
        if i % 300 == 0:
            print(i, flush=True)
    res = {"names": names, "len_bins": LEN_LABELS}
    for c in range(4):
        res[f"class{c}"] = {
            "coherent_all": np.round(coh[c].sum(0) / coh[c].sum(0)[0], 4).tolist(),
            "per_row_all": np.round(per[c].sum(0) / per[c].sum(0)[0], 4).tolist(),
            "coherent_by_len": {LEN_LABELS[b]: np.round(coh[c, b] / max(coh[c, b, 0], 1e-12), 4).tolist() for b in range(len(LEN_LABELS))},
            "coherent_share_by_len_for_y": np.round(coh[c, :, 0] / coh[c, :, 0].sum(), 4).tolist(),
        }
    res["all"] = {"coherent_all": np.round(coh.sum((0, 1)) / coh.sum((0, 1))[0], 4).tolist(),
                  "per_row_all": np.round(per.sum((0, 1)) / per.sum((0, 1))[0], 4).tolist()}
    (OUT / "coherent.json").write_text(json.dumps(res, indent=1))
    for key in ["all"] + [f"class{c}" for c in range(4)]:
        print(key)
        for j, nm in enumerate(names):
            print(f"   {nm:16s} per-row {res[key]['per_row_all'][j]:.3f}  coherent {res[key]['coherent_all'][j]:.3f}")
    for c in range(4):
        print(f"class{c} coherent y-noise share by run length {LEN_LABELS}:", res[f"class{c}"]["coherent_share_by_len_for_y"])
        print("   coherent ratio in runs 64-1024:", dict(zip(names, res[f"class{c}"]["coherent_by_len"]["64-1024"])))


if __name__ == "__main__":
    main()
