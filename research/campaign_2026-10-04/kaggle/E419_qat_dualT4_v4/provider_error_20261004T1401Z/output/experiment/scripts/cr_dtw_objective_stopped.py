"""DTW round / objective agent, third pass: stopping-time-consistent bootstrap targets and a DTW-aligned teacher target.

T_h (training-side, reads future rows): y if the label run ends within h rows (tau <= t + h), else TE at row t + h.
The run end tau is observable at tau (a6 event), so min(t + h, tau) is a bounded stopping time and, for a calibrated
martingale teacher, E[T_h | F_t] = E[y | F_t] (optional stopping) -- unlike TE at the run's last row (TD, E-174),
whose index tau - 1 is not a stopping time (survival-conditioned).

T_dtw: Sakoe-Chiba banded DTW (radius R) aligns the teacher path TE to the label path y per sequence and target; each
row's target is the mean teacher value aligned to it (a label-dependent selection -- tested as the DTW card).
Writes research/campaign_2026-10-02/dtw/objective/stopped_sums.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
T = 20000
WARMUP = 99
HS = (1, 2, 4, 8, 32, 128, 512, 2048)
R = 32
FEATURES = ["one", "y", "P", "TE"] + [f"T_h{h}" for h in HS] + ["T_dtw", "TE_td", "gy", "gTE", "ghy", "ghTE"]
# H-059: a(u) = .9 - .4 g, g = exp(-u/32) (u = rows until the label changes) -> target .1y + .9TE + .4(gy - gTE);
# causal counterpart: g replaced by gh = E_w[g | class, since-bin, TE-bin] (bin means from panel_sums.npz HB)
SINCE_EDGES = np.array([0, 2, 8, 32, 128, 512, 1 << 30])
TB_MULT = np.array([-np.inf, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, np.inf])


@numba.njit(cache=True)
def banded_dtw_align(q, r, radius):
    """Mean of q over the cells aligned to each reference row by squared-distance DTW within |i - j| <= radius."""
    n = q.shape[0]
    width = 2 * radius + 1
    inf = 1e30
    cost = np.full((n, width), inf)
    move = np.zeros((n, width), np.int8)  # 0 diag, 1 from (i-1, j), 2 from (i, j-1)
    for i in range(n):
        for b in range(width):
            j = i + b - radius
            if j < 0 or j >= n:
                continue
            d = (q[i] - r[j]) ** 2
            if i == 0 and j == 0:
                cost[i, b] = d
                continue
            best = inf
            mv = 0
            if i > 0 and j > 0:  # (i-1, j-1) sits at the same band offset b in row i-1
                best = cost[i - 1, b]
            if i > 0 and b + 1 < width and cost[i - 1, b + 1] < best:  # (i-1, j)
                best = cost[i - 1, b + 1]
                mv = 1
            if b > 0 and cost[i, b - 1] < best:  # (i, j-1)
                best = cost[i, b - 1]
                mv = 2
            cost[i, b] = d + best
            move[i, b] = mv
    s = np.zeros(n)
    c = np.zeros(n)
    i, b = n - 1, radius
    while True:
        j = i + b - radius
        s[j] += q[i]
        c[j] += 1
        if i == 0 and j == 0:
            break
        mv = move[i, b]
        if mv == 0:
            i -= 1
        elif mv == 1:
            i -= 1
            b += 1
        else:
            b -= 1
    return s / np.maximum(c, 1)


def main():
    z = np.load(ROOT / "cache/valid_targets.npz")
    Y, MASK = z["y"], z["mask"].copy()
    MASK[:, :WARMUP] = False
    CLS = z["market_class"].astype(int)
    rg = z["row_group"].astype(int)
    pos = {int(g): i for i, g in enumerate(rg)}
    FOLD = np.full(len(rg), -1)
    for f in range(4):
        sp = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
        for g in sp.evaluation_groups:
            FOLD[pos[g]] = f

    def load(name):
        return np.load(ROOT / f"cache/pred_{name}_valid.npy", mmap_mode="r")

    par = [load(f"c20261001_d256_s3_f{f}") for f in range(4)]
    hft = [load(f"c20261001_ft_d256_s3_f{f}h") for f in range(4)]
    ts3 = [load(f"c20260929_since_s3_f{f}") for f in range(4)]
    ts4 = [load(f"c20260929_since_s4_f{f}") for f in range(4)]
    n, d = len(Y), len(FEATURES)
    M = np.zeros((n, 2, d, d))
    term = np.zeros((4, len(HS), 2))  # [class, h, (|y|-weight terminated within h, total)] on scored rows
    rows = np.arange(T)
    pz = np.load(OUT / "panel_sums.npz")
    hb, med = pz["HB"].sum(0), pz["med"]  # [class, k, sb, tb, stat]
    ghat = hb[..., 1] / np.maximum(hb[..., 0], 1e-12)
    import pyarrow.parquet as pq
    a6 = pq.ParquetFile(ROOT / "datasets/valid.parquet").read(columns=["a6"]).column(0).to_numpy().reshape(-1, T)
    EVENT = np.isclose(a6, float(a6.min()), atol=1e-6)
    del a6
    for i in range(n):
        f, c = FOLD[i], CLS[i]
        y = np.clip(np.asarray(Y[i], np.float64), -2, 2)
        m = MASK[i]
        P = np.clip(0.5 * (np.asarray(par[f][i], np.float64) + np.asarray(hft[f][i], np.float64)), -2, 2)
        TE = np.clip(0.5 * (np.asarray(ts3[f][i], np.float64) + np.asarray(ts4[f][i], np.float64)), -2, 2)
        lab_start = np.zeros(T, bool)
        lab_start[0] = True
        lab_start[1:] = (Y[i][1:] != Y[i][:-1]).any(-1)
        ids = np.cumsum(lab_start) - 1
        ev = EVENT[i].copy()
        ev[0] = True
        since = rows - np.flatnonzero(ev)[np.cumsum(ev) - 1]
        sb = np.searchsorted(SINCE_EDGES, since, side="right") - 1
        tau = np.append(np.flatnonzero(lab_start)[1:], T)[ids]  # first row of the next run (T if none)
        for k in range(2):
            yk, Tk = y[:, k], TE[:, k]
            cols = [np.ones(T), yk, P[:, k], Tk]
            for j, h in enumerate(HS):
                alive = (tau > rows + h) & (rows + h < T)
                cols.append(np.where(alive, Tk[np.minimum(rows + h, T - 1)], yk))
                if k == 0:
                    term[c, j] += [np.abs(yk[m])[~alive[m]].sum(), np.abs(yk[m]).sum()]
            cols.append(banded_dtw_align(Tk, yk, R))
            cols.append(Tk[tau - 1])
            g = np.exp(-(tau - rows) / 32.0)
            tb = np.clip(np.searchsorted(TB_MULT * med[c], Tk, side="right") - 1, 0, 11)
            gh = ghat[c, k, sb, tb]
            cols += [g * yk, g * Tk, gh * yk, gh * Tk]
            feats = np.stack(cols, 1)[m]
            w = np.abs(yk[m])
            M[i, k] = (feats * w[:, None]).T @ feats
        if i % 300 == 0:
            print(i, flush=True)
    np.savez(OUT / "stopped_sums.npz", M=M, term=term, CLS=CLS, FOLD=FOLD, features=np.array(FEATURES))
    print("done")


if __name__ == "__main__":
    main()
