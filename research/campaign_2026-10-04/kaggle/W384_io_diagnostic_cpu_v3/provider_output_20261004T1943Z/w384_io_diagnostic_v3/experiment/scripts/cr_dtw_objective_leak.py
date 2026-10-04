"""DTW round / objective agent: label leakage of in-sample distillation targets and duration-aware readout room.

(1) rho = slope of (TE_in - TE_oof) on (y - TE_oof): how much of the label residual an in-sample teacher (fold g != f
model, trained on this valid sequence) has absorbed relative to the out-of-fold teacher (fold f model) on the same rows.
Effective distillation weight with an in-sample teacher is about a(1 - rho).
(2) cross-fitted per-bin affine readout of P on class x since-bin x TE-bin (HB sums of panel_sums.npz): room for a
duration/hazard-aware output head beyond the GRU.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-02/dtw/objective"
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
ts = [[np.load(ROOT / f"cache/pred_c20260929_since_s{s}_f{f}_valid.npy", mmap_mode="r") for f in range(4)] for s in (3, 4)]
S = np.zeros((4, 3))  # class: sum w (din * r), sum w r^2, sum w din^2
for i in range(len(Y)):
    f, c, m = FOLD[i], CLS[i], MASK[i]
    y = np.clip(np.asarray(Y[i], np.float64), -2, 2)[m]
    te = lambda g: np.clip(0.5 * (np.asarray(ts[0][g][i], np.float64) + np.asarray(ts[1][g][i], np.float64)), -2, 2)[m]
    oof = te(f)
    r = y - oof
    w = np.abs(y)
    for g in range(4):
        if g == f:
            continue
        din = te(g) - oof
        S[c] += [(w * din * r).sum(), (w * r * r).sum(), (w * din * din).sum()]
rho = S[:, 0] / S[:, 1]
print("rho (share of the OOF label residual absorbed by in-sample teachers) by class:", rho.round(4),
      "pooled", round(float(S[:, 0].sum() / S[:, 1].sum()), 4))
print("corr(TE_in - TE_oof, y - TE_oof) by class:", (S[:, 0] / np.sqrt(S[:, 1] * S[:, 2])).round(4))

H = np.load(OUT / "panel_sums.npz")["HB"]  # [fold, class, k, sb, tb, stat] w wg wgg wr wgr wy wP wPP wyP wyy wrr


def wp(Hs, a, b):
    out = []
    for k in range(2):
        h = Hs[:, k]
        w, wy, wp_, wpp, wyp, wyy = (h[..., j] for j in (0, 5, 6, 7, 8, 9))
        A, B = a[:, k], b[:, k]
        n, sy = w.sum(), wy.sum()
        sp = (A * w + B * wp_).sum()
        spp = (A * A * w + 2 * A * B * wp_ + B * B * wpp).sum()
        syp = (A * wy + B * wyp).sum()
        syy = wyy.sum()
        out.append((syp / n - sy * sp / n ** 2) / np.sqrt((syy / n - (sy / n) ** 2) * (spp / n - (sp / n) ** 2)))
    return float(np.mean(out))


deltas = []
for f in range(4):
    tr = H[np.arange(4) != f].sum(0)
    w, wy, wp_, wpp, wyp = (tr[..., j] for j in (0, 5, 6, 7, 8))
    lam = 0.05 * np.maximum(w, 1e-12)  # ridge toward the identity map on thin bins
    vp = wpp - wp_ ** 2 / np.maximum(w, 1e-12)
    cv = wyp - wy * wp_ / np.maximum(w, 1e-12)
    b = np.where(w > 0, (cv + lam) / (vp + lam), 1.0)
    a = np.where(w > 0, (wy - b * wp_) / np.maximum(w, 1e-12), 0.0)
    deltas.append(wp(H[f], a, b) - wp(H[f], np.zeros_like(a), np.ones_like(b)))
print("cross-fitted per-bin affine readout (class x since x TE-bin, 288 bins) dWP by fold:", np.round(deltas, 5),
      "mean", round(float(np.mean(deltas)), 5))
