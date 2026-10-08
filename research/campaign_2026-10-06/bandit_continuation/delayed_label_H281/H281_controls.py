#!/usr/bin/env python3
"""H-281 controls: label-shuffle and oracle checks for the recalibration gain."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/delayed_label_H281")
import importlib
ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
from scipy.signal import lfilter

split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
IDX = np.asarray(EVAL)
E529 = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][IDX], dtype=np.float64)
MC = np.asarray(targets["market_class"][IDX])
MASK = np.asarray(targets["mask"][IDX], dtype=bool) & (np.arange(20000) >= 99)[None, :]
X = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[IDX], dtype=np.float64)
CLIP = 2.0


def moments(pred, Yuse):
    S = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S[i, k] = scorer.moments(Yuse[i, :, k], pred[i, :, k], MASK[i])
    return S


def recal(x, y, sel, L, hl, mode, oracle=False):
    lam = 0.5 ** (1.0 / hl)
    yc = np.clip(y, -CLIP, CLIP)
    w = np.abs(yc) * sel
    r = yc - x
    f = lambda v: lfilter([1.0], [1.0, -lam], v)
    ew, exx, ex, exr, er = f(w), f(w * x * x), f(w * x), f(w * x * r), f(w * r)
    kappa = 0.05 * ew
    if oracle:  # in-sample sequence-level fit using ALL rows (uses future labels)
        tot_w = w.sum(); tot_xx = (w * x * x).sum(); tot_x = (w * x).sum()
        tot_xr = (w * x * r).sum(); tot_r = (w * r).sum()
        ew = np.full_like(x, tot_w); exx = np.full_like(x, tot_xx); ex = np.full_like(x, tot_x)
        exr = np.full_like(x, tot_xr); er = np.full_like(x, tot_r); kappa = 0.05 * ew
    else:
        sh = lambda v: np.concatenate([np.zeros(L), v[:-L]]) if L > 0 else v
        ew, exx, ex, exr, er, kappa = (sh(v) for v in (ew, exx, ex, exr, er, kappa))
    if mode == "affine":
        a11, a12, a22 = exx + kappa, ex, ew + kappa
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-300, np.inf, det)
        c = (a22 * exr - a12 * er) / det
        a = (a11 * er - a12 * exr) / det
        return a + (1 + c) * x
    c = exr / (exx + kappa + 1e-300)
    return (1 + c) * x


def run(Yuse, L, hl, mode, oracle=False):
    out = np.zeros_like(X)
    for i in range(468):
        for k in range(2):
            out[i, :, k] = recal(X[i, :, k], Yuse[i, :, k], MASK[i], L, hl, mode, oracle)
    return out


base = scorer.pooled(moments(X, Y))["wp"]
print("baseline", base, flush=True)
rng = np.random.default_rng(0)
# control A: labels from a random OTHER sequence of the same class (breaks label-prediction link)
perm = np.arange(468)
for i in range(468):
    cands = np.where((MC == MC[i]) & (np.arange(468) != i))[0]
    perm[i] = rng.choice(cands)
Ysh = Y[perm]
for mode in ("affine", "scale"):
    o = run(Ysh, 1, 2000, mode)
    print("shuffled-label L1 hl2000", mode, scorer.pooled(moments(o, Y))["wp"] - base, flush=True)
# control B: oracle in-sample per-sequence fit (uses all labels of the sequence)
for mode in ("affine", "scale"):
    out = np.zeros_like(X)
    for i in range(468):
        for k in range(2):
            out[i, :, k] = recal(X[i, :, k], Y[i, :, k], MASK[i], 0, 2000, mode, oracle=True)
    print("oracle in-sample per-sequence", mode, scorer.pooled(moments(out, Y))["wp"] - base, flush=True)
# control C: constant per-sequence fit with no label information (identity) sanity
print("identity", scorer.pooled(moments(X * 1.0, Y))["wp"] - base)
