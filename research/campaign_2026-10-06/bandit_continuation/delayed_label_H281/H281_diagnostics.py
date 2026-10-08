#!/usr/bin/env python3
"""H-281 diagnostics: within-sequence vs pooled gain; leaky look-ahead control; scale/offset stats."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
import json
split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
IDX = np.asarray([int(g) for g in split["evaluation"]["valid"]])
E529 = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][IDX], dtype=np.float64)
MASK = np.asarray(targets["mask"][IDX], dtype=bool) & (np.arange(20000) >= 99)[None, :]
X = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[IDX], dtype=np.float64)
CLIP = 2.0


def moments(pred):
    S = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S[i, k] = scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i])
    return S


def causal_affine(x, y, sel, L, hl, leak=False):
    lam = 0.5 ** (1.0 / hl)
    yc = np.clip(y, -CLIP, CLIP)
    w = np.abs(yc) * sel
    r = yc - x
    f = lambda v: lfilter([1.0], [1.0, -lam], v)
    ew, exx, ex, exr, er = f(w), f(w * x * x), f(w * x), f(w * x * r), f(w * r)
    kappa = 0.05 * ew
    if leak:  # uses stats through t+L (future labels): diagnostic only
        sh = lambda v: np.concatenate([v[L:], np.zeros(L)])
    else:
        sh = lambda v: np.concatenate([np.zeros(L), v[:-L]])
    ew, exx, ex, exr, er, kappa = (sh(v) for v in (ew, exx, ex, exr, er, kappa))
    a11, a12, a22 = exx + kappa, ex, ew + kappa
    det = a11 * a22 - a12 * a12
    det = np.where(np.abs(det) < 1e-300, np.inf, det)
    c = (a22 * exr - a12 * er) / det
    a = (a11 * er - a12 * exr) / det
    return a + (1 + c) * x, c, a


def within_seq_wp(pred):
    vals = []
    for i in range(468):
        for k in range(2):
            vals.append(scorer.pooled(scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i])[None, None].repeat(1, 0).reshape(1, 1, 6) if False else np.stack([scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i]), scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i])])[None, :, :].squeeze(0)[None]) if False else 0)
    return None


base = scorer.pooled(moments(X))["wp"]
out_causal = np.zeros_like(X); out_leak = np.zeros_like(X); cs = []; as_ = []
for i in range(468):
    for k in range(2):
        o, c, a = causal_affine(X[i, :, k], Y[i, :, k], MASK[i], 1, 2000)
        out_causal[i, :, k] = o
        cs.append(np.median(c[MASK[i]])); as_.append(np.median(a[MASK[i]]))
        o2, _, _ = causal_affine(X[i, :, k], Y[i, :, k], MASK[i], 1, 2000, leak=True)
        out_leak[i, :, k] = o2
print("baseline pooled", base)
print("causal L1 hl2000 affine pooled delta", scorer.pooled(moments(out_causal))["wp"] - base)
print("LEAK (future labels, L=1) affine pooled delta", scorer.pooled(moments(out_leak))["wp"] - base)
print("median per-seq c (scale-1), median a: ", np.median(cs), np.median(as_), "IQR c", np.percentile(cs, [25, 75]), "a", np.percentile(as_, [25, 75]))


def within_mean_pearson(pred):
    """Average over sequences/targets of the metric-weighted within-sequence Pearson."""
    out = []
    for i in range(468):
        for k in range(2):
            m = MASK[i]
            y = np.clip(Y[i, m, k], -CLIP, CLIP); p = np.clip(pred[i, m, k], -CLIP, CLIP)
            w = np.abs(y)
            if w.sum() <= 0:
                continue
            my, mp = (w * y).sum() / w.sum(), (w * p).sum() / w.sum()
            cov = (w * (y - my) * (p - mp)).sum()
            vy = (w * (y - my) ** 2).sum(); vp = (w * (p - mp) ** 2).sum()
            if vy > 0 and vp > 0:
                out.append(cov / np.sqrt(vy * vp))
    return float(np.mean(out))


print("within-seq weighted corr baseline", within_mean_pearson(X), "causal", within_mean_pearson(out_causal), "leak", within_mean_pearson(out_leak))
