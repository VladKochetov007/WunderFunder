#!/usr/bin/env python3
"""H-281 steps 2-3: strictly causal delayed-label recalibration and lagged-residual check.

Causality (index arithmetic). For a reveal delay L >= 1 and row t of a sequence:
- the label y_s becomes available at row s + L, so at row t the usable labels are
  s <= t - L;
- E[t] = lfilter over increments inc_s is computed with zero initial state per
  sequence, so E[t] depends only on rows 0..t;
- the recalibration at row t uses S_t = E[t - L] (zero stats for t < L), which
  depends only on rows 0..t-L;
- the output is a + (1 + c) * x_t, where x_t is the baseline prediction at row t
  (causal by construction in the exported predictions) and (a, c) come from S_t only.
No row s > t - L enters the fit used at row t. Reset is per sequence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
OUT = CAMP / "delayed_label_H281"
SEED = 271828
CLIP = 2.0
WARMUP = 99
KAPPA_FRAC = 0.05
HALF_LIVES = [500, 2000, 8000]
DELAYS = [1, 10, 50]
LAG_GRID = [1, 2, 5, 10, 20, 50]
t_start = time.time()

spec = importlib.util.spec_from_file_location(
    "campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
EVAL_IDX = np.asarray(EVAL, dtype=np.int64)
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL])
targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][EVAL_IDX], dtype=np.float64)
MC = np.asarray(targets["market_class"][EVAL_IDX])
MASK = np.asarray(targets["mask"][EVAL_IDX], dtype=bool)
MASK &= (np.arange(20000) >= WARMUP)[None, :]

S4 = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[EVAL_IDX], dtype=np.float64)
D272 = np.asarray(np.load(E529 / "exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/pred_multiplicative.npy", mmap_mode="r")[EVAL_IDX], dtype=np.float64)
PAIR = 0.5 * S4 + 0.5 * D272
PREDS = {"H272S4_hidden_only": S4, "equal_pair_S4hid_D272mult": PAIR}


def moments_of(pred):
    S = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S[i, k] = scorer.moments(Y[i, :, k], pred[i, :, k], MASK[i])
    return S


def recal_seq(x, y, sel, L, half_life, mode):
    """x, y: (T,) for one sequence and target; sel: bool (T,). Returns output (T,)."""
    lam = 0.5 ** (1.0 / half_life)
    yc = np.clip(y, -CLIP, CLIP)
    w = np.abs(yc) * sel
    r = yc - x
    ew = lfilter([1.0], [1.0, -lam], w)
    exx = lfilter([1.0], [1.0, -lam], w * x * x)
    ex = lfilter([1.0], [1.0, -lam], w * x)
    exr = lfilter([1.0], [1.0, -lam], w * x * r)
    er = lfilter([1.0], [1.0, -lam], w * r)
    kappa = KAPPA_FRAC * ew

    def shift(v):
        out = np.zeros_like(v)
        out[L:] = v[:-L]
        return out

    ew, exx, ex, exr, er, kappa = (shift(v) for v in (ew, exx, ex, exr, er, kappa))
    if mode == "affine":
        a11 = exx + kappa
        a12 = ex
        a22 = ew + kappa
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-300, np.inf, det)
        c = (a22 * exr - a12 * er) / det
        a = (a11 * er - a12 * exr) / det
        return a + (1.0 + c) * x
    c = exr / (exx + kappa + 1e-300)
    return (1.0 + c) * x


def run(pred, L, half_life, mode):
    out = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        for k in range(2):
            out[i, :, k] = recal_seq(pred[i, :, k], Y[i, :, k], MASK[i], L, half_life, mode)
    return out


def lagged_residual_corr(pred, L):
    """Metric-weighted Pearson of r_t and r_{t-L}, pooled per class; r = clip(pred) - clip(y)."""
    res = {}
    for label in ("all", 0, 1, 2, 3):
        sw = np.zeros(2)
        sa = np.zeros(2)
        sb = np.zeros(2)
        sab = np.zeros(2)
        sa2 = np.zeros(2)
        sb2 = np.zeros(2)
        for i in range(pred.shape[0]):
            if label != "all" and MC[i] != label:
                continue
            for k in range(2):
                r = np.clip(pred[i, :, k], -CLIP, CLIP) - np.clip(Y[i, :, k], -CLIP, CLIP)
                cur = r[L:]
                lag = r[:-L]
                ok = MASK[i, L:] & MASK[i, :-L]
                wgt = np.abs(np.clip(Y[i, L:, k], -CLIP, CLIP)) * ok
                sw[k] += wgt.sum()
                sa[k] += (wgt * cur).sum()
                sb[k] += (wgt * lag).sum()
                sab[k] += (wgt * cur * lag).sum()
                sa2[k] += (wgt * cur * cur).sum()
                sb2[k] += (wgt * lag * lag).sum()
        cors = {}
        for k, tname in ((0, "t0"), (1, "t1")):
            ma, mb = sa[k] / sw[k], sb[k] / sw[k]
            cov = sab[k] / sw[k] - ma * mb
            va = sa2[k] / sw[k] - ma * ma
            vb = sb2[k] / sw[k] - mb * mb
            cors[tname] = float(cov / np.sqrt(va * vb))
        res[str(label)] = cors
    return res


results = {"baselines": {}, "recal": {}, "lagged_residual_corr": {}, "delays": DELAYS,
           "half_lives": HALF_LIVES, "kappa_frac": KAPPA_FRAC}
for pname, pred in PREDS.items():
    S0 = moments_of(pred)
    results["baselines"][pname] = scorer.pooled(S0)
    print(pname, "baseline", results["baselines"][pname], flush=True)
    results["recal"][pname] = {}
    for L in DELAYS:
        for hl in HALF_LIVES:
            for mode in ("affine", "scale"):
                key = f"L{L}_hl{hl}_{mode}"
                out_pred = run(pred, L, hl, mode)
                S = moments_of(out_pred)
                pooled = scorer.pooled(S)
                bs = paired_component_bootstrap(S, S0, COMPS, n_boot=2000, seed=SEED, centered=True)
                per_class = {}
                for cls in (0, 1, 2, 3):
                    idx = np.where(MC == cls)[0]
                    per_class[str(cls)] = scorer.pooled(S[idx])["wp"] - scorer.pooled(S0[idx])["wp"]
                delta = pooled["wp"] - results["baselines"][pname]["wp"]
                results["recal"][pname][key] = {"pooled": pooled, "delta_wp": delta,
                                                "bootstrap_vs_uncorrected": bs,
                                                "per_class_delta_wp": per_class}
                print(pname, key, f"wp={pooled['wp']:.7f} delta={delta:+.6f}",
                      f"ci={[round(x, 6) for x in bs['ci95']]}", f"t={time.time()-t_start:.0f}s", flush=True)
                del out_pred, S

results["lagged_residual_corr"] = {str(L): lagged_residual_corr(S4, L) for L in LAG_GRID}
for L in LAG_GRID:
    print("lagged r corr L", L, results["lagged_residual_corr"][str(L)], flush=True)

(OUT / "H281_recalibration_20261008.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", f"t={time.time()-t_start:.0f}s", flush=True)
