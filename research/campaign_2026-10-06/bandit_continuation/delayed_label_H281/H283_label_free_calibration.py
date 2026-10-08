#!/usr/bin/env python3
"""H-283: label-free causal scale calibration on the fold0 468 panel.

Causality proof (serving). Features at row t are built from rows <= t only:
- input features: EW(|x_s|) and EW(x_s), EW(x_s^2) over s <= t, zero initial state
  per sequence (scipy lfilter), so they use inputs at rows 0..t, which are available
  when predict() is called at row t;
- own-prediction features use the base predictions at rows s <= t-1 (shifted by one);
- step_in_seq is the row index itself.
Calibrator outputs at row t are a function of (features_t, base prediction_t) and
frozen parameters. No label enters any feature. Labels are used only at fit time, on
training folds of the 141-component split (seed 271828), and parameters are frozen
before scoring the held-out fold. The market-class variant (c) uses a causal feature
bin, not the label-derived market_class.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.signal import lfilter

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMP = ROOT / "research/campaign_2026-10-06/bandit_continuation"
E529 = CAMP / "continuous_execution_E529"
OUT = CAMP / "delayed_label_H281"
SEED = 271828
N_FOLDS = 5
CLIP = 2.0
WARMUP = 99
HALF_LIVES = [200, 1000, 5000]
FEAT_COLS = ["i0_dp0", "i0_dp1", "i0_dv0", "a0"]
PROXY_IDX = None  # set below: EW abs-mean of i0_dp0 at HL 1000
t_start = time.time()
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
EVAL_IDX = np.asarray(EVAL, dtype=np.int64)
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL])
UNIQ = np.unique(COMPS)
perm = np.random.default_rng(SEED).permutation(len(UNIQ))
fold_of_comp = {int(UNIQ[p]): i % N_FOLDS for i, p in enumerate(perm)}
FOLD_SEQ = np.asarray([fold_of_comp[int(c)] for c in COMPS])

targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
Y = np.asarray(targets["y"][EVAL_IDX], dtype=np.float64)
MASK = np.asarray(targets["mask"][EVAL_IDX], dtype=bool) & (np.arange(20000) >= WARMUP)[None, :]

S4 = np.asarray(np.load(E529 / "exports/H272_PARENT_V2_CLEANFOLD_VALID/hidden_only.npy", mmap_mode="r")[EVAL_IDX], dtype=np.float64)
D272 = np.asarray(np.load(E529 / "exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/pred_multiplicative.npy", mmap_mode="r")[EVAL_IDX], dtype=np.float64)
BASES = {"H272S4_hidden_only": S4.astype(np.float32), "equal_pair": (0.5 * S4 + 0.5 * D272).astype(np.float32)}
del D272, S4

pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
# Input-only columns; labels t0/t1 are never read in this script.
input_cols = FEAT_COLS
SEL_I, SEL_T = np.nonzero(MASK)
N = len(SEL_I)
seq_rows = np.searchsorted(SEL_I, np.arange(468), side="left")
seq_end = np.searchsorted(SEL_I, np.arange(468), side="right")


def ew(v, lam):
    return lfilter([1.0], [1.0, -lam], v)


def input_features(cols):
    """cols (T, 4): causal EW abs-mean and std per column and half-life -> (T, 24)."""
    out = []
    for hl in HALF_LIVES:
        lam = 0.5 ** (1.0 / hl)
        for j in range(cols.shape[1]):
            v = cols[:, j]
            m1 = ew(np.abs(v), lam)
            m2 = ew(v, lam)
            m3 = ew(v * v, lam)
            out.append(m1)
            out.append(np.sqrt(np.maximum(m3 - m2 * m2, 0.0)))
    return np.stack(out, axis=1)


def own_features(x):
    """Causal stats of the base prediction over rows <= t-1: EW mean and std -> (T, 6)."""
    u = np.concatenate([[0.0], x[:-1]])
    out = []
    for hl in HALF_LIVES:
        lam = 0.5 ** (1.0 / hl)
        m1 = ew(u, lam)
        m2 = ew(u * u, lam)
        out.append(m1)
        out.append(np.sqrt(np.maximum(m2 - m1 * m1, 0.0)))
    return np.stack(out, axis=1)


# Read the four input columns once per evaluation sequence (no label columns).
INPUT_SEL = np.zeros((N, 24), dtype=np.float32)
for i, g in enumerate(EVAL):
    tbl = pf.read_row_group(g, columns=input_cols)
    cols = np.column_stack([tbl.column(c).to_numpy().astype(np.float64) for c in input_cols])
    rows_i = np.nonzero(MASK[i])[0]
    a, b = seq_rows[i], seq_end[i]
    INPUT_SEL[a:b] = input_features(cols)[rows_i].astype(np.float32)
    del tbl, cols
print("input features done", f"t={time.time()-t_start:.0f}s", flush=True)
# proxy index: abs-mean of i0_dp0 at HL 1000 (block order: hl, col, [mean_abs, std])
PROXY_IDX = HALF_LIVES.index(1000) * 8 + 0 * 2


def build_Z(base, k):
    """Feature matrix at the selected (scored) rows for target k: (N, 24+6+1)."""
    Z = np.zeros((N, 31), dtype=np.float32)
    for i in range(468):
        a, b = seq_rows[i], seq_end[i]
        if b <= a:
            continue
        rows_t = SEL_T[a:b]
        own = own_features(base[i, :, k]).astype(np.float32)
        Z[a:b, :24] = INPUT_SEL[a:b]
        Z[a:b, 24:30] = own[rows_t]
        Z[a:b, 30] = rows_t / 20000.0
    return Z


def standardize(Z, rows):
    mu = Z[rows].mean(0)
    sd = Z[rows].std(0) + 1e-8
    return (Z - mu) / sd


def fit_mult(Zs, x, y, w, rs):
    idx = rs.choice(len(x), size=min(300_000, len(x)), replace=False)
    Zi, xi, yi, wi = Zs[idx], x[idx], y[idx], w[idx]
    wn = wi / wi.sum()
    lam = 1e-4

    def loss(beta):
        e = np.exp(np.clip(Zi @ beta, -5, 5))
        resid = xi * e - yi
        val = np.sum(wn * resid * resid) + lam * beta @ beta
        grad = 2 * (Zi * (wn * resid * xi * e)[:, None]).sum(0) + 2 * lam * beta
        return val, grad

    res = minimize(loss, np.zeros(Zs.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return res.x


def apply_mult(Zs, x, beta):
    return x * np.exp(np.clip(Zs @ beta, -5, 5))


def fit_add(Zs, x, y, w):
    A = np.column_stack([np.ones(len(x)), Zs])
    lam = 1e-3 * w.sum()
    G = A.T @ (A * w[:, None]) + lam * np.eye(A.shape[1])
    v = np.linalg.solve(G, A.T @ (w * (y - x)))
    return v


def apply_add(Zs, x, v):
    return x + v[0] + Zs @ v[1:]


def fit_bin(raw_proxy, x, y, w, edges):
    bins = np.digitize(raw_proxy, edges)
    params = {}
    for b in range(len(edges) + 1):
        m = bins == b
        if m.sum() == 0:
            params[b] = np.array([0.0, 1.0])
            continue
        A = np.column_stack([np.ones(m.sum()), x[m]])
        ww = w[m]
        lam = 1e-3 * ww.sum() + 1e-12
        theta0 = np.array([0.0, 1.0])
        G = A.T @ (A * ww[:, None]) + lam * np.eye(2)
        rhs = A.T @ (ww * y[m]) + lam * theta0
        params[b] = np.linalg.solve(G, rhs)
    return params


def apply_bin(raw_proxy, x, params, edges):
    bins = np.digitize(raw_proxy, edges)
    out = np.empty_like(x)
    for b in np.unique(bins):
        m = bins == b
        th = params.get(b, np.array([0.0, 1.0]))
        out[m] = th[0] + th[1] * x[m]
    return out


def score_full(base, pred_sel_k):
    """pred_sel_k: (N, 2) calibrated predictions at selected rows; others = base."""
    full = base.copy()
    full[SEL_I, SEL_T, :] = pred_sel_k
    S = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S[i, k] = scorer.moments(Y[i, :, k], full[i, :, k], MASK[i])
    return S, full


results = {"folds": {f: int(np.sum(FOLD_SEQ == f)) for f in range(N_FOLDS)}, "bases": {}}
rs = np.random.default_rng(SEED)
for bname, base in BASES.items():
    t_b = time.time()
    Y_sel = np.clip(Y[SEL_I, SEL_T, :], -CLIP, CLIP)
    X_sel = base[SEL_I, SEL_T, :]
    W_sel = np.abs(Y_sel)
    Z_k = [build_Z(base, k) for k in range(2)]
    S_base = np.zeros((468, 2, 6))
    for i in range(468):
        for k in range(2):
            S_base[i, k] = scorer.moments(Y[i, :, k], base[i, :, k], MASK[i])
    base_wp = scorer.pooled(S_base)
    variants = {}
    raw_proxy = Z_k[0][:, PROXY_IDX]  # causal feature for target-0 variant; per-target below
    out_cal = {m: np.zeros((N, 2)) for m in ("mult", "add", "bin", "mult_rowshuffle")}
    for f in range(N_FOLDS):
        train = FOLD_SEQ[SEL_I] != f
        test = ~train
        for k in range(2):
            Zs = standardize(Z_k[k], train)
            x, y, w = X_sel[:, k], Y_sel[:, k], W_sel[:, k]
            beta = fit_mult(Zs[train], x[train], y[train], w[train], rs)
            out_cal["mult"][test, k] = apply_mult(Zs[test], x[test], beta)
            v = fit_add(Zs[train], x[train], y[train], w[train])
            out_cal["add"][test, k] = apply_add(Zs[test], x[test], v)
            rp = Z_k[k][:, PROXY_IDX]
            edges = np.quantile(rp[train], [0.25, 0.5, 0.75])
            params = fit_bin(rp[train], x[train], y[train], w[train], edges)
            out_cal["bin"][test, k] = apply_bin(rp[test], x[test], params, edges)
            # control: feature rows shuffled across selected rows (breaks feature-target link)
            pidx = np.where(test)[0]
            pidx_shuf = rs.permutation(pidx)
            out_cal["mult_rowshuffle"][test, k] = apply_mult(Zs[pidx_shuf], x[test], beta)
    for m in ("mult", "add", "bin", "mult_rowshuffle"):
        S, _ = score_full(base, out_cal[m])
        pooled = scorer.pooled(S)
        bs = paired_component_bootstrap(S, S_base, COMPS, n_boot=2000, seed=SEED, centered=True)
        variants[m] = {"pooled": pooled, "delta_wp": pooled["wp"] - base_wp["wp"], "bootstrap": bs}
        print(bname, m, f"wp={pooled['wp']:.7f} delta={pooled['wp']-base_wp['wp']:+.6f}",
              f"ci={[round(x, 6) for x in bs['ci95']]}", flush=True)
    results["bases"][bname] = {"base": base_wp, "variants": variants}
    print(bname, "done", f"t={time.time()-t_b:.0f}s", flush=True)
    del Z_k, out_cal

(OUT / "H283_label_free_calibration_20261008.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", f"t={time.time()-t_start:.0f}s", flush=True)
