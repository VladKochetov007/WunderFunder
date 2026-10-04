"""Q2: causal piecewise readouts of the pooled OOF panel (segments reset at E rows, rows <= t only).

Fixed readouts: hold (p at segment start), running mean since segment start, segment-reset EMAs, convex blends.
Cross-fitted readouts: |y|-weighted least squares of y on causal path features, fitted per (class, elapsed-row bucket,
target) or per (class, target) on the three folds disjoint from the scored fold. Control "L_p" refits only the scale
of p per (class, bucket), so the difference to it isolates the value of the extra path features.
"""

from __future__ import annotations

import json

import numpy as np

from cr_dtw_readout_core import FEATS, NB_BUCKET, NF, OUT, accumulate_stats, apply_models, load_panel, summarize

FI = {n: i for i, n in enumerate(FEATS)}


def fixed(weights: dict[str, float]) -> np.ndarray:
    v = np.zeros(NF)
    for n, w in weights.items():
        v[FI[n]] = w
    return np.broadcast_to(v, (4, 4, NB_BUCKET, 2, NF)).copy()


def fitted(XtX, Xty, feats, pool: str, ridge: float = 1e-6) -> np.ndarray:
    """Cross-fitted coefficients (nfold, 4, NB_BUCKET, 2, NF); pool 'cb' = per class x bucket, 'c' = per class."""
    idx = [FI[n] for n in feats]
    nfold = XtX.shape[0]
    coef = np.zeros((nfold, 4, NB_BUCKET, 2, NF))
    for f in range(nfold):
        A = XtX[np.arange(nfold) != f].sum(0)
        b = Xty[np.arange(nfold) != f].sum(0)
        for c in range(4):
            for j in range(2):
                if pool == "c":
                    Ac, bc = A[c, :, j].sum(0), b[c, :, j].sum(0)
                for k in range(NB_BUCKET):
                    if pool == "cb":
                        Ac, bc = A[c, k, j], b[c, k, j]
                    S = Ac[np.ix_(idx, idx)]
                    if np.trace(S) <= 0:
                        coef[f, c, k, j, FI["p"]] = 1.0
                        continue
                    sol = np.linalg.solve(S + ridge * np.trace(S) / len(idx) * np.eye(len(idx)), bc[idx])
                    coef[f, c, k, j, idx] = sol
    return coef


def main():
    y, mask, ft, par, E, fold, cls = load_panel()
    XtX, Xty = accumulate_stats(y, mask, ft, par, E, fold, cls)
    np.savez(OUT / "q2_normal_equations.npz", XtX=XtX, Xty=Xty, feats=np.array(FEATS))
    models: dict[str, np.ndarray] = {
        "base_ft": fixed({"p": 1}),
        "base_par": fixed({"par": 1}),
        "base_mix": fixed({"p": 0.5, "par": 0.5}),
        "hold": fixed({"hold": 1}),
        "runmean": fixed({"runmean": 1}),
        "sema.5": fixed({"sema.5": 1}),
        "sema.2": fixed({"sema.2": 1}),
        "sema.05": fixed({"sema.05": 1}),
        "sema.01(jump+drift)": fixed({"sema.01": 1}),
        "0.8p+0.2runmean": fixed({"p": 0.8, "runmean": 0.2}),
        "0.5p+0.5runmean": fixed({"p": 0.5, "runmean": 0.5}),
        "0.9p+0.1hold": fixed({"p": 0.9, "hold": 0.1}),
        "1.1p-0.1lag1": fixed({"p": 1.1, "p_lag1": -0.1}),
        "1.1p-0.1runmean": fixed({"p": 1.1, "runmean": -0.1}),
        "mix_runmean": fixed({"runmean": 0.5, "par_runmean": 0.5}),
    }
    specs = {
        "L_p|cb": (["p"], "cb"),
        "L_p,runmean|cb": (["p", "runmean"], "cb"),
        "L_p,runmean|c": (["p", "runmean"], "c"),
        "L_p,hold|cb": (["p", "hold"], "cb"),
        "L_p,semas|cb": (["p", "sema.5", "sema.2", "sema.05", "sema.01"], "cb"),
        "L_p,lag1,lag8|cb": (["p", "p_lag1", "p_lag8"], "cb"),
        "L_p,lag1,lag8|c": (["p", "p_lag1", "p_lag8"], "c"),
        "L_ftpath|cb": (FEATS[:9], "cb"),
        "L_p,par|cb": (["p", "par"], "cb"),
        "L_p,par|c": (["p", "par"], "c"),
        "L_all|cb": (FEATS[:11], "cb"),
        "L_all+1|cb": (FEATS, "cb"),
    }
    for name, (feats, pool) in specs.items():
        models[name] = fitted(XtX, Xty, feats, pool)
    names = list(models)
    coef = np.ascontiguousarray(np.stack([models[n] for n in names], 1))
    seq = apply_models(y, mask, ft, par, E, fold, cls, coef)
    np.save(OUT / "q2_seq_sums.npy", seq.astype(np.float64))
    summ = summarize(seq, fold, cls)
    res = {n: {key: float(v[m]) for key, v in summ.items()} for m, n in enumerate(names)}
    (OUT / "q2_readouts.json").write_text(json.dumps({"models": names, "wp": res}, indent=1))
    b = res["base_ft"]
    keys = ["pooled", "public"] + [f"c{c}" for c in range(4)]
    print(f"{'model':22s} " + " ".join(f"{k:>8s}" for k in keys) + "   per-fold pooled delta")
    print(f"{'base_ft (abs)':22s} " + " ".join(f"{b[k]:8.4f}" for k in keys))
    for n in names:
        r = res[n]
        print(f"{n:22s} " + " ".join(f"{r[k] - b[k]:+8.4f}" for k in keys) + "   " +
              " ".join(f"{r[f'f{k}'] - b[f'f{k}']:+.4f}" for k in range(4)))
    # coefficient snapshot of the runmean blend per class and bucket (fold-0 fit, target 0)
    m = models["L_p,runmean|cb"]
    print("L_p,runmean|cb fold0 t0 coef [p, runmean] per bucket:")
    for c in range(4):
        print(f" c{c}", " ".join(f"{m[0, c, k, 0, FI['p']]:.2f}/{m[0, c, k, 0, FI['runmean']]:.2f}" for k in range(NB_BUCKET)))
    m = models["L_p,lag1,lag8|c"]
    print("L_p,lag1,lag8|c fold0 coef [p, lag1, lag8] per class/target:")
    for c in range(4):
        print(f" c{c}", [[round(m[0, c, 0, j, FI[n]], 3) for n in ("p", "p_lag1", "p_lag8")] for j in range(2)])


if __name__ == "__main__":
    main()
