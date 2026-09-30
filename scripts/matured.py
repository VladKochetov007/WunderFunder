"""Is the ensemble's error on rows whose label has already MATURED predictive of its current error?

    .venv/bin/python -u scripts/matured.py [--members ft256dH+ft256ds4H] [--lag-events 4]

Label law (class 3, research findings): targets are fixed by the mid over the next 4 i0 book events. So the label
of row s is known (causally computable) once the lag-th book event after s has happened: c(s). At row t the
latest matured row is s*(t) = max{s : c(s) <= t}. Upper-bound test with TRUE labels (the rebuild is ~97.5% exact
in class 3): features at t = residual of D at s*(t), and |y|-weighted EWMAs of matured residuals (half-lives in
matured rows 50, 500). Correlation with the current residual on scored rows of pure OOF sequences, per class and
fold, and a cross-fitted (fold 0 <-> 1) linear correction D' = D + b . features: pooled WP change.
Book event = any change of the 22 i0 price codes. --lag-events 8 is a safety margin for classes 0-2 whose law
is only approximate.
"""

from __future__ import annotations

import argparse

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums

from wnn.data import SequenceFile

CLIP = 2.0


def ewma_matured(res_w, w, sstar, hl):
    """EWMA over matured rows (in row order of s) evaluated at s*(t): weighted by w."""
    a = 1 - 0.5 ** (1 / hl)
    n = len(res_w)
    num = np.zeros(n); den = np.zeros(n)
    m_num = m_den = 0.0
    for s in range(n):
        m_num = (1 - a) * m_num + a * res_w[s]
        m_den = (1 - a) * m_den + a * w[s]
        num[s], den[s] = m_num, m_den
    val = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    out = np.zeros(len(sstar))
    ok = sstar >= 0
    out[ok] = val[sstar[ok]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="ft256dH+ft256ds4H")
    ap.add_argument("--lag-events", type=int, default=4)
    a = ap.parse_args()
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    va = SequenceFile(ROOT / "datasets" / "valid.parquet")
    store = {}
    for k in (0, 1):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows] for r in a.members.split("+")]
        P = sum(np.asarray(p, dtype=np.float64) for p in preds) / len(preds)
        recs = []
        for j, i in enumerate(rows):
            x = va.read(int(rg[i]), targets=False, mask=False).features
            y = np.clip(tg["y"][i].astype(np.float64), -CLIP, CLIP)
            p = np.clip(P[j], -CLIP, CLIP)
            D, Dh = (y[:, 0] - y[:, 1]) / 2, (p[:, 0] - p[:, 1]) / 2
            w = np.abs(y).sum(1)
            res = D - Dh
            ev = np.flatnonzero(np.r_[False, (x[1:, 0:22] != x[:-1, 0:22]).any(1)])
            T = len(x)
            pos = np.searchsorted(ev, np.arange(T), side="right")
            kth = pos + a.lag_events - 1
            c = np.where(kth < len(ev), ev[np.minimum(kth, len(ev) - 1)], T + 10**6)  # maturity row of s
            sstar = np.searchsorted(c, np.arange(T), side="right") - 1  # latest s with c(s) <= t
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            f_last = np.where(sstar >= 0, res[np.maximum(sstar, 0)], 0.0)
            f50 = ewma_matured(res * w, w, sstar, 50)
            f500 = ewma_matured(res * w, w, sstar, 500)
            gap = np.where(sstar >= 0, np.arange(T) - sstar, T)
            recs.append((m, D, Dh, w, res, np.stack([f_last, f50, f500], -1), gap, int(cls[rg[i]]), y, p))
        store[k] = recs
        print(f"fold {k}: {len(rows)} sequences", flush=True)

    def wcorr(a_, b_, w_):
        ma, mb = (w_ * a_).sum() / w_.sum(), (w_ * b_).sum() / w_.sum()
        return (w_ * (a_ - ma) * (b_ - mb)).sum() / np.sqrt((w_ * (a_ - ma) ** 2).sum() * (w_ * (b_ - mb) ** 2).sum())

    names = ["last_matured", "ewma50", "ewma500"]
    for k in (0, 1):
        for c_ in (0, 1, 2, 3, "all"):
            sel = [r for r in store[k] if c_ == "all" or r[7] == c_]
            R = np.concatenate([r[4][r[0]] for r in sel]); W = np.concatenate([r[3][r[0]] for r in sel])
            X = np.concatenate([r[5][r[0]] for r in sel]); G = np.concatenate([r[6][r[0]] for r in sel])
            print(f"fold {k} class {c_}: " + " ".join(f"{n} {wcorr(X[:, j], R, W):+.3f}" for j, n in enumerate(names))
                  + f" | median rows since matured {np.median(G):.0f}", flush=True)
    # cross-fitted linear correction on D per class, pooled WP of the corrected (t0, t1)
    for k in (0, 1):
        tr, te = store[1 - k], store[k]
        betas = {}
        for c_ in range(4):
            s_ = [r for r in tr if r[7] == c_]
            X = np.concatenate([r[5][r[0]] for r in s_]); R = np.concatenate([r[4][r[0]] for r in s_])
            W = np.concatenate([r[3][r[0]] for r in s_])
            betas[c_] = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-3 * np.eye(3), X.T @ (W * R))
        base = corr = 0
        for r in te:
            m, _, _, _, _, X, _, c_, y, p = r
            dcorr = X @ betas[c_]
            q = p.copy(); q[:, 0] += dcorr; q[:, 1] -= dcorr
            base = base + sums(y[None], p[None], m[None]).sum(0)
            corr = corr + sums(y[None], q[None], m[None]).sum(0)
        print(f"held-out fold {k}: pooled WP {pooled(base):.5f} -> {pooled(corr):.5f} ({pooled(corr) - pooled(base):+.5f}); "
              f"betas {({c_: np.round(b_, 3).tolist() for c_, b_ in betas.items()})}", flush=True)


if __name__ == "__main__":
    main()
