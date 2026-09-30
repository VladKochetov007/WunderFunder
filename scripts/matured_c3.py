"""Deployable test of matured-label error feedback for class 3, with labels REBUILT from the data.

    .venv/bin/python -u scripts/matured_c3.py [--members ft256dH+ft256ds4H]

Class-3 book states are decoded to ask/bid tick paths (wnn.ticks.reconstruct, tables cache/class3_moves_full.pkl),
labels are rebuilt with Vlad's law (scripts/online_c3.py: F = round(mean mid over the next 4 book events)), and a
label counts as known only from the row of its 4th book event (known_at). All features at row t use rebuilt
labels with known_at <= t and the ensemble's own predictions at those rows: last matured residual and
|y|-weighted EWMAs (half-lives 50, 500 matured rows). The correction D' = D + b . features is fitted on one fold
(class 3 only) and applied to the other. Reports the rebuild accuracy, correlations with the TRUE current
residual, and the change in the GLOBAL pooled WP over all pure OOF sequences (other classes unchanged).
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
from fold_eval import ROOT, WARMUP, pooled, sums

sys.path.insert(0, str(ROOT / "scripts"))
from online_c3 import rebuilt_labels  # noqa: E402

from wnn.data import SequenceFile  # noqa: E402
from wnn.ticks import reconstruct  # noqa: E402

CLIP = 2.0


def ewma_at(values_w, w, sstar, hl):
    a = 1 - 0.5 ** (1 / hl)
    num = den = 0.0
    vn, vd = np.zeros(len(w)), np.zeros(len(w))
    for s in range(len(w)):
        num = (1 - a) * num + a * values_w[s]; den = (1 - a) * den + a * w[s]
        vn[s], vd[s] = num, den
    val = np.where(vd > 1e-9, vn / np.maximum(vd, 1e-9), 0.0)
    out = np.zeros(len(sstar)); ok = sstar >= 0
    out[ok] = val[sstar[ok]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="ft256dH+ft256ds4H")
    a = ap.parse_args()
    moves, spreads = pickle.loads((ROOT / "cache" / "class3_moves_full.pkl").read_bytes())
    tg = np.load(ROOT / "cache" / "valid_targets.npz"); rg = tg["row_group"]
    cf = np.load(ROOT / "cache" / "chain_folds.npz"); v = cf["split"] == "valid"
    info = {int(g): (int(f), bool(p)) for g, f, p in zip(cf["row_group"][v], cf["fold"][v], cf["pure"][v])}
    cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["valid"]
    va = SequenceFile(ROOT / "datasets" / "valid.parquet")
    store = {}
    ok_rows = tot_rows = 0
    for k in (0, 1):
        rows = np.array([i for i, g in enumerate(rg) if info[int(g)] == (k, True)])
        preds = [np.load(ROOT / "cache" / f"pred_{r}_f{k}_valid.npy", mmap_mode="r")[rows] for r in a.members.split("+")]
        P = sum(np.asarray(p, dtype=np.float64) for p in preds) / len(preds)
        recs = []
        for j, i in enumerate(rows):
            y = np.clip(tg["y"][i].astype(np.float64), -CLIP, CLIP)
            p = np.clip(P[j], -CLIP, CLIP)
            m = tg["mask"][i].copy(); m[:WARMUP] = False
            X = np.zeros((len(y), 3))
            if cls[rg[i]] == 3:
                x = va.read(int(rg[i]), targets=False, mask=False).features
                ask, bid, _ = reconstruct(x, moves, spreads)
                lab, known = rebuilt_labels(ask, bid)
                lab = np.clip(lab, -CLIP, CLIP)  # law() already returns target units
                good = known < len(y)
                ok_rows += int((np.abs(lab[good] - y[good]).max(1) < 1e-3).sum()); tot_rows += int(good.sum())
                Dt = (lab[:, 0] - lab[:, 1]) / 2
                Dh = (p[:, 0] - p[:, 1]) / 2
                w = np.abs(lab).sum(1)
                res = np.where(good, Dt - Dh, 0.0); w = np.where(good, w, 0.0)
                order = np.argsort(known, kind="stable")  # rows by the time their label matures
                known_sorted = known[order]
                idx = np.searchsorted(known_sorted, np.arange(len(y)), side="right") - 1  # last matured in maturity order
                sstar = np.where(idx >= 0, order[np.maximum(idx, 0)], -1)
                X[:, 0] = np.where(sstar >= 0, res[np.maximum(sstar, 0)], 0.0)
                # EWMAs over rows in maturity order
                rw, ww = (res * w)[order], w[order]
                for col, hl in ((1, 50), (2, 500)):
                    vals = ewma_at(rw, ww, np.arange(len(order)), hl)
                    X[:, col] = np.where(idx >= 0, vals[np.maximum(idx, 0)], 0.0)
            recs.append((m, y, p, X, int(cls[rg[i]])))
        store[k] = recs
        print(f"fold {k}: {len(rows)} sequences loaded", flush=True)
    print(f"rebuilt label accuracy on known class-3 rows: {ok_rows / max(tot_rows, 1):.4f}", flush=True)

    def wcorr(a_, b_, w_):
        ma, mb = (w_ * a_).sum() / w_.sum(), (w_ * b_).sum() / w_.sum()
        return (w_ * (a_ - ma) * (b_ - mb)).sum() / np.sqrt((w_ * (a_ - ma) ** 2).sum() * (w_ * (b_ - mb) ** 2).sum())

    fits = {}
    for k in (0, 1):
        c3 = [r for r in store[k] if r[4] == 3]
        R = np.concatenate([((r[1][:, 0] - r[1][:, 1]) / 2 - (r[2][:, 0] - r[2][:, 1]) / 2)[r[0]] for r in c3])
        W = np.concatenate([np.abs(r[1]).sum(1)[r[0]] for r in c3])
        X = np.concatenate([r[3][r[0]] for r in c3])
        print(f"fold {k} class 3 (rebuilt labels): corr with true residual  last {wcorr(X[:, 0], R, W):+.3f}  "
              f"ewma50 {wcorr(X[:, 1], R, W):+.3f}  ewma500 {wcorr(X[:, 2], R, W):+.3f}", flush=True)
        fits[k] = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-3 * np.eye(3), X.T @ (W * R))
    for k in (0, 1):
        beta = fits[1 - k]
        base = corr = 0
        for m, y, p, X, c_ in store[k]:
            q = p.copy()
            if c_ == 3:
                d = X @ beta; q[:, 0] += d; q[:, 1] -= d
            base = base + sums(y[None], p[None], m[None]).sum(0)
            corr = corr + sums(y[None], q[None], m[None]).sum(0)
        print(f"held-out fold {k}: GLOBAL pooled WP {pooled(base):.5f} -> {pooled(corr):.5f} ({pooled(corr) - pooled(base):+.5f}), "
              f"beta {np.round(beta, 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
