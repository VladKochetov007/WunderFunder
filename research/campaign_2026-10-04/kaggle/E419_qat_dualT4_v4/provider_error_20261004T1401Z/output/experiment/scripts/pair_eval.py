"""Score candidate second members as blend partners of an anchor, per held-out chain fold (exact moments).

    scripts/limited .venv/bin/python scripts/pair_eval.py ANCHOR PARTNER [PARTNER ...] [--lam 0.02]
Names are fold-model prefixes (ANCHOR_f0/_f1 must exist as cache/pred_<name>_valid.npy). For each fold:
WP of the 0.5/0.5 pair, of the best weight in [0.3, 0.7], and deltas vs production (0.65 anchor +
0.35 baseline x0.25) and vs the reference pair (anchor + ab62_base_s4), with chain-bootstrap SDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import decide  # noqa: E402
from wnn.gmetric import wp_from_sums  # noqa: E402

REQ = np.arange(20000) >= decide.WARMUP


def moments(specs, idx, Y, M):
    """(n_seq, 2 [scored/unscored], 2 targets, 3 + 2K + K*K) sums for K predictors."""
    P = []
    for s in specs:
        n, _, sc = s.partition("*")
        P.append((np.load(ROOT / "cache" / f"pred_{n}_valid.npy", mmap_mode="r"), float(sc or 1)))
    K = len(P)
    S = np.zeros((len(idx), 2, 2, 3 + 2 * K + K * K))
    for a in range(0, len(idx), 16):
        sl = idx[a:a + 16]
        y = np.clip(Y[sl].astype(np.float64), -2, 2)
        m = M[sl]
        ps = [sc * np.asarray(arr[sl], np.float64) for arr, sc in P]
        for r, sel in enumerate((m & REQ, (~m) & REQ)):
            for t in range(2):
                w = np.abs(y[..., t]) * sel
                yt = y[..., t]
                pt = [p[..., t] for p in ps]
                v = [w.sum(-1), (w * yt).sum(-1), (w * yt * yt).sum(-1)] + [(w * q).sum(-1) for q in pt] + \
                    [(w * yt * q).sum(-1) for q in pt] + [(w * q1 * q2).sum(-1) for q1 in pt for q2 in pt]
                S[a:a + 16, r, t] = np.stack(v, -1)
    return S, K


def wp(S, c, K, lam):
    t = (S[:, 0] + lam * S[:, 1]).sum(0)
    c = np.asarray(c, float)
    out = []
    for j in range(2):
        m = t[j]
        mp, myp = m[3:3 + K] @ c, m[3 + K:3 + 2 * K] @ c
        mpp = c @ m[3 + 2 * K:].reshape(K, K) @ c
        out.append(float(wp_from_sums(np.array([m[0], m[1], mp, m[2], mpp, myp]))))
    return 0.5 * sum(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anchor")
    ap.add_argument("partners", nargs="+")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--ref-partner", default="ab62_base_s4")
    a = ap.parse_args()
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    Y, M = d["y"], d["mask"]
    chain, fold = decide._chains()
    for k in (0, 1):
        idx = np.flatnonzero(fold == k)
        rng = np.random.default_rng(0)
        boots = [decide._resample(np.arange(len(idx)), rng, chain[idx]) for _ in range(200)]
        base = [f"{a.anchor}_f{k}", "baseline*0.25", f"{a.ref_partner}_f{k}"]
        for p in a.partners:
            S, K = moments(base + [f"{p}_f{k}"], idx, Y, M)
            prod, refpair = [0.65, 0.35, 0, 0], [0.5, 0, 0.5, 0]
            ws = np.linspace(0.3, 0.7, 9)
            best_w = max(ws, key=lambda w: wp(S, [w, 0, 0, 1 - w], K, a.lam))
            pair = [0.5, 0, 0, 0.5]
            f = lambda c, r: (wp(S, c, K, a.lam) - wp(S, r, K, a.lam),  # noqa: E731
                              np.std([wp(S[b], c, K, a.lam) - wp(S[b], r, K, a.lam) for b in boots]))
            d1, s1 = f(pair, prod)
            d2, s2 = f(pair, refpair)
            print(f"fold {k} partner {p:22s} pair WP {wp(S, pair, K, a.lam):.5f} (best w {best_w:.2f}: {wp(S, [best_w, 0, 0, 1 - best_w], K, a.lam):.5f})"
                  f" | vs prod {d1:+.5f}+-{s1:.5f} | vs anchor+{a.ref_partner} {d2:+.5f}+-{s2:.5f}", flush=True)


if __name__ == "__main__":
    main()
