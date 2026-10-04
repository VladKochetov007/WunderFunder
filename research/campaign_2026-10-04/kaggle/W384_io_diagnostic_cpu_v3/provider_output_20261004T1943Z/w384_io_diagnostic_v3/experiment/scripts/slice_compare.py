"""Where does model A gain over B? Global WP per rows-to-next-i0-event bucket (x class group).

    scripts/limited .venv/bin/python scripts/slice_compare.py SPEC_A SPEC_B [--lam 0.02] [--folds 0,1]
SPEC is a run name or a blend 'run:coef,run:coef' (coef multiplies cache/pred_<run>_valid.npy;
use baseline:0.0875 for 0.35 x baseline x 0.25). --folds restricts to held-out chain folds, e.g.
for fold models pass A as 'ab39_ftm_f{k}' with {k} expanded per fold. Needs cache/rte_valid.npz.
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

EDGES = (1, 3, 9, 33, 129, 10**6)
NAMES = ("1-2", "3-8", "9-32", "33-128", "129+")


def parse(spec, k):
    parts = []
    for item in spec.replace("{k}", str(k)).split(","):
        name, _, coef = item.partition(":")
        parts.append((np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r"), float(coef or 1.0)))
    return parts


def bucket_sums(parts, idx, y, m, rte, lam):
    """(len(idx), n_buckets, 2 targets, 6) sums over required rows, unscored rows weighted by lam."""
    out = np.zeros((len(idx), len(NAMES), 2, 6))
    req = np.arange(y.shape[1]) >= decide.WARMUP
    for a in range(0, len(idx), 64):
        sl = idx[a:a + 64]
        p = sum(c * np.asarray(arr[sl], dtype=np.float64) for arr, c in parts)
        yc, pc = np.clip(y[sl].astype(np.float64), -2, 2), np.clip(p, -2, 2)
        sel = (m[sl] + lam * (~m[sl])) * req
        b = np.digitize(rte[sl], EDGES) - 1
        for bi in range(len(NAMES)):
            s = sel * (b == bi)
            for t in range(2):
                out[a:a + 64, bi, t] = decide._sums6(yc[..., t], pc[..., t], s)
    return out


def wp(s):
    return float(np.mean([wp_from_sums(s[..., t, :].reshape(-1, 6).sum(0)) for t in range(2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--folds", default="")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    y, m, cls = d["y"], d["mask"], d["market_class"]
    rte = np.load(ROOT / "cache" / "rte_valid.npz")["to_next"]
    chain, fold = decide._chains()
    folds = [int(v) for v in args.folds.split(",")] if args.folds else [None]
    SA, SB, keep = [], [], []
    for k in folds:
        idx = np.arange(len(y)) if k is None else np.flatnonzero(fold == k)
        SA.append(bucket_sums(parse(args.a, k), idx, y, m, rte, args.lam))
        SB.append(bucket_sums(parse(args.b, k), idx, y, m, rte, args.lam))
        keep.append(idx)
    idx = np.concatenate(keep)
    sa, sb = np.concatenate(SA), np.concatenate(SB)
    ch = chain[idx]
    uc = np.unique(ch)
    pos = np.searchsorted(uc, ch)
    rng = np.random.default_rng(0)
    ws = [np.bincount(np.searchsorted(uc, rng.choice(uc, len(uc))), minlength=len(uc))[pos] for _ in range(args.n_boot)]
    groups = {"all": np.ones(len(idx), bool), "cls0-2": cls[idx] < 3, "cls3": cls[idx] == 3}
    tot_w = sa[..., 0].sum()
    print(f"A={args.a}  B={args.b}  lam={args.lam}  folds={args.folds or 'all'}  sequences={len(idx)}")
    print("overall delta", f"{wp(sa) - wp(sb):+.4f}", f"(A {wp(sa):.4f}, B {wp(sb):.4f})")
    for g, gs in groups.items():
        cells = []
        for bi, bn in enumerate(NAMES):
            xa, xb = sa[gs][:, bi], sb[gs][:, bi]
            dlt = wp(xa) - wp(xb)
            bs = [wp(xa * w[gs][:, None, None]) - wp(xb * w[gs][:, None, None]) for w in ws]
            share = xa[..., 0].sum() / tot_w
            cells.append(f"{bn} w{share:.2f} wpB {wp(xb):.3f} d {dlt:+.4f}±{np.std(bs):.4f}")
        print(f"  {g:7s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
