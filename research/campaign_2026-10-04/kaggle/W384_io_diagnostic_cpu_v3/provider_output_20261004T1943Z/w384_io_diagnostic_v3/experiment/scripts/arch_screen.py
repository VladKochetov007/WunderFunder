"""Score architecture-screen arms against a reference on one held-out chain fold.

    scripts/limited .venv/bin/python scripts/arch_screen.py REF ARM [ARM ...] [--fold 0]
For each arm: global WP at lambda 0 and 0.02 on the fold, delta vs REF with a chain bootstrap,
residual correlation with REF (1 - diversity), and the 50/50 blend gain over REF alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import decide  # noqa: E402
from decide import build_sums, load_sums, wp  # noqa: E402


def residual_corr(a, b, idx, y, m):
    """|y|-weighted correlation of the two models' residuals on scored rows (both targets pooled)."""
    num = da = db = 0.0
    for i in idx[::2]:
        sel = m[i]
        yy = np.clip(y[i][sel].astype(np.float64), -2, 2)
        w = np.abs(yy)
        ra = np.clip(np.asarray(a[i], np.float64)[sel], -2, 2) - yy
        rb = np.clip(np.asarray(b[i], np.float64)[sel], -2, 2) - yy
        num += (w * ra * rb).sum()
        da += (w * ra * ra).sum()
        db += (w * rb * rb).sum()
    return num / np.sqrt(da * db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    y, m = d["y"], d["mask"]
    chain, fold = decide._chains()
    idx = np.flatnonzero(fold == args.fold)
    ch = chain[idx]
    uc = np.unique(ch)
    pos = np.searchsorted(uc, ch)
    rng = np.random.default_rng(0)
    ws = [np.bincount(np.searchsorted(uc, rng.choice(uc, len(uc))), minlength=len(uc))[pos].astype(float) for _ in range(200)]
    sr = load_sums(args.ref)[idx]
    pr = np.load(ROOT / "cache" / f"pred_{args.ref}_valid.npy", mmap_mode="r")
    print(f"ref {args.ref}: fold {args.fold} WP lam0 {wp(sr):.4f} lam0.02 {wp(sr, 0.02):.4f}")
    for arm in args.arms:
        sa = load_sums(arm)[idx]
        sb = build_sums(None, [(arm, 0.5), (args.ref, 0.5)])[idx]
        pa = np.load(ROOT / "cache" / f"pred_{arm}_valid.npy", mmap_mode="r")
        cells = []
        for lam in (0.0, 0.02):
            dlt = wp(sa, lam) - wp(sr, lam)
            sd = np.std([wp(sa * w[:, None, None, None], lam) - wp(sr * w[:, None, None, None], lam) for w in ws])
            bl = wp(sb, lam) - wp(sr, lam)
            cells.append(f"lam{lam}: {wp(sa, lam):.4f} d {dlt:+.4f}±{sd:.4f} blend50 {bl:+.4f}")
        rc = residual_corr(pa, pr, idx, y, m)
        print(f"{arm:24s} " + " | ".join(cells) + f" | resid corr {rc:.4f}")


if __name__ == "__main__":
    main()
