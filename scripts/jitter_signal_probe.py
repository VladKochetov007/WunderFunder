"""Does the within-level jitter (value minus coarse-level centre) carry label information? Weighted corr with y per column, vs the level index itself."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    book = np.load(ROOT / "cache/snap_codebook_v1.npz")
    pv = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as c:
        groups, Y, M = c["row_group"], c["y"], c["mask"]
    rng = np.random.default_rng(5)
    pick = rng.choice(len(groups), 40, replace=False)
    J = {j: [] for j in (int(k[1:]) for k in book)}; L = {j: [] for j in J}; W = []; T = []
    for i in pick:
        t = pv.read_row_group(int(groups[i]), columns=list(FEATURE_COLUMNS), use_threads=False)
        m = M[i]; y = np.clip(Y[i], -2, 2)
        W.append(np.abs(y)[m].sum(-1)); T.append((y[m] * np.sign(y[m]).sum(-1, keepdims=True) if False else y[m].sum(-1)))
        for j in J:
            v = t[FEATURE_COLUMNS[j]].to_numpy().astype(np.float64)
            cen = book[f"c{j}"]
            mids = (cen[1:] + cen[:-1]) / 2
            k = np.searchsorted(mids, v)
            J[j].append((v - cen[k])[m]); L[j].append(k[m].astype(float))
    W, T = np.concatenate(W), np.concatenate(T)
    def wc(a, b, w):
        a = a - (w * a).sum() / w.sum(); b = b - (w * b).sum() / w.sum()
        return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30)
    rows = []
    for j in J:
        jit, lev = np.concatenate(J[j]), np.concatenate(L[j])
        rows.append((j, wc(jit, T, W), wc(lev, T, W), jit.std()))
    rows.sort(key=lambda r: -abs(r[1]))
    print("col  corr(jitter,y)  corr(level index,y)  jitter sd   (weighted by |y|)")
    for r in rows[:12]:
        print(f"{r[0]:3d} {r[1]:+.4f} {r[2]:+.4f} {r[3]:.5f}")
    print("max |corr| level index:", max(abs(r[2]) for r in rows).__round__(4), " median |corr| jitter:", float(np.median([abs(r[1]) for r in rows])).__round__(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
