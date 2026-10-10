"""Are slow-clock block increments partly explained by fast-block increments since the previous slow write? (round-3 idea, label-free)

Columns are grouped into blocks by identical change masks. For every block with an update rate below 20%, the increment of each
column at its update rows is regressed (ridge, sequence-grouped CV) on the summed fast-block (rate 25-35%) increments accumulated
since the previous write of that slow block. Reports out-of-sequence R^2 per slow block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    rng = np.random.default_rng(0)
    picks = rng.choice(pf.num_row_groups, 40, replace=False)
    seqs = []
    for g in picks:
        t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
        seqs.append(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[:8000])
    ch = np.concatenate([np.diff(s, axis=0) != 0 for s in seqs]).mean(0)
    sig = {}
    for j in range(len(FEATURE_COLUMNS)):
        key = tuple(np.flatnonzero(np.diff(seqs[0][:, j]) != 0)[:40])
        sig.setdefault(key, []).append(j)
    blocks = [v for v in sig.values() if len(v) >= 2]
    rate = {tuple(b): float(ch[b].mean()) for b in blocks}
    fast = [b for b in blocks if 0.25 < ch[b].mean() < 0.35]
    slow = [b for b in blocks if ch[b].mean() < 0.2]
    print("blocks:", sorted((round(ch[b].mean(), 2), len(b)) for b in blocks))
    fcols = [j for b in fast for j in b]
    for b in slow:
        X, Y, G = [], [], []
        for si, s in enumerate(seqs):
            d = np.diff(s, axis=0)
            upd = np.flatnonzero((d[:, b] != 0).any(1))
            cf = np.cumsum(d[:, fcols], axis=0)
            prev = 0
            for u in upd[1:]:
                p = upd[np.searchsorted(upd, u) - 1]
                X.append(cf[u] - cf[p]); Y.append(d[u, b]); G.append(si)
        X, Y, G = np.array(X), np.array(Y), np.array(G)
        folds = G % 5
        r2 = []
        for k in range(Y.shape[1]):
            pred = np.zeros(len(Y))
            for f in range(5):
                m = Ridge(alpha=1.0).fit(X[folds != f], Y[folds != f, k])
                pred[folds == f] = m.predict(X[folds == f])
            r2.append(1 - ((Y[:, k] - pred) ** 2).sum() / ((Y[:, k] - Y[:, k].mean()) ** 2).sum())
        print(f"slow block rate {ch[b].mean():.2f} ({len(b)} cols, {len(Y)} updates): median R2 {np.median(r2):.3f} max {np.max(r2):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
