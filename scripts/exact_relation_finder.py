"""BNP-style: find sparse exact linear relations among the 112 columns (orthogonal matching pursuit on raw, unscaled levels)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import OrthogonalMatchingPursuit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    X = np.vstack([np.column_stack([pf.read_row_group(g, columns=list(FEATURE_COLUMNS), use_threads=False)[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[:6000] for g in (3, 400, 900, 1500)])
    sd = X.std(0)
    found = []
    for j in range(112):
        others = [k for k in range(112) if k != j]
        for nnz in (1, 2, 3, 4):
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=nnz, fit_intercept=True).fit(X[:, others], X[:, j])
            res = X[:, j] - omp.predict(X[:, others])
            rel = float(res.std() / (sd[j] + 1e-12))
            if rel < 1e-3 or nnz == 4:
                idx = np.flatnonzero(omp.coef_)
                found.append((j, nnz, rel, [(others[i], round(float(omp.coef_[i]), 4)) for i in idx]))
                break
    exact = [f for f in found if f[2] < 1e-3]
    print(f"columns expressible by <=4 other columns with relative residual < 1e-3: {len(exact)}")
    for j, nnz, rel, terms in exact[:40]:
        print(f"  col {j} = {terms}  (n={nnz}, rel.resid {rel:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
