"""BNP-style de-obfuscation check: linear dependencies and effective rank among the 112 columns (levels and update-row deltas)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    xs = []
    for g in (3, 400, 900, 1500):
        t = pf.read_row_group(g, columns=list(FEATURE_COLUMNS), use_threads=False)
        xs.append(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)[:6000])
    for name, mats in (("levels", xs), ("deltas", [np.diff(x, axis=0) for x in xs])):
        X = np.vstack(mats)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        s = np.linalg.svd(X, compute_uv=False)
        ev = s**2 / (s**2).sum()
        c = np.cumsum(ev)
        print(f"{name}: components for 90/99/99.9% variance: {[int(np.searchsorted(c, q)) + 1 for q in (0.9, 0.99, 0.999)]}, smallest singular values {np.round(s[-4:] / s[0], 6).tolist()}")
        C = np.corrcoef(X.T)
        np.fill_diagonal(C, 0)
        # leave-one-column-out R^2 via the inverse correlation matrix diagonal
        P = np.linalg.pinv(np.corrcoef(X.T), rcond=1e-10)
        r2 = 1 - 1 / np.diag(P)
        print(f"  columns with R^2 from all others > 0.999: {int((r2 > 0.999).sum())}; > 0.99: {int((r2 > 0.99).sum())}; > 0.9: {int((r2 > 0.9).sum())}; median {np.median(r2):.3f}")
        print("  max |corr| pairs:", sorted(((round(float(abs(C[i, j])), 4), i, j) for i in range(112) for j in range(i + 1, 112) if abs(C[i, j]) > 0.98), reverse=True)[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
