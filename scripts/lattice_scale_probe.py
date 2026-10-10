"""BNP 'denominator search': do shared-vocabulary columns sit on a common lattice v = k * step + offset (k integer)?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    X = np.vstack([np.column_stack([pf.read_row_group(g, columns=list(FEATURE_COLUMNS), use_threads=False)[c].to_numpy() for c in FEATURE_COLUMNS])[:8000].astype(np.float64) for g in range(0, 600, 12)])
    rows = []
    for j in range(112):
        u = np.unique(X[:, j])
        if len(u) < 3:
            rows.append((j, len(u), None, None)); continue
        d = np.diff(u)
        step = np.quantile(d, 0.05)
        k = d / step
        frac = np.abs(k - np.round(k))
        rows.append((j, len(u), float(step), float(np.mean(frac < 0.02))))
    print("col  n_unique  step(5% quantile gap)  share_of_gaps_integer_multiple")
    for r in rows:
        if r[2] is not None and r[1] < 20000:
            print(f"{r[0]:3d} {r[1]:8d} {r[2]:.6g} {r[3]:.3f}")
    ok = [r for r in rows if r[3] is not None and r[3] > 0.95 and r[1] < 20000]
    print("columns on a clean integer lattice (>95% of gaps are integer multiples of the step), n_unique < 20000:", len(ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
