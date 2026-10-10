"""Label-free features of the first 99 rows of each sequence (fraction of changing rows, log1p std of first differences per column)."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from utils import FEATURE_COLUMNS
from lib import S
pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
for fold in (0, 1):
    groups = np.load(S / f"flat_f{fold}.npz")["groups"]
    F = []
    for g in groups:
        t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS])[:99].astype(np.float64)
        dx = np.diff(x, axis=0)
        F.append(np.concatenate([(dx != 0).mean(0), np.log1p(np.abs(dx).std(0))]))
    np.save(S / f"warm_f{fold}.npy", np.array(F)); print(fold, len(F))
