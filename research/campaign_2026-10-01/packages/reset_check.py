"""Package contract check: warm-up returns None for steps 0-98, two finite outputs from step 99, state resets on a
new seq_ix (A -> B -> A reproduces A exactly), repeated runs are bitwise identical.
    reset_check.py SUBMISSION.zip
"""
import importlib, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import DataPoint  # noqa: E402

tmp = tempfile.mkdtemp()
zipfile.ZipFile(sys.argv[1]).extractall(tmp)
sys.path.insert(0, tmp)
sol = importlib.import_module("solution")
pf = pq.ParquetFile(ROOT / "cache/valid_subset20.parquet")
feat_cols = [c for c in pf.schema_arrow.names if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
def seq(k, n=3000):
    t = pf.read_row_group(k)
    x = np.stack([t.column(c).to_numpy() for c in feat_cols], 1).astype(np.float32)[:n]
    return int(t.column("seq_ix")[0].as_py()), x
def run(model, sid, x):
    out = []
    for i, row in enumerate(x):
        p = model.predict(DataPoint(sid, i, i >= 99, row))
        out.append(None if p is None else np.asarray(p, np.float64))
    return out
model = sol.PredictionModel()
(sa, xa), (sb, xb) = seq(0), seq(1)
a1 = run(model, sa, xa); run(model, sb, xb); a2 = run(model, sa, xa)
assert all(p is None for p in a1[:99]), "warm-up must return None"
arr1 = np.array(a1[99:]); arr2 = np.array(a2[99:])
assert arr1.shape[1] == 2 and np.isfinite(arr1).all(), "two finite predictions from step 99"
assert np.array_equal(arr1, arr2), "A->B->A must reproduce A bitwise"
fresh = sol.PredictionModel(); a3 = np.array(run(fresh, sa, xa)[99:])
assert np.array_equal(arr1, a3), "fresh instance must match"
print("RESET_CHECK_PASS", arr1.shape, float(np.abs(arr1).mean()))
