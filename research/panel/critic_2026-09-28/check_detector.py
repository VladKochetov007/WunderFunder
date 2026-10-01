import importlib.util, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from utils import FEATURE_COLUMNS
from wnn.splits import valid_row_groups
pkg = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("sol", pkg / "solution.py"); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
@dataclass(frozen=True)
class DataPoint:
    seq_ix: int; step_in_seq: int; need_prediction: bool; state: np.ndarray
d = np.load(ROOT / "cache/valid_targets.npz"); rg, mc = d["row_group"], d["market_class"]
_, held = valid_row_groups(ROOT / "datasets/valid.parquet"); hold = np.isin(rg, held)
pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
m = sol.PredictionModel()
inv = {id(g): n for n, g in m.groups.items()}
want = {c: m.class_group[c] for c in range(4)}
got = []; margins = []
t0 = time.time()
for i, g in enumerate(rg):
    b = next(pf.iter_batches(batch_size=100, row_groups=[int(g)], columns=FEATURE_COLUMNS, use_threads=False))
    x = np.column_stack([b.column(j).to_numpy() for j in range(len(FEATURE_COLUMNS))]).astype(np.float32)
    for t in range(100):
        m.predict(DataPoint(i, t, t >= 99, x[t]))
    got.append(inv[id(m.active)])
    p = np.sort(m.probs / m.probs.sum()); margins.append(p[-1] - p[-2])
got = np.array(got); ok = np.array([got[i] == want[mc[i]] for i in range(len(rg))]); margins = np.array(margins)
print(f"routing correct: all {ok.mean():.4f} ({(~ok).sum()} wrong) holdout {ok[hold].mean():.4f}; min margin {margins.min():.3f}, 1% margin {np.quantile(margins, .01):.3f}; {time.time()-t0:.0f}s")
# argmax class accuracy (finer than group)
