import sys, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack"); sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS
pkg = ROOT / "submissions/final_keep_k4/pkg"; sys.path.insert(0, str(pkg / "v3")); import wnn_native
d = np.load(ROOT / "cache/valid_targets.npz"); rg, Y, M = d["row_group"], d["y"], d["mask"]
pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
members = json.loads((pkg / "members.json").read_text())
def wpm(y, p, m):
    y = np.clip(y, -2, 2)[m]; p = np.clip(p, -2, 2)[m]; w = np.abs(y)
    my, mp = (w*y).sum()/w.sum(), (w*p).sum()/w.sum()
    return (w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum())
for i in [int(a) for a in sys.argv[1:]]:
    t = pf.read_row_group(int(rg[i]), columns=FEATURE_COLUMNS, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    for mem in members:
        w = dict(np.load(pkg / mem["weights"])); w = {k: np.ascontiguousarray(v, np.float32) for k, v in w.items()}
        e = wnn_native.EnsemblePredictor([w], [np.float32(1.0)], fast_activations=True)
        o = np.array([e.step(r) for r in x], np.float64)
        c = np.asarray(np.load(ROOT / f"cache/pred_{mem['source']}_valid.npy", mmap_mode="r")[i], np.float64)
        diff = np.abs(o[99:] - c[99:]); worst = int(np.argmax(diff.max(1))) + 99
        dwp = np.mean([wpm(Y[i][:, k], o[:, k], M[i]) - wpm(Y[i][:, k], c[:, k], M[i]) for k in range(2)])
        print(f"idx={i} {mem['source']:<28} corr={np.corrcoef(o[99:,0],c[99:,0])[0,1]:.6f},{np.corrcoef(o[99:,1],c[99:,1])[0,1]:.6f} maxabs={diff.max():.2e} at row {worst}  seqWP native-batch={dwp:+.2e}")
