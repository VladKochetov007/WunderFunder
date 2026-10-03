# ACF / cross-correlation of targets (in tick units) over many train sequences, per lattice class.
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = int(sys.argv[1]); outdir = sys.argv[2]
pf = pq.ParquetFile(TRAIN); rng = np.random.default_rng(2); gs = np.sort(rng.choice(pf.metadata.num_row_groups, n, replace=False))
lags = np.unique(np.r_[np.arange(0, 50), np.round(np.geomspace(50, 8000, 60)).astype(int)])
def acf(x, y):
    x = x - x.mean(); y = y - y.mean(); n = len(x); out = []
    for L in lags:
        out.append(np.dot(x[:n - L], y[L:]) / n)
    return np.array(out) / np.sqrt(x.var() * y.var())
res = {}
for g in gs:
    t = pf.read_row_group(g, columns=["t0", "t1"]).to_pandas()
    a, b = t.t0.values.astype(np.float64), t.t1.values.astype(np.float64)
    c = round(lattice(a, b), 4)
    s, d = a + b, a - b
    r = np.stack([acf(a, a), acf(b, b), acf(s, s), acf(d, d), acf(a, b), acf(b, a)])
    res.setdefault(c, []).append(r)
np.savez(f"{outdir}/target_acf.npz", lags=lags, **{str(k): np.array(v) for k, v in res.items()})
names = ["acf t0", "acf t1", "acf t0+t1", "acf t0-t1", "corr t0(t), t1(t+L)", "corr t1(t), t0(t+L)"]
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for i, a in enumerate(ax.flat):
    for c, v in sorted(res.items()):
        a.plot(np.maximum(lags, 0.5), np.median(np.array(v)[:, i], 0), label=f"c={c} n={len(v)}")
    a.set_xscale("log"); a.set_title(names[i]); a.grid(alpha=.3); a.axhline(0, color="k", lw=.5); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{outdir}/../fig/target_acf.png", dpi=70)
for c, v in sorted(res.items()):
    m = np.median(np.array(v), 0)
    print(c, len(v))
    for L in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]:
        j = np.searchsorted(lags, L)
        print(f"  L={lags[j]:5d}", np.round(m[:, j], 3))
