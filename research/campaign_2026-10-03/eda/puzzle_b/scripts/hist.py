# Pooled per-column histograms + exact-value frequencies (top-200 per seq merged) over sampled train sequences.
import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
n = int(sys.argv[1])
s = pd.read_csv(f'{OUT}/seqscan_train.csv'); s['cls'] = s.t0_scale.round(4)
pick = s.groupby('cls').head(n // 4)
cs = [c for c in cols() if c not in ('seq_ix', 'step_in_seq', 'need_prediction')]
edges = np.linspace(-6, 10, 1601)
classes = sorted(pick.cls.unique())
H = np.zeros((len(classes), len(cs), 1600), np.int64)
top = {}
for j, (_, r) in enumerate(pick.iterrows()):
    d = seq(int(r.rg), cs)
    ki = classes.index(r.cls)
    for ci, c in enumerate(cs):
        x = d[c].values
        H[ki, ci] += np.histogram(x, edges)[0]
        u, k = np.unique(x, return_counts=True)
        o = np.argsort(-k)[:200]
        dd = top.setdefault((r.cls, c), {})
        for v, kk in zip(u[o].tolist(), k[o].tolist()):
            dd[v] = dd.get(v, 0) + kk
    if j % 20 == 0: print(j, flush=True)
np.savez(f'{OUT}/hist_train.npz', H=H, edges=edges, cols=np.array(cs), classes=np.array(classes))
rows = []
for (cl, c), dd in top.items():
    items = sorted(dd.items(), key=lambda t: -t[1])[:25]
    for v, k in items:
        rows.append((cl, c, v, k))
pd.DataFrame(rows, columns=['cls', 'col', 'value', 'count']).to_csv(f'{OUT}/topvals_train.csv', index=False)
print('done')
