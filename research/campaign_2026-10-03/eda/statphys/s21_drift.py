# Which current observables carry the drift of the centre coordinate X=(T0-T1)/2 and of Sum=T0+T1 (per class)?
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN); fn = feat_names(pf)
rng = np.random.default_rng(12)
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    Xs, Ys = [], []
    for rg in rng.choice(d.rg[d.u == u].values.astype(int), 16, replace=False):
        X, Y, uu, _ = load(pf, rg); Xs.append(X[99::3]); Ys.append(np.round(Y[99::3] / uu))
    X = np.concatenate(Xs).astype(np.float64); T = np.concatenate(Ys)
    Xc = (T[:, 0] - T[:, 1]) / 2; Sm = T[:, 0] + T[:, 1]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rX = Z.T @ (Xc - Xc.mean()) / len(Xc) / Xc.std(); rS = Z.T @ (Sm - Sm.mean()) / len(Sm) / Sm.std()
    oX = np.argsort(-np.abs(rX))[:10]; oS = np.argsort(-np.abs(rS))[:8]
    print(f'u={u}  X top:', [(fn[j], round(rX[j], 3)) for j in oX])
    print(f'        Sum top:', [(fn[j], round(rS[j], 3)) for j in oS])
