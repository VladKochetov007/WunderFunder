# Targets only change on rows where the i0 price state changes. Test fixed horizon in PRICE-EVENT time.
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
pf = pq.ParquetFile(TRAIN)
H = np.arange(1, 400)
def score(T):
    D = np.round(np.diff(T)).astype(np.int64); N = len(D); res = np.empty(len(H))
    for i, h in enumerate(H):
        m = (N // h) * h; C = D[:m].reshape(-1, h); dM = np.cumsum(C, axis=0)
        dM = dM - np.median(dM, axis=0, keepdims=True).astype(np.int64); res[i] = np.count_nonzero(dM) / m
    return res
for rg in [6698, 5522, 4178, 7727, 8610]:
    X, Y, u, _ = load(pf, rg)
    pch = np.r_[True, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]
    T = np.round(Y[pch] / u)
    nz = np.mean(np.any(np.diff(T, axis=0) != 0, axis=1))
    for k in range(2):
        r = score(T[:, k]); o = np.argsort(r)[:5]
        print(rg, f't{k}', 'n events', pch.sum(), 'P(target changes per event) %.3f' % nz, 'best h', H[o], r[o].round(3), 'median', round(np.median(r), 3))
