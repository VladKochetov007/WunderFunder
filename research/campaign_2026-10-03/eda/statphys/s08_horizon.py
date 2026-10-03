# Fixed-row-horizon test: if t(s)=M(s+h)-M(s), then along chains s, s+h, s+2h, ... the cumulative sum of
# D(s)=t(s+1)-t(s) reproduces dM up to a per-chain constant; dM is sparse integer -> count of nonzeros is minimal at true h.
import pyarrow.parquet as pq, numpy as np, sys
pf = pq.ParquetFile('datasets/train.parquet')
H = np.arange(1, 3001)
def score(T):
    D = np.round(np.diff(T)).astype(np.int64)
    N = len(D); res = np.empty(len(H))
    for i, h in enumerate(H):
        m = (N // h) * h
        C = D[:m].reshape(-1, h)            # rows = chain steps, cols = phase
        dM = np.cumsum(C, axis=0)
        # subtract per-chain mode (approx with median for speed)
        dM = dM - np.median(dM, axis=0, keepdims=True).astype(np.int64)
        res[i] = np.count_nonzero(dM) / m
    return res
rgs = [int(x) for x in sys.argv[1:]]
for rg in rgs:
    t = pf.read_row_group(rg, columns=['t0', 't1']).to_pandas()
    u = np.unique(np.abs(t.t0.values)); u = u[u > 1e-6][0]
    for c in ['t0', 't1']:
        T = t[c].values.astype(np.float64) / u
        r = score(T)
        o = np.argsort(r)[:5]
        print(rg, c, 'unit', round(u, 4), 'frac rows where target changes', round(np.mean(np.diff(T) != 0), 4),
              'best h', H[o], r[o].round(4), 'median over h', round(np.median(r), 4))
