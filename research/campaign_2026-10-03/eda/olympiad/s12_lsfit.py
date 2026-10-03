# Test exact rule k(n) = M(s(n+H)) - M(s(n)) with M additive over i0 bid/ask state, in various clocks.
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
pf = pq.ParquetFile("datasets/train.parquet")
rg = int(sys.argv[1]) if len(sys.argv)>1 else 0
cs = [f"i0_p{k}" for k in range(22)] + [f"i0_dp{k}" for k in range(4)]
df = pf.read_row_group(rg, columns=cs+["t0","t1"]).to_pandas()
q = np.abs(np.r_[df.t0.values, df.t1.values]); q = q[q>0].min()
k0 = np.round(df.t0.values/q); k1 = np.round(df.t1.values/q)
_, sA = np.unique(df[[f"i0_p{k}" for k in range(11)]].values, axis=0, return_inverse=True); sA=sA.ravel()
_, sB = np.unique(df[[f"i0_p{k}" for k in range(11,22)]].values, axis=0, return_inverse=True); sB=sB.ravel()
nA, nB = sA.max()+1, sB.max()+1
def ch(x): return np.r_[True, x[1:]!=x[:-1]]
clocks = {"rows": np.ones(len(df),bool), "A|B": ch(sA)|ch(sB), "A&B": ch(sA)&ch(sB)}
for cname, m in clocks.items():
    idx = np.where(m)[0]
    a, b = sA[idx], sB[idx]
    res = []
    for H in list(range(1,21))+[25,30,40,50,75,100,150,200]:
        n = len(idx)-H
        if n < 100: continue
        rows = np.arange(n)
        # +MA(a[n+H]) +MB(b[n+H]) -MA(a[n]) -MB(b[n])
        R = np.r_[rows, rows, rows, rows]
        C = np.r_[a[H:], nA+b[H:], a[:n], nA+b[:n]]
        V = np.r_[np.ones(n), np.ones(n), -np.ones(n), -np.ones(n)]
        Mx = coo_matrix((V,(R,C)), shape=(n, nA+nB)).tocsr()
        out = []
        for kk in (k0, k1):
            y = kk[idx[:n]]
            sol = lsqr(Mx, y, atol=1e-10, btol=1e-10, iter_lim=3000)[0]
            r = y - Mx@sol
            out.append(round(1 - r.var()/y.var(), 4))
        res.append((H, out))
    best = sorted(res, key=lambda z: -z[1][0])[:6]
    print(cname, "n events", len(idx), "best R2 (t0,t1):", best, flush=True)
