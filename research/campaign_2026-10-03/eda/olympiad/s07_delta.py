import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
cs = [f"i0_p{k}" for k in range(11)]
df = pf.read_row_group(0, columns=cs+["t0","t1"]).to_pandas()
X = df[cs].values.astype(np.float64)
d = np.diff(X, axis=0); m = np.any(d!=0, axis=1); D = d[m]
print("n changes", len(D))
Dr = np.round(D, 3)
u, cnt = np.unique(Dr, axis=0, return_counts=True)
print("unique delta vectors (1e-3)", len(u), "top counts", np.sort(cnt)[::-1][:20])
# sign patterns
S = np.sign(D).astype(int)
us, cs_ = np.unique(S, axis=0, return_counts=True)
o = np.argsort(-cs_)
for i in o[:15]: print(us[i], cs_[i])
# state sequence & transitions
states, inv = np.unique(X, axis=0, return_inverse=True)
inv = inv.ravel()
print("n states", len(states))
seq = inv[np.r_[True, inv[1:]!=inv[:-1]]]
pairs = pd.Series(list(zip(seq[:-1], seq[1:]))).value_counts()
print("n distinct transitions", len(pairs), "transitions", len(seq)-1)
deg = pd.Series([a for a,b in pairs.index]).value_counts()
print("out-degree distribution", deg.describe())
# revisit structure: how often do we return to previous state (A->B->A)
back = np.mean(seq[2:]==seq[:-2]); print("A->B->A fraction", back)
