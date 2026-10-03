# Variable-order Markov (n-gram) lookup over the sequence of DISTINCT i0 book-state symbols.
# key_m(t) = last m distinct symbols up to row t (causal). Train table -> valid scored rows, recursive shrinkage backoff.
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
M = 5; ALPHA = 10.0
def wcorr(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    my = np.sum(w*y)/w.sum(); mp = np.sum(w*p)/w.sum()
    return np.sum(w*(y-my)*(p-mp)) / np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(p-mp)**2))
symcols = {"AB": ["i0_p0","i0_p11"], "ABDP": ["i0_p0","i0_p11","i0_dp0"]}
mode = sys.argv[1] if len(sys.argv)>1 else "AB"
sc = symcols[mode]
def keys_for(df):
    s = np.zeros(len(df), np.uint64)
    for c in sc: s = s*np.uint64(2654435761) + df[c].values.view(np.uint32).astype(np.uint64)
    ch = np.r_[True, s[1:]!=s[:-1]]
    r = np.cumsum(ch)-1; runs = s[ch]
    K = np.zeros((M, len(df)), np.uint64); h = np.zeros(len(df), np.uint64)
    for j in range(M):
        idx = r - j
        v = np.where(idx>=0, runs[np.maximum(idx,0)], np.uint64(7))
        h = h*np.uint64(1000003) + v
        K[j] = h
    return K
def load(split, step):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); Ks=[]; Ys=[]; Cs=[]
    flag = "is_scored" if split=="valid" else "need_prediction"
    for rg in range(0, pf.metadata.num_row_groups, step):
        df = pf.read_row_group(rg, columns=sc+["t0","t1",flag]).to_pandas()
        K = keys_for(df); m = df[flag].values
        Ks.append(K[:, m]); Ys.append(df[["t0","t1"]].values[m].astype(np.float32))
    return np.concatenate(Ks, 1), np.concatenate(Ys)
Kt, Yt = load("train", int(sys.argv[2])); Kv, Yv = load("valid", 4)
print(mode, "train rows", Yt.shape[0], "valid rows", Yv.shape[0], flush=True)
for ti, tg in enumerate(("t0","t1")):
    gm = Yt[:, ti].mean(); pred = np.full(len(Yv), gm, np.float64); out=[]
    for j in range(M):
        g = pd.DataFrame({"k": Kt[j], "y": Yt[:, ti]}).groupby("k").y.agg(["sum","count"])
        m = pd.DataFrame({"k": Kv[j]}).join(g, on="k")
        s = m["sum"].fillna(0).values; n = m["count"].fillna(0).values
        pred = (s + ALPHA*pred)/(n + ALPHA)
        out.append((j+1, round(wcorr(Yv[:, ti], pred),4), "seen", round((n>0).mean(),3), "nkeys", len(g)))
    print(tg, out, flush=True)
