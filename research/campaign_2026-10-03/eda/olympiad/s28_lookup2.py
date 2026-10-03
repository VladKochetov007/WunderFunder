# Hierarchical exact-key lookup: current i0 book state (AB) -> + dp state at last anchor (a6 reset) -> + elapsed-time bin (a6).
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
ALPHA = 10.0
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
def wcorr(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    my = np.sum(w*y)/w.sum(); mp = np.sum(w*p)/w.sum()
    return np.sum(w*(y-my)*(p-mp)) / np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(p-mp)**2))
BINS = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
def h(*arrs):
    x = np.zeros(len(arrs[0]), np.uint64)
    for a in arrs: x = x*np.uint64(1000003) + a.astype(np.uint64)
    return x
def feats(df):
    p0 = df.i0_p0.values.view(np.uint32); p11 = df.i0_p11.values.view(np.uint32); dp = df.i0_dp0.values.view(np.uint32)
    s = df.a6.values < -5.19; s[0] = True
    run = np.cumsum(s)-1; dpa = dp[s][run]           # dp at last anchor (causal: anchor row <= t)
    ab = h(p0, p11); b = np.searchsorted(BINS, df.a6.values)
    lvl = [ab, h(ab, dpa), h(ab, dpa, b), h(ab, dpa, b, dp)]
    return np.stack(lvl)
def load(split, step):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); K=[]; Y=[]; C=[]; A6=[]
    flag = "is_scored" if split=="valid" else "need_prediction"
    for rg in range(0, pf.metadata.num_row_groups, step):
        df = pf.read_row_group(rg, columns=["i0_p0","i0_p11","i0_dp0","a6","t0","t1",flag]).to_pandas()
        a = np.abs(np.r_[df.t0, df.t1]); q = a[a>0].min(); c = int(np.argmin(np.abs(Q-q)))
        m = df[flag].values; K.append(feats(df)[:, m]); Y.append(df[["t0","t1"]].values[m]); C.append(np.full(m.sum(), c, np.int8)); A6.append(df.a6.values[m])
    return np.concatenate(K,1), np.concatenate(Y), np.concatenate(C), np.concatenate(A6)
Kt, Yt, Ct, _ = load("train", int(sys.argv[1])); Kv, Yv, Cv, A6v = load("valid", 4)
print("train", len(Yt), "valid", len(Yv), flush=True)
names = ["AB", "+dp@anchor", "+a6bin", "+dp_now"]
P = {}
for ti, tg in enumerate(("t0","t1")):
    gm = Yt[:, ti].mean(); pred = np.full(len(Yv), gm); out=[]
    for j in range(Kt.shape[0]):
        g = pd.DataFrame({"k": Kt[j], "y": Yt[:, ti]}).groupby("k").y.agg(["sum","count"])
        m = pd.DataFrame({"k": Kv[j]}).join(g, on="k")
        s = m["sum"].fillna(0).values; n = m["count"].fillna(0).values
        pred = (s + ALPHA*pred)/(n + ALPHA)
        out.append((names[j], round(wcorr(Yv[:, ti], pred),4), "seen", round((n>0).mean(),3)))
    P[tg] = pred
    print(tg, out, flush=True)
    print("   per class (final level):", {c: round(wcorr(Yv[Cv==c, ti], pred[Cv==c]),4) for c in range(4)})
# weight shares on valid
w = np.abs(np.clip(Yv,-2,2)).sum(1)
print("valid |t|-weight share by class:", {c: round(w[Cv==c].sum()/w.sum(),3) for c in range(4)})
an = A6v < -5.19
print("valid: anchor-row share of rows", round(an.mean(),3), "of weight", round(w[an].sum()/w.sum(),3))
for lo, hi in zip(np.r_[-6, BINS], np.r_[BINS, 9]):
    m = (A6v>=lo)&(A6v<hi); print(f"   a6 in [{lo},{hi}): rows {m.mean():.3f} weight {w[m].sum()/w.sum():.3f} corr t0 {wcorr(Yv[m,0], P['t0'][m]):.3f} t1 {wcorr(Yv[m,1], P['t1'][m]):.3f}")
