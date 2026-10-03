# Per-class ridge on volume/trade/a-columns (no price blocks), evaluated by elapsed-time bin; complements the state lookup?
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
def wcorr(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    my = np.sum(w*y)/w.sum(); mp = np.sum(w*p)/w.sum()
    return np.sum(w*(y-my)*(p-mp)) / np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(p-mp)**2))
F = [f"i0_v{k}" for k in range(22)] + [f"i1_v{k}" for k in range(22)] + [f"i0_dv{k}" for k in range(4)] + [f"i1_dv{k}" for k in range(4)] + [f"a{k}" for k in range(8)]
def load(split, step, maxrows):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); X=[]; Y=[]; C=[]
    flag = "is_scored" if split=="valid" else "need_prediction"
    for rg in range(0, pf.metadata.num_row_groups, step):
        df = pf.read_row_group(rg, columns=F+["t0","t1",flag]).to_pandas()
        a = np.abs(np.r_[df.t0, df.t1]); q = a[a>0].min(); c = int(np.argmin(np.abs(Q-q)))
        m = df[flag].values
        idx = np.where(m)[0]
        if split=="train": idx = idx[::5]
        X.append(df[F].values[idx].astype(np.float32)); Y.append(df[["t0","t1"]].values[idx]); C.append(np.full(len(idx), c, np.int8))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(C)
Xt, Yt, Ct = load("train", 25, None); Xv, Yv, Cv = load("valid", 4, None)
print("train", Xt.shape, "valid", Xv.shape, flush=True)
def design(X):
    S = (X < -5.19).astype(np.float32)        # sentinel indicators (elapsed==0 etc.)
    Xc = np.where(X < -5.19, 0, X)
    return np.c_[np.ones(len(X)), Xc, S[:, -3:]]
Pv = np.zeros_like(Yv)
for c in range(4):
    mt, mv = Ct==c, Cv==c
    D = design(Xt[mt]); w = np.abs(np.clip(Yt[mt],-2,2))
    for ti in range(2):
        Wd = D * w[:, ti:ti+1]
        beta = np.linalg.solve(D.T@Wd + 10*np.eye(D.shape[1]), Wd.T@Yt[mt, ti])
        Pv[mv, ti] = design(Xv[mv])@beta
    print("class", c, "valid wcorr t0 %.4f t1 %.4f" % (wcorr(Yv[mv,0], Pv[mv,0]), wcorr(Yv[mv,1], Pv[mv,1])), flush=True)
print("pooled valid wcorr t0 %.4f t1 %.4f" % (wcorr(Yv[:,0], Pv[:,0]), wcorr(Yv[:,1], Pv[:,1])))
a6 = Xv[:, F.index("a6")]
for lo, hi in [(-6,-5),(-5,-0.5),(-0.5,0.5),(0.5,1.5),(1.5,9)]:
    m = (a6>=lo)&(a6<hi); print(f"  a6 [{lo},{hi}) t0 {wcorr(Yv[m,0], Pv[m,0]):.3f} t1 {wcorr(Yv[m,1], Pv[m,1]):.3f}")
np.save("research/campaign_2026-10-03/eda/olympiad/s29_valid_pred.npy", Pv)
