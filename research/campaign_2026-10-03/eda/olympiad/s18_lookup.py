# Exact-state lookup ("target encoding") of i0 book-state symbols: train sample -> valid scored rows.
import pyarrow.parquet as pq, numpy as np, pandas as pd, gc
def wcorr(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    if w.sum() == 0: return np.nan
    my = np.sum(w*y)/w.sum(); mp = np.sum(w*p)/w.sum()
    return np.sum(w*(y-my)*(p-mp)) / np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(p-mp)**2))
cols = ["i0_p0","i0_p11","i0_dp0","i1_p0","i1_p11","i1_dp0","t0","t1"]
def load(split, step, extra):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); out=[]
    for rg in range(0, pf.metadata.num_row_groups, step):
        t = pf.read_row_group(rg, columns=cols+extra).to_pandas()
        if split=="valid": t = t[t.is_scored]
        else: t = t[t.need_prediction]
        d = pd.DataFrame({c: t[c].values.view(np.int32) for c in cols[:6]})
        d["t0"]=t.t0.values; d["t1"]=t.t1.values
        out.append(d)
    return pd.concat(out, ignore_index=True)
tr = load("train", 18, ["need_prediction"]); va = load("valid", 3, ["is_scored"])
print("train rows", len(tr), "valid rows", len(va), flush=True)
keys = {"A":["i0_p0"], "B":["i0_p11"], "DP":["i0_dp0"], "AB":["i0_p0","i0_p11"], "ABDP":["i0_p0","i0_p11","i0_dp0"],
        "i1AB":["i1_p0","i1_p11"], "i1DP":["i1_dp0"], "AB+i1AB":["i0_p0","i0_p11","i1_p0","i1_p11"]}
gm = {t: tr[t].mean() for t in ("t0","t1")}
for name, k in keys.items():
    res = []
    for tg in ("t0","t1"):
        g = tr.groupby(k)[tg].agg(["sum","count"]).reset_index()
        g["enc"] = (g["sum"] + 20*gm[tg])/(g["count"]+20)   # shrink to global mean
        m = va[k+[tg]].merge(g[k+["enc"]], on=k, how="left")
        cov = m.enc.notna().mean()
        p = m.enc.fillna(gm[tg]).values
        res.append((tg, round(wcorr(m[tg].values, p),4), "coverage", round(cov,3), "nkeys", len(g)))
    print(name, res, flush=True)
