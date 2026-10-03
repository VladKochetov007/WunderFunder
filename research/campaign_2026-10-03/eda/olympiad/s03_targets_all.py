# Per-sequence target lattice: q (quantum), integrality residual, k ranges, run counts, t0/t1 corr.
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
split = sys.argv[1]
pf = pq.ParquetFile(f"datasets/{split}.parquet")
cols = ["seq_ix","t0","t1"] + (["is_scored"] if split=="valid" else [])
rows = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols)
    t0 = t.column("t0").to_numpy().astype(np.float64); t1 = t.column("t1").to_numpy().astype(np.float64)
    a = np.abs(np.concatenate([t0, t1])); a = a[a > 0]
    q = a.min() if len(a) else np.nan
    best = None
    for div in (1, 2, 3, 4, 6, 8):
        qq = q/div
        r = max(np.abs(t0/qq-np.round(t0/qq)).max(), np.abs(t1/qq-np.round(t1/qq)).max())
        if r < 1e-3: best = (qq, div, r); break
    if best is None: best = (q, -1, r)
    qq = best[0]
    k0 = np.round(t0/qq).astype(np.int64); k1 = np.round(t1/qq).astype(np.int64)
    d = dict(rg=rg, seq=int(t.column("seq_ix")[0].as_py()), q=qq, div=best[1], resid=best[2],
             k0min=k0.min(), k0max=k0.max(), k1min=k1.min(), k1max=k1.max(),
             k0abs=np.abs(k0).mean(), k1abs=np.abs(k1).mean(),
             z0=np.mean(k0==0), z1=np.mean(k1==0), runs0=1+np.sum(np.diff(k0)!=0), runs1=1+np.sum(np.diff(k1)!=0),
             corr=np.corrcoef(t0,t1)[0,1] if t0.std()>0 and t1.std()>0 else np.nan,
             sum_abs=np.abs(k0+k1).mean(), sum0=np.mean(k0+k1==0), std0=t0.std(), std1=t1.std(), mean0=t0.mean(), mean1=t1.mean())
    if split=="valid":
        s = t.column("is_scored").to_numpy(zero_copy_only=False); d["nscored"]=int(s.sum()); d["first_scored"]=int(np.argmax(s)) if s.any() else -1
    rows.append(d)
    if rg % 1000 == 0: print(rg, d, flush=True)
pd.DataFrame(rows).to_parquet(f"research/campaign_2026-10-03/eda/olympiad/targets_{split}.parquet")
print("done")
