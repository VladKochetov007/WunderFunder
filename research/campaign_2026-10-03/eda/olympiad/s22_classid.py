# Can the target quantum class (q) be identified exactly from warm-up inputs (rows 0-98)? Value-membership vote on i0_p0/i0_p11.
import pyarrow.parquet as pq, numpy as np, pandas as pd, collections
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
tt = pd.read_parquet("research/campaign_2026-10-03/eda/olympiad/targets_train.parquet"); tt["cls"]=[int(np.argmin(np.abs(Q-q))) for q in tt.q]
tv = pd.read_parquet("research/campaign_2026-10-03/eda/olympiad/targets_valid.parquet"); tv["cls"]=[int(np.argmin(np.abs(Q-q))) for q in tv.q]
pf = pq.ParquetFile("datasets/train.parquet")
vote = collections.defaultdict(lambda: np.zeros(4))
for rg in range(0, 10607, 4):
    t = pf.read_row_group(rg, columns=["i0_p0","i0_p11"])
    c = tt.cls.iloc[rg]
    for col in ("i0_p0","i0_p11"):
        for v in np.unique(t.column(col).to_numpy().view(np.int32)): vote[(col, int(v))][c] += 1
print("table keys", len(vote), flush=True)
pv = pq.ParquetFile("datasets/valid.parquet"); ok=0; n=0; conf=np.zeros((4,4),int); undecided=0
for rg in range(pv.metadata.num_row_groups):
    t = pv.read_row_group(rg, columns=["i0_p0","i0_p11"])
    s = np.zeros(4)
    for col in ("i0_p0","i0_p11"):
        x = t.column(col).to_numpy()[:99].view(np.int32)
        for v in np.unique(x):
            vv = vote.get((col,int(v)))
            if vv is not None: s += np.log1p(vv)
    pred = int(np.argmax(s)) if s.sum()>0 else -1
    if pred<0: undecided+=1; pred=3
    conf[tv.cls.iloc[rg], pred]+=1
print("valid class-id accuracy from warm-up rows 0-98:", np.trace(conf)/conf.sum(), "undecided", undecided); print(conf)
print("class shares train", tt.cls.value_counts(normalize=True).sort_index().round(3).to_dict(), "valid", tv.cls.value_counts(normalize=True).sort_index().round(3).to_dict())
