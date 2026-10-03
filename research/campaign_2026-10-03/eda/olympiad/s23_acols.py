import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pf = pq.ParquetFile("datasets/train.parquet")
df = pf.read_row_group(500).to_pandas()
A = [f"a{k}" for k in range(8)]
print(df[A].describe().T)
print("a7 values:", np.unique(df.a7.round(5), return_counts=True))
# relation of a2/a3 diffs to events
ch0 = np.r_[False, np.diff(df.i0_p0.values)!=0]
for c in A:
    d = np.r_[0, np.diff(df[c].values)]
    print(c, "corr(diff, next-diff)", np.corrcoef(d[1:-1], d[2:])[0,1].round(3), "lag1 autocorr level", np.corrcoef(df[c].values[:-1], df[c].values[1:])[0,1].round(4),
          "mean|diff| on i0 event vs not", np.abs(d[ch0]).mean().round(4), np.abs(d[~ch0]).mean().round(4))
fig, ax = plt.subplots(8,1, figsize=(16,18), sharex=True)
sl = slice(5000, 6000)
for i,c in enumerate(A):
    ax[i].plot(np.arange(20000)[sl], df[c].values[sl], lw=.6); ax[i].set_title(c)
    ev = np.where(ch0[sl])[0]+5000
    ax[i].vlines(ev, df[c].values[sl].min(), df[c].values[sl].min()+0.1, color="r", lw=.5)
plt.tight_layout(); plt.savefig("research/campaign_2026-10-03/eda/olympiad/f04_acols_zoom.png", dpi=55)
