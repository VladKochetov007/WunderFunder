import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pf = pq.ParquetFile("datasets/train.parquet")
rgs = list(range(0, 10607, 270))[:40]
cols = [f"i0_p{k}" for k in range(22)] + [f"i0_dp{k}" for k in range(4)]
df = pd.concat([pf.read_row_group(r, columns=cols).to_pandas() for r in rgs], ignore_index=True)
u = df.drop_duplicates()
fig, ax = plt.subplots(4,6, figsize=(24,16))
ax = ax.ravel()
for i,k in enumerate(range(1,11)):
    ax[i].scatter(u["i0_p0"], u[f"i0_p{k}"], s=1); ax[i].set_title(f"i0_p0 vs i0_p{k}")
for i,k in enumerate(range(12,22)):
    ax[10+i].scatter(u["i0_p11"], u[f"i0_p{k}"], s=1); ax[10+i].set_title(f"i0_p11 vs i0_p{k}")
ax[20].scatter(u["i0_p0"], u["i0_p11"], s=1); ax[20].set_title("i0_p0 vs i0_p11")
ax[21].scatter(u["i0_dp0"], u["i0_dp1"], s=1); ax[21].set_title("dp0 vs dp1")
ax[22].scatter(u["i0_dp0"], u["i0_dp2"], s=1); ax[22].set_title("dp0 vs dp2")
ax[23].scatter(u["i0_dp0"], u["i0_dp3"], s=1); ax[23].set_title("dp0 vs dp3")
plt.tight_layout(); plt.savefig("research/campaign_2026-10-03/eda/olympiad/f02_block_scatter.png", dpi=55)
