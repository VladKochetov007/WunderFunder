# Plot raw columns of one sequence to see the stochastic structure.
import sys, numpy as np, pyarrow.parquet as pq, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = sys.argv[1]; g = int(sys.argv[2]); out = sys.argv[3]
pf = pq.ParquetFile(f); t = pf.read_row_group(g).to_pandas()
groups = [("i0_p", range(0, 11)), ("i0_p", range(11, 22)), ("i0_v", range(0, 11)), ("i0_v", range(11, 22)),
          ("i0_dp", range(4)), ("i0_dv", range(4)), ("i1_p", range(0, 11)), ("i1_dp", range(4)), ("a", range(8)), ("t", range(2))]
fig, ax = plt.subplots(len(groups), 1, figsize=(18, 2.2 * len(groups)), sharex=True)
for a, (pre, rng) in zip(ax, groups):
    for k in rng:
        a.plot(t[f"{pre}{k}"].values, lw=0.4, label=f"{pre}{k}")
    a.legend(fontsize=6, ncol=11, loc="upper left"); a.set_ylabel(pre)
ax[0].set_title(f"{f} rg={g} seq={t.seq_ix.iloc[0]}")
plt.tight_layout(); plt.savefig(out, dpi=70)
