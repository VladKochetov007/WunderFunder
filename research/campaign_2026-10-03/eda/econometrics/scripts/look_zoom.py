# Zoomed raw view of one sequence window: price columns, trades, a*, targets.
import sys, numpy as np, pyarrow.parquet as pq, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
f, g, s0, s1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
t = pq.ParquetFile(f).read_row_group(g).to_pandas().iloc[s0:s1]
x = t.step_in_seq.values
groups = [("i0_p", range(0, 11)), ("i0_p", range(11, 22)), ("i0_v", range(0, 22)), ("i0_dp", range(4)), ("i0_dv", range(4)),
          ("i1_p", range(0, 22)), ("i1_dp", range(4)), ("a", range(0, 4)), ("a", range(4, 8)), ("t", range(2))]
fig, ax = plt.subplots(len(groups), 1, figsize=(18, 2.0 * len(groups)), sharex=True)
for a, (pre, rng) in zip(ax, groups):
    for k in rng:
        a.plot(x, t[f"{pre}{k}"].values, lw=0.6, drawstyle="steps-post", label=f"{pre}{k}")
    a.legend(fontsize=6, ncol=11, loc="upper left"); a.set_ylabel(pre); a.grid(alpha=.3)
ax[0].set_title(f"{f} rg={g} seq={t.seq_ix.iloc[0]} steps {s0}-{s1}")
plt.tight_layout(); plt.savefig(out, dpi=70)
