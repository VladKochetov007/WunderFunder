import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pf = pq.ParquetFile("datasets/train.parquet")
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
fig, ax = plt.subplots(2,4, figsize=(22,9))
for i, rg in enumerate([0, 1, 500, 7000]):
    df = pf.read_row_group(rg, columns=["a6","t0","t1","i0_p0","i0_p11"]).to_pandas()
    a = np.abs(np.r_[df.t0, df.t1]); q = a[a>0].min()
    k0 = np.round(df.t0.values/q).astype(int); k1 = np.round(df.t1.values/q).astype(int)
    s = df.a6.values < -5.19
    T = np.r_[True, (k0[1:]!=k0[:-1])|(k1[1:]!=k1[:-1])]
    print("rg", rg, "q", round(q,4), "a6 resets", s.sum(), "T", T.sum(), "T&reset", (T&s).sum(), "T&~reset", (T&~s).sum(), "reset&~T", (s&~T).sum())
    run = np.cumsum(s)
    g = pd.DataFrame({"run":run, "a6":df.a6.values, "k0":k0, "k1":k1}).groupby("run").agg(dur=("a6","max"), n=("a6","size"), k0=("k0","first"), k1=("k1","first"), k0n=("k0","nunique"))
    print("   runs", len(g), "runs with >1 distinct target", (g.k0n>1).sum())
    g["bin"] = pd.qcut(g.dur.rank(method="first"), 10, labels=False)
    print("   by duration decile: mean|k0|, P(k0 in{-1,0}), mean k0+k1:\n", g.groupby("bin").apply(lambda z: pd.Series({"dur":z.dur.median(), "abs_k0": np.abs(z.k0).mean(), "flat": np.mean((z.k0>=-1)&(z.k0<=0)&(z.k1>=-1)&(z.k1<=0)), "sum": (z.k0+z.k1).mean()})).round(3).T.to_string())
    ax[0,i].scatter(g.dur, g.k0 + np.random.uniform(-.2,.2,len(g)), s=2); ax[0,i].set_title(f"rg{rg} q={q:.3f}: k0 vs max a6 in anchor run"); ax[0,i].set_xlabel("a6 at end of run (elapsed until next anchor)")
    # causal view: rows' current a6 vs |k0|
    r = pd.DataFrame({"a6":df.a6.values, "absk":np.abs(k0)+np.abs(k1)})
    r["b"] = pd.qcut(r.a6.rank(method="first"), 20, labels=False)
    gg = r.groupby("b").agg(a6=("a6","median"), absk=("absk","mean"))
    ax[1,i].plot(gg.a6, gg.absk, "o-"); ax[1,i].set_title(f"rg{rg}: mean |k0|+|k1| vs current a6 (causal)"); ax[1,i].set_xlabel("a6")
plt.tight_layout(); plt.savefig("research/campaign_2026-10-03/eda/olympiad/f05_a6_anchor.png", dpi=55)
