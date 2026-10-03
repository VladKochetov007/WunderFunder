import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d = np.load("research/campaign_2026-10-03/eda/linear/univariate.npz")
TR = list(d["TR"]); F = list(d["FEATS"]); nf = len(F)
names = [f"{t}:{f}" for t in TR for f in F]
for k in ["wc0", "wc1"]:
    c = d[k]; per = d["per"][:, 0 if k == "wc0" else 1]
    idx = np.argsort(-np.abs(c))[:25]
    print(f"== top {k} (pooled weighted corr, scored valid rows; per-seq median, frac same sign)")
    for i in idx:
        p = per[:, i]; p = p[np.isfinite(p)]
        print(f"  {names[i]:16s} {c[i]: .4f}  unweighted-all {d['u'+k[1:]][i]: .4f}  perseq-med {np.median(p): .3f}  same-sign {np.mean(np.sign(p)==np.sign(c[i])):.2f}")
    # best per transform
    print("  best |corr| per transform:", {t: round(float(np.abs(c[j*nf:(j+1)*nf]).max()), 3) for j, t in enumerate(TR)})
q = d["q"]
vals, cnt = np.unique(np.round(q, 4), return_counts=True)
print("q classes", vals, cnt)
for v in vals:
    m = np.round(q, 4) == v
    print(f"q={v}: n={m.sum()} std0={d['std0'][m].mean():.3f} std1={d['std1'][m].mean():.3f} frac_sc={d['frac_sc'][m].mean():.3f} corr01={d['corr01'][m].mean():.3f} mabs0_sc={d['mabs0_sc'][m].mean():.3f}")
smd = d["smd"]; idx = np.argsort(-np.abs(smd))[:15]
print("is_scored SMD top:", [(names[i], round(float(smd[i]), 3)) for i in idx])
