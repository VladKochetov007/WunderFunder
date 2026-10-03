import numpy as np
np.set_printoptions(linewidth=200, precision=4, suppress=True)
for name in ["train","valid"]:
    d = np.load(f"scan1_{name}.npz")
    q = np.round(np.minimum(d["q0"], d["q1"]), 5)
    u, c = np.unique(q, return_counts=True)
    print(name, "q classes:", list(zip(u, c)))
    print(" q0==q1 frac", np.mean(np.isclose(d["q0"], d["q1"], rtol=1e-4)))
    for uu in u[c > 20]:
        m = q == uu
        print(f"  q={uu}: n={m.sum()} sd0={d['sd0'][m].mean():.3f} sd1={d['sd1'][m].mean():.3f} z0={d['z0'][m].mean():.3f} kmax={np.median(d['kmax0'][m]):.1f} corr01={np.nanmean(d['c01'][m]):.3f} m0={d['m0'][m].mean():.4f} amin7={d['amin'][m][:,7].mean():.3f}")
    print(" corr01 overall", np.nanmean(d["c01"]), np.nanquantile(d["c01"], [.05,.5,.95]))
