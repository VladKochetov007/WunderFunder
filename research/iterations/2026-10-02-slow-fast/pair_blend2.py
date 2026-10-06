"""Pair-level composite: compare base pairs (fullft / wise05 / gglob98 / wise025) alone, + baseline, + baseline + subset pairs.
All deltas vs the fullft pair alone. Optional argv: extra member specs 'name:weight' added on top of 'base pair + baseline'."""
import sys
import numpy as np

exec(open("comp_eval.py").read().split("public={")[0])
L = lambda n, sp: np.load(R + f"cache/pred_{n}_{sp}.npy", mmap_mode="r")


def comp(spec):  # spec: list of (weight, [names], scale); names averaged
    Sv = np.zeros((2, 6))
    St = np.zeros((2, 6))
    srcs = [(w * sc / len(ns), [(L(n, "valid"), L(n, "train")) for n in ns]) for w, ns, sc in spec]
    for i in iv:
        P = sum(c * sum(np.asarray(v[i]) for v, _ in arr) for c, arr in srcs)
        for j in range(2):
            Sv[j] += sums(Yv[i, :, j], P[:, j], Mv[i].astype(float))
    for i in it:
        q = np.asarray(prop[rgt[i]]).astype(float)
        q[:99] = 0
        P = sum(c * sum(np.asarray(t[i]) for _, t in arr) for c, arr in srcs)
        for j in range(2):
            St[j] += sums(Yt[i, :, j], P[:, j], q)
    return np.array([wp(Sv), wp(St)])


pr = lambda z: [f"cmp_{z}_s3", f"cmp_{z}_s4"]
b0 = comp([(1, pr("fullft"), 1)])
print("fullft pair: valid %.4f trainHO %.4f" % tuple(b0), flush=True)


def rep(name, spec):
    r = comp(spec) - b0
    print(f"{name:46s} d_valid {r[0]:+.4f} d_trainHO {r[1]:+.4f}", flush=True)


bases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fullft", "wise05", "gglob98", "wise025"]
extra = [a.split(":") for a in sys.argv[2:]]
for bp_name in bases:
    bp = pr(bp_name)
    if bp_name != "fullft":
        rep(f"{bp_name} pair alone", [(1, bp, 1)])
    rep(f"{bp_name} + base", [(0.75, bp, 1), (0.25, ["orgbase"], 0.25)])
    rep(f"{bp_name} + base + za/zvol/zpxev", [(0.4, bp, 1), (0.25, ["orgbase"], 0.25), (0.12, pr("za"), 1), (0.12, pr("zvol"), 1), (0.11, pr("zpxev"), 1)])
    for name, w in extra:
        w = float(w)
        rep(f"{bp_name} + base + {name} {w}", [(0.75 - w, bp, 1), (0.25, ["orgbase"], 0.25), (w, pr(name), 1)])
