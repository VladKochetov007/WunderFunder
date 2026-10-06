"""Per-class blend weights: P = (1-a_c)*pair + a_c*0.25*baseline, a_c chosen per market class on one part
(train-HO or valid fold 0) and evaluated on the other. Uses pooled metric over all classes."""
import numpy as np

exec(open("comp_eval.py").read().split("public={")[0])
L = lambda n, sp: np.load(R + f"cache/pred_{n}_{sp}.npy", mmap_mode="r")
clsv = tv["market_class"]
clst = tt["market_class"]
A = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]


def part_sums(split):
    """per class, per a: metric sums (2 targets x 6)"""
    if split == "valid":
        idx, Y, M, cls = iv, Yv, None, clsv
        F = [L("cmp_fullft_s3", "valid"), L("cmp_fullft_s4", "valid")]
        B = L("orgbase", "valid")
    else:
        idx, Y, M, cls = it, Yt, None, clst
        F = [L("cmp_fullft_s3", "train"), L("cmp_fullft_s4", "train")]
        B = L("orgbase", "train")
    S = np.zeros((4, len(A), 2, 6))
    for i in idx:
        if split == "valid":
            w = Mv[i].astype(float)
        else:
            w = np.asarray(prop[rgt[i]]).astype(float)
            w[:99] = 0
        p = 0.5 * (np.asarray(F[0][i]) + np.asarray(F[1][i]))
        b = np.asarray(B[i]) * 0.25
        c = int(cls[i])
        for k, a in enumerate(A):
            q = (1 - a) * p + a * b
            for j in range(2):
                S[c, k, j] += sums(Y[i, :, j], q[:, j], w)
    return S


Sv, St = part_sums("valid"), part_sums("train")


def score(S, choice):
    tot = sum(S[c, choice[c]] for c in range(4))
    return wp(tot)


for name, fit, ev in (("fit trainHO -> eval valid", St, Sv), ("fit valid -> eval trainHO", Sv, St)):
    best = [0, 0, 0, 0]
    for _ in range(3):  # coordinate ascent over classes on the fit part
        for c in range(4):
            sc = []
            for k in range(len(A)):
                ch = list(best)
                ch[c] = k
                sc.append(score(fit, ch))
            best[c] = int(np.argmax(sc))
    k25 = A.index(0.25)
    glob = [k25] * 4
    none = [0] * 4
    print(name, "per-class a:", [A[k] for k in best])
    print(f"   eval: pair {score(ev, none):.4f}  global a=.25 {score(ev, glob):.4f}  per-class {score(ev, best):.4f}", flush=True)
