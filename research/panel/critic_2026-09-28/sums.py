"""Per-sequence WP sums (scored rows) for member blends. Independent of scripts/decide.py.

Output: npz with, per blend name, an array (1873, 2 targets, 6) of
[sum w, sum w y, sum w p, sum w y^2, sum w p^2, sum w y p] over mask rows, clipped to [-2, 2].
"""
import sys
import numpy as np

C = "cache/"
d = np.load(C + "valid_targets.npz")
Y, M = d["y"], d["mask"]
N = len(Y)

FIN = {
    "s4g": "ab67_final_since_s4_gcorr", "s3g": "ab67_final_since_s3_gcorr",
    "ch": "ab84_final_ch500_s5_gclp", "on": "ab84_final_onorm_s5_gclp",
    "e3": "ab83_final_ema6_s3_gclp", "e4": "ab83_final_ema6_s4_gclp",
}
FOLD = {f"{a}_f{k}": f"ab82_ft_{b}_f{k}" for k in (0, 1) for a, b in
        [("s3", "since4s3"), ("b4", "base4s4"), ("ch", "ch500"), ("on", "onorm"), ("e3", "ema6s3"), ("e4", "ema6s4")]}

BLENDS = {
    # final models
    "F_inc": ["s4g", "s3g"], "F_k4": ["s4g", "s3g", "ch", "on"], "F_k3ch": ["s4g", "s3g", "ch"],
    "F_k3on": ["s4g", "s3g", "on"], "F_A": ["e3", "e4", "ch", "on"],
    "F_s4g": ["s4g"], "F_s3g": ["s3g"], "F_ch": ["ch"], "F_on": ["on"],
}
for k in (0, 1):
    f = lambda *xs: [f"{x}_f{k}" for x in xs]
    BLENDS.update({
        f"P{k}_inc": f("s3", "b4"), f"P{k}_k4": f("s3", "b4", "ch", "on"),
        f"P{k}_A": f("e3", "e4", "ch", "on"),
        # alternative 'incumbent-like' bases (two since-weighted members) and the same +ch+on lever
        f"P{k}_se": f("s3", "e4"), f"P{k}_se_k4": f("s3", "e4", "ch", "on"),
        f"P{k}_ee": f("e3", "e4"), f"P{k}_ee_k4": f("e3", "e4", "ch", "on"),
        f"P{k}_s3": f("s3"), f"P{k}_b4": f("b4"), f"P{k}_ch": f("ch"), f"P{k}_on": f("on"),
    })

names = {**FIN, **FOLD}
arrs = {a: np.load(C + f"pred_{n}_valid.npy", mmap_mode="r") for a, n in names.items()}
for a, x in arrs.items():
    assert x.shape == (N, 20000, 2), (a, x.shape)

out = {b: np.zeros((N, 2, 6)) for b in BLENDS}
CH = 25
for s in range(0, N, CH):
    sl = slice(s, min(N, s + CH))
    y = np.clip(Y[sl].astype(np.float64), -2, 2)
    m = M[sl]
    w = np.abs(y) * m[..., None]
    wy = w * y
    cache = {}
    for b, mem in BLENDS.items():
        p = 0.0
        for a in mem:
            if a not in cache:
                cache[a] = np.asarray(arrs[a][sl], dtype=np.float64)
            p = p + cache[a]
        p = np.clip(p / len(mem), -2, 2)
        wp = w * p
        out[b][sl] = np.stack([w.sum(1), wy.sum(1), wp.sum(1), (wy * y).sum(1), (wp * p).sum(1), (wy * p).sum(1)], -1)
    if s % 200 == 0:
        print(s, flush=True)
    for a, v in cache.items():
        if not np.isfinite(v).all():
            print("NONFINITE", a, s)
np.savez(sys.argv[1], **out)
print("done")
