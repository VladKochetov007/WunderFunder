import sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack"); sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import valid_row_groups
d = np.load(ROOT / "cache/valid_targets.npz"); Y, M, rg = d["y"], d["mask"], d["row_group"]
_, held = valid_row_groups(ROOT / "datasets/valid.parquet"); H = np.flatnonzero(np.isin(rg, held))
PK = {  # package: ([(member, coef)], public score)
 "s4_base35": ([("ab33_final_s4", .65), ("baseline", .0875)], .6586),
 "s4ft_base35": ([("ab40_final_s4_ft", .65), ("baseline", .0875)], .6600),
 "s4lpft_base35": ([("ab51_final_lpft", .65), ("baseline", .0875)], .6601),
 "ema4_lpft_base35": ([("ab60_final_ema4_p1_lpft", .65), ("baseline", .0875)], .6618),
 "mixed_ema6_shared": ([("ab72_final_ema6_s3_ep5", .5), ("ab67_final_since_s4", .5)], .6616),
 "since2_gclp": ([("ab67_final_since_s4_gcorr", .5), ("ab67_final_since_s3_gcorr", .5)], .6622),
 "keep_k4": ([("ab67_final_since_s4_gcorr", .25), ("ab67_final_since_s3_gcorr", .25), ("ab84_final_ch500_s5_gclp", .25), ("ab84_final_onorm_s5_gclp", .25)], np.nan),
}
arr = {}
def get(n):
    if n not in arr: arr[n] = np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r")
    return arr[n]
def sums(y, p, m):
    w = np.abs(y) * m[..., None]; wy, wp = w * y, w * p
    return np.stack([w.sum(1), wy.sum(1), wp.sum(1), (wy * y).sum(1), (wp * p).sum(1), (wy * p).sum(1)], -1)
def wp(T):
    w = T[..., 0]; my, mp = T[..., 1] / w, T[..., 2] / w
    return (T[..., 5] / w - my * mp) / np.sqrt(np.maximum(T[..., 3] / w - my * my, 1e-300) * np.maximum(T[..., 4] / w - mp * mp, 1e-300))
res = {}
for name, (mem, pub) in PK.items():
    S = np.zeros((len(H), 2, 6))
    for a in range(0, len(H), 23):
        ix = H[a:a + 23]
        y = np.clip(Y[ix].astype(np.float64), -2, 2)
        p = np.clip(sum(c * np.asarray(get(n)[ix], np.float64) for n, c in mem), -2, 2)
        S[a:a + 23] = sums(y, p, M[ix])
    g = float(wp(S.sum(0)).mean()); ps = wp(S).mean(-1); ps = float(np.nanmean(ps))
    res[name] = (g, ps, pub); print(f"{name:<20} holdout global={g:.5f} per-seq={ps:.5f} public={pub}")
from scipy.stats import spearmanr, pearsonr
names = [n for n in PK if not np.isnan(PK[n][1])]
g = np.array([res[n][0] for n in names]); s = np.array([res[n][1] for n in names]); p = np.array([res[n][2] for n in names])
print("spearman vs public: global", spearmanr(g, p)[0], " per-seq", spearmanr(s, p)[0])
print("pearson  vs public: global", pearsonr(g, p)[0], " per-seq", pearsonr(s, p)[0])
print("mixed - since2: global %+.5f per-seq %+.5f public %+.4f" % (res["mixed_ema6_shared"][0]-res["since2_gclp"][0], res["mixed_ema6_shared"][1]-res["since2_gclp"][1], .6616-.6622))
print("keep_k4 - since2: global %+.5f per-seq %+.5f" % (res["keep_k4"][0]-res["since2_gclp"][0], res["keep_k4"][1]-res["since2_gclp"][1]))
