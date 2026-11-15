"""OOF blend analysis of valid-only fine-tuned (V) members vs train-only bases, chain folds of valid (n=2).
V member OOF: rows of fold f come from the model fine-tuned on the other fold. Exposed valid; fixed grids."""
import json, sys
from pathlib import Path
import numpy as np
C = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
with np.load(C / "valid_targets.npz") as t:
    groups, mask, Y = t["row_group"], t["mask"], t["y"]
cf = np.load(C / "chain_folds.npz"); v = cf["split"] == "valid"
fmap = dict(zip(cf["row_group"][v].tolist(), (cf["fold"][v] % 2).tolist()))
sf = np.array([fmap[int(g)] for g in groups]); rf = np.repeat(sf, mask.sum(1))
y = np.clip(Y[mask].astype(np.float64), -2, 2); w = np.abs(y)
def load(n): return np.load(C / f"pred_{n}_valid.npy", mmap_mode="r")[mask].astype(np.float64)
def oof(prefix):
    a, b = load(f"{prefix}_f0"), load(f"{prefix}_f1"); return np.where((rf == 0)[:, None], a, b)
B = {"tl5": load("q14t_tlead_s5"), "tl6": load("q14t_tlead_s6"), "fbc": load("q14t_fbc_s5"), "kd": load("q14s_local_kd90_s3")}
V = {"tl5": oof("q15v_V"), "tl6": oof("q15v_V6"), "fbc": oof("q15v_Vfbc"), "kd": oof("q15v_Vkd")}
def wp(p):
    res = []
    for f in (0, 1, None):
        sel = np.ones(len(y), bool) if f is None else rf == f; r = []
        for j in range(2):
            yy, pp, ww = y[sel, j], np.clip(p[sel, j], -2, 2), w[sel, j]; W = ww.sum()
            my, mp = (ww * yy).sum() / W, (ww * pp).sum() / W
            r.append((ww * (yy - my) * (pp - mp)).sum() / np.sqrt((ww * (yy - my) ** 2).sum() * (ww * (pp - mp) ** 2).sum()))
        res.append(round(float(np.mean(r)), 6))
    return res  # [fold0, fold1, all]
tl3 = lambda D: .34 * D["fbc"] + .33 * D["tl5"] + .33 * D["tl6"]
q = lambda D: .75 * tl3(D) + .25 * D["kd"]
eq = lambda D: (D["fbc"] + D["tl5"] + D["tl6"] + D["kd"]) / 4
out = {"single_base": {k: wp(x) for k, x in B.items()}, "single_V": {k: wp(x) for k, x in V.items()},
       "tl3c_base": wp(tl3(B)), "tl3c_V": wp(tl3(V)), "tl3c_kd_base": wp(q(B)), "tl3c_kd_V": wp(q(V)),
       "eq4_base": wp(eq(B)), "eq4_V": wp(eq(V)), "mix_base_V": {}}
for a in (.25, .5, .75):
    out["mix_base_V"][a] = wp(a * q(V) + (1 - a) * q(B))
out["corr_V_vs_base"] = {k: float(np.mean([np.corrcoef(V[k][:, j], B[k][:, j])[0, 1] for j in range(2)])) for k in B}
out["corr_V_members"] = {f"{a}|{b}": float(np.mean([np.corrcoef(V[a][:, j], V[b][:, j])[0, 1] for j in range(2)]))
                         for i, a in enumerate(V) for b in list(V)[i + 1:]}
print(json.dumps(out, indent=1)); Path(__file__).with_name("OOF_BLEND_R1.json").write_text(json.dumps(out, indent=1))
