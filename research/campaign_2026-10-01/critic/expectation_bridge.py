"""Critic: historical holdout-delta -> public-delta bridge on the 11 public packages (forecast: expectation_mc.py).

Uses research/campaign_2026-10-01/data_eval/seqsums_public_packages_holdout.npz (exact-mask sums on the 368 hash-holdout
sequences) and the user-reported public scores stored with it. CPU only, seconds.

    WNN_CPU_PCT=7 WNN_MEM_MAX=4G CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
        research/campaign_2026-10-01/critic/expectation_bridge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from de_public_proxy import sequence_table, wp  # noqa: E402

PUB143 = np.array([0.1832, 0.1597, 0.1984, 0.4588])
z = np.load(ROOT / "research/campaign_2026-10-01/data_eval/seqsums_public_packages_holdout.npz")
names, public, sums, rows = list(z["names"]), z["public"], z["sums"], z["rows"]
table = sequence_table()
cls = table["market_class"][rows]
comp = table["component"][rows]
share = np.bincount(cls, minlength=4) / len(cls)
w_pub = (PUB143 / share)[cls]
idx = {n: i for i, n in enumerate(names)}
out = {"holdout_wp": {}, "pairs": {}}
for n in names:
    out["holdout_wp"][n] = {"all": wp(sums[idx[n]]), "pub143": wp(sums[idx[n]], w_pub), "public": float(public[idx[n]])}

# Adjacent same-lineage steps that were actually submitted in sequence (E-224, E-227, E-267), plus EMA6 under both
# CPU-path scenarios. These are the paired package deltas closest in kind to SP2 vs incumbent.
steps = [("final_s4_base35", "final_s4ft_base35"), ("final_s4ft_base35", "final_s4lpft_base35"),
         ("final_s4lpft_base35", "final_ema4_lpft_base35"), ("final_ema4_lpft_base35", "final_since2_gclp"),
         ("final_since2_gclp", "final_mixed_ema6_shared"), ("final_since2_gclp", "final_ema6_shared"),
         ("final_tv_e2", "blend_base_final_tv"), ("final_since2_gclp", "final_mixed_ema6_shared__single"),
         ("final_since2_gclp", "final_ema6_shared__single")]
rng = np.random.default_rng(271828)
ucomp, inv = np.unique(comp, return_inverse=True)
for a, b in steps:
    d_all = out["holdout_wp"][b]["all"] - out["holdout_wp"][a]["all"]
    d_pub = out["holdout_wp"][b]["pub143"] - out["holdout_wp"][a]["pub143"]
    boot = []
    for _ in range(400):  # component bootstrap SD of the holdout delta
        pick = rng.integers(0, len(ucomp), len(ucomp))
        cnt = np.bincount(pick, minlength=len(ucomp))[inv].astype(float)
        boot.append(wp(sums[idx[b]], cnt) - wp(sums[idx[a]], cnt))
    out["pairs"][f"{b}-{a}"] = {"holdout_all": d_all, "holdout_pub143": d_pub, "holdout_sd": float(np.std(boot)),
                                "public": float(public[idx[b]] - public[idx[a]])}


def through_origin(x, y):
    x, y = np.asarray(x), np.asarray(y)
    b = float((x * y).sum() / (x * x).sum())
    r = y - b * x
    return b, float(np.sqrt((r ** 2).sum() / (len(x) - 1))), r.tolist()


fits = {}
for scen, keys in {"pair": [0, 1, 2, 3, 4, 5, 6], "single": [0, 1, 2, 3, 6, 7, 8], "pair_no_tv": [0, 1, 2, 3, 4, 5]}.items():
    ks = [list(out["pairs"])[k] for k in keys]
    for kind in ("holdout_all", "holdout_pub143"):
        x = [out["pairs"][k][kind] for k in ks]
        y = [out["pairs"][k]["public"] for k in ks]
        b, s, r = through_origin(x, y)
        fits[f"{scen}/{kind}"] = {"slope": b, "resid_sd": s, "n": len(ks), "residuals": dict(zip(ks, r))}
out["bridge_fits"] = fits

print(json.dumps(out, indent=1))
