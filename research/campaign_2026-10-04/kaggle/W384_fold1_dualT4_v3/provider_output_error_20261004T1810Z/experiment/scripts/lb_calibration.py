"""Calibrate holdout proxies against known public leaderboard scores, then rank new candidates.

    scripts/limited .venv/bin/python scripts/lb_calibration.py

Every scored package's members were trained without the 368 reserved holdout sequences, so all
of them can be scored cleanly there. For each proxy variant (mask softness, class mix, sequence
subset) the script fits public = a + b * local over the scored packages, reports rank agreement
and leave-one-out error, and applies the best variants to candidate blends.
"""

from __future__ import annotations

import json
from itertools import combinations

import decide
import numpy as np
from ensemble_scan import parse_blend
from holdout_compare import blend_sums

from wnn.splits import valid_row_groups

SCORED = {
    "baseline_x025": ("baseline", 0.6503),
    "ab18": ("ab18_wmse_struct", 0.6488),
    "final_tv_e2": ("ab21_final_tv_e2", 0.6548),
    "blend_base_final_tv": ("ab21_final_tv_e2*0.5+baseline*0.5", 0.6569),
    "final_s4_base35": ("ab33_final_s4*0.65+baseline*0.35", 0.6586),
    "final_s4ft_base35": ("ab40_final_s4_ft*0.65+baseline*0.35", 0.6600),
    "final_s4lpft_base35": ("ab51_final_lpft*0.65+baseline*0.35", 0.6601),
    "final_ema4_lpft_base35": ("ab60_final_ema4_p1_lpft*0.65+baseline*0.35", 0.6618),
    "final_since2_gclp": ("ab67_final_since_s4_gcorr+ab67_final_since_s3_gcorr", 0.6622),
    "final_mixed_ema6_shared": ("ab72_final_ema6_s3_ep5+ab67_final_since_s4", 0.6616),
}
PAIR = "ab67_final_since_s4_gcorr+ab67_final_since_s3_gcorr"
DIV = "ab84_final_ch500_s5_gclp+ab84_final_onorm_s5_gclp"
CANDIDATES = {
    "keep_k4": f"{PAIR}+{DIV}",
    "keep_k3_ch500": f"{PAIR}+ab84_final_ch500_s5_gclp",
    "keep_k3_onorm": f"{PAIR}+ab84_final_onorm_s5_gclp",
    "new_k4": f"ab83_final_ema6_s3_gclp+ab83_final_ema6_s4_gclp+{DIV}",
    "keep_k4_base10": f"{PAIR}+{DIV}+baseline*0.1",
    "pair_base15": f"{PAIR}+baseline*0.15",
}


def main() -> None:
    with np.load(decide.ROOT / "cache" / "valid_targets.npz") as v:
        y, mask, rg, cls = v["y"], v["mask"], v["row_group"], v["market_class"]
    with np.load(decide.ROOT / "cache" / "train_targets.npz") as t:
        train_mix = np.bincount(t["market_class"], minlength=4) / len(t["market_class"])
    _, held = valid_row_groups(decide.ROOT / "datasets" / "valid.parquet")
    hold = np.isin(rg, held)
    idx = np.flatnonzero(hold)
    chain, _ = decide._chains()
    cf = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    in_train = set(cf["chain"][cf["split"] == "train"].tolist()) | set(chain[~hold].tolist())
    iso = np.array([c not in in_train for c in chain[idx]])
    adv = np.load(decide.ROOT / "cache" / "adv_valid_score.npy")[idx]
    c = cls[idx]
    counts = np.bincount(c, minlength=4)

    def mix_w(mix):
        return mix[c] * len(c) / counts[c]

    unlike = np.zeros(len(idx), bool)
    for k in range(4):
        sel = c == k
        unlike[sel] = adv[sel] >= np.quantile(adv[sel], 1 / 3)
    variants = {}
    for lam in (0.0, 0.02, 0.05):
        variants[f"holdout lam{lam:g}"] = (lam, np.ones(len(idx)))
        variants[f"train-mix lam{lam:g}"] = (lam, mix_w(train_mix))
        variants[f"class3x2 lam{lam:g}"] = (lam, np.where(c == 3, 2.0, 1.0))
        variants[f"class3x0.5 lam{lam:g}"] = (lam, np.where(c == 3, 0.5, 1.0))
        variants[f"iso-only lam{lam:g}"] = (lam, iso.astype(float))
        variants[f"nbr-only lam{lam:g}"] = (lam, (~iso).astype(float))
        variants[f"train-unlike lam{lam:g}"] = (lam, unlike.astype(float))

    sums = {name: blend_sums(parse_blend(spec), idx, y, mask) for name, (spec, _) in SCORED.items()}
    cand = {name: blend_sums(parse_blend(spec), idx, y, mask) for name, spec in CANDIDATES.items()}
    inc = sums["final_since2_gclp"]
    lb = np.array([pub for _, pub in SCORED.values()])
    names = list(SCORED)

    def score(s, lam, w):
        return decide.wp(s * w[:, None, None, None], lam)

    rows = []
    for vname, (lam, w) in variants.items():
        loc = np.array([score(sums[n], lam, w) for n in names])
        rho = float(np.corrcoef(np.argsort(np.argsort(loc)), np.argsort(np.argsort(lb)))[0, 1])
        loo = []
        for i in range(len(names)):
            keep = np.arange(len(names)) != i
            b, a = np.polyfit(loc[keep], lb[keep], 1)
            loo.append(a + b * loc[i] - lb[i])
        top = [i for i in range(len(names)) if lb[i] >= 0.6586]
        pairs = [(i, j) for i, j in combinations(top, 2) if abs(lb[i] - lb[j]) >= 0.0004]
        sign = np.mean([np.sign(loc[i] - loc[j]) == np.sign(lb[i] - lb[j]) for i, j in pairs])
        b, a = np.polyfit(loc, lb, 1)
        pred = {k: float(a + b * score(s, lam, w)) for k, s in cand.items()}
        d_inc = {k: float(b * (score(s, lam, w) - score(inc, lam, w))) for k, s in cand.items()}
        rows.append({"variant": vname, "spearman": rho, "loo_rmse": float(np.sqrt(np.mean(np.square(loo)))),
                     "top_pair_sign_acc": float(sign), "n_top_pairs": len(pairs), "slope": float(b),
                     "pred_public": pred, "pred_delta_vs_incumbent": d_inc})
    rows.sort(key=lambda r: r["loo_rmse"])
    for r in rows[:10]:
        d = r["pred_delta_vs_incumbent"]
        print(f"{r['variant']:22s} rho {r['spearman']:.2f} loo {r['loo_rmse'] * 1e4:.1f}e-4 "
              f"topsign {r['top_pair_sign_acc']:.2f}/{r['n_top_pairs']} slope {r['slope']:.2f} | "
              + " ".join(f"{k} {v * 1e4:+.1f}" for k, v in d.items()))
    out = decide.ROOT / "research" / "artifacts" / "lb_calibration.json"
    out.write_text(json.dumps({"isolated_n": int(iso.sum()), "variants": rows}, indent=1))


if __name__ == "__main__":
    main()
