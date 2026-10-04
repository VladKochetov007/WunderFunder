"""Public-like local proxies: subset/class-reweighted variants tested against 11 public-scored packages.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_proxy_study.py

Reads the sums cached by scripts/de_proxy_cache.py. Every proxy is a per-sequence weight that
depends only on market class, component size and purity (plus the historical adversarial
train-likeness score and soft-mask lambda of the existing 21-variant library). Calibrators map
proxy WP to public WP: affine (fit slope and intercept) or shift-only (slope 1). Proxy selection is
repeated inside every outer fold (nested leave-one-submission-out and held-out family/lineage).
Writes research/campaign_2026-10-01/data_eval/proxy_study.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from de_public_proxy import proxy_weights, sequence_table, wp  # noqa: E402

OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"
FAMILY = {  # post hoc labels of scripts/campaign_calibration_audit.py
    "baseline_x025": "early", "ab18": "early", "final_tv_e2": "early", "blend_base_final_tv": "early",
    "final_s4_base35": "s4", "final_s4ft_base35": "s4", "final_s4lpft_base35": "s4",
    "final_ema4_lpft_base35": "ema4", "final_since2_gclp": "since",
    "final_mixed_ema6_shared": "ema6", "final_ema6_shared": "ema6",
}
LINEAGE = {  # shared literal members/parents (expectation audit): at most three groups
    "baseline_x025": "base", "ab18": "ab18", "final_tv_e2": "base", "blend_base_final_tv": "base",
    "final_s4_base35": "base", "final_s4ft_base35": "base", "final_s4lpft_base35": "base",
    "final_ema4_lpft_base35": "base", "final_since2_gclp": "since",
    "final_mixed_ema6_shared": "since", "final_ema6_shared": "since",
}
LAMS = (0.0, 0.02, 0.05)


def mix_weight(cls: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sequence weight moving the sequence class shares of this panel to ``target``."""
    share = np.bincount(cls, minlength=4) / len(cls)
    return (np.asarray(target) / share)[cls]


def subset_to_mix(cls: np.ndarray, subset: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Zero outside ``subset``; inside, class-reweighted so the subset's sequence shares equal ``target``."""
    share = np.bincount(cls[subset], minlength=4) / subset.sum()
    return np.where(subset, (np.asarray(target) / np.maximum(share, 1e-12))[cls], 0.0)


def mix_1p_share(s: float, valid_share: np.ndarray) -> np.ndarray:
    """Valid class proportions for classes 0-2 with the class-3 share set to s."""
    rest = valid_share[:3] / valid_share[:3].sum()
    return np.concatenate([(1 - s) * rest, [s]])


def library(t: dict, idx: np.ndarray) -> tuple[dict, dict, dict]:
    """(existing 21-variant weights, chain/class weights declared before results, post hoc weights)."""
    sub = {k: v[idx] for k, v in t.items()}
    cls = sub["market_class"]
    with np.load(ROOT / "cache" / "train_targets.npz") as z:
        train_mix = np.bincount(z["market_class"], minlength=4) / len(z["market_class"])
    adv = np.load(ROOT / "cache" / "adv_valid_score.npy")[idx]
    unlike = np.zeros(len(idx), bool)
    for c in range(4):
        s = cls == c
        unlike[s] = adv[s] >= np.quantile(adv[s], 1 / 3)
    old = {
        "holdout": np.ones(len(idx)),
        "train_mix": mix_weight(cls, train_mix),
        "class3_x2": np.where(cls == 3, 2.0, 1.0),
        "class3_x0_5": np.where(cls == 3, 0.5, 1.0),
        "isolated": sub["isolated"].astype(float),
        "linked": (~sub["isolated"]).astype(float),
        "train_unlike": unlike.astype(float),
    }
    single_mix = np.bincount(t["market_class"][t["component_len"] == 1], minlength=4)
    single_mix = single_mix / single_mix.sum()
    new = {
        "single_len1": proxy_weights(sub, {"max_component_len": 1}),
        "short_len_le2": proxy_weights(sub, {"max_component_len": 2}),
        "short_len_le3": proxy_weights(sub, {"max_component_len": 3}),
        "long_len_ge4": proxy_weights(sub, {"min_component_len": 4}),
        "pure": proxy_weights(sub, {"pure_only": True}),
        "inv_len": proxy_weights(sub, {"component_len_power": -1.0}),
        "inv_sqrt_len": proxy_weights(sub, {"component_len_power": -0.5}),
        "single_chain_class_mix": mix_weight(cls, single_mix),
        "class3_x1_5": np.where(cls == 3, 1.5, 1.0),
        "class3_x3": np.where(cls == 3, 3.0, 1.0),
        "class23_x2": np.where(cls >= 2, 2.0, 1.0),
    }
    # declared after inspecting the results above: exposure to dev-valid chain neighbours that
    # final packages trained on, and class-matched chain subsets separating class mix from chain effects
    dev = np.bincount(t["component"][~t["holdout"]], minlength=t["component"].max() + 1)[sub["component"]]
    holdout_mix = np.bincount(cls, minlength=4) / len(cls)
    posthoc = {
        "ndev0": (dev == 0).astype(float),
        "ndev_le1": (dev <= 1).astype(float),
        "ndev_ge1": (dev >= 1).astype(float),
        "ndev_le1_to_holdout_mix": subset_to_mix(cls, dev <= 1, holdout_mix),
        "short_len_le3_to_holdout_mix": subset_to_mix(cls, sub["component_len"] <= 3, holdout_mix),
        "long_len_ge4_to_short_mix": subset_to_mix(
            cls, sub["component_len"] >= 4, np.bincount(cls[sub["component_len"] <= 3], minlength=4) / (sub["component_len"] <= 3).sum()),
    }
    return old, new, posthoc


def affine(x, y):
    b, a = np.polyfit(x, y, 1)
    return lambda z: a + b * z


def shift(x, y):
    d = float(np.mean(y - x))
    return lambda z: z + d


def inner_rmse(x, y, train, calib):
    res = [calib(x[train[train != i]], y[train[train != i]])(x[i]) - y[i] for i in train]
    return float(np.sqrt(np.mean(np.square(res))))


def nested(local: np.ndarray, public: np.ndarray, groups: np.ndarray, calib) -> dict:
    """Outer: hold out each group; inner: select the proxy by LOSO RMSE on the rest, refit, predict."""
    rows = []
    for g in dict.fromkeys(groups):
        held, train = np.flatnonzero(groups == g), np.flatnonzero(groups != g)
        inner = np.array([inner_rmse(x, public, train, calib) for x in local])
        k = int(np.argmin(inner))
        f = calib(local[k, train], public[train])
        rows += [(int(i), k, float(f(local[k, i]) - public[i])) for i in held]
    err = np.array([r[2] for r in sorted(rows)])
    chosen = [r[1] for r in sorted(rows)]
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "bias": float(err.mean()),
            "errors": err.tolist(), "chosen": chosen}


def class_sums(s: np.ndarray, cls: np.ndarray, lam: float) -> np.ndarray:
    """(4, 2, 6) per-class pooled sums of (N, 2, 2, 6) sequence sums."""
    t = s[:, 0] + lam * s[:, 1]
    return np.stack([t[cls == c].sum(0) for c in range(4)])


def mix_wp(cs: np.ndarray, pi: np.ndarray, base_share: np.ndarray) -> float:
    from wnn.gmetric import wp_from_sums

    w = pi / base_share
    return float(wp_from_sums(np.einsum("c,ckj->kj", w, cs)).mean())


def fit_mix(cs_list, public, base_share):
    """Class sequence shares pi (simplex) making raw proxy WP match public levels, no affine map."""
    def softmax(z):
        e = np.exp(np.clip(z - z.max(), -30, 0))
        return e / e.sum()

    def loss(z):
        pi = softmax(z)
        return sum((mix_wp(cs, pi, base_share) - p) ** 2 for cs, p in zip(cs_list, public, strict=True))

    best = None
    for start in (np.zeros(4), np.log(base_share), np.array([0, 0, 0, 1.0]), np.array([0, 0, 1.0, 1.5])):
        r = minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return softmax(best.x)


def main() -> None:
    t = sequence_table()
    z = np.load(OUT / "seqsums_public_packages_holdout.npz")
    idx, names_all, public_all, sums_all = z["rows"], list(z["names"]), z["public"], z["sums"]
    ref = np.load(OUT / "seqsums_reference_fourfold.npz")["sums"]
    cls_h = t["market_class"][idx]
    old, new, posthoc = library(t, idx)
    report = {"reference_subsets": {}, "package_by_class": {}, "scenarios": {}}

    # reference (clean four-fold) by subset
    cls, L, W = t["market_class"], t["component_len"], ref[:, 0, :, 0].sum(1)
    subsets = {"all": np.ones(len(cls), bool), **{f"class{c}": cls == c for c in range(4)},
               "len1": L == 1, "len2": L == 2, "len3": L == 3, "len4_5": (L >= 4) & (L <= 5),
               "len6_9": (L >= 6) & (L <= 9), "len10plus": L >= 10, "len_ge2": L >= 2, "len_ge4": L >= 4,
               "pure": t["pure"], "linked_to_train": ~t["pure"]}
    subsets.update({f"class{c}_len1": (cls == c) & (L == 1) for c in range(4)})
    subsets.update({f"class{c}_len_ge2": (cls == c) & (L >= 2) for c in range(4)})
    subsets.update({f"class{c}_pure": (cls == c) & t["pure"] for c in range(4)})
    subsets.update({f"class{c}_linked": (cls == c) & ~t["pure"] for c in range(4)})
    for name, s in subsets.items():
        report["reference_subsets"][name] = {
            "n_sequences": int(s.sum()), "n_components": int(len(np.unique(t["component"][s]))),
            "weight_share": float(W[s].sum() / W.sum()), "class_mix": (np.bincount(cls[s], minlength=4) / s.sum()).tolist(),
            "wp": wp(ref[s])}
    # class-matched chain-length effect: reweight len>=2 to the len1 class mix
    s1, s2 = L == 1, L >= 2
    w = np.zeros(len(cls))
    w[s2] = mix_weight(cls[s2], np.bincount(cls[s1], minlength=4) / s1.sum())
    report["reference_subsets"]["len_ge2_reweighted_to_len1_class_mix"] = {"wp": wp(ref, w)}
    w = np.zeros(len(cls))
    w[s1] = mix_weight(cls[s1], np.bincount(cls[s2], minlength=4) / s2.sum())
    report["reference_subsets"]["len1_reweighted_to_len_ge2_class_mix"] = {"wp": wp(ref, w)}

    # reference level if the scored population had another class mix (sequence shares)
    with np.load(ROOT / "cache" / "train_targets.npz") as zt:
        train_mix = np.bincount(zt["market_class"], minlength=4) / len(zt["market_class"])
    mixes = {"valid": np.bincount(cls, minlength=4) / len(cls), "train": train_mix,
             "single_chain": np.bincount(cls[L == 1], minlength=4) / (L == 1).sum(),
             "short_le3": np.bincount(cls[L <= 3], minlength=4) / (L <= 3).sum()}
    levels = {k: {"mix": m.tolist(), "wp": wp(ref, mix_weight(cls, m))} for k, m in mixes.items()}
    for mult in (1.25, 1.5, 2.0, 3.0):
        levels[f"class3_x{mult:g}"] = {"wp": wp(ref, np.where(cls == 3, mult, 1.0))}
    lo, hi = 1.0, 20.0  # class-3 multiplier that brings the reference to the incumbent's public score
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if wp(ref, np.where(cls == 3, mid, 1.0)) > 0.6622 else (lo, mid)
    levels["class3_multiplier_for_0.6622"] = {"multiplier": lo, "class3_sequence_share": float(
        lo * (cls == 3).sum() / (lo * (cls == 3).sum() + (cls != 3).sum()))}
    report["reference_class_mix_levels"] = levels
    report["reference_short_le3_to_valid_mix"] = wp(ref, subset_to_mix(cls, L <= 3, mixes["valid"]))
    report["reference_long_ge4_to_short_mix"] = wp(ref, subset_to_mix(cls, L >= 4, mixes["short_le3"]))

    for scenario in ("pair", "single"):
        names = [n for n in names_all if "__" not in n]
        sums = []
        for n in names:
            key = f"{n}__single" if scenario == "single" and f"{n}__single" in names_all else n
            sums.append(sums_all[names_all.index(key)])
        public = np.array([public_all[names_all.index(n)] for n in names])
        variants = [(f"{k}_lam{lam:g}", w, lam, "old") for k, w in old.items() for lam in LAMS]
        variants += [(f"{k}_lam{lam:g}", w, lam, "new") for k, w in new.items() for lam in LAMS]
        variants += [(f"{k}_lam{lam:g}", w, lam, "posthoc") for k, w in posthoc.items() for lam in LAMS]
        local = np.array([[wp(s, w, lam) for s in sums] for _, w, lam, _ in variants])
        vnames = [v[0] for v in variants]
        kind = np.array([v[3] for v in variants])
        fam = np.array([FAMILY[n] for n in names])
        lin = np.array([LINEAGE[n] for n in names])
        loso = np.arange(len(names))
        res = {"names": names, "public": public.tolist(), "variants": vnames,
               "local": local.tolist(), "nested": {}, "fixed": {}}
        lam0 = np.array([v.endswith("_lam0") for v in vnames])
        libs = {"old21": kind == "old", "new33": kind == "new", "union54": kind != "posthoc",
                "union54_lam0": (kind != "posthoc") & lam0, "new_lam0": (kind == "new") & lam0,
                "all72_incl_posthoc": np.ones(len(variants), bool)}
        for cal_name, cal in (("affine", affine), ("shift", shift)):
            for lib_name, keep in libs.items():
                for grp_name, grp in (("loso", loso), ("family", fam), ("lineage", lin)):
                    r = nested(local[keep], public, grp, cal)
                    r["chosen"] = [np.array(vnames)[keep][k] for k in r["chosen"]]
                    res["nested"][f"{cal_name}|{lib_name}|{grp_name}"] = r
            # each fixed variant without selection (outer LOSO / family only)
            for j, v in enumerate(vnames):
                res["fixed"].setdefault(v, {})
                for grp_name, grp in (("loso", loso), ("family", fam), ("lineage", lin)):
                    r = nested(local[j : j + 1], public, grp, cal)
                    res["fixed"][v][f"{cal_name}|{grp_name}"] = r["rmse"]
                res["fixed"][v][f"{cal_name}|fullfit_slope"] = float(np.polyfit(local[j], public, 1)[0])
                res["fixed"][v]["mean_public_minus_local"] = float(np.mean(public - local[j]))
                res["fixed"][v]["sd_public_minus_local"] = float(np.std(public - local[j], ddof=1))
        # class-mix level match: pi fitted on training packages, no affine map
        base_share = np.bincount(cls_h, minlength=4) / len(cls_h)
        mix = {}
        for lam in LAMS:
            cs = [class_sums(s, cls_h, lam) for s in sums]
            full = fit_mix(cs, public, base_share)
            out = {"full_fit_pi": full.tolist(),
                   "full_fit_residuals": [mix_wp(c, full, base_share) - p for c, p in zip(cs, public, strict=True)]}
            for grp_name, grp in (("loso", loso), ("family", fam), ("lineage", lin)):
                err = np.zeros(len(names))
                for g in dict.fromkeys(grp):
                    held, train = np.flatnonzero(grp == g), np.flatnonzero(grp != g)
                    pi = fit_mix([cs[i] for i in train], public[train], base_share)
                    for i in held:
                        err[i] = mix_wp(cs[i], pi, base_share) - public[i]
                out[f"{grp_name}_rmse"] = float(np.sqrt(np.mean(err**2)))
                out[f"{grp_name}_errors"] = err.tolist()
            mix[f"lam{lam:g}"] = out
        res["class_mix_level_match"] = mix
        # package WP by class on the holdout panel
        res["package_by_class"] = {n: [wp(s[cls_h == c]) for c in range(4)] + [wp(s)] for n, s in zip(names, sums, strict=True)}
        report["scenarios"][scenario] = res
    (OUT / "proxy_study.json").write_text(json.dumps(report, indent=1) + "\n")
    valid_mix = np.asarray(mixes["valid"])
    mult = round(levels["class3_multiplier_for_0.6622"]["multiplier"], 2)
    implied = mix_1p_share(levels["class3_multiplier_for_0.6622"]["class3_sequence_share"], valid_mix)
    tool = [
        {"name": "public_like_class_mix",
         "spec": {"class_mix": np.round(implied, 4).tolist()},
         "why": "implied public class mix: valid proportions with class-3 share set so the clean four-fold reference "
                "scores the incumbent public 0.6622 (scripts/de_class_mix.py; same as class-3 weight x" f"{mult:g} on all valid)"},
        {"name": "short_chain_len_le3", "spec": {"max_component_len": 3},
         "why": "best pre-declared fixed proxy on the historical 11-package panel (shift/affine, LOSO/family/lineage)"},
        {"name": "train_class_mix", "spec": {"class_mix": np.round(mixes["train"], 4).tolist()},
         "why": "public test drawn with the train class mix"},
        {"name": "single_chain_class_mix", "spec": {"class_mix": np.round(mixes["single_chain"], 4).tolist()},
         "why": "stress: class mix of single-sequence chains (66% class 3)"},
    ]
    summary = {"proxies_for_tool": tool, "reference_class_mix_levels": levels,
               "reference_subsets": report["reference_subsets"], "calibration": {}}
    for scenario, res in report["scenarios"].items():
        fixed_keep = ("holdout_lam0", "train_mix_lam0", "class3_x1_5_lam0", "class3_x3_lam0", "single_chain_class_mix_lam0",
                      "single_len1_lam0", "short_len_le2_lam0", "short_len_le3_lam0", "long_len_ge4_lam0", "pure_lam0",
                      "train_unlike_lam0.05", "ndev_le1_lam0", "ndev_le1_to_holdout_mix_lam0",
                      "short_len_le3_to_holdout_mix_lam0", "long_len_ge4_to_short_mix_lam0")
        summary["calibration"][scenario] = {
            "nested_rmse": {k: v["rmse"] for k, v in res["nested"].items()},
            "nested_errors_affine_loso": {lib: res["nested"][f"affine|{lib}|loso"]["errors"]
                                          for lib in ("old21", "new33", "all72_incl_posthoc")},
            "fixed_variants": {k: res["fixed"][k] for k in fixed_keep},
            "class_mix_level_match": {k: {kk: v[kk] for kk in ("full_fit_pi", "loso_rmse", "family_rmse", "lineage_rmse")}
                                      for k, v in res["class_mix_level_match"].items()},
            "package_order": res["names"],
        }
    if (OUT / "proxy.json").exists():  # ledger entries are written by hand after review
        summary["ledger_entries"] = json.loads((OUT / "proxy.json").read_text()).get("ledger_entries", [])
    (OUT / "proxy.json").write_text(json.dumps(summary, indent=1) + "\n")
    for scenario, res in report["scenarios"].items():
        print("==", scenario)
        for k, r in res["nested"].items():
            print(f"{k:40s} rmse {r['rmse']*1e4:6.2f}e-4 bias {r['bias']*1e4:+6.2f}e-4")
        for lam, r in res["class_mix_level_match"].items():
            print("mix", lam, np.round(r["full_fit_pi"], 3), "loso", round(r["loso_rmse"] * 1e4, 2), "family",
                  round(r["family_rmse"] * 1e4, 2), "lineage", round(r["lineage_rmse"] * 1e4, 2))


if __name__ == "__main__":
    main()
