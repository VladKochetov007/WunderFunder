"""Implied public class mix: class-reweighted pooled Global WP matched to public scores.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/de_class_mix.py

Reweighting rule (same as scripts/screen_score.py ``reweighted``): every sequence of class c gets weight
mix_c / local_share_c, where local_share is the class share of sequences in the scored panel. Families:
  1p: valid class mix with the class-3 share replaced by s (classes 0-2 keep their valid proportions);
  2p: class-2 and class-3 shares free, classes 0:1 keep their valid ratio;
  3p: free simplex.
(a) clean four-fold reference (since pair): s solving WP = 0.6622 (minus an incumbent-vs-reference
    offset), component bootstrap. (b) 11 public packages on the 368 holdout sequences: least squares
    level match without affine map, nested LOSO / family / lineage holdout, component bootstrap of the fit.
Writes research/campaign_2026-10-01/data_eval/class_mix.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from de_proxy_study import FAMILY, LINEAGE  # noqa: E402
from de_public_proxy import sequence_table  # noqa: E402

from wnn.gmetric import wp_from_sums  # noqa: E402

OUT = ROOT / "research" / "campaign_2026-10-01" / "data_eval"
PUBLIC_INCUMBENT = 0.6622
GRID = np.linspace(0.20, 0.80, 1201)


def class_sums(seq_sums: np.ndarray, cls: np.ndarray, counts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(4, 2, 6) scored-row sums per class and (4,) sequence counts; ``counts`` = bootstrap multiplicities."""
    c = np.ones(len(cls)) if counts is None else counts
    t = seq_sums[:, 0] * c[:, None, None]
    return np.stack([t[cls == k].sum(0) for k in range(4)]), np.bincount(cls, weights=c, minlength=4)


def mix_1p(s, valid_share):
    s = np.atleast_1d(s)[:, None]
    rest = valid_share[:3] / valid_share[:3].sum()
    return np.concatenate([(1 - s) * rest[None], s], 1)


def wp_mix(cs: np.ndarray, n: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """WP for one or many mixes (M, 4) on one panel's class sums."""
    mix = np.atleast_2d(mix)
    f = mix / (n / n.sum())[None]
    return wp_from_sums(np.einsum("mc,ckj->mkj", f, cs)).mean(-1)


def solve_1p(cs, n, target, valid_share):
    """Class-3 share whose reweighted WP equals target (WP falls monotonically in s); nan if off-grid."""
    v = wp_mix(cs, n, mix_1p(GRID, valid_share))
    above = v >= target
    if above.all() or not above.any():
        return float("nan")
    i = int(np.argmin(above)) - 1
    return float(GRID[i] + (GRID[i + 1] - GRID[i]) * (v[i] - target) / (v[i] - v[i + 1]))


def softmax(z):
    e = np.exp(np.clip(z - z.max(), -30, 0))
    return e / e.sum()


def fit(panels, public, valid_share, k: int) -> np.ndarray:
    """Least-squares level match of class mix (1p, 2p or 3p family) to public scores."""
    if k == 1:
        r = minimize_scalar(lambda s: sum((wp_mix(cs, n, mix_1p(s, valid_share))[0] - p) ** 2
                                          for (cs, n), p in zip(panels, public, strict=True)), bounds=(0.05, 0.95), method="bounded")
        return mix_1p(r.x, valid_share)[0]
    r01 = valid_share[:2] / valid_share[:2].sum()

    def to_mix(z):
        if k == 2:
            p = softmax(np.array([0.0, z[0], z[1]]))  # [classes 0+1, class 2, class 3]
            return np.array([p[0] * r01[0], p[0] * r01[1], p[1], p[2]])
        return softmax(np.concatenate([[0.0], z]))

    def loss(z):
        m = to_mix(z)
        return sum((wp_mix(cs, n, m)[0] - p) ** 2 for (cs, n), p in zip(panels, public, strict=True))

    start = np.log(valid_share / valid_share[0])[1:] if k == 3 else np.log(
        np.array([valid_share[2], valid_share[3]]) / valid_share[:2].sum())
    best = None
    for jitter in (0.0, 0.5, -0.5):
        r = minimize(loss, start + jitter, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 6000})
        best = r if best is None or r.fun < best.fun else best
    return to_mix(best.x)


def main() -> None:
    t = sequence_table()
    cls_all = t["market_class"]
    valid_share = np.bincount(cls_all, minlength=4) / len(cls_all)
    rng = np.random.default_rng(271828)
    report = {"rule": "sequence weight mix_c / local_share_c (sequence shares); means unchanged",
              "valid_share": valid_share.tolist()}

    # (a) clean four-fold reference
    ref = np.load(OUT / "seqsums_reference_fourfold.npz")["sums"]
    comp = t["component"]
    ids, inv = np.unique(comp, return_inverse=True)
    cs, n = class_sums(ref, cls_all)
    offsets = (-0.002, 0.0, 0.002, 0.004)
    a = {"wp_valid_mix": float(wp_mix(cs, n, valid_share)[0]), "by_offset": {}}
    boots = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=2000)
    for off in offsets:
        target = PUBLIC_INCUMBENT - off
        s_hat = solve_1p(cs, n, target, valid_share)
        bs = np.array([solve_1p(*class_sums(ref, cls_all, b[inv].astype(float)), target, valid_share) for b in boots])
        a["by_offset"][f"{off:+g}"] = {
            "target_reference_wp": target, "class3_share": s_hat, "mix": mix_1p(s_hat, valid_share)[0].tolist(),
            "boot_sd": float(np.nanstd(bs, ddof=1)), "boot_q025": float(np.nanquantile(bs, 0.025)),
            "boot_q975": float(np.nanquantile(bs, 0.975)), "boot_nan": int(np.isnan(bs).sum())}
    a["note"] = ("offset = assumed clean-local advantage of the full incumbent over the fold reference; "
                 "one equation identifies only the 1p family")
    report["reference"] = a

    # (b) 11 public packages on the holdout panel
    z = np.load(OUT / "seqsums_public_packages_holdout.npz")
    idx, names_all = z["rows"], list(z["names"])
    cls_h, comp_h = cls_all[idx], comp[idx]
    hid, hinv = np.unique(comp_h, return_inverse=True)
    hb = rng.multinomial(len(hid), np.full(len(hid), 1 / len(hid)), size=300)
    report["packages"] = {}
    for scenario in ("pair", "single"):
        names = [x for x in names_all if "__" not in x]
        keys = [f"{x}__single" if scenario == "single" and f"{x}__single" in names_all else x for x in names]
        sums = [z["sums"][names_all.index(k)] for k in keys]
        public = np.array([z["public"][names_all.index(x)] for x in names])
        panels = [class_sums(s, cls_h) for s in sums]
        res = {"names": names, "public": public.tolist()}
        for k in (1, 2, 3):
            m = fit(panels, public, valid_share, k)
            pred = np.array([wp_mix(cs_, n_, m)[0] for cs_, n_ in panels])
            out = {"mix": m.tolist(), "in_sample_rmse": float(np.sqrt(np.mean((pred - public) ** 2))),
                   "residuals": (pred - public).tolist()}
            for gname, groups in (("loso", np.arange(len(names))), ("family", np.array([FAMILY[x] for x in names])),
                                  ("lineage", np.array([LINEAGE[x] for x in names]))):
                err, mixes = np.zeros(len(names)), []
                for g in dict.fromkeys(groups):
                    held, train = np.flatnonzero(groups == g), np.flatnonzero(groups != g)
                    mg = fit([panels[i] for i in train], public[train], valid_share, k)
                    mixes.append(mg.tolist())
                    for i in held:
                        err[i] = wp_mix(*panels[i], mg)[0] - public[i]
                out[f"{gname}_rmse"] = float(np.sqrt(np.mean(err**2)))
                out[f"{gname}_errors"] = err.tolist()
                out[f"{gname}_fold_mixes"] = mixes
            if k <= 2:
                bm = []
                for b in hb[: 300 if k == 1 else 100]:
                    w = b[hinv].astype(float)
                    bm.append(fit([class_sums(s, cls_h, w) for s in sums], public, valid_share, k))
                bm = np.array(bm)
                out["boot_mix_mean"] = bm.mean(0).tolist()
                out["boot_mix_sd"] = bm.std(0, ddof=1).tolist()
                out["boot_class3_q025_q975"] = [float(np.quantile(bm[:, 3], 0.025)), float(np.quantile(bm[:, 3], 0.975))]
            res[f"{k}p"] = out
        report["packages"][scenario] = res
    (OUT / "class_mix.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"reference": {o: (v["class3_share"], v["boot_sd"]) for o, v in a["by_offset"].items()},
                      **{sc: {k: (np.round(r[k]["mix"], 3).tolist(), round(r[k]["loso_rmse"] * 1e4, 1),
                                  round(r[k]["family_rmse"] * 1e4, 1), round(r[k]["lineage_rmse"] * 1e4, 1),
                                  np.round(r[k].get("boot_mix_sd", []), 3).tolist())
                              for k in ("1p", "2p", "3p")} for sc, r in report["packages"].items()}}, indent=1))


if __name__ == "__main__":
    main()
