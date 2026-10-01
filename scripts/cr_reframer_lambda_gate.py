"""R06 probe: does a softer public mask (lambda > 0 on unscored required rows) expose a training-weight mismatch?

On the clean folds' evaluation sequences, with the incumbent-like mix (0.5 ft_since_s3 + 0.5 ft_since_s4) and the
fold's own OOS propensity cache:
  * pooled WP at lambda in {0, .02, .05} (inclusion = is_scored + lambda * unscored, step >= 99),
  * pre-registered LB probe: predictions on rows with propensity < tau replaced by a per-class constant,
  * |y|-weighted share of the scoring measure vs of the parents' training weight (p^2 + .03) on low-propensity rows.

    WNN_CPU_PCT=6 WNN_MEM_MAX=6G scripts/limited .venv/bin/python scripts/cr_reframer_lambda_gate.py OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.evaluation import paired_component_bootstrap  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402

spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

MEMBERS = ["c20260929_ft_since_s3_f{fold}", "c20260929_ft_since_s4_f{fold}"]
LAMBDAS = [0.0, 0.02, 0.05]
TAUS = [0.01, 0.02]


def main() -> None:
    out_path = Path(sys.argv[1])
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    position = {int(g): i for i, g in enumerate(targets["row_group"])}
    steps = np.arange(20_000)
    arms = ["base"] + [f"gate{t:g}" for t in TAUS]
    mom = {(a, lam): [] for a in arms for lam in LAMBDAS}
    share = {lam: np.zeros((len(TAUS), 2)) for lam in LAMBDAS}  # [tau, (low, total)] of |y| * inclusion
    train_share = np.zeros((len(TAUS), 2))
    classes, components = [], []
    for fold in range(4):
        split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", ROOT)
        comp = {g: c for s, g, c in split.components if s == "valid"}
        preds = [np.load(ROOT / "cache" / f"pred_{m.format(fold=fold)}_valid.npy", mmap_mode="r") for m in MEMBERS]
        prop = np.load(ROOT / "cache" / f"propcache_c20260929_prop_f{fold}_valid.npy", mmap_mode="r")
        groups = [position[g] for g in split.evaluation_groups]
        cls_of = np.array([int(targets["market_class"][i]) for i in groups])
        # per-class constant: mean prediction over required rows of that class in this fold (causal: class is known)
        const = np.zeros((4, 2))
        for c in range(4):
            acc = [np.asarray(sum(0.5 * np.asarray(pr[i, 99:], np.float64) for pr in preds)).mean(0)
                   for i, k in zip(groups, cls_of) if k == c]
            const[c] = np.mean(acc, 0)
        for g, i, c in zip(split.evaluation_groups, groups, cls_of):
            y = np.asarray(targets["y"][i], np.float64)
            m = np.asarray(targets["mask"][i], bool) & (steps >= 99)
            req = steps >= 99
            p = sum(0.5 * np.asarray(pr[i], np.float64) for pr in preds).astype(np.float32).astype(np.float64)
            pr_i = np.asarray(prop[i], np.float64)
            variants = {"base": p}
            for t in TAUS:
                q = p.copy()
                q[(pr_i < t) & req] = const[c]
                variants[f"gate{t:g}"] = q
            wy = np.abs(np.clip(y, -2, 2)).mean(1)
            for lam in LAMBDAS:
                inc = (m + lam * (req & ~m)).astype(np.float64)
                for a in arms:
                    mom[(a, lam)].append(np.stack([cs.moments(y[:, k], variants[a][:, k], inc) for k in range(2)]))
                for j, t in enumerate(TAUS):
                    low = (pr_i < t) & req
                    share[lam][j] += [np.sum(wy * inc * low), np.sum(wy * inc)]
            tw = (pr_i ** 2 + 0.03) * req
            for j, t in enumerate(TAUS):
                low = (pr_i < t) & req
                train_share[j] += [np.sum(wy * tw * low), np.sum(wy * tw)]
            classes.append(int(c))
            components.append(comp[g])
        print("fold", fold, flush=True)
    classes, components = np.array(classes), np.array(components)
    report = {"members": MEMBERS, "n_sequences": len(classes), "lambda": {}}
    for lam in LAMBDAS:
        base = np.stack(mom[("base", lam)])
        entry = {"base_wp": cs.pooled(base)["wp"],
                 "base_by_class": {int(c): cs.pooled(base[classes == c])["wp"] for c in range(4)},
                 "low_prop_measure_share": {f"{t:g}": float(share[lam][j, 0] / share[lam][j, 1]) for j, t in enumerate(TAUS)}}
        for a in arms[1:]:
            cand = np.stack(mom[(a, lam)])
            b = paired_component_bootstrap(cand, base, components, n_boot=1000, seed=271828, centered=True)
            entry[a] = {"delta": b["mean_diff"], "boot_sd": b["bootstrap_sd"],
                        "by_class_delta": {int(c): cs.pooled(cand[classes == c])["wp"] - entry["base_by_class"][c]
                                           for c in range(4)}}
        report["lambda"][f"{lam:g}"] = entry
        print(lam, json.dumps(entry), flush=True)
    report["train_weight_low_prop_share"] = {f"{t:g}": float(train_share[j, 0] / train_share[j, 1]) for j, t in enumerate(TAUS)}
    out_path.write_text(json.dumps(report, indent=1))
    print(report["train_weight_low_prop_share"])


if __name__ == "__main__":
    main()
