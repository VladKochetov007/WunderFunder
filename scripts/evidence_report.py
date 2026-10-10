"""Effect-size report for scratch-teacher recipe changes on one fold, with honest uncertainty.

For every model with a saved prediction on the fold: official WP, the recipe factors that differ from the default scratch
recipe (read from the saved checkpoint arguments), and its effect against the default-recipe baseline runs:

- ``delta``: WP minus the mean WP of the baseline runs (default recipe, different seeds);
- ``boot``: paired component-bootstrap interval of the difference to one baseline run, i.e. how consistent the difference is
  across evaluation units (chain components);
- ``seed_sd``: run-to-run training noise estimated from the baseline runs, added in quadrature (a single run is one draw);
- ``shrunk``: empirical-Bayes posterior mean of the effect (removes the winner's curse: the largest observed effects are
  the most inflated), using the spread of all observed effects to estimate the true-effect variance.

Public-vs-local check: rank correlation of local fold scores with public scores for recipes that exist in both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

DEFAULT = {"optim": "muon", "lr": 0.001, "muon_lr_mult": 1.0, "wd": 0.01, "epochs": 4, "hidden": 256, "layers": 2,
           "cell": "gru", "enc": "linear", "sched": "onecycle", "loss": "wmse", "dropout": 0.0, "batch": 32, "chunk": 1000,
           "arch": "gru", "rnn_init": "default", "readout_activation": "gelu", "ema_teacher": 0.0, "swa_frac": 0.0,
           "clip": 1.0, "class_weight": "", "label_weight_power": 1.0, "label_weight_floor": 0.0, "head_lr_mult": 1.0,
           "since_weight": 32.0, "prop_power": 2.0, "aux_struct": 0.3, "aux_next": 0.3, "raw_only": False,
           "heuristics": False, "online_norm": False, "muon_scope": "rnn", "thin_prob": 0.0, "seed": None}
LABELS = {"muon_lr_mult": "muon x{}", "lr": "lr {}", "hidden": "h{}", "layers": "{}L", "epochs": "{}ep", "sched": "{}",
          "loss": "loss {}", "ema_teacher": "ema", "swa_frac": "swa {}", "rnn_init": "init {}", "arch": "{}", "cell": "{}",
          "enc": "enc {}", "dropout": "drop {}", "optim": "{}", "wd": "wd {}", "readout_activation": "{}", "raw_only": "raw-only",
          "class_weight": "cw {}", "label_weight_power": "lwp {}", "label_weight_floor": "lwf {}", "head_lr_mult": "head x{}",
          "since_weight": "since {}", "prop_power": "prop {}", "chunk": "chunk {}", "batch": "b{}", "clip": "clip {}",
          "heuristics": "heuristics", "online_norm": "onorm", "muon_scope": "scope {}", "thin_prob": "thin {}"}
PUBLIC = {"c20261003_combo_s4": 0.6660, "c20261003_combo_s3": 0.6655, "c20261004_combo_head_s3": 0.6654,
          "c20261004_combo_parent_s3": 0.6641, "c20261002_te07_s3": 0.6636, "c20261002_d07_s3": 0.6635,
          "c20261002_d07_s4": 0.6639, "c20261007_c1_s3": 0.6647, "c20261007_c1_s4": 0.6654}


def factors(args: dict) -> dict:
    out = {}
    for key, default in DEFAULT.items():
        if key == "seed":
            continue
        value = args.get(key, default)
        if key == "ema_teacher":
            value = float(value) if args.get("eval_ema") else 0.0
        if key == "optim":
            value = value or "adamw"
        if value is None:
            value = default
        if isinstance(default, float) and isinstance(value, (int, float)):
            same = abs(float(value) - default) < 1e-9
        else:
            same = value == default
        if not same:
            out[key] = value
    return out


STUDENT_KEYS = ("init", "epochs", "lr", "optim", "sched", "distill_alpha", "distill_family_weight", "distill_cache_variant",
                "distill_boot_h", "distill_family_target_mode", "seed", "batch", "chunk", "wd", "loss", "muon_lr_mult")


def student_factors(args: dict) -> dict:
    return {k: args.get(k) for k in STUDENT_KEYS if args.get(k) is not None}


def label(diff: dict) -> str:
    if not diff:
        return "default"
    parts = [LABELS.get(k, k + " {}").format(v if not isinstance(v, float) else round(v, 4)) for k, v in diff.items()]
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--prefix", default="vt_")
    parser.add_argument("--extra-prefix", default="combo_scratch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--students", action="store_true", help="report distilled/warm-started runs instead of scratch ones")
    parser.add_argument("--baselines", nargs="*", default=[], help="run names that form the control group (students mode)")
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    rows = []
    moments = {}
    for _stem, (name, array, order) in sorted(models.items()):
        if not (name.startswith(args.prefix) or name.startswith(args.extra_prefix)) or "__ep" in name or "dsq" in name:
            continue
        checkpoint = next(iter(sorted((ROOT / "artifacts/runs" / name).glob("model_ep*.pt"))[-1:]), None)
        if checkpoint is None:
            continue
        ck_args = torch.load(checkpoint, map_location="cpu", weights_only=False).get("args", {})
        is_student = bool(ck_args.get("distill") or ck_args.get("init") or ck_args.get("distill_cache_root"))
        if is_student != args.students:
            continue  # students and warm-started runs are a different population
        p = np.asarray(array[order], dtype=np.float32)
        wp = score(y, mask, p)["wp"]
        moments[name] = paired_component_moments(y, p, p, mask)[0]
        del p
        diff = factors(ck_args) if not args.students else student_factors(ck_args)
        rows.append({"run": name, "wp": wp, "factors": diff, "label": label(diff), "seed": ck_args.get("seed")})
    if args.students:
        base = [r for r in rows if r["run"] in args.baselines]
        ref_factors = base[0]["factors"] if base else {}
        for r in rows:
            r["factors"] = {k: v for k, v in r["factors"].items() if ref_factors.get(k) != v or k not in ref_factors} if r not in base else {}
            r["label"] = label(r["factors"])
    else:
        base = [r for r in rows if not r["factors"]]
    base_wp = np.array([r["wp"] for r in base])
    base_mean = float(base_wp.mean()) if len(base_wp) else float("nan")
    seed_sd = float(max(base_wp.std(ddof=1), 0.0008)) if len(base_wp) > 1 else 0.001
    reference = moments[base[0]["run"]] if base else None
    for r in rows:
        r["delta"] = r["wp"] - base_mean
        if reference is not None and r["run"] != base[0]["run"]:
            boot = paired_component_bootstrap(moments[r["run"]], reference, comp, n_boot=args.n_boot, seed=17, centered=True)
            lo, hi = boot["ci95"]
            r["boot_sd"] = float((hi - lo) / 3.92)
        else:
            r["boot_sd"] = 0.0
        r["sd_total"] = float(np.sqrt(r["boot_sd"] ** 2 + seed_sd ** 2 + seed_sd ** 2 / max(len(base), 1)))
        r["ci95"] = [r["delta"] - 1.96 * r["sd_total"], r["delta"] + 1.96 * r["sd_total"]]
    effects = [r for r in rows if r["factors"]]
    # diverged / catastrophic runs (|delta| > 0.015) are real but would swamp the true-effect variance estimate: leave them out of it
    regular = [r for r in effects if abs(r["delta"]) < 0.015]
    if len(regular) > 3:
        deltas = np.array([r["delta"] for r in regular])
        variances = np.array([r["sd_total"] ** 2 for r in regular])
        tau2 = max(float(deltas.var() - variances.mean()), 2.5e-7)  # floor: true-effect sd at least 0.0005
        for r in effects:
            r["shrunk"] = r["delta"] * tau2 / (tau2 + r["sd_total"] ** 2) if abs(r["delta"]) < 0.015 else r["delta"]
        shrink_info = {"tau": float(np.sqrt(tau2)), "mean_sd_total": float(np.sqrt(variances.mean()))}
    else:
        shrink_info = {}
    pairs = [(PUBLIC[s], next(r["wp"] for r in [{"wp": score(y, mask, np.asarray(models[s][1][models[s][2]], dtype=np.float32))["wp"]}]))
             for s in PUBLIC if s in models]
    public = {}
    if len(pairs) >= 4:
        from scipy.stats import pearsonr, spearmanr

        a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        public = {"n": len(pairs), "spearman": float(spearmanr(a, b)[0]), "pearson": float(pearsonr(a, b)[0]),
                  "pairs": [[float(x), float(z)] for x, z in pairs]}
    report = {"fold": args.fold, "baseline_runs": [r["run"] for r in base], "baseline_mean": base_mean, "seed_sd": seed_sd,
              "n_models": len(rows), "shrinkage": shrink_info, "public_vs_local": public,
              "rows": sorted(rows, key=lambda r: -r["wp"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, default=float))
    for r in report["rows"][:60]:
        print(f"{r['wp']:.5f} d={r['delta']:+.4f} ci=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"shrunk={r.get('shrunk', float('nan')):+.4f}  {r['label'][:60]}  <{r['run']}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
