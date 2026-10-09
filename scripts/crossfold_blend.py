"""Cross-fold blend/stack transfer test: fit blend weights on one fold, score on another.

Models are paired across two prediction banks by recipe stem (the run name with its
fold token removed), so each blend member exists as an independently fold-trained
model on both folds.  Weights are fitted on fold A from exact weighted-Pearson
sufficient statistics (weights |clip(y)|, scored rows >= 99) and then evaluated on
fold B with the official streaming metric.  The honest reference on B is the single
model chosen on A.  Both directions are reported.  Both folds are historically
exposed, so this measures transfer of a selection rule, not a fresh holdout score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import (  # noqa: E402
    GROUP_CHUNK,
    SCORED_FROM_ROW,
    aligned,
    load_bank,
    score,
    sha256,
)

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

FOLD_TOKEN = re.compile(r"_f[0-9](?=_|$)")


def stem(name: str) -> str:
    return FOLD_TOKEN.sub("", name, count=1)


def load_fold(bank: Path, fold: int, split: Path, targets_path: Path):
    rows = load_bank(bank, fold, skip_missing=True)
    split_value = json.loads(split.read_text())
    groups = [int(g) for g in split_value["evaluation"]["valid"]]
    component_of = {int(r["row_group"]): int(r["component"]) for r in split_value["sequences"]["valid"]}
    with np.load(targets_path, allow_pickle=False) as cache:
        position = {int(g): i for i, g in enumerate(cache["row_group"])}
        index = [position[g] for g in groups]
        y = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= (np.arange(y.shape[1]) >= SCORED_FROM_ROW)[None, :]
    models = {}
    split_sha = sha256(split)
    for row in rows:
        result = aligned(row, groups, split_sha)
        if isinstance(result, str):
            continue
        array, order = result
        models[stem(row["name"])] = (row["name"], array, order)
    return y, mask, np.asarray([component_of[g] for g in groups]), models


def moments(y, mask, predictions):
    """Per-target weighted sums: W, Sy, Syy, Sp[i], Syp[i], Spp[i,j]."""
    m = len(predictions)
    stats = []
    for k in range(2):
        acc = {"W": 0.0, "Sy": 0.0, "Syy": 0.0, "Sp": np.zeros(m), "Syp": np.zeros(m), "Spp": np.zeros((m, m))}
        for start in range(0, len(y), GROUP_CHUNK):
            part = slice(start, start + GROUP_CHUNK)
            sel = mask[part]
            yk = np.clip(y[part][..., k][sel].astype(np.float64), -2, 2)
            w = np.abs(yk)
            x = np.stack([np.clip(np.asarray(p[part][..., k][sel], dtype=np.float64), -2, 2) for p in predictions], 1)
            acc["W"] += w.sum()
            acc["Sy"] += (w * yk).sum()
            acc["Syy"] += (w * yk * yk).sum()
            acc["Sp"] += w @ x
            acc["Syp"] += (w * yk) @ x
            acc["Spp"] += (x * w[:, None]).T @ x
        stats.append(acc)
    return stats


def blend_wp(a: np.ndarray, stats, per_target: bool = False) -> float:
    values = []
    for k, s in enumerate(stats):
        wk = a[k] if per_target else a
        W = s["W"]
        my, mp = s["Sy"] / W, wk @ s["Sp"] / W
        cov = wk @ s["Syp"] / W - my * mp
        vy = s["Syy"] / W - my * my
        vp = wk @ s["Spp"] @ wk / W - mp * mp
        values.append(cov / np.sqrt(max(vy * vp, 1e-30)))
    return float(np.mean(values))


def fit_weights(stats, m: int, per_target: bool) -> np.ndarray:
    shape = (2, m) if per_target else (m,)
    x0 = np.full(shape, 1.0 / m).ravel()
    cons = (
        [{"type": "eq", "fun": lambda x, k=k: x.reshape(2, m)[k].sum() - 1} for k in range(2)]
        if per_target else [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    )
    result = minimize(
        lambda x: -blend_wp(x.reshape(shape), stats, per_target), x0, method="SLSQP",
        bounds=[(0.0, 1.0)] * x0.size, constraints=cons, options={"maxiter": 500, "ftol": 1e-12},
    )
    return result.x.reshape(shape)


def materialize(models_b, keys, weights, per_target: bool):
    first = models_b[keys[0]]
    out = None
    for i, key in enumerate(keys):
        _, array, order = models_b[key]
        p = np.asarray(array[order], dtype=np.float32)
        w = weights[:, i] if per_target else np.array([weights[i], weights[i]])
        term = p * w.astype(np.float32)[None, None, :]
        out = term if out is None else out + term
    del first
    return out


def evaluate_direction(fold_a, fold_b, keys, max_members: int):
    ya, ma, _, models_a = fold_a
    yb, mb, comp_b, models_b = fold_b
    preds_a = [np.asarray(models_a[k][1][models_a[k][2]], dtype=np.float32) for k in keys]
    stats_a = moments(ya, ma, preds_a)
    singles_a = {k: score(ya, ma, p)["wp"] for k, p in zip(keys, preds_a, strict=True)}
    del preds_a
    ranked = sorted(keys, key=lambda k: -singles_a[k])
    m = len(keys)
    rules = {"best_single": {ranked[0]: 1.0}}
    for top in (2, 3, 4, 6, 8):
        if top <= m:
            rules[f"equal_top{top}"] = {k: 1.0 / top for k in ranked[:top]}
    chosen, counts = [], {}
    for _ in range(max_members):
        best = max(keys, key=lambda k: blend_wp(
            np.array([(counts.get(j, 0) + (j == k)) / (len(chosen) + 1) for j in keys]), stats_a))
        chosen.append(best)
        counts[best] = counts.get(best, 0) + 1
    rules["greedy_equal"] = {k: c / len(chosen) for k, c in counts.items()}
    w_shared = fit_weights(stats_a, m, per_target=False)
    rules["optimized_nonneg"] = {k: float(w) for k, w in zip(keys, w_shared, strict=True) if w > 1e-4}
    w_target = fit_weights(stats_a, m, per_target=True)
    fit_a = {name: blend_wp(np.array([rule.get(k, 0.0) for k in keys]), stats_a) for name, rule in rules.items()}
    fit_a["optimized_per_target"] = blend_wp(w_target, stats_a, per_target=True)

    results = {}
    reference = None
    for name, rule in rules.items():
        rk = list(rule)
        p = materialize(models_b, rk, np.array([rule[k] for k in rk]), per_target=False)
        results[name] = {"fit_on_a": fit_a[name], "score_on_b": score(yb, mb, p), "weights": rule}
        if name == "best_single":
            reference = paired_component_moments(yb, p, p, mb)[0]
        else:
            cand = paired_component_moments(yb, p, p, mb)[0]
            results[name]["b_minus_best_single_of_a"] = paired_component_bootstrap(
                cand, reference, comp_b, n_boot=1000, seed=271828, centered=True)
        del p
    nz = [i for i in range(m) if w_target[:, i].max() > 1e-4]
    p = materialize(models_b, [keys[i] for i in nz], w_target[:, nz], per_target=True)
    cand = paired_component_moments(yb, p, p, mb)[0]
    results["optimized_per_target"] = {
        "fit_on_a": fit_a["optimized_per_target"], "score_on_b": score(yb, mb, p),
        "weights": {keys[i]: w_target[:, i].round(4).tolist() for i in nz},
        "b_minus_best_single_of_a": paired_component_bootstrap(cand, reference, comp_b, n_boot=1000, seed=271828, centered=True),
    }
    oracle = max(keys, key=lambda k: score(yb, mb, np.asarray(models_b[k][1][models_b[k][2]], dtype=np.float32))["wp"])
    return {"single_scores_a": singles_a, "rules": results, "oracle_best_single_on_b": oracle}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-a", type=Path, required=True)
    parser.add_argument("--fold-a", type=int, required=True)
    parser.add_argument("--bank-b", type=Path, required=True)
    parser.add_argument("--fold-b", type=int, required=True)
    parser.add_argument("--targets", type=Path, default=ROOT / "cache/valid_targets.npz")
    parser.add_argument("--max-members", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits"
    fold_a = load_fold(args.bank_a, args.fold_a, split / f"fold{args.fold_a}.json", args.targets)
    fold_b = load_fold(args.bank_b, args.fold_b, split / f"fold{args.fold_b}.json", args.targets)
    keys = sorted(set(fold_a[3]) & set(fold_b[3]))
    if len(keys) < 2:
        raise SystemExit(f"need at least two recipes present on both folds, found {keys}")
    report = {
        "schema": "wnn.crossfold-blend-transfer.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "metric_id": "global_wp_clip2_v1",
        "paired_recipes": {k: [fold_a[3][k][0], fold_b[3][k][0]] for k in keys},
        "exposure": "both folds historically exposed; measures transfer of the selection rule",
        f"fit_f{args.fold_a}_score_f{args.fold_b}": evaluate_direction(fold_a, fold_b, keys, args.max_members),
        f"fit_f{args.fold_b}_score_f{args.fold_a}": evaluate_direction(fold_b, fold_a, keys, args.max_members),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    for direction in [k for k in report if k.startswith("fit_")]:
        print(direction)
        for name, row in report[direction]["rules"].items():
            delta = row.get("b_minus_best_single_of_a", {})
            print(f"  {name:22s} fitA {row['fit_on_a']:.5f}  scoreB {row['score_on_b']['wp']:.5f}"
                  f"  dB {delta.get('mean_diff', 0):+.5f} ci {delta.get('ci95')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
