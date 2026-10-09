"""Regime-sliced model comparison and regime-gated blending test on one fold of the prediction bank.

Slices: market class of the sequence, clipped-|y| tier of the row (metric weight lives in the tails) and
position in the sequence.  For each model and slice it reports the within-slice weighted Pearson and the
calibration slope of y on the prediction (slope < 1: over-confident, > 1: under-confident).  The gating test
splits the evaluation sequences by component parity, fits non-negative blend weights per class on one half
and scores on the other half against (a) the best single model of the fit half and (b) a single global
blend fit on the same half, so any gain is out-of-sample over sequences (the fold itself is exposed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import blend_wp, fit_weights, load_fold, moments  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

from wnn.data import market_class  # noqa: E402

TIERS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.01)]


def weighted_stats(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    total = w.sum()
    if total <= 0:
        return float("nan"), float("nan")
    my, mp = (w * y).sum() / total, (w * p).sum() / total
    cov = (w * (y - my) * (p - mp)).sum() / total
    vp = (w * (p - mp) ** 2).sum() / total
    vy = (w * (y - my) ** 2).sum() / total
    return float(cov / np.sqrt(max(vy * vp, 1e-30))), float(cov / max(vp, 1e-30))


def slice_table(y, mask, preds, names, classes):
    rows = {}
    clipped = np.clip(y, -2, 2)
    position = np.broadcast_to(np.arange(y.shape[1])[None, :, None], y.shape)
    edges = np.quantile(position[0, mask[0]] if mask[0].any() else position[0], [0.25, 0.5, 0.75])
    slices = {f"class{c}": (classes == c)[:, None, None] & mask[..., None] for c in sorted(set(classes.tolist()))}
    for lo, hi in TIERS:
        magnitude = np.abs(clipped)
        slices[f"abs_y_{lo}-{hi}"] = (magnitude >= lo) & (magnitude < hi) & mask[..., None]
    bounds = [0, *edges.tolist(), y.shape[1]]
    for i in range(4):
        slices[f"pos_q{i + 1}"] = ((position >= bounds[i]) & (position < bounds[i + 1]) & mask[..., None])
    for sname, sel in slices.items():
        sel = np.broadcast_to(sel, y.shape)
        yy = clipped[sel].astype(np.float64)
        w = np.abs(yy)
        entry = {"rows": int(sel.sum()), "weight_share": float(w.sum())}
        for name, p in zip(names, preds, strict=True):
            r, slope = weighted_stats(yy, np.clip(p[sel].astype(np.float64), -2, 2), w)
            entry[name] = {"wp": round(r, 5), "slope": round(slope, 3)}
        rows[sname] = entry
    total = sum(r["weight_share"] for r in rows if False) or 1.0
    return rows, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--max-models", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    ranked = sorted(models, key=lambda k: -score(y, mask, np.asarray(models[k][1][models[k][2]], dtype=np.float32))["wp"])
    keep = ranked[: args.max_models]
    preds = [np.asarray(models[k][1][models[k][2]], dtype=np.float32) for k in keep]
    classes = np.array([market_class(y[i][mask[i]]) for i in range(len(y))])
    table, _ = slice_table(y, mask, preds, keep, classes)

    half = (comp % 2) == 0
    results = {}
    for fit_half, test_half, label in ((half, ~half, "fit_even_score_odd"), (~half, half, "fit_odd_score_even")):
        yf, mf = y[fit_half], mask[fit_half]
        pf = [p[fit_half] for p in preds]
        yt, mt = y[test_half], mask[test_half]
        pt = [p[test_half] for p in preds]
        singles = [score(yf, mf, p)["wp"] for p in pf]
        best = int(np.argmax(singles))
        w_global = fit_weights(moments(yf, mf, pf), len(keep), per_target=False)
        global_blend = sum(w * p for w, p in zip(w_global, pt, strict=True))
        gated = np.zeros_like(pt[0])
        cls_f, cls_t = classes[fit_half], classes[test_half]
        class_weights = {}
        for c in sorted(set(cls_t.tolist())):
            idx_f = np.where(cls_f == c)[0]
            idx_t = np.where(cls_t == c)[0]
            if len(idx_f) < 20:
                gated[idx_t] = global_blend[idx_t]
                continue
            w_c = fit_weights(moments(yf[idx_f], mf[idx_f], [p[idx_f] for p in pf]), len(keep), per_target=False)
            class_weights[int(c)] = {k: round(float(w), 3) for k, w in zip(keep, w_c, strict=True) if w > 1e-3}
            gated[idx_t] = sum(w * p[idx_t] for w, p in zip(w_c, pt, strict=True))
        results[label] = {
            "best_single_on_fit_half": keep[best],
            "score_best_single": score(yt, mt, pt[best])["wp"],
            "score_global_blend": score(yt, mt, global_blend)["wp"],
            "score_class_gated_blend": score(yt, mt, gated)["wp"],
            "class_weights": class_weights,
            "fit_half_global_blend_wp": blend_wp(w_global, moments(yf, mf, pf)),
        }
    report = {"schema": "wnn.regime-blend-analysis.v1", "fold": args.fold, "models": keep,
              "class_counts": {int(c): int((classes == c).sum()) for c in sorted(set(classes.tolist()))},
              "slices": table, "gating_test": results,
              "exposure": "fold historically exposed; split-by-sequence gating test measures transfer of the rule"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({"class_counts": report["class_counts"], "gating": {
        k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk.startswith("score")}
        for k, v in results.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
