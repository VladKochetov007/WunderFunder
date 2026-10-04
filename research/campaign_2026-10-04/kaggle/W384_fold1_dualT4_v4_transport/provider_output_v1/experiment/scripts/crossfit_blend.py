"""Cross-fitted convex blend weights for fold members, scored later by screen_score.py.

For each target fold k, weights are chosen on the evaluation rows of the *other* listed folds (where every member
is out-of-fold) to maximise mean pooled |y|-weighted Pearson of the blend, then applied to fold k's rows. The
blend is written as ``cache/pred_<OUT>_f<k>_valid.npy`` with a sidecar bound to fold k's manifest, so no weight
ever saw fold k labels.

    crossfit_blend.py OUT --folds 0 1 --members c20260929_ft_since_s3_f{fold} c20260929_ft_since_s4_f{fold} ...

Fitting uses |y|-weighted second moments of [y, p_1..p_K] per target (members clipped to ±2 individually), so
each objective evaluation is O(K^2); the final score clips the blend itself, exactly like the metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.data import WARMUP  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

MANIFEST = "research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"


def fold_arrays(fold: int, members: list[str], targets, position):
    split = load_campaign_split(ROOT / MANIFEST.format(fold=fold), ROOT)
    preds = []
    for pattern in members:
        name = pattern.format(fold=fold)
        path = ROOT / "cache" / f"pred_{name}_valid.npy"
        side = json.loads(path.with_suffix(".json").read_text())
        if side.get("split_manifest_sha256") != split.sha256:
            raise ValueError(f"{name}: not fitted on clean fold {fold}")
        preds.append(np.load(path, mmap_mode="r"))
    rows = [position[g] for g in split.evaluation_groups]
    return split, preds, rows


def second_moments(preds, rows, targets) -> np.ndarray:
    """|y|-weighted uncentred sums of outer products of z = [1, y, p_1..p_K], one matrix per target."""
    k = len(preds)
    acc = np.zeros((2, k + 2, k + 2))
    for i in rows:
        sel = targets["mask"][i].copy()
        sel[:WARMUP] = False
        if not sel.any():
            continue
        y = np.clip(np.asarray(targets["y"][i][sel], dtype=np.float64), -2, 2)
        for t in range(2):
            z = np.empty((sel.sum(), k + 2))
            z[:, 0] = 1.0
            z[:, 1] = y[:, t]
            for j, p in enumerate(preds):
                z[:, 2 + j] = np.clip(np.asarray(p[i][sel, t], dtype=np.float64), -2, 2)
            acc[t] += (z * np.abs(y[:, t])[:, None]).T @ z
    return acc


def blend_corr(w: np.ndarray, acc: np.ndarray) -> float:
    scores = []
    for m in acc:
        n = m[0, 0]
        mean = m[0] / n
        cov = m / n - np.outer(mean, mean)
        cyp = cov[1, 2:] @ w
        vp = w @ cov[2:, 2:] @ w
        scores.append(cyp / np.sqrt(cov[1, 1] * vp))
    return float(np.mean(scores))


def fit_weights(acc: np.ndarray) -> np.ndarray:
    k = acc.shape[1] - 2

    def unpack(v):
        e = np.exp(v - v.max())
        return e / e.sum()

    res = minimize(lambda v: -blend_corr(unpack(v), acc), np.zeros(k), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    return unpack(res.x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    args = ap.parse_args()
    tz = np.load(ROOT / "cache/valid_targets.npz")
    targets = {"y": tz["y"], "mask": tz["mask"]}
    position = {int(g): i for i, g in enumerate(tz["row_group"])}
    data = {f: fold_arrays(f, args.members, targets, position) for f in args.folds}
    moments = {f: second_moments(data[f][1], data[f][2], targets) for f in args.folds}
    report = {"members": args.members, "folds": {}}
    for f in args.folds:
        others = sum(moments[g] for g in args.folds if g != f)
        w = fit_weights(others)
        split, preds, rows = data[f]
        name = f"{args.out}_f{f}"
        path = ROOT / "cache" / f"pred_{name}_valid.npy"
        out = np.lib.format.open_memmap(path.with_name(path.name + ".partial"), mode="w+", dtype=np.float32,
                                        shape=preds[0].shape)
        for i in range(preds[0].shape[0]):
            out[i] = sum(wj * np.asarray(p[i], dtype=np.float64) for wj, p in zip(w, preds)).astype(np.float32)
        out.flush()
        del out
        path.with_name(path.name + ".partial").replace(path)
        path.with_suffix(".json").write_text(json.dumps({
            "run": name, "split": "valid", "epoch": None, "kind": "crossfit_blend",
            "weights": dict(zip(args.members, map(float, w))), "fitted_on_folds": [g for g in args.folds if g != f],
            "split_manifest_sha256": split.sha256, "prediction_sha256": file_sha256(path),
            "fit_objective_other_folds": blend_corr(w, others),
            "equal_weight_objective_other_folds": blend_corr(np.full(len(w), 1 / len(w)), others)}, indent=1) + "\n")
        report["folds"][f] = {"weights": w.tolist(), "fit_corr": blend_corr(w, others),
                              "equal_corr": blend_corr(np.full(len(w), 1 / len(w)), others)}
        print(f"fold {f} weights {np.round(w, 4).tolist()} fit {report['folds'][f]['fit_corr']:.6f} "
              f"equal {report['folds'][f]['equal_corr']:.6f}", flush=True)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
