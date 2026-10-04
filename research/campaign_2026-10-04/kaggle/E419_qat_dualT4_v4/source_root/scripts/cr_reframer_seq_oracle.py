"""R02 probe: are per-sequence level/scale oracles real latents or in-sample overfit?

For each clean fold's evaluation sequences, the incumbent-like mix (0.5 ft_since_s3 + 0.5 ft_since_s4, two-stage FT)
is corrected per sequence by an oracle fitted on targets:
  level_D  : shift D = (t0 - t1)/2 by the |y|-weighted mean residual of D
  level_k  : shift each target by its own |y_k|-weighted mean residual
  scale_k  : rescale each target around its sequence mean by the within-sequence weighted LS slope (clamped [0, 3])
Each oracle is fitted (a) in-sample on all scored rows, (b) on odd episodes -> applied to even and vice versa,
(c) on the other temporal half, (d) causally: first half -> second half only. Episodes are maximal constant-(t0, t1)
runs. Output: pooled Global WP deltas vs the uncorrected mix with chain-component bootstrap SD, per class.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_reframer_seq_oracle.py OUT.json
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
HALF = 10_000


def fit(y, p, sel):
    """Oracle parameters from rows ``sel``: (cD, c0, c1, b0, b1, m0, m1) or None when too thin."""
    if sel.sum() < 20:
        return None
    yy, pp = y[sel], p[sel]
    w = np.abs(yy)
    wd = w.mean(1)
    if w[:, 0].sum() < 1 or w[:, 1].sum() < 1 or wd.sum() < 1:
        return None
    dy, dp = (yy[:, 0] - yy[:, 1]) / 2, (pp[:, 0] - pp[:, 1]) / 2
    cD = np.dot(wd, dy - dp) / wd.sum()
    out = [cD]
    for k in range(2):
        out.append(np.dot(w[:, k], yy[:, k] - pp[:, k]) / w[:, k].sum())
    for k in range(2):
        mp = np.dot(w[:, k], pp[:, k]) / w[:, k].sum()
        my = np.dot(w[:, k], yy[:, k]) / w[:, k].sum()
        vp = np.dot(w[:, k], (pp[:, k] - mp) ** 2)
        b = np.dot(w[:, k], (pp[:, k] - mp) * (yy[:, k] - my)) / vp if vp > 1e-9 else 1.0
        out.append(float(np.clip(b, 0.0, 3.0)))
    for k in range(2):
        out.append(np.dot(w[:, k], pp[:, k]) / w[:, k].sum())
    return np.array(out)


def apply(kind, theta, p, alpha=1.0):
    """Apply an oracle shrunk by ``alpha`` toward the identity."""
    q = p.copy()
    if theta is None:
        return q
    if kind == "level_D":
        q[:, 0] += alpha * theta[0]
        q[:, 1] -= alpha * theta[0]
    elif kind == "level_k":
        q[:, 0] += alpha * theta[1]
        q[:, 1] += alpha * theta[2]
    elif kind == "scale_k":
        for k in range(2):
            q[:, k] = theta[5 + k] + (1 + alpha * (theta[3 + k] - 1)) * (q[:, k] - theta[5 + k])
    return q


KINDS = ["level_D", "level_k", "scale_k"]
ALPHAS = [float(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
SCHEMES = ["insample", "episode_xfit", "half_xfit", "causal_half"] + [
    f"{s}@{a:g}" for a in ALPHAS for s in ("episode_xfit", "causal_half")]


def corrected(y, p, sel, episodes, steps):
    out = {}
    odd = (episodes % 2 == 1)
    second = steps >= HALF
    for kind in KINDS:
        th = fit(y, p, sel)
        out[f"{kind}:insample"] = apply(kind, th, p)
        q = p.copy()
        for part in (odd, ~odd):
            q[part] = apply(kind, fit(y, p, sel & ~part), p[part])
        out[f"{kind}:episode_xfit"] = q
        q = p.copy()
        for part in (second, ~second):
            q[part] = apply(kind, fit(y, p, sel & ~part), p[part])
        out[f"{kind}:half_xfit"] = q
        q = p.copy()
        q[second] = apply(kind, fit(y, p, sel & ~second), p[second])
        out[f"{kind}:causal_half"] = q
        for a in ALPHAS:
            q = p.copy()
            for part in (odd, ~odd):
                q[part] = apply(kind, fit(y, p, sel & ~part), p[part], a)
            out[f"{kind}:episode_xfit@{a:g}"] = q
            q = p.copy()
            q[second] = apply(kind, fit(y, p, sel & ~second), p[second], a)
            out[f"{kind}:causal_half@{a:g}"] = q
    return out


def main() -> None:
    out_path = Path(sys.argv[1])
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    order = [int(g) for g in targets["row_group"]]
    position = {g: i for i, g in enumerate(order)}
    steps = np.arange(20_000)
    arms = ["base"] + [f"{k}:{s}" for k in KINDS for s in SCHEMES]
    mom = {a: [] for a in arms}
    classes, components, folds = [], [], []
    for fold in range(4):
        split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", ROOT)
        comp = {g: c for s, g, c in split.components if s == "valid"}
        preds = [np.load(ROOT / "cache" / f"pred_{m.format(fold=fold)}_valid.npy", mmap_mode="r") for m in MEMBERS]
        for g in split.evaluation_groups:
            i = position[g]
            y = np.clip(np.asarray(targets["y"][i], dtype=np.float64), -2, 2)
            sel = np.asarray(targets["mask"][i], dtype=bool) & (steps >= 99)
            p = sum(0.5 * np.asarray(pr[i], dtype=np.float64) for pr in preds).astype(np.float32).astype(np.float64)
            p = np.clip(p, -2, 2)
            change = np.any(y[1:] != y[:-1], axis=1)
            episodes = np.concatenate([[0], np.cumsum(change)])
            variants = corrected(y, p, sel, episodes, steps)
            variants["base"] = p
            for a in arms:
                q = np.clip(variants[a], -2, 2)
                mom[a].append(np.stack([cs.moments(y[:, k], q[:, k], sel.astype(np.float64)) for k in range(2)]))
            classes.append(int(targets["market_class"][i]))
            components.append(comp[g])
            folds.append(fold)
        print("fold", fold, "done", flush=True)
    classes, components, folds = np.array(classes), np.array(components), np.array(folds)
    m = {a: np.stack(v) for a, v in mom.items()}
    report = {"members": MEMBERS, "n_sequences": len(classes), "arms": {}}
    base_wp = cs.pooled(m["base"])["wp"]
    report["base_wp"] = base_wp
    report["base_by_class"] = {int(c): cs.pooled(m["base"][classes == c])["wp"] for c in range(4)}
    for a in arms[1:]:
        boot = paired_component_bootstrap(m[a], m["base"], components, n_boot=1000, seed=271828, centered=True)
        report["arms"][a] = {
            "wp": cs.pooled(m[a])["wp"],
            "delta": boot["mean_diff"],
            "boot_sd": boot["bootstrap_sd"],
            "by_class_delta": {int(c): cs.pooled(m[a][classes == c])["wp"] - report["base_by_class"][int(c)]
                               for c in range(4)},
            "by_fold_delta": {int(f): cs.pooled(m[a][folds == f])["wp"] - cs.pooled(m["base"][folds == f])["wp"]
                              for f in range(4)},
        }
        print(f"{a:24s} delta {boot['mean_diff']:+.5f} sd {boot['bootstrap_sd']:.5f}", flush=True)
    out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
