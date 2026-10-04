"""Surprise miner falsifier: is the class-3 winner tier (price moves through vs F at the near quote)
causally observable beyond the since-FT pair?

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_tier.py ROWS.npz OUT.json

Class-3 scored rows, fold-crossfit. Label (training only): winner = max(t0, t1) >= 1 tick. Classifier MLP on
causal row features + p vs a baseline using p alone; then a remap of p by class-3 (predicted-winner decile x
sign of p_D), fitted on other folds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_surprise_hazard import CAUSAL, features  # noqa: E402
from cr_surprise_lib import recal, wp  # noqa: E402

torch.set_num_threads(3)


def auc(score, lab):
    o = np.argsort(score)
    r = np.empty(len(o))
    r[o] = np.arange(1, len(o) + 1)
    n1 = lab.sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(lab) - n1)))


def fit_clf(x, y, x_app, epochs=2):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64),
                              torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 8192):
            b = perm[i:i + 8192]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[b])[:, 0], Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.from_numpy(x_app))[:, 0].numpy()


def main() -> None:
    z = np.load(sys.argv[1])
    d = {k: z[k] for k in CAUSAL + ["cls", "y0", "y1", "ft0", "ft1", "seq", "seq_fold"]}
    s3 = d["cls"] == 3
    d = {k: (v[s3] if k != "seq_fold" else v) for k, v in d.items()}
    fold = d["seq_fold"][d["seq"]]
    x = features(d)
    lab = np.maximum(d["y0"], d["y1"]) > 0.3  # >= 1 tick of 0.657
    xp = x[:, -8:-4]  # p and |p| only
    s_full, s_p = np.zeros(len(lab)), np.zeros(len(lab))
    for k in range(4):
        fit, app = fold != k, fold == k
        s_full[app] = fit_clf(x[fit], lab[fit], x[app])
        s_p[app] = fit_clf(xp[fit], lab[fit], xp[app])
        print("fold", k, flush=True)
    out = {"winner_share_rows": float(lab.mean()), "auc_full": auc(s_full, lab), "auc_p_only": auc(s_p, lab),
           "base_c3": wp(d, "ft")}
    pD = d["ft0"] - d["ft1"]
    for name, sc in (("full", s_full), ("p_only", s_p)):
        edges = np.quantile(sc, np.linspace(0, 1, 11)[1:-1])
        groups = np.searchsorted(edges, sc) * 2 + (pD > 0)
        q0, q1 = recal(d, groups, "ft", fold)
        out[f"remap_gain_c3_{name}"] = wp(d, "ft", None, q0, q1) - out["base_c3"]
        out[f"remap_gain_c3_{name}_by_fold"] = [wp(d, "ft", fold == k, q0, q1) - wp(d, "ft", fold == k)
                                                 for k in range(4)]
    print(json.dumps(out, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
