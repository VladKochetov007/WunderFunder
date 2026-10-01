"""Surprise miner falsifier: is the late-run information (rows just before the next quote move) causally
observable beyond what the since-FT pair already encodes?

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_hazard.py ROWS.npz OUT.json

Fold-crossfit (fit folds != k, apply fold k): (1) MLP hazard on causal row features -> log1p(rows to next
a6 event), a feature-only label; (2) per-class remap of p by predicted-hazard decile; (3) per-class residual
MLP p' = p + g(features, p) trained with |y|-weighted MSE (upper bound for these summary features).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_surprise_lib import recal, wp  # noqa: E402

torch.set_num_threads(3)
CAUSAL = ["since_E", "since_cc", "since_tr0", "since_vol0", "since_px1", "since_tr1", "run_cc", "run_tr0",
          "run_px1", "run_vol0", "spread", "t", "run"] + [f"a{i}" for i in range(8)]


def features(d):
    cols = [np.log1p(np.maximum(d[k].astype(np.float64), 0)) if k.startswith(("since", "run", "t")) else
            d[k].astype(np.float64) for k in CAUSAL]
    cols += [d["ft0"], d["ft1"], np.abs(d["ft0"]), np.abs(d["ft1"])]
    cols += [(d["cls"] == c).astype(np.float64) for c in range(4)]
    x = np.column_stack(cols).astype(np.float32)
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def fit_mlp(x, y, w, x_app, epochs=2, out=1, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64),
                              torch.nn.GELU(), torch.nn.Linear(64, out))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X, Y, W = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w.astype(np.float32))
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8192):
            b = perm[i:i + 8192]
            loss = (W[b, None] * (net(X[b]) - Y[b].view(len(b), -1)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.from_numpy(x_app)).numpy()


def main() -> None:
    z = np.load(sys.argv[1])
    d = {k: z[k] for k in CAUSAL + ["cls", "y0", "y1", "ft0", "ft1", "seq", "seq_fold", "nc_until_E"]}
    fold = d["seq_fold"][d["seq"]]
    x = features(d)
    lab = np.log1p(d["nc_until_E"].astype(np.float64))
    haz = np.zeros(len(lab))
    res0, res1 = d["ft0"].astype(np.float64).copy(), d["ft1"].astype(np.float64).copy()
    yy = np.column_stack([np.clip(d["y0"], -2, 2), np.clip(d["y1"], -2, 2)]).astype(np.float64)
    ww = np.abs(yy).mean(1)
    for k in range(4):
        fit, app = fold != k, fold == k
        haz[app] = fit_mlp(x[fit], lab[fit], np.ones(fit.sum()), x[app])[:, 0]
        r = fit_mlp(x[fit], yy[fit] - np.column_stack([d["ft0"], d["ft1"]])[fit], ww[fit], x[app], out=2)
        res0[app] += r[:, 0]
        res1[app] += r[:, 1]
        print("fold", k, flush=True)
    out = {"hazard_r2": float(1 - ((lab - haz) ** 2).mean() / lab.var()),
           "hazard_corr": float(np.corrcoef(lab, haz)[0, 1]), "base": wp(d, "ft")}
    cls = d["cls"].astype(int)
    for c in range(4):
        s = cls == c
        out[f"hazard_r2_c{c}"] = float(1 - ((lab[s] - haz[s]) ** 2).mean() / lab[s].var())
    edges = np.quantile(haz, np.linspace(0, 1, 11)[1:-1])
    groups = cls * 10 + np.searchsorted(edges, haz)
    q0, q1 = recal(d, groups, "ft", fold)
    q0c, q1c = recal(d, cls, "ft", fold)
    out["remap_class_x_hazard_decile_crossfit"] = wp(d, "ft", None, q0, q1) - out["base"]
    out["remap_class_only_crossfit"] = wp(d, "ft", None, q0c, q1c) - out["base"]
    out["residual_mlp_crossfit"] = wp(d, "ft", None, res0, res1) - out["base"]
    for c in range(4):
        s = cls == c
        out[f"residual_mlp_c{c}"] = wp(d, "ft", s, res0, res1) - wp(d, "ft", s)
        out[f"remap_hazard_c{c}"] = wp(d, "ft", s, q0, q1) - wp(d, "ft", s)
    for k in range(4):
        out[f"residual_mlp_fold{k}"] = wp(d, "ft", fold == k, res0, res1) - wp(d, "ft", fold == k)
    print(json.dumps(out, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
