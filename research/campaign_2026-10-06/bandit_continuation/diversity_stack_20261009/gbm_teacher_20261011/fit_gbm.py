"""Fit LightGBM (weighted MSE on clipped targets) on the fold-1 train sample; prune corr>=0.9 first."""
import sys; sys.path.append("/tmp/claude-1000/pylib/lgb")
import json, sys, time
from pathlib import Path
import numpy as np, lightgbm as lgb

sample, outdir = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(sample, allow_pickle=False)
X, y, isw, comp, names = d["X"], d["y"], d["isw"], d["comp"], list(d["names"])
print("sample", X.shape, flush=True)
rng = np.random.default_rng(0)
keep = list(range(X.shape[1])); knames = names  # corr>=0.9 pruning done beforehand on a pilot sample (prune.py)
n_before = 865
comps = np.unique(comp); hold = set(rng.choice(comps, max(1, len(comps) // 10), replace=False).tolist())
isho = np.array([c in hold for c in comp])
params = dict(objective="regression", learning_rate=0.08, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.5,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=63, num_threads=1, verbose=-1)
def wpearson(yv, p, w):
    W = w.sum(); my = (w * yv).sum() / W; mp = (w * p).sum() / W
    return ((w * (yv - my) * (p - mp)).sum() / W) / np.sqrt(((w * (yv - my) ** 2).sum() / W) * ((w * (p - mp) ** 2).sum() / W))
rep = {"kept_features": knames, "n_rows": int(len(X)), "n_features_before": n_before}
for k in range(2):
    w = np.abs(y[:, k]) * isw
    t = time.time()
    dtr = lgb.Dataset(X[~isho], y[~isho, k], weight=w[~isho], feature_name=[f"f{i}" for i in range(X.shape[1])])
    dva = lgb.Dataset(X[isho], y[isho, k], weight=w[isho], reference=dtr)
    m = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)])
    p = m.predict(X[isho], num_iteration=m.best_iteration)
    rep[f"t{k}"] = {"best_iter": m.best_iteration, "holdout_train_wpearson": float(wpearson(y[isho, k], np.clip(p, -2, 2), w[isho])), "sec": time.time() - t}
    print(k, rep[f"t{k}"], flush=True)
    m.save_model(str(outdir / f"lgb_t{k}.txt"), num_iteration=m.best_iteration)
(outdir / "fit_report.json").write_text(json.dumps(rep, indent=1))

