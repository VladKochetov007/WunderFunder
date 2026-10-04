"""Scout falsifier F5 (IceCube-3rd move): cross-fitted gradient-boosted stacker over member disagreement.

Features per target: 6 cached members of that target and of the other target (parents, head-FT, two-stage FT for
seeds 3 and 4), market class, log1p(rows since last a6 event), a6, step. Trained |y|-weighted on three folds'
evaluation rows (every SUB-th scored row), scored on all scored rows of the held fold. Compared against the
two-stage pair mean, the six-member mean and a cross-fitted |y|-weighted linear stacker on the same features.

    OMP_NUM_THREADS=3 WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_scout_tree_stacker.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import wp_from_sums  # noqa: E402

OUT = ROOT / "research/campaign_2026-10-01/creative/cr_scout/probe_stacker.json"
NAMES = ("c20260929_since_s3_f{k}", "c20260929_since_s4_f{k}", "c20260929_ft_since_s3_f{k}h",
         "c20260929_ft_since_s4_f{k}h", "c20260929_ft_since_s3_f{k}", "c20260929_ft_since_s4_f{k}")
SUB = 16


def sums(y, p):
    y, p = np.clip(y, -2, 2), np.clip(p, -2, 2)
    w = np.abs(y)
    return np.array([w.sum(), (w * y).sum(), (w * p).sum(), (w * y * y).sum(), (w * p * p).sum(), (w * y * p).sum()])


def main() -> None:
    z = np.load(ROOT / "cache/valid_targets.npz")
    row_group, cls_all = z["row_group"], z["market_class"]
    f = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    a6 = np.stack([f.read_row_group(int(g), columns=["a6"]).column("a6").to_numpy() for g in row_group])
    ev = np.isclose(a6, a6.min())
    t_idx = np.arange(a6.shape[1])
    last = np.maximum.accumulate(np.where(ev, t_idx[None, :], -1), axis=1)
    since = np.log1p(np.where(last >= 0, t_idx[None, :] - last, 20000)).astype(np.float32)
    pos = {int(g): i for i, g in enumerate(row_group)}
    folds = {}
    for k in range(4):
        man = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{k}.json").read_text())
        folds[k] = np.array(sorted(pos[int(g)] for g in man["evaluation"]["valid"]))

    def build(k, sub):
        idx = folds[k]
        m = z["mask"][idx]
        if sub > 1:
            m = m & (np.arange(m.shape[1])[None, :] % sub == 0)
        P = np.stack([np.load(ROOT / f"cache/pred_{n.format(k=k)}_valid.npy", mmap_mode="r")[idx][m] for n in NAMES], 1)
        y = z["y"][idx][m]
        ss, tt = np.nonzero(m)
        extra = np.stack([cls_all[idx][ss].astype(np.float32), since[idx][m], a6[idx][m],
                          tt.astype(np.float32) / 20000], 1)
        return P.astype(np.float32), extra, y

    train_parts = {k: build(k, SUB) for k in range(4)}
    report = {"members": NAMES, "sub": SUB, "folds": {}}
    for k in range(4):
        P, extra, y = build(k, 1)
        res = {}
        for name, pred in (("pair_two_stage", P[:, 4:6].mean(1)), ("mean6", P.mean(1))):
            res[name] = float(np.mean([wp_from_sums(sums(y[:, j], pred[:, j])) for j in range(2)]))
        lin, tree = np.zeros_like(y), np.zeros_like(y)
        for j in range(2):
            feats = lambda P_, e_: np.concatenate([P_[:, :, j], P_[:, :, 1 - j], e_], 1)  # noqa: E731
            Xtr = np.concatenate([feats(*train_parts[i][:2]) for i in range(4) if i != k])
            ytr = np.concatenate([train_parts[i][2][:, j] for i in range(4) if i != k])
            wtr = np.abs(np.clip(ytr, -2, 2))
            keep = wtr > 0
            Xtr, ytr, wtr = Xtr[keep], ytr[keep], wtr[keep]
            # linear: |y|-weighted least squares on member preds + one-hot class
            def lin_design(X):
                oh = np.eye(4, dtype=np.float32)[X[:, 12].astype(int)]
                return np.concatenate([X[:, :12], oh], 1)
            A = lin_design(Xtr) * np.sqrt(wtr)[:, None]
            beta = np.linalg.lstsq(A, ytr * np.sqrt(wtr), rcond=None)[0]
            X = feats(P, extra)
            lin[:, j] = lin_design(X) @ beta
            hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=200, l2_regularization=1.0,
                                                categorical_features=[12], random_state=0)
            hgb.fit(Xtr, ytr, sample_weight=wtr)
            tree[:, j] = hgb.predict(X)
        res["linear_stack"] = float(np.mean([wp_from_sums(sums(y[:, j], lin[:, j])) for j in range(2)]))
        res["tree_stack"] = float(np.mean([wp_from_sums(sums(y[:, j], tree[:, j])) for j in range(2)]))
        res["tree_plus_pair"] = float(np.mean([wp_from_sums(sums(y[:, j], 0.5 * tree[:, j] + 0.5 * P[:, 4:6, j].mean(1)))
                                               for j in range(2)]))
        cls = extra[:, 0].astype(int)
        res["per_class_tree_minus_pair"] = {
            c: float(np.mean([wp_from_sums(sums(y[cls == c, j], tree[cls == c, j]))
                              - wp_from_sums(sums(y[cls == c, j], P[cls == c, 4:6, j].mean(1))) for j in range(2)]))
            for c in range(4)}
        report["folds"][k] = res
        print(k, json.dumps({a: (round(b, 5) if isinstance(b, float) else {c: round(v, 4) for c, v in b.items()})
                             for a, b in res.items()}), flush=True)
        OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
