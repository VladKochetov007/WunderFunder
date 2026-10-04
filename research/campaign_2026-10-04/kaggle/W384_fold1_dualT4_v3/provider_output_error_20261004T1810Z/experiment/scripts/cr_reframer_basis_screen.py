"""R12/R13 screen: cross-fitted linear D-correction of class-3 rows from decoded level-reversion features.

Features (causal, class-3 grammar decoding of i0 and i1 tick paths, see cr_reframer_basis.py):
  r0_h  = mid0 - EWMA_h(mid0)                 (i0's own displacement from its recent level; invisible in shape codes)
  r1_h  = mid1 - EWMA_h(mid1)                 (i1 displacement, EWMA restarted at unknown i1 transitions)
  ec_h  = (mid1 - mid0) - EWMA_h(mid1 - mid0) (cross-instrument error-correction term)
and the same terms gated by a causal "long stretch" flag (rows since the last i0 mid move > 32).
The correction delta = F @ beta is added to t0 and subtracted from t1 (a D shift). beta is the |y|-weighted least
squares fit of (Dy - Dg) on F over class-3 scored rows of the *other* folds (leave-one-fold-out). Pooled Global WP
over all 1873 clean-fold evaluation sequences (classes 0-2 unchanged), chain-component bootstrap.

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G WNN_THREADS=3 scripts/limited .venv/bin/python scripts/cr_reframer_basis_screen.py OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cr_reframer_basis import ewma_reset  # noqa: E402
from wnn.evaluation import paired_component_bootstrap  # noqa: E402

spec = importlib.util.spec_from_file_location("campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

BASE = ["r0_64", "r0_512", "r1_512", "ec512", "ec4096"]
SETS = {
    "i0_only": ["r0_64", "r0_512"],
    "i1_only": ["r1_512", "ec512", "ec4096"],
    "all": BASE,
    "all_gated": BASE + [f"{f}*long" for f in BASE],
}
_SF = _MV = None


def _init():
    global _SF, _MV
    from wnn.data import SequenceFile
    _SF = SequenceFile(ROOT / "datasets" / "valid.parquet")
    with (ROOT / "cache" / "class3_moves_full.pkl").open("rb") as source:
        _MV = pickle.load(source)


def _one(args):
    i, g, fold = args
    from wnn.ticks import reconstruct
    s = _SF.read(int(g))
    x = s.features
    m = s.scored.copy()
    m[:99] = False
    a, b, _ = reconstruct(x, *_MV)
    x1 = x.copy()
    x1[:, 0:22] = x[:, 52:74]
    a1, b1, _ = reconstruct(x1, *_MV)
    mid0, mid1 = (a + b) / 2.0, (a1 + b1) / 2.0
    T = len(x)
    ev1 = np.flatnonzero((x1[1:, :22] != x1[:-1, :22]).any(1)) + 1
    codes = [tuple(r) for r in np.round(x1[np.r_[ev1 - 1, ev1], :22], 4)]
    n1 = len(ev1)
    unk = np.zeros(T, np.int64)
    unk[ev1] = [(c0, c1) not in _MV[0] for c0, c1 in zip(codes[:n1], codes[n1:], strict=True)]
    seg = np.cumsum(unk)
    zero = np.zeros(T, np.int64)
    basis = mid1 - mid0
    feats = {
        "r0_64": mid0 - ewma_reset(mid0, zero, 64),
        "r0_512": mid0 - ewma_reset(mid0, zero, 512),
        "r1_512": mid1 - ewma_reset(mid1, seg, 512),
        "ec512": basis - ewma_reset(basis, seg, 512),
        "ec4096": basis - ewma_reset(basis, seg, 4096),
    }
    t = np.arange(T)
    moved = np.r_[False, np.diff(mid0) != 0]
    last_move = np.maximum.accumulate(np.where(moved, t, 0))
    long = ((t - last_move) > 32).astype(np.float64)
    for f in BASE:
        feats[f"{f}*long"] = feats[f] * long
    preds = [np.load(ROOT / "cache" / f"pred_c20260929_ft_since_s{sd}_f{fold}_valid.npy", mmap_mode="r") for sd in (3, 4)]
    p = sum(0.5 * np.asarray(pr[i], np.float64) for pr in preds).astype(np.float32).astype(np.float64)
    names = BASE + [f"{f}*long" for f in BASE]
    F = np.stack([feats[k] for k in names], 1)
    return i, fold, s.targets[m].astype(np.float64), p[m], F[m]


def main():
    out_path = Path(sys.argv[1])
    targets = cs.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    rg, cls = np.asarray(targets["row_group"]).astype(int), np.asarray(targets["market_class"]).astype(int)
    fold_of, comp_of = {}, {}
    from wnn.splits import load_campaign_split
    for f in range(4):
        split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
        for g in split.evaluation_groups:
            fold_of[int(g)] = f
        comp_of.update({int(g): c for s_, g, c in split.components if s_ == "valid"})
    idx = [i for i in range(len(rg)) if int(rg[i]) in fold_of]
    c3 = [i for i in idx if cls[i] == 3]
    data = {}
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "2")), initializer=_init) as ex:
        for i, f, y, p, F in ex.map(_one, [(i, int(rg[i]), fold_of[int(rg[i])]) for i in c3], chunksize=4):
            data[i] = (f, y, p, F)
    print("decoded", len(data), flush=True)
    names = BASE + [f"{f}*long" for f in BASE]
    col = {k: j for j, k in enumerate(names)}
    steps = np.arange(20_000)
    # base moments for every evaluation sequence
    base_m, seq_cls, seq_comp, seq_fold, order = [], [], [], [], []
    for i in idx:
        f = fold_of[int(rg[i])]
        if i in data:
            _, y, p, _ = data[i]
            sel = np.ones(len(y))
        else:
            preds = [np.load(ROOT / "cache" / f"pred_c20260929_ft_since_s{sd}_f{f}_valid.npy", mmap_mode="r") for sd in (3, 4)]
            p = sum(0.5 * np.asarray(pr[i], np.float64) for pr in preds).astype(np.float32).astype(np.float64)
            y = np.asarray(targets["y"][i], np.float64)
            sel = (np.asarray(targets["mask"][i], bool) & (steps >= 99)).astype(np.float64)
        base_m.append(np.stack([cs.moments(y[:, k], p[:, k], sel) for k in range(2)]))
        seq_cls.append(int(cls[i]))
        seq_comp.append(comp_of[int(rg[i])])
        seq_fold.append(f)
        order.append(i)
    base_m = np.stack(base_m)
    seq_cls, seq_comp, seq_fold = np.array(seq_cls), np.array(seq_comp), np.array(seq_fold)
    report = {"n_sequences": len(order), "n_class3": len(data), "base_wp": cs.pooled(base_m)["wp"],
              "base_class3_wp": cs.pooled(base_m[seq_cls == 3])["wp"], "sets": {}}
    for set_name, feats in SETS.items():
        cols = [col[k] for k in feats]
        betas = {}
        for hold in range(4):
            XtX = np.zeros((len(cols), len(cols)))
            Xty = np.zeros(len(cols))
            for f, y, p, F in data.values():
                if f == hold:
                    continue
                yc = np.clip(y, -2, 2)
                w = np.abs(yc).mean(1)
                r = (yc[:, 0] - yc[:, 1]) / 2 - (p[:, 0] - p[:, 1]) / 2
                X = F[:, cols]
                XtX += X.T @ (X * w[:, None])
                Xty += X.T @ (w * r)
            betas[hold] = np.linalg.solve(XtX + 1e-6 * np.eye(len(cols)) * np.trace(XtX) / len(cols), Xty)
        cand = base_m.copy()
        for n, i in enumerate(order):
            if i not in data:
                continue
            f, y, p, F = data[i]
            delta = F[:, cols] @ betas[f]
            q = p.copy()
            q[:, 0] += delta
            q[:, 1] -= delta
            ones = np.ones(len(y))
            cand[n] = np.stack([cs.moments(y[:, k], q[:, k], ones) for k in range(2)])
        boot = paired_component_bootstrap(cand, base_m, seq_comp, n_boot=1000, seed=271828, centered=True)
        entry = {"features": feats, "delta": boot["mean_diff"], "boot_sd": boot["bootstrap_sd"],
                 "class3_delta": cs.pooled(cand[seq_cls == 3])["wp"] - report["base_class3_wp"],
                 "by_fold_delta": {int(f): cs.pooled(cand[seq_fold == f])["wp"] - cs.pooled(base_m[seq_fold == f])["wp"]
                                   for f in range(4)},
                 "beta_by_holdout": {int(h): b.round(5).tolist() for h, b in betas.items()}}
        report["sets"][set_name] = entry
        print(set_name, json.dumps({k: v for k, v in entry.items() if k != "beta_by_holdout"}), flush=True)
    out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
