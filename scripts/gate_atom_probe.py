"""Two post-hoc probes on saved predictions (Grok round 2, ideas R1 and R3).

1. Clock gating (causal): output p_t when a chosen block of columns updated at row t, otherwise repeat the previous output.  Blocks are
   represented by one column each (columns inside a block update on identical rows).
2. Run atoms (oracle, diagnostic only - needs the labels): broadcast, inside each true constant-label run, (a) the mean prediction,
   (b) the prediction at the first row, (c) the prediction at row 4 of the run.  Shows how much loss is within-run wander versus the
   wrong level at the start of the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

from utils import FEATURE_COLUMNS  # noqa: E402
from wnn.data import market_class  # noqa: E402

REPS = {"A fast 31%": 0, "B 70%": 22, "B2 71%": 33, "C 40%": 44, "D 15%": 48, "E 12%": 52, "E2 12%": 63, "F 41%": 74, "F2 41.5%": 85,
        "G 24%": 96, "H 25%": 100, "I 100%": 104}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, _comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    by_name = {name: (array, order) for name, array, order in models.values()}
    array, order = by_name[args.run]
    p = np.asarray(array[order], dtype=np.float32)
    n, t, _ = y.shape
    base = score(y, mask, p)["wp"]
    print(f"run {args.run} fold {args.fold}: baseline WP {base:.5f}")
    # update masks per representative column
    import json

    groups = [int(g) for g in json.loads(split.read_text())["evaluation"]["valid"]]
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        pos = {int(g): i for i, g in enumerate(cache["row_group"])}
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    cols = list(REPS.values())
    upd = np.zeros((n, t, len(cols)), dtype=bool)
    for i, g in enumerate(groups):
        tab = pf.read_row_group(pos[g], columns=[FEATURE_COLUMNS[c] for c in cols], use_threads=False)
        x = np.column_stack([tab[FEATURE_COLUMNS[c]].to_numpy() for c in cols])
        upd[i, 1:] = x[1:] != x[:-1]
        upd[i, 0] = True
    print("clock gating (causal): hold the previous output unless a block updates")
    names = list(REPS)
    cands = [(nm, upd[..., j]) for j, nm in enumerate(names)]
    cands.append(("A or C", upd[..., 0] | upd[..., 3]))
    cands.append(("A or C or D", upd[..., 0] | upd[..., 3] | upd[..., 4]))
    cands.append(("any of A..H (not 100%)", upd[..., :-1].any(-1)))
    for nm, gate in cands:
        out = p.copy()
        for k in range(1, t):
            out[:, k] = np.where(gate[:, k, None], p[:, k], out[:, k - 1])
        wp = score(y, mask, out)["wp"]
        print(f"  gate {nm:<24} update share {gate[mask].mean():5.1%}  WP {wp:.5f}  ({wp - base:+.5f})")
    # run atoms
    changed = np.zeros((n, t), bool)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    changed[:, 0] = True
    run_id = np.cumsum(changed, axis=1) - 1
    print("run atoms (oracle boundaries, diagnostic)")
    mean_out = np.empty_like(p)
    start_out = np.empty_like(p)
    start4_out = np.empty_like(p)
    for i in range(n):
        rid = run_id[i]
        starts = np.flatnonzero(changed[i])
        k = len(starts)
        counts = np.bincount(rid, minlength=k).astype(np.float32)
        for c in range(2):
            sums = np.bincount(rid, weights=p[i, :, c], minlength=k)
            mean_out[i, :, c] = (sums / np.maximum(counts, 1))[rid]
            start_out[i, :, c] = p[i, starts, c][rid]
            idx4 = np.minimum(starts + 4, np.r_[starts[1:], t] - 1)
            start4_out[i, :, c] = p[i, idx4, c][rid]
    for nm, q in (("mean prediction over the run", mean_out), ("prediction at run start", start_out), ("prediction at run row 4", start4_out)):
        wp = score(y, mask, q)["wp"]
        print(f"  {nm:<30} WP {wp:.5f} ({wp - base:+.5f})")
    cls = np.array([market_class(y[i][mask[i]]) for i in range(n)])
    idx = np.where(cls == 3)[0]
    for nm, q in (("class 3 baseline", p), ("class 3 run-mean", mean_out)):
        print(f"  {nm:<30} WP {score(y[idx], mask[idx], q[idx])['wp']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
