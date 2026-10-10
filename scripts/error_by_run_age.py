"""Sign-error mass versus the age of the current constant-label run (rows since the label last changed).

Labels are piecewise constant.  For each scored row the age is the number of rows since the label vector last changed.  The
script reports, per age bucket, the share of metric weight (|clip(y,2)|), the weighted sign accuracy and the share of total
weighted sign-error mass, to show how much of the loss sits right after a change (unknowable from the past) versus deep inside
a run (learnable).  Read-only analysis of saved predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import load_fold  # noqa: E402


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
    changed = np.zeros((n, t), dtype=bool)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    changed[:, 0] = True
    age = np.zeros((n, t), dtype=np.int32)
    for i in range(1, t):
        age[:, i] = np.where(changed[:, i], 0, age[:, i - 1] + 1)
    yc = np.clip(y, -2, 2)
    w = np.abs(yc) * mask[..., None]
    wrong = (np.sign(p) != np.sign(yc)) & (np.abs(yc) > 0)
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 100000]
    total_w, total_err = float(w.sum()), float((w * wrong).sum())
    runs = changed[mask].sum()
    print(f"run {args.run} fold {args.fold}: scored rows {int(mask.sum())}, label changes in scored rows {int(runs)}, "
          f"mean run length {mask.sum() / max(runs, 1):.1f}")
    print("age bucket      weight%  sign acc  error-mass%")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (age >= lo) & (age < hi)
        wb = w * sel[..., None]
        ws = float(wb.sum())
        if ws == 0:
            continue
        err = float((wb * wrong).sum())
        print(f"[{lo:>5},{hi:>6})  {100 * ws / total_w:7.2f}  {1 - err / ws:8.4f}  {100 * err / total_err:10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
