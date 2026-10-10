"""Where does the loss live: sign, magnitude, or both?  Oracle swaps on one fold, per market class.

For a prediction p and label y (clipped at 2 as in the official metric) the script reports official WP of
  p                      the model,
  sign(p) * |y|          oracle magnitude (direction from the model),
  |p| * sign(y)          oracle direction (magnitude from the model),
  sign(y) * |y|          sanity (must be 1.0),
  c * p                  best global rescale (should equal p: Pearson is scale invariant),
and the sign accuracy of p on scored rows, weighted by |y|.  Read-only analysis of saved predictions.
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
from grade_prediction_bank import score  # noqa: E402

from wnn.data import market_class  # noqa: E402


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
    yc = np.clip(y, -2, 2)
    classes = np.array([market_class(y[i][mask[i]]) for i in range(len(y))])
    variants = {
        "model": p,
        "sign(p)*|y|  (oracle magnitude)": np.sign(p) * np.abs(yc),
        "|p|*sign(y)  (oracle direction)": np.abs(p) * np.sign(yc),
        "sign(y)*|y|  (sanity)": yc,
    }
    print(f"run {args.run}, fold {args.fold}; official WP per class (all = whole fold)")
    header = "variant".ljust(34) + "  all     " + "  ".join(f"c{c}" .ljust(7) for c in sorted(set(classes.tolist())))
    print(header)
    for name, q in variants.items():
        row = [score(y, mask, q)["wp"]]
        for c in sorted(set(classes.tolist())):
            idx = np.where(classes == c)[0]
            row.append(score(y[idx], mask[idx], q[idx])["wp"])
        print(name.ljust(34), "  ".join(f"{v:7.4f}" for v in row))
    w = np.abs(yc) * mask[..., None]
    agree = (np.sign(p) == np.sign(yc)) & (np.abs(yc) > 0)
    print("weighted sign accuracy, all:", round(float((w * agree).sum() / w.sum()), 4),
          "| per class:", {int(c): round(float((w[classes == c] * agree[classes == c]).sum() / w[classes == c].sum()), 4)
                           for c in sorted(set(classes.tolist()))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
