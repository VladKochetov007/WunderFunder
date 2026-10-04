"""Evaluate pickled per-class symbolic residual terms on scored validation rows.

    scripts/limited .venv/bin/python scripts/symreg_apply.py OUT.npy residual_c0_ab20_gru128_e2:2 residual_c1_ab20_gru128_e2:2
Each argument is a pickle stem in cache/symreg_<stem>.pkl and the number of terms to keep.
Columns hold one class's terms on its rows and zero elsewhere, aligned to the scored rows of
cache/heur_valid.npz (the order scripts/lift_probe.py --npy expects).
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np

from wnn.symreg import predict_terms

ROOT = Path(__file__).resolve().parents[1]


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    h = np.load(ROOT / "cache" / "heur_valid.npz")
    sc = h["scored"]
    seq = h["seq"][sc]
    cls = np.load(ROOT / "cache" / "valid_targets.npz")["market_class"][seq]
    X = np.concatenate([h["X"][sc], h["raw"][sc]], 1)
    cols = []
    for spec in specs:
        stem, _, k = spec.partition(":")
        res = pickle.load(open(ROOT / "cache" / f"symreg_{stem}.pkl", "rb"))
        c = int(re.search(r"_c(\d)", stem).group(1))
        rows = cls == c
        vals = np.zeros((len(seq), int(k or len(res.terms))), np.float32)
        vals[rows] = predict_terms(res, X[rows], int(k or len(res.terms)))
        cols.append(vals)
        print(stem, [t.expr for t in res.terms[: vals.shape[1]]])
    np.save(out_path, np.concatenate(cols, 1))


if __name__ == "__main__":
    main()
