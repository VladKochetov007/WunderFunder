"""Paired, class-stratified bootstrap comparison of runs on identical sequences.

    python scripts/compare_runs.py base_run other_run [other_run ...] [--split dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.evaluation import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def load(name, split):
    z = np.load(ROOT / "artifacts" / "runs" / name / f"scores_{split}.npz")
    return z["row_groups"], z["classes"], z["scores"].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    g0, c0, s0 = load(args.base, args.split)
    for other in args.others:
        g1, _, s1 = load(other, args.split)
        if not np.array_equal(g0, g1):
            raise SystemExit(f"{other}: different sequences than {args.base}")
        res = {"base": args.base, "other": other, "overall": paired_bootstrap(s1, s0, strata=c0)}
        for c in np.unique(c0):
            sel = c0 == c
            res[f"class{int(c)}"] = paired_bootstrap(s1[sel], s0[sel], n_boot=1000)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
