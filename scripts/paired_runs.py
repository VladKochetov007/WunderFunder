"""Paired component-bootstrap comparison of run pairs on one fold (candidate minus reference), official WP.

Usage: paired_runs.py --fold 1 --pair cand_run:ref_run [--pair ...]. Reads cache/pred_<run>_valid.npy produced by predict_run.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import load_fold  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--pair", action="append", required=True, help="candidate:reference run names")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    models = {name: (name, array, order) for name, array, order in models.values()}
    out = []
    for spec in args.pair:
        cand, ref = spec.split(":")
        a, b = models[cand], models[ref]
        pa, pb = np.asarray(a[1][a[2]], dtype=np.float32), np.asarray(b[1][b[2]], dtype=np.float32)
        moments = paired_component_moments(y, pa, pb, mask)
        boot = paired_component_bootstrap(moments[0], moments[1], comp, n_boot=args.n_boot, seed=17, centered=True)
        out.append({"candidate": cand, "reference": ref, **{k: boot[k] for k in boot if k in ("delta", "ci95", "candidate", "reference")}})
        print(json.dumps(out[-1], default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
