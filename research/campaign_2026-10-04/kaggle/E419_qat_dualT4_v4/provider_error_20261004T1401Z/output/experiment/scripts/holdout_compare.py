"""Compare two final-model blends on the reserved validation holdout (exact clipped global WP).

    scripts/limited .venv/bin/python scripts/holdout_compare.py \
        --candidate "a+b+c" --reference "ab67_final_since_s4_gcorr+ab67_final_since_s3_gcorr"

Members are full-valid prediction files ``cache/pred_<name>_valid.npy`` (``baseline`` is scaled by
0.25). The holdout shares chains with final training data, so it is a sanity check on the final
fits, not the selection evidence (that comes from ``ensemble_scan.py`` on pure fold chains).
"""

from __future__ import annotations

import argparse

import decide
import numpy as np
from ensemble_scan import LAMBDAS, parse_blend

from wnn.splits import valid_row_groups


def blend_sums(blend: dict[str, float], idx: np.ndarray, y, mask) -> np.ndarray:
    preds = {
        name: np.load(decide.ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
        for name in blend
    }
    scale = {name: 0.25 if name == "baseline" else 1.0 for name in blend}
    need = np.arange(y.shape[1]) >= decide.WARMUP
    out = np.zeros((len(idx), 2, 2, 6))
    for start in range(0, len(idx), 64):
        rows = idx[start : start + 64]
        p = np.clip(sum(w * scale[n] * np.asarray(preds[n][rows], np.float64) for n, w in blend.items()), -2, 2)
        t = np.clip(y[rows].astype(np.float64), -2, 2)
        for k in range(2):
            out[start : start + len(rows), 0, k] = decide._sums6(t[..., k], p[..., k], need & mask[rows])
            out[start : start + len(rows), 1, k] = decide._sums6(t[..., k], p[..., k], need & ~mask[rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()
    with np.load(decide.ROOT / "cache" / "valid_targets.npz") as valid:
        y, mask, row_groups = valid["y"], valid["mask"], valid["row_group"]
    _, held = valid_row_groups(decide.ROOT / "datasets" / "valid.parquet")
    idx = np.flatnonzero(np.isin(row_groups, held))
    chain, _ = decide._chains()
    cand = blend_sums(parse_blend(args.candidate), idx, y, mask)
    ref = blend_sums(parse_blend(args.reference), idx, y, mask)
    rng = np.random.default_rng(41)
    boots = [decide._resample(np.arange(len(idx)), rng, chain[idx]) for _ in range(args.boot)]
    for lam in LAMBDAS:
        d = decide.wp(cand, lam) - decide.wp(ref, lam)
        sd = np.std([decide.wp(cand[b], lam) - decide.wp(ref[b], lam) for b in boots])
        print(f"holdout n={len(idx)} lam={lam:.2f} cand {decide.wp(cand, lam):.5f} ref {decide.wp(ref, lam):.5f} "
              f"delta {d:+.5f} ± {sd:.5f}")


if __name__ == "__main__":
    main()
