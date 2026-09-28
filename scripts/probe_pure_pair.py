"""Compare two fold-specific pair blends with exact clipped scorer moments."""

from __future__ import annotations

import argparse

import decide
import numpy as np


def pair_sums(
    anchor: np.ndarray,
    partner: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Return scored and unscored six-moment sums for a 50/50 pair."""
    out = np.zeros((len(idx), 2, 2, 6), dtype=np.float64)
    need = np.arange(y.shape[1]) >= decide.WARMUP
    for start in range(0, len(idx), 32):
        sl = idx[start : start + 32]
        target = np.clip(y[sl].astype(np.float64), -2, 2)
        pred = np.clip(
            0.5 * (np.asarray(anchor[sl], np.float64) + np.asarray(partner[sl], np.float64)),
            -2,
            2,
        )
        for target_id in range(2):
            out[start : start + len(sl), 0, target_id] = decide._sums6(
                target[..., target_id], pred[..., target_id], need & mask[sl]
            )
            out[start : start + len(sl), 1, target_id] = decide._sums6(
                target[..., target_id], pred[..., target_id], need & ~mask[sl]
            )
    return out


def main() -> None:
    """Score the same anchor with a candidate or reference second member."""
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", help="fold-specific run prefix")
    parser.add_argument("candidate", help="fold-specific run prefix")
    parser.add_argument("reference", help="fold-specific run prefix")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--pure-chains", action="store_true")
    parser.add_argument("--per-fold", action="store_true")
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()

    valid = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y, mask = valid["y"], valid["mask"]
    chain, fold = decide._chains()
    selected = np.isin(fold, [int(value) for value in args.folds.split(",")])
    if args.pure_chains:
        chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
        train_chains = chain_data["chain"][chain_data["split"] == "train"]
        selected &= ~np.isin(chain, train_chains)

    candidate_sums = []
    reference_sums = []
    ordered_idx = []
    for k in sorted(set(fold[selected])):
        idx = np.flatnonzero(selected & (fold == k))
        pred = {
            name: np.load(decide.ROOT / "cache" / f"pred_{name}_f{k}_valid.npy", mmap_mode="r")
            for name in (args.anchor, args.candidate, args.reference)
        }
        candidate_sums.append(pair_sums(pred[args.anchor], pred[args.candidate], y, mask, idx))
        reference_sums.append(pair_sums(pred[args.anchor], pred[args.reference], y, mask, idx))
        ordered_idx.append(idx)
    idx = np.concatenate(ordered_idx)
    candidate = np.concatenate(candidate_sums)
    reference = np.concatenate(reference_sums)
    rng = np.random.default_rng(41)
    boots = [decide._resample(np.arange(len(idx)), rng, chain[idx]) for _ in range(args.boot)]
    for lam in (0.0, 0.02, 0.05):
        delta = decide.wp(candidate, lam) - decide.wp(reference, lam)
        sd = np.std([decide.wp(candidate[b], lam) - decide.wp(reference[b], lam) for b in boots])
        print(f"n={len(idx)} lam={lam:.2f} delta={delta:+.6f} chain_boot_sd={sd:.6f}")
        if args.per_fold:
            for k in sorted(set(fold[idx])):
                selected_fold = fold[idx] == k
                fold_delta = decide.wp(candidate[selected_fold], lam) - decide.wp(
                    reference[selected_fold], lam
                )
                print(f"  fold={k} n={selected_fold.sum()} delta={fold_delta:+.6f}")


if __name__ == "__main__":
    main()
