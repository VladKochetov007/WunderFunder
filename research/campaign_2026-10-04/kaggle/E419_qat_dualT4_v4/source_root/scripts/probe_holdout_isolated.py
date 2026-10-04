"""Compare final 50/50 pairs on reserved and chain-isolated holdout rows."""

from __future__ import annotations

import argparse

import decide
import numpy as np
from probe_pure_pair import pair_sums

from wnn.splits import valid_row_groups


def main() -> None:
    """Report exact clipped pair deltas with chain-bootstrap uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_a")
    parser.add_argument("candidate_b")
    parser.add_argument("reference_a")
    parser.add_argument("reference_b")
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()

    data = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y, mask, groups = data["y"], data["mask"], data["row_group"]
    dev_groups, hold_groups = valid_row_groups(decide.ROOT / "datasets" / "valid.parquet")
    chain, _ = decide._chains()
    chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    training_chain = np.isin(chain_data["split"], ["train"])
    chain_by_group = dict(
        zip(
            chain_data["row_group"][chain_data["split"] == "valid"].tolist(),
            chain_data["chain"][chain_data["split"] == "valid"].tolist(),
            strict=True,
        )
    )
    trained_chains = set(chain_data["chain"][training_chain].tolist())
    trained_chains.update(chain_by_group[int(group)] for group in dev_groups)
    holdout = np.flatnonzero(np.isin(groups, np.asarray(hold_groups)))
    pure = holdout[~np.isin(chain[holdout], list(trained_chains))]

    predictions = {
        name: np.load(decide.ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
        for name in (
            args.candidate_a,
            args.candidate_b,
            args.reference_a,
            args.reference_b,
        )
    }
    for label, idx in (("holdout", holdout), ("pure_holdout", pure)):
        candidate = pair_sums(
            predictions[args.candidate_a], predictions[args.candidate_b], y, mask, idx
        )
        reference = pair_sums(
            predictions[args.reference_a], predictions[args.reference_b], y, mask, idx
        )
        rng = np.random.default_rng(41)
        boots = [decide._resample(np.arange(len(idx)), rng, chain[idx]) for _ in range(args.boot)]
        for lam in (0.0, 0.02, 0.05):
            delta = decide.wp(candidate, lam) - decide.wp(reference, lam)
            sd = np.std(
                [decide.wp(candidate[b], lam) - decide.wp(reference[b], lam) for b in boots]
            )
            print(f"{label} n={len(idx)} lam={lam:.2f} delta={delta:+.6f} chain_boot_sd={sd:.6f}")


if __name__ == "__main__":
    main()
