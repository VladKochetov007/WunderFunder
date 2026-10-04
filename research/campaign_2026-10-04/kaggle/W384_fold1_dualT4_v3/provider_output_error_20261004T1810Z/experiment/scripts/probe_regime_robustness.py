"""Check OOF model deltas across label-free sequence activity regimes."""

from __future__ import annotations

import argparse

import decide
import numpy as np
import polars as pl


def main() -> None:
    """Compare two fold families using only their held-out chain folds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--strata", choices=["activity", "shift"], default="activity")
    parser.add_argument("--pure-chains", action="store_true")
    args = parser.parse_args()

    chain, fold = decide._chains()
    valid = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    groups = valid["row_group"]
    cls = valid["market_class"]
    stats = pl.read_parquet(
        decide.ROOT / "cache" / "seqstats_valid.parquet",
        columns=["row_group", "ev_i0_bid", "ev_i0_ask"],
    ).sort("row_group")
    if not np.array_equal(stats["row_group"].to_numpy(), groups):
        raise ValueError("sequence statistics do not match validation row order")
    activity = (stats["ev_i0_bid"] + stats["ev_i0_ask"]).to_numpy()
    stratifier = (
        activity
        if args.strata == "activity"
        else np.load(decide.ROOT / "cache" / "adv_valid_score.npy")
    )

    wanted = [int(value) for value in args.folds.split(",")]
    selected = np.isin(fold, wanted)
    if args.pure_chains:
        chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
        training_chains = chain_data["chain"][chain_data["split"] == "train"]
        selected &= ~np.isin(chain, training_chains)
    if args.stress:
        selected &= decide._stress_keep()
    candidate = np.zeros((len(groups), 2, 2, 6))
    reference = np.zeros_like(candidate)
    for k in wanted:
        idx = fold == k
        candidate[idx] = decide.load_sums(f"{args.candidate}_f{k}")[idx]
        reference[idx] = decide.load_sums(f"{args.reference}_f{k}")[idx]

    # Use each market class's label-free activity distribution to define terciles.
    tercile = np.full(len(groups), -1, dtype=np.int8)
    for k in range(4):
        idx = selected & (cls == k)
        cuts = np.quantile(stratifier[idx], [1 / 3, 2 / 3])
        tercile[idx] = np.searchsorted(cuts, stratifier[idx], side="right")

    rng = np.random.default_rng(27)
    regimes = {"all": selected}
    regimes.update({f"class{k}": selected & (cls == k) for k in range(4)})
    regimes.update({f"{args.strata}{k}": selected & (tercile == k) for k in range(3)})
    regimes.update({f"without_class{k}": selected & (cls != k) for k in range(4)})
    regimes.update({f"without_{args.strata}{k}": selected & (tercile != k) for k in range(3)})
    for label, mask in regimes.items():
        idx = np.flatnonzero(mask)
        delta = decide.wp(candidate[idx], args.lam) - decide.wp(reference[idx], args.lam)
        boots = [
            decide.wp(candidate[pick], args.lam) - decide.wp(reference[pick], args.lam)
            for pick in (decide._resample(idx, rng, chain) for _ in range(100))
        ]
        print(f"{label:>18} n={len(idx):4d} delta={delta:+.6f} boot_sd={np.std(boots):.6f}")


if __name__ == "__main__":
    main()
