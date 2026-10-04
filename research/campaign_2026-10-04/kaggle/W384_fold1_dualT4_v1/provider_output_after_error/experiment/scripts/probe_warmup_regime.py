"""Cross-fit a causal first-99-row regime correction on pure-chain OOF pairs."""

from __future__ import annotations

import argparse

import decide
import numpy as np


def warmup_event_rates() -> np.ndarray:
    """Return three event rates observed before the first predicted row."""
    with np.load(decide.ROOT / "cache" / "valid_rowinfo.npz") as rows:
        rates = np.empty((rows["since_book"].shape[0], 3), dtype=np.float64)
        for column, key in enumerate(("since_book", "since_trade", "since_i1")):
            rates[:, column] = (rows[key][:, 1 : decide.WARMUP] == 0).mean(axis=1)
    return rates


def pair_row(anchor: np.ndarray, candidate: np.ndarray, index: int) -> np.ndarray:
    """Read the fixed equal-weight pair for one sequence."""
    return 0.5 * (
        np.asarray(anchor[index], dtype=np.float64) + np.asarray(candidate[index], dtype=np.float64)
    )


def fit_correction(
    features: np.ndarray,
    residual_sum: np.ndarray,
    weight_sum: np.ndarray,
    fit_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit 1% diagonal ridge slopes and shrink all coefficients to 25%."""
    mean = features[fit_idx].mean(axis=0)
    scale = np.maximum(features[fit_idx].std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(fit_idx)), (features[fit_idx] - mean) / scale))
    beta = np.zeros((x.shape[1], 2), dtype=np.float64)
    for target in range(2):
        mass = weight_sum[fit_idx, target]
        gram = x.T @ (mass[:, None] * x)
        ridge = 0.01 * np.diag(gram)
        ridge[0] = 0.0
        gram[np.diag_indices_from(gram)] += ridge + 1e-8 * max(gram.trace(), 1.0)
        beta[:, target] = 0.25 * np.linalg.solve(gram, x.T @ residual_sum[fit_idx, target])
    return mean, scale, beta


def score_fold(
    anchor: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
    corrections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scorer moments with and without the constant regime correction."""
    before = np.zeros((len(idx), 2, 2, 6), dtype=np.float64)
    after = np.zeros_like(before)
    for local, sequence in enumerate(idx):
        target = np.clip(y[sequence, decide.WARMUP :].astype(np.float64), -2, 2)
        prediction = pair_row(anchor, candidate, int(sequence))[decide.WARMUP :]
        corrected = prediction + corrections[local]
        scored = mask[sequence, decide.WARMUP :]
        for target_id in range(2):
            for mask_id, selected in enumerate((scored, ~scored)):
                before[local, mask_id, target_id] = decide._sums6(
                    target[:, target_id], np.clip(prediction[:, target_id], -2, 2), selected
                )
                after[local, mask_id, target_id] = decide._sums6(
                    target[:, target_id], np.clip(corrected[:, target_id], -2, 2), selected
                )
    return before, after


def main() -> None:
    """Use opposite pure-chain folds to fit and evaluate the causal correction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="ab62_base_s4")
    parser.add_argument("--candidate", default="ab68_ema6_since_ep5")
    args = parser.parse_args()

    data = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y, mask = data["y"], data["mask"]
    chain, fold = decide._chains()
    chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    train_chains = chain_data["chain"][chain_data["split"] == "train"]
    pure = ~np.isin(chain, train_chains)
    rates = warmup_event_rates()
    if len(rates) != len(y):
        raise ValueError("event-rate cache and target cache have different sequence counts")
    features = np.full((len(y), 5), np.nan, dtype=np.float64)
    features[:, :3] = rates
    residual_sum = np.zeros((len(y), 2), dtype=np.float64)
    weight_sum = np.zeros_like(residual_sum)
    models = {}
    for k in (0, 1):
        models[k] = (
            np.load(decide.ROOT / "cache" / f"pred_{args.anchor}_f{k}_valid.npy", mmap_mode="r"),
            np.load(decide.ROOT / "cache" / f"pred_{args.candidate}_f{k}_valid.npy", mmap_mode="r"),
        )
        anchor, candidate = models[k]
        for sequence in np.flatnonzero(pure & (fold == k)):
            pair = pair_row(anchor, candidate, int(sequence))
            features[sequence, 3:] = pair[: decide.WARMUP].mean(axis=0)
            target = np.clip(y[sequence, decide.WARMUP :].astype(np.float64), -2, 2)
            scored = mask[sequence, decide.WARMUP :]
            weight = np.abs(target) * scored[:, None]
            residual_sum[sequence] = (weight * (target - pair[decide.WARMUP :])).sum(axis=0)
            weight_sum[sequence] = weight.sum(axis=0)

    before_parts = []
    after_parts = []
    for eval_fold in (0, 1):
        fit_idx = np.flatnonzero(pure & (fold == 1 - eval_fold))
        eval_idx = np.flatnonzero(pure & (fold == eval_fold))
        mean, scale, beta = fit_correction(features, residual_sum, weight_sum, fit_idx)
        x_eval = np.column_stack((np.ones(len(eval_idx)), (features[eval_idx] - mean) / scale))
        before, after = score_fold(*models[eval_fold], y, mask, eval_idx, x_eval @ beta)
        before_parts.append(before)
        after_parts.append(after)
        print(f"eval_fold={eval_fold} n={len(eval_idx)} beta={beta.tolist()}", flush=True)
        for lam in (0.0, 0.02):
            print(
                f"fold{eval_fold} lam={lam:.2f} delta={decide.wp(after, lam) - decide.wp(before, lam):+.6f}",
                flush=True,
            )
    before = np.concatenate(before_parts)
    after = np.concatenate(after_parts)
    for lam in (0.0, 0.02):
        print(f"pooled lam={lam:.2f} delta={decide.wp(after, lam) - decide.wp(before, lam):+.6f}")


if __name__ == "__main__":
    main()
