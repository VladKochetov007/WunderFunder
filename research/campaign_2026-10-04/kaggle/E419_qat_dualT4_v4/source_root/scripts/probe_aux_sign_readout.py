"""Cross-fit a small correction from an already trained next-change sign head."""

from __future__ import annotations

import argparse

import decide
import numpy as np


def rows(
    anchor: np.ndarray,
    candidate: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
):
    """Yield exact-target, pair-prediction, bounded-logit and mask chunks."""
    for start in range(0, len(idx), 32):
        sl = idx[start : start + 32]
        target = np.clip(y[sl].astype(np.float64), -2, 2)
        pred = 0.5 * (
            np.asarray(anchor[sl], dtype=np.float64) + np.asarray(candidate[sl], dtype=np.float64)
        )
        signal = np.tanh(np.asarray(aux[sl], dtype=np.float64) / 2)
        yield sl, target, pred, signal, mask[sl]


def fit_beta(
    anchor: np.ndarray,
    candidate: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Fit a fixed 25%-shrunk, 1%-ridge weighted residual slope per target."""
    numerator = np.zeros(2, dtype=np.float64)
    denominator = np.zeros(2, dtype=np.float64)
    weight_sum = np.zeros(2, dtype=np.float64)
    for _, target, pred, signal, scored in rows(anchor, candidate, aux, y, mask, idx):
        weight = np.abs(target) * scored[..., None]
        numerator += (weight * signal[..., None] * (target - pred)).sum((0, 1))
        denominator += (weight * signal[..., None] ** 2).sum((0, 1))
        weight_sum += weight.sum((0, 1))
    return 0.25 * numerator / (denominator + 0.01 * weight_sum)


def score_sums(
    anchor: np.ndarray,
    candidate: np.ndarray,
    aux: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    idx: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact scorer moments before and after the cross-fitted correction."""
    before = np.zeros((len(idx), 2, 2, 6), dtype=np.float64)
    after = np.zeros_like(before)
    need = np.arange(y.shape[1]) >= decide.WARMUP
    offset = 0
    for sl, target, pred, signal, scored in rows(anchor, candidate, aux, y, mask, idx):
        original = np.clip(pred, -2, 2)
        corrected = np.clip(pred + signal[..., None] * beta, -2, 2)
        for target_id in range(2):
            for mask_id, selected in enumerate((scored & need, ~scored & need)):
                before[offset : offset + len(sl), mask_id, target_id] = decide._sums6(
                    target[..., target_id], original[..., target_id], selected
                )
                after[offset : offset + len(sl), mask_id, target_id] = decide._sums6(
                    target[..., target_id], corrected[..., target_id], selected
                )
        offset += len(sl)
    return before, after


def main() -> None:
    """Fit on one pure-chain OOF fold and score the other, then reverse."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="ab62_base_s4")
    parser.add_argument("--candidate", default="ab68_ema6_since_ep5")
    parser.add_argument("--aux-run", default="ab68_ema6_since")
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()

    data = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y, mask = data["y"], data["mask"]
    chain, fold = decide._chains()
    chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    train_chains = chain_data["chain"][chain_data["split"] == "train"]
    pure = ~np.isin(chain, train_chains)
    models = {}
    for k in (0, 1):
        models[k] = (
            np.load(decide.ROOT / "cache" / f"pred_{args.anchor}_f{k}_valid.npy", mmap_mode="r"),
            np.load(
                decide.ROOT / "cache" / f"pred_{args.candidate}_f{k}_valid.npy",
                mmap_mode="r",
            ),
            np.load(
                decide.ROOT / "cache" / f"auxsign_{args.aux_run}_f{k}_ep{args.epoch}_valid.npy",
                mmap_mode="r",
            ),
        )
    all_before = []
    all_after = []
    for eval_fold in (0, 1):
        fit_fold = 1 - eval_fold
        fit_idx = np.flatnonzero(pure & (fold == fit_fold))
        eval_idx = np.flatnonzero(pure & (fold == eval_fold))
        beta = fit_beta(*models[fit_fold], y, mask, fit_idx)
        before, after = score_sums(*models[eval_fold], y, mask, eval_idx, beta)
        all_before.append(before)
        all_after.append(after)
        print(f"fit_fold={fit_fold} eval_fold={eval_fold} beta={beta.tolist()}")
        for lam in (0.0, 0.02):
            print(
                f"fold{eval_fold} n={len(eval_idx)} lam={lam:.2f} "
                f"delta={decide.wp(after, lam) - decide.wp(before, lam):+.6f}"
            )
    before = np.concatenate(all_before)
    after = np.concatenate(all_after)
    for lam in (0.0, 0.02):
        print(f"pooled lam={lam:.2f} delta={decide.wp(after, lam) - decide.wp(before, lam):+.6f}")


if __name__ == "__main__":
    main()
