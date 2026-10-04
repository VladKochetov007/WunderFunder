"""Cross-fit a small scored-row residual map on held-out chain-fold predictions.

This tests whether the hidden scoring selection leaves a stable, causal
conditional-mean correction after a GRU and propensity model. The correction
uses only current predictions, class, and propensity at inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROP_EDGES = np.array([0.02, 0.1, 0.3, 0.6])
DIR_EDGES = np.array([-0.5, 0.5])
N_CELLS = 4 * (len(PROP_EDGES) + 1) * (len(DIR_EDGES) + 1)
PRIOR_WEIGHT = 20_000.0


def cell_ids(prediction: np.ndarray, propensity: np.ndarray, market: int) -> np.ndarray:
    """Assign each row to a fixed causal class, propensity, and direction bin."""
    prop_bin = np.searchsorted(PROP_EDGES, propensity, side="right")
    direction = prediction[:, 0] - prediction[:, 1]
    dir_bin = np.searchsorted(DIR_EDGES, direction, side="right")
    return (market * 5 + prop_bin) * 3 + dir_bin


def fit_bins(
    indices: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    market: np.ndarray,
    predictions: np.ndarray,
    propensity: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return shrunk scored residual means and scored/unscored contrast."""
    mass = np.zeros((2, N_CELLS, 2), dtype=np.float64)
    residual = np.zeros_like(mass)
    for i in indices:
        yi = np.clip(target[i].astype(np.float64), -2, 2)
        pi = np.clip(np.asarray(predictions[i], np.float64), -2, 2)
        bins = cell_ids(pi, propensity[i], int(market[i]))[decide.WARMUP :]
        yi, pi = yi[decide.WARMUP :], pi[decide.WARMUP :]
        is_scored = mask[i, decide.WARMUP :]
        for selected, row_mask in enumerate((is_scored, ~is_scored)):
            for k in range(2):
                w = np.abs(yi[:, k]) * row_mask
                mass[selected, :, k] += np.bincount(bins, weights=w, minlength=N_CELLS)
                residual[selected, :, k] += np.bincount(
                    bins, weights=w * (yi[:, k] - pi[:, k]), minlength=N_CELLS
                )
    scored = residual[0] / np.maximum(mass[0], 1)
    unscored = residual[1] / np.maximum(mass[1], 1)
    contrast = float(np.sum(mass[0] * np.abs(scored - unscored)) / mass[0].sum())
    correction = residual[0] / (mass[0] + PRIOR_WEIGHT)
    return correction, contrast


def grade_bins(
    indices: np.ndarray,
    correction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    market: np.ndarray,
    predictions: np.ndarray,
    propensity: np.ndarray,
    base_sums: np.ndarray,
    changed_sums: np.ndarray,
) -> None:
    """Accumulate exact clipped six sums for the unchanged and corrected outputs."""
    required = np.arange(target.shape[1]) >= decide.WARMUP
    for i in indices:
        yi = np.clip(target[i].astype(np.float64), -2, 2)
        pi = np.clip(np.asarray(predictions[i], np.float64), -2, 2)
        bins = cell_ids(pi, propensity[i], int(market[i]))
        adjusted = np.clip(pi + correction[bins], -2, 2)
        for selected, row_mask in enumerate((mask[i] & required, ~mask[i] & required)):
            for k in range(2):
                base_sums[i, selected, k] = decide._sums6(yi[:, k], pi[:, k], row_mask)
                changed_sums[i, selected, k] = decide._sums6(yi[:, k], adjusted[:, k], row_mask)


def main() -> None:
    """Cross-fit within each of the two out-of-sample chain folds."""
    data = np.load(ROOT / "cache" / "valid_targets.npz")
    target, mask, market = data["y"], data["mask"], data["market_class"]
    propensity = np.load(ROOT / "cache" / "propcache_prop_v1_valid.npy", mmap_mode="r")
    chain, fold = decide._chains()
    base_sums = np.zeros((len(target), 2, 2, 6))
    changed_sums = np.zeros_like(base_sums)
    rng = np.random.default_rng(2709)
    for k, name in ((0, "ab62_since_f0"), (1, "ab62_since_f1")):
        prediction = np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
        indices = np.flatnonzero(fold == k)
        chains = rng.permutation(np.unique(chain[indices]))
        halves = (
            indices[np.isin(chain[indices], chains[: len(chains) // 2])],
            indices[np.isin(chain[indices], chains[len(chains) // 2 :])],
        )
        for side in (0, 1):
            fit = halves[side]
            grade = halves[1 - side]
            correction, contrast = fit_bins(fit, target, mask, market, prediction, propensity)
            grade_bins(
                grade,
                correction,
                target,
                mask,
                market,
                prediction,
                propensity,
                base_sums,
                changed_sums,
            )
            print(
                f"fold{k} fit_half{side} contrast={contrast:.4f} "
                f"max_correction={np.abs(correction).max():.4f}",
                flush=True,
            )
        for lam in (0.0, 0.02):
            delta = decide.wp(changed_sums[indices], lam) - decide.wp(base_sums[indices], lam)
            print(f"fold{k} lambda={lam:.2f} delta={delta:+.6f}", flush=True)
    used = np.flatnonzero(fold < 2)
    for lam in (0.0, 0.02):
        delta = decide.wp(changed_sums[used], lam) - decide.wp(base_sums[used], lam)
        print(f"pooled lambda={lam:.2f} delta={delta:+.6f}", flush=True)


if __name__ == "__main__":
    main()
