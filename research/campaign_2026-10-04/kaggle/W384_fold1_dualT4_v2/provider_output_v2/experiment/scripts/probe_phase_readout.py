"""Cross-fit a small event-phase residual readout on held-out chain folds."""

from __future__ import annotations

import argparse

import decide
import numpy as np


def weighted_pearson(y: np.ndarray, p: np.ndarray) -> float:
    """Score two clipped targets using the organizer's pooled weighted Pearson."""
    scores = []
    for target in range(2):
        yy = np.clip(y[:, target].astype(np.float64), -2, 2)
        pp = np.clip(p[:, target].astype(np.float64), -2, 2)
        weight = np.abs(yy)
        total = weight.sum()
        mean_y = np.dot(weight, yy) / total
        mean_p = np.dot(weight, pp) / total
        cov = np.dot(weight, (yy - mean_y) * (pp - mean_p)) / total
        var_y = np.dot(weight, (yy - mean_y) ** 2) / total
        var_p = np.dot(weight, (pp - mean_p) ** 2) / total
        scores.append(cov / np.sqrt(var_y * var_p))
    return float(np.mean(scores))


def fit(y: np.ndarray, p: np.ndarray, phase: np.ndarray, fraction: float) -> np.ndarray:
    """Weighted residual slope and intercept, shrunk toward no correction."""
    beta = np.zeros((4, 2, 2), np.float64)
    for group in range(4):
        sel = phase == group
        for target in range(2):
            yy = np.clip(y[sel, target].astype(np.float64), -2, 2)
            pp = np.clip(p[sel, target].astype(np.float64), -2, 2)
            weight = np.abs(yy)
            n = weight.sum()
            swp = np.dot(weight, pp)
            swp2 = np.dot(weight, pp * pp)
            residual = yy - pp
            swr = np.dot(weight, residual)
            swpr = np.dot(weight, pp * residual)
            gram = np.array([[n, swp], [swp, swp2]])
            gram.flat[::3] += n * 0.01
            beta[group, target] = fraction * np.linalg.solve(gram, [swr, swpr])
    return beta


def apply(p: np.ndarray, phase: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply phase parameters without any label or future information."""
    out = p.copy()
    for target in range(2):
        out[:, target] += beta[phase, target, 0] + beta[phase, target, 1] * p[:, target]
    return out


def main() -> None:
    """Fit on fold 0 and score fold 1, then reverse the folds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", default="ab62_since", nargs="?")
    parser.add_argument("--fraction", type=float, default=0.25)
    args = parser.parse_args()

    h = np.load(decide.ROOT / "cache" / "heur_valid.npz")
    seq, step, scored = (h[name] for name in ("seq", "step", "scored"))
    keep = scored & (step >= 99)
    seq, step = seq[keep], step[keep]
    phase = np.searchsorted(np.log1p([3, 16, 64]), h["X"][keep, 0], side="right").astype(np.int8)
    _, fold = decide._chains()
    valid = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y = valid["y"][seq, step]
    p = np.zeros_like(y)
    for k in (0, 1):
        rows = fold[seq] == k
        prediction = np.load(
            decide.ROOT / "cache" / f"pred_{args.prefix}_f{k}_valid.npy", mmap_mode="r"
        )
        p[rows] = prediction[seq[rows], step[rows]]
    selected = np.isin(fold[seq], [0, 1])
    corrections = p.copy()
    for eval_fold in (0, 1):
        train = fold[seq] == 1 - eval_fold
        eval_rows = fold[seq] == eval_fold
        beta = fit(y[train], p[train], phase[train], args.fraction)
        corrections[eval_rows] = apply(p[eval_rows], phase[eval_rows], beta)
        before = weighted_pearson(y[eval_rows], p[eval_rows])
        after = weighted_pearson(y[eval_rows], corrections[eval_rows])
        print(f"fold{eval_fold}: {after - before:+.6f} ({after:.6f} vs {before:.6f})")
        print(f"beta_fit_on_fold{1 - eval_fold}: {np.round(beta, 4).tolist()}")
    before = weighted_pearson(y[selected], p[selected])
    after = weighted_pearson(y[selected], corrections[selected])
    print(f"pooled: {after - before:+.6f} ({after:.6f} vs {before:.6f})")


if __name__ == "__main__":
    main()
