"""Cross-fit a bounded causal activity gate between two strong GRUs."""

from __future__ import annotations

import argparse

import decide
import numpy as np


def activity(age: np.ndarray) -> np.ndarray:
    """Smooth current/past time since the last book event."""
    return 1.0 / (1.0 + age.astype(np.float64) / 32.0)


def fit_gate(
    anchor: np.ndarray,
    partner: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    age: np.ndarray,
    idx: np.ndarray,
) -> tuple[float, float]:
    """Fit a 25%-shrunk one-parameter weighted residual correction."""
    total_feature = 0.0
    total_weight = 0.0
    for sequence in idx:
        target = np.clip(y[sequence, decide.WARMUP :].astype(np.float64), -2, 2)
        weight = np.abs(target) * mask[sequence, decide.WARMUP :, None]
        feature = activity(age[sequence, decide.WARMUP :])
        total_feature += (weight * feature[:, None]).sum()
        total_weight += weight.sum()
    center = total_feature / total_weight

    numerator = 0.0
    denominator = 0.0
    for sequence in idx:
        target = np.clip(y[sequence, decide.WARMUP :].astype(np.float64), -2, 2)
        weight = np.abs(target) * mask[sequence, decide.WARMUP :, None]
        a = np.asarray(anchor[sequence, decide.WARMUP :], dtype=np.float64)
        b = np.asarray(partner[sequence, decide.WARMUP :], dtype=np.float64)
        basis = (activity(age[sequence, decide.WARMUP :]) - center)[:, None] * (a - b)
        numerator += (weight * basis * (target - 0.5 * (a + b))).sum()
        denominator += (weight * basis * basis).sum()
    return center, float(np.clip(0.25 * numerator / max(denominator, 1e-8), -0.1, 0.1))


def score_fold(
    anchor: np.ndarray,
    partner: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    age: np.ndarray,
    idx: np.ndarray,
    center: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact clipped scorer moments for fixed and gated pairs."""
    before = np.zeros((len(idx), 2, 2, 6), dtype=np.float64)
    after = np.zeros_like(before)
    for local, sequence in enumerate(idx):
        target = np.clip(y[sequence, decide.WARMUP :].astype(np.float64), -2, 2)
        a = np.asarray(anchor[sequence, decide.WARMUP :], dtype=np.float64)
        b = np.asarray(partner[sequence, decide.WARMUP :], dtype=np.float64)
        basis = (activity(age[sequence, decide.WARMUP :]) - center)[:, None] * (a - b)
        fixed = np.clip(0.5 * (a + b), -2, 2)
        gated = np.clip(0.5 * (a + b) + delta * basis, -2, 2)
        scored = mask[sequence, decide.WARMUP :]
        for target_id in range(2):
            for mask_id, selected in enumerate((scored, ~scored)):
                before[local, mask_id, target_id] = decide._sums6(
                    target[:, target_id], fixed[:, target_id], selected
                )
                after[local, mask_id, target_id] = decide._sums6(
                    target[:, target_id], gated[:, target_id], selected
                )
    return before, after


def main() -> None:
    """Fit on each pure-chain OOF fold and evaluate on the other fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="ab68_ema6_since_ep5")
    parser.add_argument("--partner", default="ab62_base_s4")
    args = parser.parse_args()

    data = np.load(decide.ROOT / "cache" / "valid_targets.npz")
    y, mask = data["y"], data["mask"]
    with np.load(decide.ROOT / "cache" / "valid_rowinfo.npz") as rowinfo:
        age = rowinfo["since_book"]
    if len(age) != len(y):
        raise ValueError("book-event age and targets have different sequence counts")
    chain, fold = decide._chains()
    chain_data = np.load(decide.ROOT / "cache" / "chain_folds.npz")
    train_chains = chain_data["chain"][chain_data["split"] == "train"]
    pure = ~np.isin(chain, train_chains)
    models = {
        k: (
            np.load(decide.ROOT / "cache" / f"pred_{args.anchor}_f{k}_valid.npy", mmap_mode="r"),
            np.load(decide.ROOT / "cache" / f"pred_{args.partner}_f{k}_valid.npy", mmap_mode="r"),
        )
        for k in (0, 1)
    }
    before_parts = []
    after_parts = []
    for eval_fold in (0, 1):
        fit_fold = 1 - eval_fold
        fit_idx = np.flatnonzero(pure & (fold == fit_fold))
        eval_idx = np.flatnonzero(pure & (fold == eval_fold))
        center, delta = fit_gate(*models[fit_fold], y, mask, age, fit_idx)
        before, after = score_fold(*models[eval_fold], y, mask, age, eval_idx, center, delta)
        before_parts.append(before)
        after_parts.append(after)
        print(f"fit_fold={fit_fold} eval_fold={eval_fold} center={center:.6f} delta={delta:+.6f}")
        for lam in (0.0, 0.02):
            print(
                f"fold{eval_fold} lam={lam:.2f} gain={decide.wp(after, lam) - decide.wp(before, lam):+.6f}"
            )
    before = np.concatenate(before_parts)
    after = np.concatenate(after_parts)
    for lam in (0.0, 0.02):
        print(f"pooled lam={lam:.2f} gain={decide.wp(after, lam) - decide.wp(before, lam):+.6f}")


if __name__ == "__main__":
    main()
