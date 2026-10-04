"""Screen a training-only class-3 future-path denoising target on pure chains."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from online_c3 import law
from wnn.data import SequenceFile
from wnn.ticks import reconstruct

ROOT = Path(__file__).resolve().parents[1]
MAX_HALF_TICK = 8
STATES = np.arange(-MAX_HALF_TICK, MAX_HALF_TICK + 1)


def book_events(features: np.ndarray, moves: dict, spreads: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the quote path at book events and return half-tick midpoint moves."""
    starts = np.r_[0, np.flatnonzero(np.any(features[1:, :22] != features[:-1, :22], axis=1)) + 1]
    ask, bid, _ = reconstruct(features[starts], moves, spreads)
    increments = np.diff(ask + bid).astype(np.int16)
    return starts, ask, bid, increments


def fit_kernel(source: SequenceFile, groups: np.ndarray, moves: dict, spreads: dict) -> np.ndarray:
    """Estimate P(next midpoint increment | preceding increment) on train only."""
    count = np.full((len(STATES), len(STATES)), 0.5, dtype=np.float64)
    for position, group in enumerate(groups, 1):
        features = source.read(int(group), targets=False, mask=False).features
        _, _, _, increments = book_events(features, moves, spreads)
        first, second = increments[:-1], increments[1:]
        keep = (np.abs(first) <= MAX_HALF_TICK) & (np.abs(second) <= MAX_HALF_TICK)
        flat = (first[keep] + MAX_HALF_TICK) * len(STATES) + second[keep] + MAX_HALF_TICK
        count += np.bincount(flat, minlength=count.size).reshape(count.shape)
        if position % 50 == 0:
            print(f"train sequences decoded: {position}/{len(groups)}", flush=True)
    return count / count.sum(axis=1, keepdims=True)


def payoff_moments(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate three unknown events following the observed first move."""
    m1 = np.full((7, 2, len(STATES), 2), np.nan, dtype=np.float64)
    m2 = np.full_like(m1, np.nan)
    h2, h3, h4 = np.meshgrid(STATES, STATES, STATES, indexing="ij")
    h2, h3, h4 = h2.ravel(), h3.ravel(), h4.ravel()
    path_weight = kernel[h2 + MAX_HALF_TICK, h3 + MAX_HALF_TICK] * kernel[h3 + MAX_HALF_TICK, h4 + MAX_HALF_TICK]
    path_offset = (3 * h2 + 2 * h3 + h4) / 8.0
    for spread in range(1, 7):
        for ask_parity in (0, 1):
            ask = np.full(h2.shape, ask_parity, dtype=np.int64)
            bid = ask - spread
            mid = (ask + bid) / 2.0
            for first_index, first_move in enumerate(STATES):
                weights = path_weight * kernel[first_index, h2 + MAX_HALF_TICK]
                future_reference = np.rint(mid + first_move / 2.0 + path_offset)
                targets = np.clip(law(future_reference, bid, ask), -2.0, 2.0)
                m1[spread, ask_parity, first_index] = weights @ np.abs(targets)
                m2[spread, ask_parity, first_index] = weights @ (np.abs(targets) * targets)
    return m1, m2


def fold_metrics(
    fold: int,
    row_groups: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    source: SequenceFile,
    moves: dict,
    spreads: dict,
    m1_table: np.ndarray,
    m2_table: np.ndarray,
) -> dict:
    """Calculate coverage, residual variance and tilted covariance bias."""
    first = np.load(ROOT / "cache" / f"pred_ab62_base_s4_f{fold}_valid.npy", mmap_mode="r")
    second = np.load(ROOT / "cache" / f"pred_ab62_since_f{fold}_valid.npy", mmap_mode="r")
    sums = np.zeros((2, 11), dtype=np.float64)
    # Columns: full mass, eligible mass, pseudo mass, true wy, true wp, true wyp,
    # pseudo m2, pseudo m1p, pseudo m2p, squared true-pseudo, squared true-pair.
    for position, index in enumerate(row_groups, 1):
        features = source.read(int(index), targets=False, mask=False).features
        starts, ask, bid, increments = book_events(features, moves, spreads)
        if len(starts) < 5:
            continue
        event_id = np.searchsorted(starts, np.arange(len(features)), side="right") - 1
        good = masks[index] & (np.arange(len(features)) >= 99) & (event_id + 4 < len(starts))
        spread = (ask - bid)[event_id]
        first_move = increments[np.minimum(event_id, len(increments) - 1)]
        good &= (spread >= 1) & (spread <= 6) & (np.abs(first_move) <= MAX_HALF_TICK)
        yi = np.clip(targets[index].astype(np.float64), -2.0, 2.0)
        for target_id in range(2):
            sums[target_id, 0] += np.abs(yi[masks[index], target_id]).sum()
        if not np.any(good):
            continue
        event = event_id[good]
        moment1 = m1_table[spread[good], ask[event] & 1, first_move[good] + MAX_HALF_TICK]
        moment2 = m2_table[spread[good], ask[event] & 1, first_move[good] + MAX_HALF_TICK]
        y = yi[good]
        pred = 0.5 * (np.asarray(first[index, good], np.float64) + np.asarray(second[index, good], np.float64))
        pseudo = moment2 / np.maximum(moment1, 1e-8)
        weight = np.abs(y)
        for target_id in range(2):
            w, yy, pp, mm1, mm2, qq = (
                weight[:, target_id], y[:, target_id], pred[:, target_id],
                moment1[:, target_id], moment2[:, target_id], pseudo[:, target_id]
            )
            sums[target_id, 1:] += [
                w.sum(), mm1.sum(), (w * yy).sum(), (w * pp).sum(),
                (w * yy * pp).sum(), mm2.sum(), (mm1 * pp).sum(),
                (mm2 * pp).sum(), (w * (yy - qq) ** 2).sum(),
                (w * (yy - pp) ** 2).sum(),
            ]
        if position % 50 == 0:
            print(f"fold {fold}: {position}/{len(row_groups)} sequences decoded", flush=True)
    by_target = []
    for row in sums:
        full, true_mass, pseudo_mass, wy, wp, wyp, m2, m1p, m2p, pseudo_err, pair_err = row
        true_cov = wyp - wy * wp / true_mass
        pseudo_cov = m2p - m2 * m1p / pseudo_mass
        by_target.append({
            "target_weight_coverage": true_mass / full,
            "noise_mse_ratio_vs_pair": pseudo_err / pair_err,
            "covariance_bias_share": abs(pseudo_cov - true_cov) / abs(true_cov),
            "weight_mass_bias_share": abs(pseudo_mass - true_mass) / true_mass,
            "true_covariance": true_cov,
            "pseudo_covariance": pseudo_cov,
        })
    return {"fold": fold, "class3_sequences": len(row_groups), "targets": by_target}


def main() -> None:
    """Run the fixed training-only label-quality screen on both pure folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "research/artifacts/rb_class3_screen.json")
    args = parser.parse_args()
    with (ROOT / "cache/class3_moves_full.pkl").open("rb") as handle:
        moves, spreads = pickle.load(handle)
    classes = np.load(ROOT / "research/artifacts/market-classes-v1.npz")
    train_candidates = np.flatnonzero(classes["train"] == 3)
    train_groups = np.random.default_rng(41).choice(train_candidates, args.train_sequences, replace=False)
    kernel = fit_kernel(SequenceFile(ROOT / "datasets/train.parquet"), train_groups, moves, spreads)
    m1, m2 = payoff_moments(kernel)
    valid = np.load(ROOT / "cache/valid_targets.npz")
    fold_map = np.load(ROOT / "cache/chain_folds.npz")
    train_chains = fold_map["chain"][fold_map["split"] == "train"]
    valid_rows = fold_map["split"] == "valid"
    fold_of = np.full(len(classes["valid"]), -1, dtype=np.int16)
    chain_of = np.full(len(fold_of), -1, dtype=np.int64)
    fold_of[fold_map["row_group"][valid_rows]] = fold_map["fold"][valid_rows]
    chain_of[fold_map["row_group"][valid_rows]] = fold_map["chain"][valid_rows]
    source = SequenceFile(ROOT / "datasets/valid.parquet")
    result = {"train_sequences": args.train_sequences, "seed": 41, "kernel_support_half_ticks": [-8, 8], "folds": []}
    for fold in (0, 1):
        groups = np.flatnonzero((classes["valid"] == 3) & (fold_of == fold) & ~np.isin(chain_of, train_chains))
        result["folds"].append(fold_metrics(fold, groups, valid["y"], valid["mask"], source, moves, spreads, m1, m2))
    result["passes_gate"] = all(
        target["target_weight_coverage"] >= 0.6
        and target["noise_mse_ratio_vs_pair"] <= 0.7
        and target["covariance_bias_share"] <= 0.05
        and target["weight_mass_bias_share"] <= 0.05
        for fold in result["folds"] for target in fold["targets"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
