"""Check whether an auxiliary sign logit adds out-of-chain pair information."""

from __future__ import annotations

import json
from pathlib import Path

import decide
import numpy as np
from probe_pure_pair import pair_sums
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def fit_weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fit a small weighted linear readout with fixed ridge stabilization."""
    scale = np.sqrt(w / w.mean())
    lhs = x * scale[:, None]
    rhs = y * scale
    return np.linalg.solve(lhs.T @ lhs + np.eye(x.shape[1]) * 1e-3, lhs.T @ rhs)


def main() -> None:
    """Write a reproducible one-fold, chain-cross-fitted diagnostic."""
    data = np.load(ROOT / "cache/valid_targets.npz")
    y, mask, cls = data["y"], data["mask"], data["market_class"]
    chain, fold = decide._chains()
    cf = np.load(ROOT / "cache/chain_folds.npz")
    trained_chains = cf["chain"][cf["split"] == "train"]
    seq = np.flatnonzero((fold == 0) & ~np.isin(chain, trained_chains))
    class_seq = seq[cls[seq] == 3]
    base = np.load(ROOT / "cache/pred_ab62_base_s4_f0_valid.npy", mmap_mode="r")
    since = np.load(ROOT / "cache/pred_ab62_since_f0_valid.npy", mmap_mode="r")
    logit = np.load(ROOT / "cache/auxsign_ab78_auxsign01_f0_ep3_valid.npy", mmap_mode="r")
    pair = np.asarray(0.5 * (base[class_seq] + since[class_seq]), np.float32)
    z = np.asarray(logit[class_seq], np.float32).clip(-8, 8)
    target = np.asarray(np.clip(y[class_seq], -2, 2), np.float32)
    p = 0.5 * (pair[..., 0] - pair[..., 1])
    truth = 0.5 * (target[..., 0] - target[..., 1])
    active = mask[class_seq] & (np.arange(y.shape[1]) >= decide.WARMUP) & (truth != 0)
    w = np.abs(truth[active]).astype(np.float64)
    direction = (truth[active] > 0).astype(np.int8)
    report: dict[str, object] = {
        "pure_fold0_sequences": int(len(seq)),
        "class3_sequences": int(len(class_seq)),
        "scored_nonzero_direction_rows": int(active.sum()),
        "weighted_auc_pair_contrast": float(roc_auc_score(direction, p[active], sample_weight=w)),
        "weighted_auc_aux_sign_logit": float(roc_auc_score(direction, z[active], sample_weight=w)),
    }
    pair_right = (p[active] > 0) == direction
    sign_right = (z[active] > 0) == direction
    report["weighted_direction_accuracy_pair"] = float(np.average(pair_right, weights=w))
    report["weighted_direction_accuracy_aux_sign"] = float(np.average(sign_right, weights=w))
    confidence_conflict = (p * z < 0) & (np.abs(z) >= 2.2) & (np.abs(p) < 0.4)
    scored_conflict = confidence_conflict[active]
    report["confidence_conflict_weight_share"] = float(w[scored_conflict].sum() / w.sum())
    report["confidence_conflict_sign_head_accuracy"] = float(
        np.average(sign_right[scored_conflict], weights=w[scored_conflict])
    ) if scored_conflict.any() else None
    rng = np.random.default_rng(41)
    ids = np.unique(chain[class_seq])
    rng.shuffle(ids)
    side = np.isin(chain[class_seq], ids[: len(ids) // 2])
    readouts: dict[str, np.ndarray] = {}
    for name, columns in (
        ("pair_calibration", (np.ones_like(p), p, p**3)),
        ("pair_plus_sign", (np.ones_like(p), p, p**3, z)),
    ):
        x = np.stack(columns, axis=-1)
        estimate = np.zeros_like(p)
        coefficients = []
        for train_side in (False, True):
            train = (side == train_side)[:, None] & active
            test = side != train_side
            beta = fit_weighted(
                x[train].astype(np.float64),
                truth[train].astype(np.float64),
                np.abs(truth[train]).astype(np.float64),
            )
            coefficients.append(beta.tolist())
            estimate[test] = x[test] @ beta
        readouts[name] = estimate
        report[name + "_coefficients"] = coefficients
    basis = np.stack((np.ones_like(p), p, p**3), axis=-1)
    residual_estimate = np.zeros_like(p)
    residual_coefficients = []
    for train_side in (False, True):
        train = (side == train_side)[:, None] & active
        test = side != train_side
        train_weight = np.abs(truth[train]).astype(np.float64)
        z_projection = fit_weighted(
            basis[train].astype(np.float64),
            z[train].astype(np.float64),
            train_weight,
        )
        z_residual = z[train] - basis[train] @ z_projection
        slope = float(
            np.sum(train_weight * z_residual * (truth[train] - p[train]))
            / (np.sum(train_weight * z_residual**2) + 1e-3)
        )
        residual_coefficients.append(slope)
        residual_estimate[test] = p[test] + 0.25 * slope * (
            z[test] - basis[test] @ z_projection
        )
    readouts["residual_sign_25pct"] = residual_estimate
    report["residual_sign_coefficients"] = residual_coefficients
    baseline = 0.5 * (np.asarray(base, np.float32) + np.asarray(since, np.float32))
    scores: dict[str, dict[str, float]] = {}
    soft_sign = 0.5 * np.tanh(z / 2)
    for name, contrast in (
        ("pair", p),
        *readouts.items(),
        ("soft_sign_25pct", 0.75 * p + 0.25 * soft_sign),
        ("soft_sign_50pct", 0.5 * p + 0.5 * soft_sign),
        ("confident_disagreement_abstain", np.where(confidence_conflict, 0.0, p)),
    ):
        if name == "pair":
            candidate = baseline
        else:
            candidate = baseline.copy()
            difference = (contrast - p).astype(np.float32)
            candidate[class_seq, :, 0] += difference
            candidate[class_seq, :, 1] -= difference
        sums = pair_sums(candidate, candidate, y, mask, seq)
        scores[name] = {
            "lambda0": decide.wp(sums, 0.0),
            "lambda002": decide.wp(sums, 0.02),
        }
    report["pure_pair_scores"] = scores
    path = ROOT / "research/artifacts/brainteaser_aux_sign_readout.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
