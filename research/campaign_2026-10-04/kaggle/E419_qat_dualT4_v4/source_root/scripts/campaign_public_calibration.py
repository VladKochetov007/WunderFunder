"""Nested leave-one-submission-out calibration of local proxy to public score.

All public scores are user-reported portal evidence. The historical 368-sequence
proxy is repeatedly inspected and chain-linked; this script measures empirical
transfer error, not specification authority or a clean confirmation fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import decide  # noqa: E402
from ensemble_scan import parse_blend  # noqa: E402
from holdout_compare import blend_sums  # noqa: E402
from lb_calibration import CANDIDATES, SCORED  # noqa: E402

from wnn.splits import valid_row_groups  # noqa: E402

SCORED_BASE = SCORED | {
    "final_ema6_shared": (
        "ab72_final_ema6_s3_ep5+ab72_final_ema6_s4_ep5",
        0.6612,
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept, declining degenerate proxy values."""
    if np.ptp(x) <= 1e-9:
        raise ValueError("proxy scores have no resolvable spread")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def inner_error(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    """Estimate a proxy variant's error without its prospective held-out package."""
    residuals = []
    for i in train:
        fit = train[train != i]
        slope, intercept = linear_fit(x[fit], y[fit])
        residuals.append(intercept + slope * x[i] - y[i])
    return float(np.sqrt(np.mean(np.square(residuals))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ema-membership", choices=("pair", "single"), default="pair")
    args = parser.parse_args()
    scored = dict(SCORED_BASE)
    if args.ema_membership == "single":
        member0 = "ab72_final_ema6_s3_ep5"
        for name in ("final_mixed_ema6_shared", "final_ema6_shared"):
            scored[name] = (member0, scored[name][1])
    with np.load(ROOT / "cache" / "valid_targets.npz") as cache:
        y, mask, rg, classes = cache["y"], cache["mask"], cache["row_group"], cache["market_class"]
    with np.load(ROOT / "cache" / "train_targets.npz") as cache:
        train_mix = np.bincount(cache["market_class"], minlength=4) / len(cache["market_class"])
    _, held = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    idx = np.flatnonzero(np.isin(rg, held))
    class_id = classes[idx]
    counts = np.bincount(class_id, minlength=4)
    train_mix_weight = train_mix[class_id] * len(idx) / counts[class_id]
    chain, _ = decide._chains()
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    in_fit = set(cf["chain"][cf["split"] == "train"].tolist()) | set(chain[~np.isin(rg, held)].tolist())
    isolated = np.array([component not in in_fit for component in chain[idx]])
    adversary = np.load(ROOT / "cache" / "adv_valid_score.npy")[idx]
    unlike = np.zeros(len(idx), bool)
    for market_class in range(4):
        selected = class_id == market_class
        unlike[selected] = adversary[selected] >= np.quantile(adversary[selected], 1 / 3)
    weights = {
        "holdout": np.ones(len(idx)),
        "train_mix": train_mix_weight,
        "class3_x2": np.where(class_id == 3, 2.0, 1.0),
        "class3_x0_5": np.where(class_id == 3, 0.5, 1.0),
        "isolated": isolated.astype(float),
        "linked": (~isolated).astype(float),
        "train_unlike": unlike.astype(float),
    }
    variants = [(label, lam, w) for label, w in weights.items() for lam in (0.0, 0.02, 0.05)]
    names = list(scored)
    public = np.array([scored[name][1] for name in names], dtype=np.float64)
    sums = {
        name: blend_sums(parse_blend(scored[name][0]), idx, y, mask)
        for name in names
    }
    local = np.array(
        [[decide.wp(sums[name] * w[:, None, None, None], lam) for name in names] for _, lam, w in variants]
    )
    outer_rows = []
    all_packages = np.arange(len(names))
    for held_out in all_packages:
        train = all_packages[all_packages != held_out]
        inner = np.array([inner_error(proxy, public, train) for proxy in local])
        chosen = int(np.argmin(inner))
        slope, intercept = linear_fit(local[chosen, train], public[train])
        prediction = intercept + slope * local[chosen, held_out]
        outer_rows.append(
            {
                "name": names[held_out],
                "public": float(public[held_out]),
                "predicted": float(prediction),
                "error": float(prediction - public[held_out]),
                "chosen_variant": f"{variants[chosen][0]}_lambda{variants[chosen][1]:g}",
                "inner_rmse": float(inner[chosen]),
            }
        )
    outer_residuals = np.array([row["error"] for row in outer_rows])
    full_inner = np.array([inner_error(proxy, public, all_packages) for proxy in local])
    selected = int(np.argmin(full_inner))
    slope, intercept = linear_fit(local[selected], public)
    label, lam, w = variants[selected]
    candidate_forecasts = {
        name: float(intercept + slope * decide.wp(blend_sums(parse_blend(spec), idx, y, mask) * w[:, None, None, None], lam))
        for name, spec in CANDIDATES.items()
    }
    top = np.flatnonzero(public >= 0.6586)
    pairs = [(i, j) for i, j in combinations(top, 2) if abs(public[i] - public[j]) >= 0.0004]
    report = {
        "metric_id": "global_wp_clip2_v1",
        "ema_membership_scenario": args.ema_membership,
        "ema_membership_caveat": "EMA6 packages collapse to identical member0 on v2 CPU; user-reported public scores differ, and portal CPU level is not independently known",
        "source": "user-reported public submissions; local 368-sequence inspected, chain-linked holdout",
        "n_submissions": len(names),
        "n_proxy_variants": len(variants),
        "submission_order": names,
        "public_scores_user_reported": public.tolist(),
        "submission_prediction_specs": {name: scored[name][0] for name in names},
        "local_archive_sha256": {
            name: sha256(ROOT / "submissions" / name / f"{name}.zip")
            for name in ("final_ema6_shared", "final_mixed_ema6_shared", "final_since2_gclp")
        },
        "ema_member0_npz_sha256": {
            name: sha256(ROOT / "submissions" / name / "pkg" / "weights_0.npz")
            for name in ("final_ema6_shared", "final_mixed_ema6_shared")
        },
        "candidate_prediction_specs": CANDIDATES,
        "proxy_variants": [
            {"weight_id": label, "soft_mask_lambda": lam} for label, lam, _ in variants
        ],
        "local_scores_by_variant_by_submission": local.tolist(),
        "holdout_row_groups": rg[idx].astype(int).tolist(),
        "isolated_row_groups": rg[idx][isolated].astype(int).tolist(),
        "prediction_file_sha256": {
            name: sha256(ROOT / "cache" / f"pred_{name}_valid.npy")
            for name in sorted({member for spec in [*(entry[0] for entry in scored.values()), *CANDIDATES.values()] for member in parse_blend(spec)})
        },
        "input_file_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "cache" / "valid_targets.npz",
                ROOT / "cache" / "train_targets.npz",
                ROOT / "cache" / "chain_folds.npz",
                ROOT / "cache" / "adv_valid_score.npy",
            )
        },
        "calibration_script_sha256": sha256(Path(__file__)),
        "isolated_sequences": int(isolated.sum()),
        "nested_loso": outer_rows,
        "nested_loso_rmse": float(np.sqrt(np.mean(outer_residuals**2))),
        "nested_loso_mae": float(np.mean(np.abs(outer_residuals))),
        "nested_loso_bias": float(np.mean(outer_residuals)),
        "nested_loso_error_range": [float(outer_residuals.min()), float(outer_residuals.max())],
        "full_selected_variant": f"{label}_lambda{lam:g}",
        "full_variant_loso_rmse": float(full_inner[selected]),
        "full_slope": slope,
        "full_intercept": intercept,
        "top_pair_sign_accuracy": float(np.mean([np.sign(local[selected, i] - local[selected, j]) == np.sign(public[i] - public[j]) for i, j in pairs])),
        "candidate_forecasts_exploratory": candidate_forecasts,
        "caveat": "candidate forecasts omit mechanism-specific transfer discount and winner's-curse correction; never use alone for promotion",
    }
    output = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / f"nested_public_calibration_{args.ema_membership}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("nested_loso_rmse", "nested_loso_bias", "full_selected_variant", "candidate_forecasts_exploratory")}, indent=2))


if __name__ == "__main__":
    main()
