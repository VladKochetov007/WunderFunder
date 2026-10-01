"""Score merged-model pair replacements on untouched development chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import wp_from_sums, wp_sums  # noqa: E402
from wnn.splits import valid_row_groups  # noqa: E402

TARGETS = ROOT / "cache" / "valid_targets.npz"
FOLDS = ROOT / "cache" / "chain_folds.npz"
PLOT_DIR = ROOT / "artifacts" / "plots"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_stats(
    targets: np.ndarray,
    predictions: np.ndarray,
    scored: np.ndarray,
    required: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return pooled WP sufficient sums for each sequence and target."""
    stats = np.empty((len(targets), 2, 6), dtype=np.float64)
    unscored = required & ~scored
    for target in range(2):
        selected = wp_sums(targets[..., target], predictions[..., target], scored)
        stress = wp_sums(targets[..., target], predictions[..., target], unscored)
        stats[:, target] = selected + lam * stress
    return stats


def score_sums(stats: np.ndarray) -> float:
    """Convert pooled per-target sufficient sums into the official mean WP."""
    return float(np.mean(wp_from_sums(np.asarray(stats).sum(axis=0))))


def bootstrap_delta(
    reference: np.ndarray,
    candidate: np.ndarray,
    chain_ids: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Estimate a paired score delta by resampling whole sequence chains."""
    unique_chains, inverse = np.unique(chain_ids, return_inverse=True)
    ref_by_chain = np.zeros((len(unique_chains), 2, 6), dtype=np.float64)
    cand_by_chain = np.zeros_like(ref_by_chain)
    np.add.at(ref_by_chain, inverse, reference)
    np.add.at(cand_by_chain, inverse, candidate)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_chains), size=(samples, len(unique_chains)))
    ref_scores = np.array([score_sums(ref_by_chain[row]) for row in draws])
    cand_scores = np.array([score_sums(cand_by_chain[row]) for row in draws])
    deltas = cand_scores - ref_scores
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean": float(deltas.mean()),
        "sd": float(deltas.std(ddof=1)),
        "ci95": [float(low), float(high)],
        "p_positive": float(np.mean(deltas > 0)),
        "chains": int(len(unique_chains)),
        "samples": int(samples),
    }


def pooled_score(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    required: np.ndarray,
    keep: np.ndarray,
    lam: float,
) -> float:
    """Score selected sequences with the exact clipped global WP formula."""
    stats = sequence_stats(targets[keep], predictions[keep], mask[keep], required[keep], lam)
    return score_sums(stats)


def main() -> None:
    """Evaluate predeclared equal-weight pair and three-member combinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reference_since = ROOT / "cache" / f"pred_ab82_ft_since4s3_f{args.fold}_valid.npy"
    reference_base = ROOT / "cache" / f"pred_ab82_ft_base4s4_f{args.fold}_valid.npy"

    metadata = json.loads(args.prediction.with_suffix(args.prediction.suffix + ".json").read_text())
    row_groups = np.asarray(metadata["row_groups"], dtype=np.int64)
    candidate = np.load(args.prediction, mmap_mode="r")
    target_data = np.load(TARGETS)
    all_groups = target_data["row_group"].astype(np.int64)
    positions = {int(group): i for i, group in enumerate(all_groups)}
    idx = np.array([positions[int(group)] for group in row_groups], dtype=np.int64)
    y = target_data["y"][idx]
    mask = target_data["mask"][idx].astype(bool)
    classes = target_data["market_class"][idx]
    required = np.broadcast_to(np.arange(y.shape[1]) >= 99, y.shape[:2])

    dev_groups, hold_groups = valid_row_groups(ROOT / "datasets" / "valid.parquet")
    if np.intersect1d(row_groups, hold_groups).size:
        raise ValueError("candidate predictions include reserved holdout row groups")
    if not np.isin(row_groups, dev_groups).all():
        raise ValueError("candidate predictions include row groups outside validation development")

    fold_data = np.load(FOLDS)
    valid = fold_data["split"] == "valid"
    chains_by_group = dict(
        zip(
            fold_data["row_group"][valid].tolist(),
            fold_data["chain"][valid].tolist(),
            strict=True,
        )
    )
    chain_ids = np.array([chains_by_group[int(group)] for group in row_groups])

    since = np.load(reference_since, mmap_mode="r")[idx]
    base = np.load(reference_base, mmap_mode="r")[idx]
    merged = np.asarray(candidate)
    predictions = {
        "reference_pair": (since + base) * 0.5,
        "replace_base_with_merged": (since + merged) * 0.5,
        "replace_since_with_merged": (base + merged) * 0.5,
        "three_member_mean": (since + base + merged) / 3.0,
    }
    scores: dict[str, dict[str, float]] = {}
    comparisons = [
        ("replace_base_with_merged", "reference_pair"),
        ("replace_since_with_merged", "reference_pair"),
        ("three_member_mean", "reference_pair"),
    ]
    for lam in (0.0, 0.02):
        per_model = {
            name: sequence_stats(y, pred, mask, required, lam) for name, pred in predictions.items()
        }
        for name, _ in comparisons:
            scores.setdefault(name, {})[f"wp_lambda_{lam:g}"] = score_sums(per_model[name])
        for name, ref_name in comparisons:
            key = f"{name}_minus_{ref_name}_lambda_{lam:g}"
            scores.setdefault(name, {})[f"delta_lambda_{lam:g}"] = scores[name][
                f"wp_lambda_{lam:g}"
            ] - score_sums(per_model[ref_name])
            scores[name][f"bootstrap_lambda_{lam:g}"] = bootstrap_delta(
                per_model[ref_name],
                per_model[name],
                chain_ids,
                args.bootstrap,
                41 + int(lam * 100),
            )
            scores[name][f"comparison_key_lambda_{lam:g}"] = key

    class_deltas = {}
    for name, _ in comparisons:
        class_deltas[name] = {}
        for class_id in sorted(int(value) for value in np.unique(classes) if value >= 0):
            keep = classes == class_id
            ref_score = pooled_score(y, predictions["reference_pair"], mask, required, keep, 0.0)
            candidate_score = pooled_score(y, predictions[name], mask, required, keep, 0.0)
            class_deltas[name][str(class_id)] = {
                "sequences": int(keep.sum()),
                "reference_wp": ref_score,
                "candidate_wp": candidate_score,
                "delta": candidate_score - ref_score,
            }

    result = {
        "experiment": "merged teammate reset-diff architecture with local since weighting",
        "fold": int(metadata["fold"]),
        "development_sequences": int(len(row_groups)),
        "metric": "official pooled global clipped |target|-weighted Pearson",
        "reference": [reference_since.name, reference_base.name],
        "candidate_checkpoint_sha256": metadata["checkpoint_sha256"],
        "candidate_predictions_sha256": sha256(args.prediction),
        "reference_since_sha256": sha256(reference_since),
        "reference_base_sha256": sha256(reference_base),
        "scores": scores,
        "class_deltas_lambda_0": class_deltas,
    }
    result_path = EXPERIMENT_DIR / f"analysis_f{args.fold}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_plot(scores, args.fold)
    logging.info(json.dumps(result, indent=2))


def make_plot(scores: dict[str, dict[str, float]], fold: int) -> None:
    """Save predeclared ensemble scores and paired chain-bootstrap deltas."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    lambdas = ["0", "0.02"]
    strategies = {
        "replace_base_with_merged": "replace base",
        "replace_since_with_merged": "replace since",
        "three_member_mean": "add third member",
    }
    colors = {
        "reference_pair": "#35678d",
        "replace_base_with_merged": "#df8e45",
        "replace_since_with_merged": "#6e9e75",
        "three_member_mean": "#8a6dae",
    }

    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
    ):
        fig, (score_ax, delta_ax) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        x = np.arange(2)
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, 4)
        reference_scores = [
            scores["replace_base_with_merged"][f"wp_lambda_{lam}"]
            - scores["replace_base_with_merged"][f"delta_lambda_{lam}"]
            for lam in lambdas
        ]
        score_ax.bar(
            x + offsets[0],
            reference_scores,
            width,
            label="reference pair",
            color=colors["reference_pair"],
        )
        for offset, (name, label) in zip(offsets[1:], strategies.items(), strict=True):
            score_ax.bar(
                x + offset,
                [scores[name][f"wp_lambda_{lam}"] for lam in lambdas],
                width,
                label=label,
                color=colors[name],
            )
        score_ax.set_xticks(x, ["exact mask", "lambda 0.02 stress"])
        score_ax.set_ylabel("Global WP")
        all_scores = reference_scores + [
            scores[name][f"wp_lambda_{lam}"] for name in strategies for lam in lambdas
        ]
        score_ax.set_ylim(min(all_scores) - 0.003, max(all_scores) + 0.001)
        score_ax.legend(frameon=False, fontsize=7)
        score_ax.set_title(f"OOF blend scores, fold {fold}")

        delta_ax.axhline(0, color="#555555", linewidth=1)
        strategy_x = np.arange(len(strategies))
        x_offsets = (-0.10, 0.10)
        for li, lam in enumerate(lambdas):
            delta = [scores[name][f"delta_lambda_{lam}"] for name in strategies]
            intervals = [scores[name][f"bootstrap_lambda_{lam}"]["ci95"] for name in strategies]
            lower = [value - interval[0] for value, interval in zip(delta, intervals, strict=True)]
            upper = [interval[1] - value for value, interval in zip(delta, intervals, strict=True)]
            delta_ax.errorbar(
                strategy_x + x_offsets[li],
                delta,
                yerr=[lower, upper],
                fmt="o",
                color="#333333" if lam == "0" else "#df8e45",
                capsize=3,
                linewidth=1.2,
                label=f"lambda {lam}",
            )
        delta_ax.set_xticks(strategy_x, list(strategies.values()))
        delta_ax.set_ylabel("Pair WP delta")
        delta_ax.set_title("Paired chain bootstrap, 95% interval")
        delta_ax.legend(frameon=False, fontsize=8)
        delta_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"merged_policy_fold{fold}.png", bbox_inches="tight")
        fig.savefig(PLOT_DIR / f"merged_policy_fold{fold}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
