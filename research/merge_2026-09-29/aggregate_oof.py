"""Pool the predeclared fold-0/1 merge comparisons with chain bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analyze_oof as analysis  # noqa: E402

TARGETS = ROOT / "cache" / "valid_targets.npz"
FOLDS = ROOT / "cache" / "chain_folds.npz"
PLOT_DIR = ROOT / "artifacts" / "plots"
RESULT_PATH = HERE / "analysis_f0_f1.json"
STRATEGIES = (
    "replace_base_with_merged",
    "replace_since_with_merged",
    "three_member_mean",
)


def main() -> None:
    """Pool OOF sufficient statistics across folds 0 and 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    target_data = np.load(TARGETS)
    all_groups = target_data["row_group"].astype(np.int64)
    positions = {int(group): i for i, group in enumerate(all_groups)}
    dev_groups, hold_groups = analysis.valid_row_groups(ROOT / "datasets" / "valid.parquet")
    fold_data = np.load(FOLDS)
    valid_rows = fold_data["split"] == "valid"
    chains_by_group = dict(
        zip(
            fold_data["row_group"][valid_rows].tolist(),
            fold_data["chain"][valid_rows].tolist(),
            strict=True,
        )
    )

    groups_all = []
    classes_all = []
    chains_all = []
    sums = {name: {lam: [] for lam in (0.0, 0.02)} for name in ("reference_pair", *STRATEGIES)}
    checkpoint_hashes = []
    prediction_hashes = []
    since_hashes = []
    base_hashes = []

    for fold in (0, 1):
        prediction_path = HERE / f"merged160_since32_f{fold}.npy"
        metadata = json.loads(
            prediction_path.with_suffix(prediction_path.suffix + ".json").read_text()
        )
        if int(metadata["fold"]) != fold:
            raise ValueError(f"fold metadata mismatch for {prediction_path}")
        row_groups = np.asarray(metadata["row_groups"], dtype=np.int64)
        if np.intersect1d(row_groups, hold_groups).size:
            raise ValueError(f"fold {fold} candidate includes reserved holdout row groups")
        if not np.isin(row_groups, dev_groups).all():
            raise ValueError(
                f"fold {fold} candidate includes groups outside validation development"
            )

        idx = np.array([positions[int(group)] for group in row_groups], dtype=np.int64)
        y = target_data["y"][idx]
        mask = target_data["mask"][idx].astype(bool)
        classes = target_data["market_class"][idx]
        required = np.broadcast_to(np.arange(y.shape[1]) >= 99, y.shape[:2])
        candidate = np.asarray(np.load(prediction_path, mmap_mode="r"))

        since_path = ROOT / "cache" / f"pred_ab82_ft_since4s3_f{fold}_valid.npy"
        base_path = ROOT / "cache" / f"pred_ab82_ft_base4s4_f{fold}_valid.npy"
        since = np.load(since_path, mmap_mode="r")[idx]
        base = np.load(base_path, mmap_mode="r")[idx]
        predictions = {
            "reference_pair": (since + base) * 0.5,
            "replace_base_with_merged": (since + candidate) * 0.5,
            "replace_since_with_merged": (base + candidate) * 0.5,
            "three_member_mean": (since + base + candidate) / 3.0,
        }

        for lam in (0.0, 0.02):
            for name, pred in predictions.items():
                sums[name][lam].append(analysis.sequence_stats(y, pred, mask, required, lam))
        groups_all.append(row_groups)
        classes_all.append(classes)
        chains_all.append(np.array([chains_by_group[int(group)] for group in row_groups]))
        checkpoint_hashes.append(metadata["checkpoint_sha256"])
        prediction_hashes.append(analysis.sha256(prediction_path))
        since_hashes.append(analysis.sha256(since_path))
        base_hashes.append(analysis.sha256(base_path))

    if np.intersect1d(groups_all[0], groups_all[1]).size:
        raise ValueError("fold 0 and fold 1 candidate groups overlap")
    if set(map(int, groups_all[0])).intersection(map(int, groups_all[1])):
        raise ValueError("fold 0 and fold 1 candidate groups overlap")

    groups = np.concatenate(groups_all)
    classes = np.concatenate(classes_all)
    chain_ids = np.concatenate(chains_all)
    pooled = {
        name: {lam: np.concatenate(parts, axis=0) for lam, parts in by_lam.items()}
        for name, by_lam in sums.items()
    }

    results = {}
    for name in STRATEGIES:
        results[name] = {}
        for lam in (0.0, 0.02):
            ref_stats = pooled["reference_pair"][lam]
            cand_stats = pooled[name][lam]
            results[name][f"reference_wp_lambda_{lam:g}"] = analysis.score_sums(ref_stats)
            results[name][f"wp_lambda_{lam:g}"] = analysis.score_sums(cand_stats)
            results[name][f"delta_lambda_{lam:g}"] = (
                results[name][f"wp_lambda_{lam:g}"] - results[name][f"reference_wp_lambda_{lam:g}"]
            )
            results[name][f"bootstrap_lambda_{lam:g}"] = analysis.bootstrap_delta(
                ref_stats,
                cand_stats,
                chain_ids,
                args.bootstrap,
                81 + int(lam * 100),
            )

    class_deltas = {}
    for name in STRATEGIES:
        class_deltas[name] = {}
        for class_id in sorted(int(value) for value in np.unique(classes) if value >= 0):
            keep = classes == class_id
            ref_score = analysis.score_sums(pooled["reference_pair"][0.0][keep])
            cand_score = analysis.score_sums(pooled[name][0.0][keep])
            class_deltas[name][str(class_id)] = {
                "sequences": int(keep.sum()),
                "reference_wp": ref_score,
                "candidate_wp": cand_score,
                "delta": cand_score - ref_score,
            }

    result = {
        "experiment": "merged teammate reset-diff architecture with local since weighting",
        "folds": [0, 1],
        "development_sequences": int(len(groups)),
        "metric": "official pooled global clipped |target|-weighted Pearson",
        "reference_by_fold": [
            ["pred_ab82_ft_since4s3_f0_valid.npy", "pred_ab82_ft_base4s4_f0_valid.npy"],
            ["pred_ab82_ft_since4s3_f1_valid.npy", "pred_ab82_ft_base4s4_f1_valid.npy"],
        ],
        "candidate_checkpoint_sha256_by_fold": checkpoint_hashes,
        "candidate_predictions_sha256_by_fold": prediction_hashes,
        "reference_since_sha256_by_fold": since_hashes,
        "reference_base_sha256_by_fold": base_hashes,
        "scores": results,
        "class_deltas_lambda_0": class_deltas,
        "bootstrap_samples": args.bootstrap,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_plot(results)
    print(json.dumps(result, indent=2))


def make_plot(scores: dict[str, dict[str, dict[str, float]]]) -> None:
    """Save pooled fold-0/1 blend scores and chain-bootstrap intervals."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "replace_base_with_merged": "replace base",
        "replace_since_with_merged": "replace since",
        "three_member_mean": "add third member",
    }
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.axhline(0, color="#555555", linewidth=1)
        x = np.arange(len(labels))
        offsets = (-0.10, 0.10)
        for li, lam in enumerate(("0", "0.02")):
            delta = [scores[name][f"delta_lambda_{lam}"] for name in labels]
            intervals = [scores[name][f"bootstrap_lambda_{lam}"]["ci95"] for name in labels]
            lower = [v - ci[0] for v, ci in zip(delta, intervals, strict=True)]
            upper = [ci[1] - v for v, ci in zip(delta, intervals, strict=True)]
            ax.errorbar(
                x + offsets[li],
                delta,
                yerr=[lower, upper],
                fmt="o",
                capsize=4,
                color="#333333" if lam == "0" else "#df8e45",
                label=f"lambda {lam}",
            )
        ax.set_xticks(x, list(labels.values()))
        ax.set_ylabel("Pooled pair WP delta")
        ax.set_title("Fold 0 + 1, paired bootstrap by chain (95% CI)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "merged_policy_pooled_f0_f1.png", bbox_inches="tight")
        fig.savefig(PLOT_DIR / "merged_policy_pooled_f0_f1.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
