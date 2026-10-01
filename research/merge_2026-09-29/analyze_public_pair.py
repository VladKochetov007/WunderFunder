"""Compare the merged model against the exact pair used by public incumbent ZIP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analyze_oof as analysis  # noqa: E402

ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 1.0 / 3.0, 0.40, 0.50)


def main() -> None:
    """Score the public since/since pair and its equal-weight three-member extension."""
    target_data = np.load(ROOT / "cache" / "valid_targets.npz")
    all_groups = target_data["row_group"].astype(np.int64)
    positions = {int(group): index for index, group in enumerate(all_groups)}
    fold_data = np.load(ROOT / "cache" / "chain_folds.npz")
    valid = fold_data["split"] == "valid"
    chain_by_group = dict(
        zip(
            fold_data["row_group"][valid].tolist(),
            fold_data["chain"][valid].tolist(),
            strict=True,
        )
    )
    since4_path = ROOT / "cache" / "pred_ab67_final_since_s4_gcorr_valid.npy"
    since3_path = ROOT / "cache" / "pred_ab67_final_since_s3_gcorr_valid.npy"
    since4 = np.load(since4_path, mmap_mode="r")
    since3 = np.load(since3_path, mmap_mode="r")

    stats: dict[float, dict[float, list[np.ndarray]]] = {
        alpha: {0.0: [], 0.02: []} for alpha in ALPHAS
    }
    reference_stats: dict[float, list[np.ndarray]] = {0.0: [], 0.02: []}
    chains: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    fold_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {"0": {}, "1": {}}
    hashes: list[str] = []
    prediction_hashes: list[str] = []

    for fold in (0, 1):
        pred_path = HERE / f"merged160_since32_f{fold}.npy"
        metadata = json.loads(pred_path.with_suffix(".npy.json").read_text())
        row_groups = np.asarray(metadata["row_groups"], dtype=np.int64)
        if int(metadata["fold"]) != fold:
            raise ValueError(f"fold metadata mismatch for {pred_path}")
        indices = np.asarray([positions[int(group)] for group in row_groups], dtype=np.int64)
        y = target_data["y"][indices]
        mask = target_data["mask"][indices].astype(bool)
        class_ids = target_data["market_class"][indices]
        candidate = np.load(pred_path, mmap_mode="r")
        if candidate.shape != y.shape:
            raise ValueError(f"prediction shape {candidate.shape} does not match target {y.shape}")
        reference = (since4[indices] + since3[indices]) * 0.5
        pair = reference
        required = np.broadcast_to(np.arange(y.shape[1]) >= 99, y.shape[:2])

        for lam in (0.0, 0.02):
            ref_stats = analysis.sequence_stats(y, pair, mask, required, lam)
            reference_stats[lam].append(ref_stats)
            fold_scores[str(fold)][f"lambda_{lam:g}"] = {}
            for alpha in ALPHAS:
                blend = (1.0 - alpha) * pair + alpha * candidate
                candidate_stats = analysis.sequence_stats(y, blend, mask, required, lam)
                stats[alpha][lam].append(candidate_stats)
                fold_scores[str(fold)][f"lambda_{lam:g}"][f"alpha_{alpha:.4f}"] = {
                    "reference_wp": analysis.score_sums(ref_stats),
                    "candidate_wp": analysis.score_sums(candidate_stats),
                    "delta": analysis.score_sums(candidate_stats) - analysis.score_sums(ref_stats),
                }
        chains.append(np.asarray([chain_by_group[int(group)] for group in row_groups]))
        classes.append(class_ids)
        hashes.append(metadata["checkpoint_sha256"])
        prediction_hashes.append(analysis.sha256(pred_path))

    result_scores: dict[str, dict[str, object]] = {}
    all_chains = np.concatenate(chains)
    all_classes = np.concatenate(classes)
    class_ids = sorted(int(value) for value in np.unique(all_classes) if value >= 0)
    class_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for alpha in ALPHAS:
        result_scores[f"alpha_{alpha:.4f}"] = {}
        alpha_class_deltas: dict[str, dict[str, float]] = {}
        for lam in (0.0, 0.02):
            ref_stats = np.concatenate(reference_stats[lam])
            cand_stats = np.concatenate(stats[alpha][lam])
            result_scores[f"alpha_{alpha:.4f}"][f"lambda_{lam:g}"] = {
                "reference_wp": analysis.score_sums(ref_stats),
                "candidate_wp": analysis.score_sums(cand_stats),
                "delta": analysis.score_sums(cand_stats) - analysis.score_sums(ref_stats),
                "bootstrap": analysis.bootstrap_delta(
                    ref_stats,
                    cand_stats,
                    all_chains,
                    2000,
                    241 + int(alpha * 1000) + int(lam * 100),
                ),
            }
            if lam == 0.0:
                for class_id in class_ids:
                    keep = all_classes == class_id
                    alpha_class_deltas[str(class_id)] = {
                        "sequences": int(keep.sum()),
                        "reference_wp": analysis.score_sums(ref_stats[keep]),
                        "candidate_wp": analysis.score_sums(cand_stats[keep]),
                        "delta": analysis.score_sums(cand_stats[keep])
                        - analysis.score_sums(ref_stats[keep]),
                    }
        class_deltas[f"alpha_{alpha:.4f}"] = alpha_class_deltas

    result = {
        "experiment": "merge model added to exact public final_since2_gclp member pair",
        "reference_members": ["ab67_final_since_s4_gcorr", "ab67_final_since_s3_gcorr"],
        "candidate_blend": "(1-alpha) times public pair plus alpha times merged model",
        "tested_merged_model_weights": list(ALPHAS),
        "folds": [0, 1],
        "sequences": int(len(all_chains)),
        "candidate_checkpoint_sha256_by_fold": hashes,
        "candidate_predictions_sha256_by_fold": prediction_hashes,
        "reference_prediction_sha256": {
            since4_path.name: analysis.sha256(since4_path),
            since3_path.name: analysis.sha256(since3_path),
        },
        "metric": "official pooled global clipped |target|-weighted Pearson",
        "fold_scores": fold_scores,
        "scores_by_alpha": result_scores,
        "class_deltas_lambda_0_by_alpha": class_deltas,
    }
    output = HERE / "analysis_public_pair_f0_f1.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_plot(result_scores)
    print(json.dumps(result, indent=2))


def make_plot(scores: dict[str, dict[str, object]]) -> None:
    """Save public-pair blend deltas across merged-model weights."""
    plot_dir = ROOT / "artifacts" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        ax.axhline(0, color="#555555", linewidth=1)
        for key, color, label in (
            ("lambda_0", "#35678d", "exact mask"),
            ("lambda_0.02", "#df8e45", "lambda 0.02 stress"),
        ):
            xs, ys, low, high = [], [], [], []
            for alpha_key, item in scores.items():
                alpha = float(alpha_key.removeprefix("alpha_"))
                result = item[key]
                delta = float(result["delta"])
                interval = result["bootstrap"]["ci95"]
                xs.append(alpha)
                ys.append(delta)
                low.append(delta - interval[0])
                high.append(interval[1] - delta)
            ax.errorbar(xs, ys, yerr=[low, high], fmt="o-", capsize=3, color=color, label=label)
        ax.set_xlabel("Merged model weight in final blend")
        ax.set_ylabel("Global WP delta vs public pair")
        ax.set_title(
            "Public incumbent pair plus merged member\nOOF folds 0–1, chain bootstrap 95% CI"
        )
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "merged_public_pair_weight_scan_f0_f1.png", bbox_inches="tight")
        fig.savefig(plot_dir / "merged_public_pair_weight_scan_f0_f1.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
