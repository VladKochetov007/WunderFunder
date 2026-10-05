"""Render frozen E486 crossed-activation grade JSON; never reruns inference."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = ROOT / "research/campaign_2026-10-05/activation_counterfactual"
EDA = HERE / "eda/cross_v1"
SCORES = EDA / "scores.json"
EXPORT = EDA / "selected_exports.npz"
SEALED = HERE / "e486_cpu_cross_v1.sealed.json"
ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e486-activation-counterfactual-cpu-cross-v1.json"
)
REPRODUCTION = (
    ROOT / "research/campaign_2026-10-05/checks/counterfactual_activation_export_reproduction.json"
)
REPORT = HERE / "REPORT.md"
SEEDS = (271831, 271832, 271833)
BETAS = (0.8, 0.9, 0.95)
SHAPES = ("baseline", "sine")
CELL_ORDER = (
    ("baseline", "baseline", "A: base→base"),
    ("baseline", "sine", "B: base→sine"),
    ("sine", "baseline", "C: sine→base"),
    ("sine", "sine", "D: sine→sine"),
)
CONTRASTS = (
    "D_minus_A_matched_diagonal",
    "B_minus_A_inference_sine_on_base_fit",
    "C_minus_D_remove_sine_on_sine_fit",
    "C_minus_A_train_shape_under_base_eval",
    "D_minus_B_train_shape_under_sine_eval",
    "training_by_inference_interaction",
)
CONTRAST_LABELS = {
    "D_minus_A_matched_diagonal": "D − A: matched-shape diagonal",
    "B_minus_A_inference_sine_on_base_fit": "B − A: sine inference on base fit",
    "C_minus_D_remove_sine_on_sine_fit": "C − D: remove sine on sine fit",
    "C_minus_A_train_shape_under_base_eval": "C − A: train shape under base inference",
    "D_minus_B_train_shape_under_sine_eval": "D − B: train shape under sine inference",
    "training_by_inference_interaction": "(D − B) − (C − A): train × inference interaction",
}
EXPECTED_SCORES = "52c0f1edb1ef74b07aa79bf0d7dd16e8ab559692bb44d60b35fc14ab2e17ff40"
EXPECTED_EXPORT = "c5b67506da03dc2f5c014d0a3a99c5bdd4df12a31fd64bf172b07dd2d7dc5b26"
EXPECTED_SEALED = "2f6f533c0c6863943670974012a9fbfc326d542d68aeb118ee80af187727d910"
EXPECTED_SEALED_SPEC = "f4d68023e0abbf8b340398fad669e8d34d0262fd982f889e2de6a47694bd7042"
EXPECTED_ATTEMPT = "0b697c23db1c7090582d21dec7e92f39cac8d86597a7ce1a7a1feca3df2036e2"
EXPECTED_REPRODUCTION = "35218d0b08b8f3ad3297f72e66e7db6cee4ab5d7ddefb7c13f5a5cc3b2f0ef0c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    scores = json.loads(SCORES.read_text())
    sealed = json.loads(SEALED.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    reproduction = json.loads(REPRODUCTION.read_text())
    for path, expected, label in (
        (SCORES, EXPECTED_SCORES, "scores"),
        (EXPORT, EXPECTED_EXPORT, "export"),
        (SEALED, EXPECTED_SEALED, "sealed spec"),
        (ATTEMPT, EXPECTED_ATTEMPT, "runner attempt"),
        (REPRODUCTION, EXPECTED_REPRODUCTION, "raw-export reproduction"),
    ):
        require(sha256_file(path) == expected, f"E486 {label} identity changed")
    require(
        attempt.get("id") == "e486-activation-counterfactual-cpu-cross-v1"
        and attempt.get("status") == "finished"
        and attempt.get("exit_code") == 0
        and attempt.get("teardown_verified") is True
        and attempt.get("peak_observed_vram_bytes") == 0
        and attempt.get("unit_properties", {}).get("Result") == "success"
        and attempt.get("unit_properties", {}).get("ExecMainStatus") in (0, "0"),
        "E486 runner attempt was not a clean CPU completion",
    )
    for before_key, after_key in (("initial_memory_events", "final_memory_events"),):
        before, after = attempt.get(before_key, {}), attempt.get(after_key, {})
        require(
            all(before.get(key) == after.get(key) for key in ("oom", "oom_kill", "oom_group_kill")),
            "E486 worker OOM counters changed",
        )
    parent_before = attempt.get("initial_parent_memory_events", {})
    parent_after = attempt.get("final_parent_memory_events", {})
    require(
        all(
            parent_before.get(key) == parent_after.get(key)
            for key in ("oom", "oom_kill", "oom_group_kill")
        ),
        "E486 parent OOM counters changed during grade",
    )
    artifacts = attempt.get("artifacts", [])
    require(
        [item.get("path") for item in artifacts]
        == [
            "research/campaign_2026-10-05/activation_counterfactual/eda/cross_v1/scores.json",
            "research/campaign_2026-10-05/activation_counterfactual/eda/cross_v1/selected_exports.npz",
        ]
        and [item.get("sha256") for item in artifacts] == [EXPECTED_SCORES, EXPECTED_EXPORT],
        "E486 runner artifact set/hash differs",
    )
    require(
        scores.get("cell_count") == 36
        and len(scores.get("cells", {})) == 36
        and scores.get("contrast_count") == 54
        and len(scores.get("contrasts_by_seed_momentum", {})) == 9
        and scores.get("new_fits") == 0
        and scores.get("new_checkpoints") == 0,
        "E486 result matrix/counts are incomplete",
    )
    proof = scores.get("proof", {})
    require(
        proof.get("test_rows_loaded") is True
        and proof.get("test_rows_scored") is True
        and proof.get("e486_sealed_spec_sha256") == EXPECTED_SEALED_SPEC
        and attempt.get("spec_sha256") == EXPECTED_SEALED_SPEC,
        "E486 proof does not bind the sealed cross run",
    )
    require(
        reproduction.get("study") == "E486"
        and reproduction.get("status") == "all_checked_values_match"
        and reproduction.get("cells") == 36
        and reproduction.get("global_contrasts") == 54
        and reproduction.get("new_training_fits") == 0
        and reproduction.get("result_sha256") == EXPECTED_SCORES
        and reproduction.get("export_sha256") == EXPECTED_EXPORT,
        "E486 independent arithmetic reproduction does not bind result",
    )
    return scores, sealed, attempt, reproduction


def cell_id(seed: int, train: str, infer: str, beta: float) -> str:
    return f"seed{seed}__train_{train}__eval_{infer}__beta{beta}"


def build_report(
    scores: dict[str, Any], attempt: dict[str, Any], reproduction: dict[str, Any]
) -> str:
    lines = [
        "# E486 crossed training and inference activation shapes",
        "",
        "This CPU-only diagnostic re-evaluates all 18 completed E483 checkpoints under both their trained and swapped activation equations. It contains 36 scored cells and all 54 registered global paired contrasts, with each seed and Muon momentum kept separate. It adds no fitted model or checkpoint. The independent raw-export arithmetic check matches all 36 cells and 54 contrasts to maximum absolute error **1.116e−14**; it does not rerun inference.",
        "",
        "The 16 historically exposed components are reused. A diagonal D−A cell pair repeats the earlier matched-shape E483 comparison; it is a consistency check, not new evidence. Swapping the activation changes the causal hidden trajectory, output amplitude, and clipping, so the interaction is a training-shape × inference-shape diagnostic. It does not isolate pure optimization or prove a representation mechanism.",
        "",
        "## Run receipt",
        "",
        f"The frozen CPU grade finished in {attempt['elapsed_seconds']:.2f}s with peak worker RAM {attempt['peak_ram_bytes']:,} bytes, zero GPU memory, no worker or parent OOM-counter increase, and verified teardown. It scored existing checkpoints only; no fit was repeated.",
        "",
        f"Root's raw-export reproducer checked {reproduction['cells']} cells and {reproduction['global_contrasts']} global contrasts (maximum absolute error {reproduction['maximum_absolute_error']:.3e}). The result and selected-export files are byte-bound below.",
        "",
        "## All 36 cells",
        "",
        "A = train baseline / infer baseline; B = train baseline / infer sine; C = train sine / infer baseline; D = train sine / infer sine. Each row shows the global WP plus both target slices.",
        "",
        "| Seed | Muon β₁ | Cell | Global WP | Target 0 | Target 1 | Selected rows |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    cell_labels = {
        ("baseline", "baseline"): "A",
        ("baseline", "sine"): "B",
        ("sine", "baseline"): "C",
        ("sine", "sine"): "D",
    }
    for seed in SEEDS:
        for beta in BETAS:
            for train, infer, _label in CELL_ORDER:
                row = scores["cells"][cell_id(seed, train, infer, beta)]
                lines.append(
                    f"| {seed} | {beta:.2f} | {cell_labels[(train, infer)]} | {row['global_wp']:.6f} | {row['target0_wp']:.6f} | {row['target1_wp']:.6f} | {row['selected_rows']} |"
                )
    lines.extend(
        [
            "",
            "![All 36 global score cells](eda/cross_v1/all36_global_cells.png)",
            "",
            "![All 36 target 0 score cells](eda/cross_v1/all36_target0_cells.png)",
            "",
            "![All 36 target 1 score cells](eda/cross_v1/all36_target1_cells.png)",
            "",
            "## All 54 paired global contrasts",
            "",
            "Intervals are paired component-bootstrap 95% CIs over 16 components, 2,000 common draws per seed/momentum pair (bootstrap seed 271828). Contrasts are dependent by construction; the training-by-inference term reuses the same four crossed cells.",
            "",
            "| Seed | Muon β₁ | Contrast | Δ Global WP | Paired 95% CI |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for seed in SEEDS:
        for beta in BETAS:
            group = scores["contrasts_by_seed_momentum"][f"seed{seed}__beta{beta}"]
            for name in CONTRASTS:
                row = group[name]
                lo, hi = row["ci95"]
                lines.append(
                    f"| {seed} | {beta:.2f} | {CONTRAST_LABELS[name]} | {row['mean_diff']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |"
                )
    lines.extend(
        [
            "",
            "![All 54 global paired contrasts](eda/cross_v1/all54_global_contrasts.png)",
            "",
            "## Interpretation limits",
            "",
            "The nine D−A comparisons reproduce E483's diagonal matched-shape comparison from the same checkpoints and exposed rows. B−A and C−D measure the immediate consequence of changing inference activation on fixed parameters; C−A and D−B compare trained shapes under each inference activation. The difference-in-differences asks whether that training-shape gap changes when the inference activation changes.",
            "",
            "Because swapping activation changes the recurrent dynamics and prediction/clipping distribution, a nonzero interaction is not a clean estimate of optimization alone. It shows a dependency between the trained shape and inference transform for these frozen checkpoints. The contrasts are not independent replications, the three seeds are not pooled, and this exposed-panel diagnostic does not support public-score claims or new candidate selection.",
            "",
            "## Provenance",
            "",
            f"- Score JSON SHA256: {sha256_file(SCORES)}",
            f"- Selected raw-export NPZ SHA256: {sha256_file(EXPORT)}",
            f"- Sealed specification SHA256: {sha256_file(SEALED)}",
            f"- CPU grade attempt SHA256: {sha256_file(ATTEMPT)}",
            f"- Independent arithmetic reproduction SHA256: {sha256_file(REPRODUCTION)}",
            "",
            "This renderer checks and displays saved JSON and receipt metadata. It hashes but does not load the selected-export NPZ, reload test rows, run the scorer, or recompute any score or bootstrap statistic.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_cell_heatmaps(scores: dict[str, Any]) -> None:
    labels = np.array([["A", "B"], ["C", "D"]])
    for field, filename, title in (
        ("global_wp", "all36_global_cells.png", "Global WP"),
        ("target0_wp", "all36_target0_cells.png", "Target 0 WP"),
        ("target1_wp", "all36_target1_cells.png", "Target 1 WP"),
    ):
        fig, axes = plt.subplots(len(SEEDS), len(BETAS), figsize=(12, 11), squeeze=False)
        values_all = [row[field] for row in scores["cells"].values()]
        vmin, vmax = min(values_all), max(values_all)
        image = None
        for i, seed in enumerate(SEEDS):
            for j, beta in enumerate(BETAS):
                matrix = np.empty((2, 2), dtype=float)
                for row_idx, train in enumerate(SHAPES):
                    for col_idx, infer in enumerate(SHAPES):
                        matrix[row_idx, col_idx] = scores["cells"][
                            cell_id(seed, train, infer, beta)
                        ][field]
                axis = axes[i, j]
                image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
                for row_idx in range(2):
                    for col_idx in range(2):
                        axis.text(
                            col_idx,
                            row_idx,
                            f"{labels[row_idx, col_idx]}\n{matrix[row_idx, col_idx]:.4f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=9,
                        )
                axis.set_xticks((0, 1), ("eval base", "eval sine"))
                axis.set_yticks(
                    (0, 1), ("train base", "train sine") if j == 0 else ("", "")
                )
                axis.set_title(f"seed {seed}, β₁={beta:.2f}")
        assert image is not None
        fig.suptitle(f"E486 crossed activation cells: {title}")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, label=title)
        fig.savefig(EDA / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)


def render_contrasts(scores: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=True)
    colors = ("#3569a8", "#d27635", "#32865a")
    x = np.arange(len(BETAS))
    for axis, name in zip(axes.flat, CONTRASTS, strict=True):
        for seed_idx, seed in enumerate(SEEDS):
            rows = [
                scores["contrasts_by_seed_momentum"][f"seed{seed}__beta{beta}"][name]
                for beta in BETAS
            ]
            point = np.array([row["mean_diff"] for row in rows])
            low = np.array([row["ci95"][0] for row in rows])
            high = np.array([row["ci95"][1] for row in rows])
            axis.errorbar(
                x + (seed_idx - 1) * 0.06,
                point,
                yerr=np.vstack((point - low, high - point)),
                fmt="o-",
                capsize=3,
                color=colors[seed_idx],
                label=str(seed),
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(CONTRAST_LABELS[name])
        axis.set_xticks(x, [f"{beta:.2f}" for beta in BETAS])
        axis.set_xlabel("Muon β₁")
        axis.set_ylabel("Δ Global WP")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(title="Seed", frameon=False)
    fig.suptitle("E486: all 54 paired contrast estimates and component-bootstrap CIs")
    fig.tight_layout()
    fig.savefig(EDA / "all54_global_contrasts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csvs(scores: dict[str, Any]) -> None:
    with (EDA / "all36_cells.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed",
                "muon_beta1",
                "train_shape",
                "inference_shape",
                "cell",
                "global_wp",
                "target0_wp",
                "target1_wp",
                "selected_rows",
                "checkpoint_sha256",
            )
        )
        for seed in SEEDS:
            for beta in BETAS:
                for train, infer, label in CELL_ORDER:
                    row = scores["cells"][cell_id(seed, train, infer, beta)]
                    writer.writerow(
                        (
                            seed,
                            beta,
                            train,
                            infer,
                            label[0],
                            row["global_wp"],
                            row["target0_wp"],
                            row["target1_wp"],
                            row["selected_rows"],
                            row["checkpoint_sha256"],
                        )
                    )
    with (EDA / "all54_global_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed",
                "muon_beta1",
                "contrast",
                "mean_diff",
                "ci95_low",
                "ci95_high",
                "bootstrap_sd",
                "p_le_0",
                "bootstrap_count",
                "bootstrap_seed",
            )
        )
        for seed in SEEDS:
            for beta in BETAS:
                group = scores["contrasts_by_seed_momentum"][f"seed{seed}__beta{beta}"]
                for name in CONTRASTS:
                    row = group[name]
                    writer.writerow(
                        (
                            seed,
                            beta,
                            CONTRAST_LABELS[name],
                            row["mean_diff"],
                            *row["ci95"],
                            row["bootstrap_sd"],
                            row["p_le_0"],
                            row["n_boot"],
                            row["bootstrap_seed"],
                        )
                    )


def main() -> None:
    scores, _sealed, attempt, reproduction = read_inputs()
    render_cell_heatmaps(scores)
    render_contrasts(scores)
    write_csvs(scores)
    REPORT.write_text(build_report(scores, attempt, reproduction))
    print(f"wrote {REPORT}")
    print(f"report_sha256={sha256_file(REPORT)}")
    print(f"score_sha256={sha256_file(SCORES)}")


if __name__ == "__main__":
    main()
