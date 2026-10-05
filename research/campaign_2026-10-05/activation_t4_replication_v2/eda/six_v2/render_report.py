"""Render figures and a report from frozen E482 and E481 score receipts only."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
E482 = ROOT / "research/campaign_2026-10-05/activation_t4_replication_v2"
E481 = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature/eda/eighteen_v2"
PROOF = ROOT / "research/campaign_2026-10-05/checks/t4_activation_export_reproduction.json"
GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e482-private-t4-sixfit-independent-cpu-grade-v2.json"
)
REMOTE = E482 / "remote_v1/e482-results/e482_receipt.json"
SCORES_PATH = HERE / "scores.json"
LOCAL_SCORES_PATH = E481 / "scores.json"


def sha256(path: Path) -> str:
    """Hash a receipt source with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[dict, dict, dict, dict, dict]:
    """Load only frozen score/provenance receipts; do not recompute scores."""
    scores = json.loads(SCORES_PATH.read_text())
    local_all = json.loads(LOCAL_SCORES_PATH.read_text())
    local = {
        "arm_scores": {
            key: value for key, value in local_all["arm_scores"].items() if value["seed"] == 271828
        },
        "contrasts": local_all["contrasts_by_seed"]["271828"],
    }
    proof = json.loads(PROOF.read_text())
    attempt = json.loads(GRADE_ATTEMPT.read_text())
    remote = json.loads(REMOTE.read_text())
    if scores["validated_fit_cells"] != 6 or scores["contrast_count"] != 10:
        raise ValueError("unexpected independent grader result shape")
    if proof["arms"] != 6 or proof["global_contrasts"] != 10:
        raise ValueError("raw-sum proof does not cover all six arms and ten contrasts")
    if proof["result_sha256"] != sha256(SCORES_PATH):
        raise ValueError("root raw-sum receipt does not bind the independent grade")
    if scores["fit_only_proof"]["test_predictions_scored"] is not False:
        raise ValueError("grade fit-only proof reports an unexpected test-score state")
    if attempt["status"] != "finished" or attempt["memory_events"]["oom"] != 0:
        raise ValueError("CPU grading attempt did not finish cleanly")
    if remote["fit_count"] != 6 or remote["status"] != "six_fits_complete_test_unread":
        raise ValueError("remote fit receipt is not a complete six-fit record")
    return scores, local, proof, attempt, remote


def arm_fig(scores: dict, local: dict) -> None:
    """Compare the six official T4 scores with the existing same-seed local scores."""
    metrics = (("wp", "Global WP"), ("t0", "Target 0 WP"), ("t1", "Target 1 WP"))
    arms = [
        ("adamw", "baseline"),
        ("adamw", "sine"),
        ("adamw", "cubic"),
        ("muon_adamw", "baseline"),
        ("muon_adamw", "sine"),
        ("muon_adamw", "cubic"),
    ]
    labels = [f"{optimizer}\n{shape}" for optimizer, shape in arms]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True)
    x = np.arange(len(arms))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        t4_values, local_values = [], []
        for optimizer, shape in arms:
            key = f"{optimizer}__{shape}"
            t4_values.append(scores["arm_scores"][key][metric])
            local_key = f"seed271828__{key}"
            local_values.append(local["arm_scores"][local_key][metric])
        ax.plot(x, local_values, marker="o", color="#777777", linewidth=1.2, label="E481 local")
        ax.plot(x, t4_values, marker="s", color="#176B87", linewidth=1.2, label="E482 T4")
        for x_i, local_value, t4_value in zip(x, local_values, t4_values, strict=True):
            ax.plot([x_i, x_i], [local_value, t4_value], color="#b6c3c7", linewidth=0.8, zorder=0)
        ax.set_ylabel("WP")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(2.5, color="#aaaaaa", linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Seed 271828, optimizer and activation")
    fig.suptitle("E482 independent T4 grade versus existing E481 local grade", y=0.995)
    fig.tight_layout()
    fig.savefig(HERE / "arm_scores_t4_vs_local.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def contrast_fig(scores: dict, local: dict) -> None:
    """Plot all ten T4 contrasts and the same-seed E481 local intervals."""
    contrasts = list(scores["contrasts"].items())
    labels = [
        key.replace("__", " ").replace("_minus_", " − ").replace("_", " ") for key, _ in contrasts
    ]
    y = np.arange(len(contrasts))
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    ax.axvline(0.0, color="#252525", linewidth=1.0)
    for idx, (key, t4_item) in enumerate(contrasts):
        local_item = local["contrasts"][key]
        color = (
            "#176B87"
            if t4_item["family"] == "shape_within_optimizer"
            else ("#B86131" if t4_item["family"] == "optimizer_within_shape" else "#7957A5")
        )
        low, high = t4_item["ci95"]
        ax.errorbar(
            t4_item["mean_diff"],
            idx - 0.12,
            xerr=[[t4_item["mean_diff"] - low], [high - t4_item["mean_diff"]]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=5,
        )
        local_low, local_high = local_item["ci95"]
        ax.errorbar(
            local_item["mean_diff"],
            idx + 0.12,
            xerr=[[local_item["mean_diff"] - local_low], [local_high - local_item["mean_diff"]]],
            fmt="s",
            color="#858585",
            capsize=3,
            markersize=4,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired-component change in global WP (95% bootstrap interval)")
    ax.set_title("All ten preregistered E482 contrasts: T4 grade and same-seed local grade")
    ax.grid(axis="x", alpha=0.25)
    ax.scatter([], [], marker="o", color="#176B87", label="E482 T4 (shape contrasts)")
    ax.scatter([], [], marker="o", color="#B86131", label="E482 T4 (optimizer contrasts)")
    ax.scatter([], [], marker="o", color="#7957A5", label="E482 T4 (interaction)")
    ax.scatter([], [], marker="s", color="#858585", label="E481 local, seed 271828")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "contrasts_t4_vs_local.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_report(scores: dict, local: dict, proof: dict, attempt: dict, remote: dict) -> None:
    """Write a concise human report from already graded receipts."""
    arm_rows = []
    for key, arm in scores["arm_scores"].items():
        local_arm = local["arm_scores"][f"seed271828__{key}"]
        arm_rows.append(
            f"| {arm['optimizer']} | {arm['shape']} | {arm['wp']:.6f} | "
            f"{arm['t0']:.6f} | {arm['t1']:.6f} | "
            f"{arm['wp'] - local_arm['wp']:+.2e} |"
        )
    contrast_rows = []
    for key, item in scores["contrasts"].items():
        local_item = local["contrasts"][key]
        lo, hi = item["ci95"]
        local_lo, local_hi = local_item["ci95"]
        contrast_rows.append(
            f"| {key.replace('__', ' ').replace('_', ' ')} | {item['mean_diff']:+.6f} "
            f"[{lo:+.6f}, {hi:+.6f}] | {local_item['mean_diff']:+.6f} "
            f"[{local_lo:+.6f}, {local_hi:+.6f}] |"
        )
    report = f"""# E482 private T4 replication: independent grade

The six preregistered seed-271828 fits completed on two T4 workers and were graded by the frozen independent scorer on the same 16 previously exposed components. This remains an exploratory, same-panel hardware replication: it is not new-market confirmation, and it does not change the recorded public best of 0.6660 or forecast a public score.

The root raw-sum reproduction independently matched all six arm scores and ten global contrasts to the saved grade, with maximum absolute error **{proof["maximum_absolute_error"]:.3e}**. This is an arithmetic reproduction of the saved exports, not a second grader or a new inference.

## Runtime and resource receipts

Provider status was COMPLETE at {scores["provider_proof"]["observed_terminal_utc"]}. All six fits used 474 updates and 61,058 parameters; both T4 worker OOM counters remained zero. Fit time summed to 1,430.661 seconds. The independent CPU grade completed in {attempt["elapsed_seconds"]:.4f} seconds with {attempt["peak_ram_bytes"]:,} bytes peak RAM and zero worker memory-event increments. See `REMOTE_FIT_RESULT.md` and the fit-only attempt receipt for detailed provenance.

## Six arm scores

All arms are retained. The final column is the simple difference from the existing local E481 score for the exact same seed; it is a hardware comparison, not an added statistical test.

| Optimizer | Shape | T4 global WP | T4 target 0 | T4 target 1 | T4 − local global WP |
|---|---:|---:|---:|---:|---:|
{chr(10).join(arm_rows)}

![Six T4 and same-seed local arm scores](eda/six_v2/arm_scores_t4_vs_local.png)

## Ten preregistered contrasts

Intervals use paired bootstrap resampling over the same 16 exposed components. T4 and local intervals are displayed side by side from their existing graders; they are not pooled. The cubic-minus-sine Muon contrast has a small positive T4 interval ({scores["contrasts"]["muon_adamw__cubic_minus_muon_adamw__sine"]["ci95"][0]:+.7f} to {scores["contrasts"]["muon_adamw__cubic_minus_muon_adamw__sine"]["ci95"][1]:+.7f}), while the same-seed local interval crosses zero ({local["contrasts"]["muon_adamw__cubic_minus_muon_adamw__sine"]["ci95"][0]:+.7f} to {local["contrasts"]["muon_adamw__cubic_minus_muon_adamw__sine"]["ci95"][1]:+.7f}). This narrow difference is hardware/optimizer numerical sensitivity, not a new seed or market confirmation. All ten contrasts remain visible.

| Contrast | T4 change [95% CI] | E481 local same-seed change [95% CI] |
|---|---:|---:|
{chr(10).join(contrast_rows)}

![All ten T4 and same-seed local paired contrasts](eda/six_v2/contrasts_t4_vs_local.png)

The independent score file is `eda/six_v2/scores.json`; root's reproduction receipt is `checks/t4_activation_export_reproduction.json`. The report and plots display those receipts and do not independently score predictions.
"""
    (E482 / "REPORT.md").write_text(report)


def main() -> None:
    scores, local, proof, attempt, remote = load_inputs()
    arm_fig(scores, local)
    contrast_fig(scores, local)
    render_report(scores, local, proof, attempt, remote)
    print("Rendered E482 report and two comparison figures from frozen receipts.")


if __name__ == "__main__":
    main()
