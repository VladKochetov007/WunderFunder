"""Render the independently graded E481 recurrent candidate-shape results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"
RESULT = HERE / "eda/eighteen_v2/scores.json"
EXPORT = HERE / "eda/eighteen_v2/selected_exports.npz"
GRADE_SPEC = HERE / "e481_cpu_grade_v2.sealed.json"
GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e481-candidate-curvature-cpu-test-grade-v2.json"
)
OUTPUT = HERE / "eda/eighteen_v2"
SEEDS = (271828, 271829, 271830)
OPTIMIZERS = ("adamw", "muon_adamw")
SHAPES = ("baseline", "sine", "cubic")


@dataclass(frozen=True)
class Arm:
    key: str
    seed: int
    optimizer: str
    shape: str
    wp: float
    t0: float
    t1: float


@dataclass(frozen=True)
class Contrast:
    seed: int
    name: str
    family: str
    candidate: str
    reference: str
    delta: float
    low: float
    high: float
    n_components: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> tuple[dict[str, Any], list[Arm], list[Contrast], dict[str, Any]]:
    result = json.loads(RESULT.read_text())
    grade = json.loads(GRADE_ATTEMPT.read_text())
    spec = json.loads(GRADE_SPEC.read_text())
    if result.get("arm_count") != 18 or len(result.get("arm_scores", {})) != 18:
        raise ValueError("E481 result must contain all 18 fit arms")
    if (
        result.get("contrast_count") != 30
        or sum(map(len, result.get("contrasts_by_seed", {}).values())) != 30
    ):
        raise ValueError("E481 result must contain all 30 registered global contrasts")
    if tuple(result.get("fit_seeds", [])) != SEEDS:
        raise ValueError("E481 fit seed list differs from registered seeds")
    if result.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("E481 metric identity differs")
    if sha256(RESULT) != grade_artifact_hash(grade, RESULT) or sha256(
        EXPORT
    ) != grade_artifact_hash(grade, EXPORT):
        raise ValueError("grade attempt artifact hashes do not match the score/export files")
    if sha256(EXPORT) != result["selected_exports_sha256"]:
        raise ValueError("E481 result export hash does not match selected exports")
    if (
        grade.get("id") != "e481-candidate-curvature-cpu-test-grade-v2"
        or grade.get("status") != "finished"
    ):
        raise ValueError("E481 v2 grade attempt identity/status differs")
    if (
        grade.get("exit_code") != 0
        or grade.get("signals") != []
        or grade.get("teardown_verified") is not True
    ):
        raise ValueError("E481 v2 grade attempt did not exit cleanly with teardown")
    unit = grade.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("E481 v2 grade systemd unit reports failure")
    for before_key, after_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = grade.get(before_key, {}), grade.get(after_key, {})
        for counter in ("oom", "oom_kill", "oom_group_kill"):
            if before.get(counter) != after.get(counter):
                raise ValueError(f"E481 v2 grade OOM counter changed: {before_key}/{counter}")
    declared = set(spec.get("artifacts", []))
    expected_artifacts = {
        "research/campaign_2026-10-05/recurrent_activation_curvature/eda/eighteen_v2/scores.json",
        "research/campaign_2026-10-05/recurrent_activation_curvature/eda/eighteen_v2/selected_exports.npz",
    }
    if declared != expected_artifacts or grade_artifact_hash(grade, RESULT) != sha256(RESULT):
        raise ValueError("sealed E481 score-grade artifact binding differs")
    if grade_artifact_hash(grade, EXPORT) != sha256(EXPORT):
        raise ValueError("sealed E481 export-grade artifact binding differs")

    arms: list[Arm] = []
    for key, item in result["arm_scores"].items():
        arms.append(
            Arm(
                key,
                int(item["seed"]),
                item["optimizer"],
                item["shape"],
                float(item["wp"]),
                float(item["t0"]),
                float(item["t1"]),
            )
        )
    if {(a.seed, a.optimizer, a.shape) for a in arms} != {
        (seed, optimizer, shape) for seed in SEEDS for optimizer in OPTIMIZERS for shape in SHAPES
    }:
        raise ValueError("E481 arm matrix is incomplete or has duplicate cells")
    arms.sort(key=lambda a: (a.seed, OPTIMIZERS.index(a.optimizer), SHAPES.index(a.shape)))

    contrasts: list[Contrast] = []
    for seed in SEEDS:
        entries = result["contrasts_by_seed"][str(seed)]
        if len(entries) != 10:
            raise ValueError(f"E481 seed {seed} must have ten contrasts")
        for name, item in entries.items():
            contrasts.append(
                Contrast(
                    seed,
                    name,
                    item["family"],
                    item["candidate"],
                    item["reference"],
                    float(item["mean_diff"]),
                    float(item["ci95"][0]),
                    float(item["ci95"][1]),
                    int(item["n_components"]),
                )
            )
    return result, arms, contrasts, grade


def grade_artifact_hash(grade: dict[str, Any], path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    for artifact in grade.get("artifacts", []):
        if artifact.get("path") == rel:
            return str(artifact.get("sha256"))
    raise ValueError(f"grade attempt does not declare artifact {rel}")


def plot_arms(arms: list[Arm], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.2), sharey=True)
    colors = {"baseline": "#4c78a8", "sine": "#f58518", "cubic": "#54a24b"}
    markers = {"adamw": "o", "muon_adamw": "s"}
    labels = {"baseline": "baseline", "sine": "sine α=0.1", "cubic": "cubic α=0.1"}
    offset = {"adamw": -0.12, "muon_adamw": 0.12}
    for ax, seed in zip(axes, SEEDS, strict=True):
        for index, shape in enumerate(SHAPES):
            for optimizer in OPTIMIZERS:
                arm = next(
                    a
                    for a in arms
                    if a.seed == seed and a.shape == shape and a.optimizer == optimizer
                )
                ax.scatter(
                    index + offset[optimizer],
                    arm.wp,
                    color=colors[shape],
                    marker=markers[optimizer],
                    s=76,
                    edgecolor="#222",
                    linewidth=0.6,
                    zorder=3,
                )
        ax.set_title(f"Fit seed {seed}")
        ax.set_xticks(range(3), [labels[s] for s in SHAPES], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Global weighted Pearson")
    axes[0].text(
        0.02,
        0.98,
        "circles: AdamW; squares: Muon+AdamW",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    fig.suptitle("E481: all 18 recurrent candidate-shape fits", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_contrasts(contrasts: list[Contrast], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 13.5), sharex=True)
    colors = {
        "shape_within_optimizer": "#4c78a8",
        "optimizer_within_shape": "#f58518",
        "shape_by_optimizer_interaction": "#54a24b",
    }
    for ax, seed in zip(axes, SEEDS, strict=True):
        entries = [c for c in contrasts if c.seed == seed]
        y = np.arange(len(entries))
        for yi, item in zip(y, entries, strict=True):
            ax.errorbar(
                item.delta,
                yi,
                xerr=[[item.delta - item.low], [item.high - item.delta]],
                fmt="o",
                color=colors[item.family],
                capsize=2.5,
                markersize=4.5,
                linewidth=1,
            )
        ax.axvline(0, color="#222", linewidth=0.9)
        ax.set_yticks(
            y,
            [item.name.replace("__", ":").replace("_minus_", " − ") for item in entries],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_title(f"Fit seed {seed}", loc="left")
        ax.grid(axis="x", alpha=0.22)
    axes[-1].set_xlabel(
        "Paired global weighted-Pearson difference (95% component-bootstrap interval)"
    )
    fig.suptitle("E481: all 30 registered contrasts, shown separately by fit seed", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_report(
    result: dict[str, Any], arms: list[Arm], contrasts: list[Contrast], grade: dict[str, Any]
) -> None:
    score_plot = OUTPUT / "e481_scores.png"
    contrast_plot = OUTPUT / "e481_contrasts.png"
    plot_arms(arms, score_plot)
    plot_contrasts(contrasts, contrast_plot)
    result_hash, export_hash, attempt_hash = sha256(RESULT), sha256(EXPORT), sha256(GRADE_ATTEMPT)
    lines = [
        "# E481: recurrent candidate curvature and optimizer interaction",
        "",
        "E481 compares the baseline candidate `u`, bounded sine `u + 0.1 sin(πu)`, and matched-center cubic `u + 0.1πu(1−u²)` in a two-layer reset-after GRU. The experiment contains three fixed fit seeds, two optimizers, and all three shapes (18 completed fits). It reuses the same 16 previously exposed components; this is descriptive proxy evidence, not new holdout evidence or a public forecast.",
        "",
        "![All 18 E481 scores](eda/eighteen_v2/e481_scores.png)",
        "",
        "![All 30 paired contrasts](eda/eighteen_v2/e481_contrasts.png)",
        "",
        "## Arm scores",
        "",
        "| Fit seed | Optimizer | Candidate | Global WP | t0 WP | t1 WP |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm.seed} | {arm.optimizer} | {arm.shape} | {arm.wp:.6f} | {arm.t0:.6f} | {arm.t1:.6f} |"
        )
    lines += [
        "",
        "## All 30 registered global contrasts",
        "",
        "Each difference is candidate minus reference. The 95% intervals use 2,000 shared component-bootstrap draws within the same fit seed; each interval conditions on that seed's fitted checkpoints and the same 16 components.",
        "",
        "| Fit seed | Family | Candidate − reference | Difference | 95% interval |",
        "|---:|---|---|---:|---:|",
    ]
    for item in contrasts:
        lines.append(
            f"| {item.seed} | {item.family} | `{item.candidate}` − `{item.reference}` | {item.delta:+.6f} | [{item.low:+.6f}, {item.high:+.6f}] |"
        )

    lines += [
        "",
        "## What changed across shapes",
        "",
        "The sine and cubic candidates move together relative to baseline within each optimizer/seed more often than they differ from each other. The key departure is seed 271828 with AdamW: sine and cubic are respectively −0.031079 (95% CI [−0.044033, −0.015963]) and −0.034952 ( [−0.049418, −0.017843]) versus baseline. Under Muon+AdamW for the same seed, both are modestly positive: +0.002691 ([+0.000489, +0.005917]) for sine and +0.003204 ([+0.000865, +0.006721]) for cubic. Thus the fitted response to candidate shape depends strongly on optimizer in that seed.",
        "",
        "For seeds 271829 and 271830, both candidate shapes have positive point estimates over baseline under both optimizers. Among the four sine-versus-baseline comparisons for seeds 271829 and 271830, three intervals are above zero; the Muon+AdamW interval at seed 271830 crosses zero. Cubic-versus-baseline has the same pattern. Cubic-minus-sine differences are much smaller: under AdamW they are negative at seeds 271828 and 271829 (the latter interval excludes zero), while seed 271830 is uncertain; all three Muon+AdamW cubic-minus-sine intervals cross zero. This suggests optimizer-conditioned shape effects in the proxy, but does not establish a stable winner or a causal mechanistic explanation.",
        "",
        "The registered cubic-minus-sine-by-optimizer interaction is positive with intervals above zero for seeds 271828 and 271829, and crosses zero for 271830. This is consistent with the cubic being relatively less unfavorable under Muon than under AdamW for the first two seeds. The result is seed-sensitive and is based on three predeclared fits per condition, not an estimate over a population of training seeds.",
        "",
        "## Grade and limits",
        "",
        f"- Independent CPU grade attempt: `{grade['id']}` finished with exit code 0, no signals, and verified teardown; elapsed {grade['elapsed_seconds']:.1f} s, peak RAM {grade['peak_ram_bytes']:,} bytes, observed VRAM {grade['peak_observed_vram_bytes']:,} bytes.",
        "- Worker OOM counters did not increase. Parent counters were nonzero before this attempt and remained unchanged.",
        "- The v1 grader stopped at fit-only proof validation due to an unavailable helper API before test scoring; the successful score is from the independently reviewed v2 grader. The underlying fit attempt completed normally.",
        f"- Score JSON SHA-256: `{result_hash}`; paired selected-export SHA-256: `{export_hash}`; grade attempt SHA-256: `{attempt_hash}`.",
        "- Bootstrap unit is one numeric-sorted component, with shared component draws across arms and contrasts within each seed. No seeds are pooled; component intervals do not include training-seed or model-selection uncertainty.",
        "- The candidate-map interval bound is an amplitude invariant for the exact reset-after recurrence, not a gradient-stability guarantee. These proxy scores do not establish public leaderboard performance.",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result, arms, contrasts, grade = load_and_validate()
    render_report(result, arms, contrasts, grade)
    print(
        json.dumps(
            {
                "arms": len(arms),
                "contrasts": len(contrasts),
                "score_sha256": sha256(RESULT),
                "export_sha256": sha256(EXPORT),
                "score_plot": str(OUTPUT / "e481_scores.png"),
                "contrast_plot": str(OUTPUT / "e481_contrasts.png"),
                "report": str(HERE / "REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
