"""Render the frozen E479 adaptive recurrent activation summary."""

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
STUDY = ROOT / "research/campaign_2026-10-05/recurrent_activation_adaptive"
RESULT = STUDY / "eda/sixteen_v2/RESULT.json"
REPRO = ROOT / "research/campaign_2026-10-05/checks/adaptive_activation_export_reproduction.json"
OUT = STUDY / "eda/sixteen_v2"
TRAJECTORY = STUDY / "eda/coefficient_trajectories.png"


@dataclass(frozen=True)
class Arm:
    key: str
    seed: int
    optimizer: str
    condition: str
    global_wp: float
    t0: float
    t1: float


@dataclass(frozen=True)
class Contrast:
    key: str
    difference: float
    low: float
    high: float
    seed: int
    components: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[dict[str, Any], list[Arm], list[Contrast], dict[str, Any]]:
    raw = json.loads(RESULT.read_text())
    repro = json.loads(REPRO.read_text())
    if raw.get("kind") != "e479_learned_candidate_score_v1" and "arms" not in raw:
        raise ValueError("unexpected E479 result schema")
    if raw.get("status") != "scored_exposed_components_descriptive_only":
        raise ValueError("E479 result is not a completed descriptive score")
    if len(raw["arms"]) != 16 or len(raw["contrasts"]) != 28:
        raise ValueError("expected all 16 arms and 28 registered contrasts")
    if sha256(RESULT) != repro["result_sha256"]:
        raise ValueError("reproduction receipt is bound to a different result file")
    export = Path(raw["export"]["path"])
    if not export.is_absolute():
        export = ROOT / export
    if sha256(export) != raw["export"]["sha256"] or sha256(export) != repro["export_sha256"]:
        raise ValueError("paired export hash mismatch")
    if (
        repro.get("status") != "all_checked_values_match"
        or repro.get("arms") != 16
        or repro.get("global_contrasts") != 28
    ):
        raise ValueError("independent reproduction receipt is incomplete")

    arms: list[Arm] = []
    for key, item in raw["arms"].items():
        arms.append(
            Arm(
                key,
                int(item["seed"]),
                str(item["optimizer"]),
                str(item["condition"]),
                float(item["global_wp"]),
                float(item["target_wp"]["t0"]),
                float(item["target_wp"]["t1"]),
            )
        )
    arms.sort(
        key=lambda a: (
            a.seed,
            a.optimizer,
            ["fixed0", "fixed01", "learned0", "learned01"].index(a.condition),
        )
    )

    contrasts: list[Contrast] = []
    for key, item in raw["contrasts"].items():
        contrasts.append(
            Contrast(
                key,
                float(item["difference"]),
                float(item["ci95"][0]),
                float(item["ci95"][1]),
                int(item["seed"]),
                int(item["n_components"]),
            )
        )
    contrasts.sort(key=lambda c: c.key)
    return raw, arms, contrasts, repro


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def plot_scores(arms: list[Arm], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    conditions = ["fixed0", "fixed01", "learned0", "learned01"]
    labels = ["fixed α=0", "fixed α=.1", "learned from 0", "learned from .1"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    offsets = {"adamw": -0.12, "muon_adamw": 0.12}
    for ax, seed in zip(axes, sorted({a.seed for a in arms}), strict=True):
        for opt in ("adamw", "muon_adamw"):
            group = [a for a in arms if a.seed == seed and a.optimizer == opt]
            for i, (condition, _label, color) in enumerate(
                zip(conditions, labels, colors, strict=True)
            ):
                arm = next(a for a in group if a.condition == condition)
                ax.scatter(
                    i + offsets[opt],
                    arm.global_wp,
                    s=74,
                    color=color,
                    marker="o" if opt == "adamw" else "s",
                    edgecolor="#222222",
                    linewidth=0.55,
                    zorder=3,
                )
        ax.set_title(f"Fit seed {seed}")
        ax.set_xticks(range(4), labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Global weighted Pearson")
    axes[0].text(
        0.02,
        0.98,
        "circles: AdamW   squares: Muon+AdamW",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    fig.suptitle("E479: all 16 learned recurrent candidate-shape fits", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_contrasts(contrasts: list[Contrast], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 12.0))
    y = np.arange(len(contrasts))
    delta = np.array([c.difference for c in contrasts])
    lo = np.array([c.low for c in contrasts])
    hi = np.array([c.high for c in contrasts])
    colors = ["#e45756" if c.seed == 271828 else "#4c78a8" for c in contrasts]
    for yi, est, low, high, color in zip(y, delta, lo, hi, colors, strict=True):
        ax.errorbar(
            est,
            yi,
            xerr=[[est - low], [high - est]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.2,
            markersize=4.5,
            linewidth=1.1,
        )
    ax.axvline(0, color="#333333", linewidth=1, alpha=0.8)
    ax.set_yticks(
        y,
        [c.key.replace("seed271828:", "s828 ").replace("seed271829:", "s829 ") for c in contrasts],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Paired global weighted-Pearson difference (95% component-bootstrap interval)")
    ax.set_title("E479: all 28 registered contrasts; seed colors kept separate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(
    raw: dict[str, Any], arms: list[Arm], contrasts: list[Contrast], repro: dict[str, Any]
) -> None:
    score_plot = OUT / "e479_scores.png"
    contrast_plot = OUT / "e479_contrasts.png"
    plot_scores(arms, score_plot)
    plot_contrasts(contrasts, contrast_plot)

    lines = [
        "# E479: learned bounded recurrent candidate activation",
        "",
        "E479 tests a per-layer learned coefficient in the reset-after GRU candidate map `u + α sin(πu)`, where `u=tanh(v)`. It completed all 16 registered fits (two fit seeds × two optimizers × four fixed/learned initialization conditions). Each score is on the same previously exposed set of 16 components. These results are descriptive; they do not establish a public-score gain or seed-population uncertainty.",
        "",
        "![All 16 arm scores](eda/sixteen_v2/e479_scores.png)",
        "",
        "![All 28 paired contrasts](eda/sixteen_v2/e479_contrasts.png)",
        "",
        "## Arm scores",
        "",
        "| Fit seed | Optimizer | Candidate coefficient condition | Global WP | t0 WP | t1 WP |",
        "|---:|---|---|---:|---:|---:|",
    ]
    names = {
        "fixed0": "fixed α=0",
        "fixed01": "fixed α=0.1",
        "learned0": "learned, initialized α=0",
        "learned01": "learned, initialized α=0.1",
    }
    for a in arms:
        lines.append(
            f"| {a.seed} | {a.optimizer} | {names[a.condition]} | {a.global_wp:.6f} | {a.t0:.6f} | {a.t1:.6f} |"
        )
    lines += [
        "",
        "## All registered paired contrasts",
        "",
        "Differences are left condition minus right condition, using shared component resampling within a fit seed. Positive favors the left condition. Intervals condition on the fitted checkpoints and these 16 components.",
        "",
        "| Contrast | Difference | 95% interval | Components |",
        "|---|---:|---:|---:|",
    ]
    for c in contrasts:
        lines.append(
            f"| `{c.key}` | {c.difference:+.6f} | [{c.low:+.6f}, {c.high:+.6f}] | {c.components} |"
        )

    # Explicit selected effects preserve sign reversal and the requested tiny learned-vs-fixed context.
    lines += [
        "",
        "## Interpretation",
        "",
        "Learned-versus-fixed effects are small in absolute terms and depend on optimizer and seed. For seed 271828 with AdamW, learning from α=0.1 exceeds the fixed α=0.1 control by **+0.000650** global WP (95% component interval [ +0.000061, +0.001260 ]). This is the notable positive learned-shape contrast, but it does not repeat: seed 271829 AdamW is −0.000417, while Muon+AdamW is negative for both seeds at the α=0.1 start (−0.000415 and −0.000160). At the zero start, three of four within-seed/optimizer contrasts are negative; the remaining seed-271829 Muon contrast is near zero (+0.000044).",
        "",
        "The fixed α=0.1 shape itself has mixed effects: it is negative versus fixed zero for seed-271828 AdamW (−0.031079), but positive for the other three seed/optimizer pairs (+0.002712, +0.005782, +0.010675). Optimizer contrasts reverse sign across seeds: Muon+AdamW leads for each shape under seed 271828, while it trails AdamW for seed 271829. The learned coefficient is therefore not a stable optimizer-independent improvement in this small two-seed study.",
        "",
        "The training-only trajectory figure below shows the learned-from-zero coefficients staying close to zero and learned-from-0.1 coefficients remaining near 0.09–0.10, with modest layer/optimizer/seed-dependent movement. That is consistent with learned-vs-fixed performance gaps generally being small; it is not evidence that coefficients are causally irrelevant. The scalar bound applies to the candidate map, not to the full recurrent system.",
        "",
        f"![Training-only learned coefficient trajectories]({TRAJECTORY.relative_to(STUDY)})",
        "",
        "## Reproduction and provenance",
        "",
        f"- Frozen score artifact SHA-256: `{sha256(RESULT)}`.",
        f"- Paired raw prediction export SHA-256: `{sha256(Path(raw['export']['path']))}`.",
        f"- Registered fit source SHA-256: `{raw['provenance']['actor_sha256']}`; seal `{raw['provenance']['seal_name']}` (`{raw['provenance']['seal_sha256']}`).",
        f"- Independent pure-NumPy recomputation: {repro['status']}; checked {repro['arms']} arm scores and {repro['global_contrasts']} contrasts, maximum absolute discrepancy `{repro['maximum_absolute_error']:.3g}`. It uses the existing raw export and an independent raw-sum computation; no refit or grader import.",
        "- Bootstrap resampling unit: numeric-sorted test component; 2,000 shared draws within each seed. No pooling across fit seeds.",
        "- Limitations: only two fit seeds and 16 exposed components; intervals quantify component resampling conditional on these checkpoints, not run-to-run or seed uncertainty. This exploratory proxy result does not forecast public performance.",
        "",
    ]
    (STUDY / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, arms, contrasts, repro = load_inputs()
    write_report(raw, arms, contrasts, repro)
    print(
        json.dumps(
            {
                "arms": len(arms),
                "contrasts": len(contrasts),
                "result_sha256": sha256(RESULT),
                "report": str(STUDY / "REPORT.md"),
                "score_plot": str(OUT / "e479_scores.png"),
                "contrast_plot": str(OUT / "e479_contrasts.png"),
                "reproduction_max_abs_error": repro["maximum_absolute_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
