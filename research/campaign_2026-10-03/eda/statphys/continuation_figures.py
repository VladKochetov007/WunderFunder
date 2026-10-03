from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parent


def render_figures() -> list[str]:
    """Plot measured proxy gains and the increasingly strict hazard controls."""
    result = json.loads((ROOT / "continuation_results.json").read_text())
    followup = json.loads((ROOT / "continuation_hazard_control.json").read_text())
    names = result["models"]
    scores = [result["scores"][name] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["Current", "+ instant / clock", "+ event memory", "+ row memory", "+ stale memory"]
    axes[0].bar(np.arange(5), scores, color=["#78889a", "#78889a", "#287c6f", "#4688aa", "#bd9468"])
    axes[0].set_xticks(np.arange(5), labels, rotation=20, ha="right")
    axes[0].set_ylim(0.49, 0.60)
    axes[0].set_ylabel("Sampled global weighted Pearson")
    axes[0].set_title("Recent feature history improves this ridge proxy")
    for index, value in enumerate(scores):
        axes[0].text(index, value + 0.0015, f"{value:.4f}", ha="center")
    controls = ["current_instant_clock", "delayed_event_memory", "row_memory"]
    for index, name in enumerate(controls):
        delta = result["deltas"][name]
        lower, upper = delta["sequence_bootstrap_95"]
        axes[1].plot([lower, upper], [index, index], color="#287c6f", lw=3)
        axes[1].plot(delta["event_minus_control"], index, "o", color="#1c493f")
    axes[1].set_yticks(range(3), ["Instant / clock", "Stale memory", "Row memory"])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Event-memory WP minus matched comparison")
    axes[1].set_title("Event-clock advantage over row clock is unresolved")
    figure.suptitle("48 train / 32 valid sequences; 48,270 scored rows; no baseline claim")
    figure.tight_layout()
    first = "f24_causal_memory_controls.png"
    figure.savefig(ROOT / first, dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    hazard = result["hazard"]
    values = [
        hazard["real_age"]["young_hazard"],
        hazard["real_age"]["old_hazard"],
        hazard["delayed_age"]["young_hazard"],
        hazard["delayed_age"]["old_hazard"],
    ]
    axes[0].bar(range(4), values, color=["#287c6f", "#287c6f", "#bd9468", "#bd9468"])
    axes[0].set_xticks(range(4), ["Age 1–3", "Age 8–31", "Stale 1–3", "Stale 8–31"])
    axes[0].set_ylabel("Next-row price-state event probability")
    axes[0].set_title(
        "Exact price state + recent activity matching\nOriginal arms have different support"
    )
    for index, (_name, entry) in enumerate(followup["results"].items()):
        lower, upper = entry["contrast_bootstrap_95"]
        axes[1].plot([lower, upper], [index, index], lw=3, color="#287c6f")
        axes[1].plot(entry["real_minus_stale"], index, "o", color="#1c493f")
        axes[1].annotate(
            f"n={entry['common_strata']} strata",
            (upper, index),
            xytext=(5, 7),
            textcoords="offset points",
        )
    axes[1].set_yticks(range(2), ["Common support", "+ a2 / a3 bins"])
    axes[1].set_ylim(-0.35, 1.35)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Real minus stale old–young hazard difference")
    axes[1].set_title("Same strata in both age controls\n95% sequence bootstrap, exploratory")
    figure.tight_layout()
    second = "f25_conditional_age_controls.png"
    figure.savefig(ROOT / second, dpi=130)
    plt.close(figure)
    return [first, second]


def main() -> None:
    """Render final artifacts and save their reproducibility metadata."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    figures = render_figures()
    artifacts = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file()
        and (
            path.name.startswith("continuation_")
            or path.name.startswith("f2")
            or path.name
            in [
                "HYPOTHESES.json",
                "HYPOTHESES_FOLLOWUP.json",
                "RECOVERY_SUMMARY.txt",
                "causal_features.py",
                "prepare_continuation.py",
                "eda_statphys.py",
            ]
        )
    )
    manifest = {
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifacts
        },
        "datasets": {
            name: {
                "bytes": Path(f"datasets/{name}.parquet").stat().st_size,
                "whole_file_hashed": False,
                "rows_per_sequence": 20000,
            }
            for name in ["train", "valid"]
        },
        "sample_provenance": "continuation_samples.json",
        "frozen_procedures": ["HYPOTHESES.json", "HYPOTHESES_FOLLOWUP.json"],
        "figures": figures,
    }
    (ROOT / "ARTIFACT_PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "figures_ready_pending_visual_inspection_and_report"
    state["artifacts"] = sorted(set(state["artifacts"] + list(manifest["artifacts"])))
    state["commands"] += [
        "continuation_hazard_control.py: completed H3",
        "continuation_figures.py: completed plots/provenance",
    ]
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    logging.info("Rendered %s", figures)


if __name__ == "__main__":
    main()
