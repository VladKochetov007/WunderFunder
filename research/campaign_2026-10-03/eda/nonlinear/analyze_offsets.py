from __future__ import annotations

import csv
import hashlib
import json
import logging
import os

import numpy as np
from continuation_common import MODELS, OUT, weighted_pearson

os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def analyze() -> dict[str, object]:
    """Compute frozen verdicts and paired diagnostics, preserving negatives."""
    with np.load(OUT / "evaluation_moments_v1.npz") as data:
        moments = data["moments"]
        groups = data["row_groups"]
        sources = data["sources"].tolist()
    provenance = json.loads((OUT / "provenance_evaluation_v1.json").read_text())
    scores = weighted_pearson(moments.sum(axis=0))
    halves = np.stack([weighted_pearson(moments[fold::2].sum(axis=0)) for fold in range(2)])
    # Only exact-prefix duplicate components are observable in this clean room.
    # They are not the campaign's connected components; intervals are descriptive.
    components: dict[str, list[int]] = {}
    for index, record in enumerate(provenance):
        if not record["excluded_exact_warmup_duplicate"]:
            components.setdefault(record["warmup_inputs_sha256"], []).append(index)
    component_moments = np.stack([moments[indices].sum(axis=0) for indices in components.values()])
    rng = np.random.default_rng(260105)
    boot = []
    for _ in range(400):
        sampled = rng.integers(0, len(component_moments), len(component_moments))
        boot.append(weighted_pearson(component_moments[sampled].sum(axis=0)))
    bootstrap = np.stack(boot)
    records = []
    for source, source_name in enumerate(sources):
        for model, model_name in enumerate(MODELS):
            delta = scores[source, model] - scores[source, 1]
            boot_delta = bootstrap[:, source, model] - bootstrap[:, source, 1]
            interval = np.quantile(boot_delta, [0.025, 0.975], axis=0)
            records.append(
                {
                    "source": source_name,
                    "model": model_name,
                    "wp_t0": float(scores[source, model, 0]),
                    "wp_t1": float(scores[source, model, 1]),
                    "wp_average": float(scores[source, model].mean()),
                    "delta_t0_vs_affine": float(delta[0]),
                    "delta_t1_vs_affine": float(delta[1]),
                    "delta_average_vs_affine": float(delta.mean()),
                    "t1_descriptive_cluster_interval": interval[:, 1].tolist(),
                }
            )
    verdicts = {}
    for source, source_name in enumerate(sources):
        delta = scores[source, 2, 1] - scores[source, 1, 1]
        half_delta = halves[:, source, 2, 1] - halves[:, source, 1, 1]
        nonlinear_delta = (scores[source, 5] - scores[source, 2]).mean()
        nonlinear_halves = (halves[:, source, 5] - halves[:, source, 2]).mean(axis=1)
        verdicts[source_name] = {
            "offset_t1_delta": float(delta),
            "offset_t1_half_deltas": half_delta.tolist(),
            "offset_passes_frozen_rule": bool(delta >= 0.005 and np.all(half_delta > 0)),
            "threshold_average_delta_vs_linear_offset": float(nonlinear_delta),
            "threshold_average_half_deltas": nonlinear_halves.tolist(),
            "threshold_passes_frozen_rule": bool(
                nonlinear_delta >= 0.003 and np.all(nonlinear_halves > 0)
            ),
        }
    result = {
        "test_id": "causal_residual_offsets_v1",
        "scores": records,
        "verdicts": verdicts,
        "halves_wp": halves.tolist(),
        "evaluation_sequences": len(groups),
        "excluded_exact_duplicates": sum(r["excluded_exact_warmup_duplicate"] for r in provenance),
        "evaluation_scored_rows": sum(
            r["scored_or_required_rows"]
            for r in provenance
            if not r["excluded_exact_warmup_duplicate"]
        ),
        "observable_exact_prefix_components": len(components),
        "uncertainty_caveat": "400 paired bootstraps over exact-prefix components; descriptive only, assumes independence of those components. Campaign dependence unknown; no campaign-valid confidence interval.",
    }
    (OUT / "results_offsets_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "scores_offsets_v1.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (OUT / "sequence_diagnostics_v1.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_group", "seq_ix", "scored_rows", "source", "model", "wp_t0", "wp_t1"])
        for index, record in enumerate(provenance):
            if moments[index, 0, 0, 0, 0] == 0:
                continue
            sequence_scores = weighted_pearson(moments[index])
            for source, name in enumerate(sources):
                for model, model_name in enumerate(MODELS):
                    writer.writerow(
                        [
                            int(groups[index]),
                            record["seq_ix"],
                            record["scored_or_required_rows"],
                            name,
                            model_name,
                            *sequence_scores[source, model],
                        ]
                    )
    plot_results(scores, sources)
    LOGGER.info("Frozen verdicts: %s", json.dumps(verdicts, indent=2))
    hashes = {}
    for name in [
        "continuation_common.py",
        "probe_offsets.py",
        "analyze_offsets.py",
        "HYPOTHESES.json",
        "sampling_v1.json",
        "evaluation_moments_v1.npz",
        "results_offsets_v1.json",
    ]:
        hashes[name] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    (OUT / "artifact_hashes_v1.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return result


def plot_results(scores: np.ndarray, sources: list[str]) -> None:
    """Show matched controls and learned state-offset mechanism."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    positions = np.arange(len(MODELS))
    for source, axis in enumerate(axes):
        for target in range(2):
            axis.plot(positions, scores[source, :, target], "o-", label=f"t{target}")
        axis.set_xticks(positions, MODELS, rotation=25)
        axis.set_title(f"Calibration: {sources[source]}")
        axis.set_ylabel("Global clipped weighted Pearson")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Frozen causal test: 64 sequence-disjoint evaluation row groups")
    figure.tight_layout()
    figure.savefig(OUT / "fig09_offsets_controls_v1.png", dpi=130)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    labels = ["a2", "a3", "activity100", "trade100", "log event age"]
    for source, axis in enumerate(axes):
        with np.load(OUT / f"calibration_{sources[source]}_v1.npz") as fitted:
            offset = fitted["offset"][2:]
        axis.bar(np.arange(5) - 0.16, offset[:, 0], width=0.32, label="t0")
        axis.bar(np.arange(5) + 0.16, offset[:, 1], width=0.32, label="t1")
        axis.set_xticks(np.arange(5), labels, rotation=25)
        axis.set_title(sources[source])
        axis.set_ylabel("Residual offset per source-standardized state")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.legend()
    figure.suptitle("Input-state offsets; no target-derived unit class")
    figure.tight_layout()
    figure.savefig(OUT / "fig10_offset_coefficients_v1.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze()
