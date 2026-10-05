"""Summarize the complete fixed screen using the existing component scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import _centered_moments

COMPARISONS = (
    ("gru_spline", "gru_mlp"),
    ("ssm_learned_mlp", "ssm_fixed_mlp"),
    ("ssm_learned_spline", "ssm_learned_mlp"),
    ("gru_attention_mlp", "gru_pointwise_mlp"),
    ("ssm_attention_mlp", "ssm_pointwise_mlp"),
    ("ssm_attention_spline", "ssm_attention_mlp"),
    ("ssm_attention_spline", "ssm_learned_spline"),
)


def main() -> None:
    """Require all ten arms; save raw conditional intervals and compact figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    report = json.loads((run / "result.json").read_text())
    if len(report["arms"]) != 10 or "source_sha256" not in report:
        raise ValueError("the fixed ten-arm fit and final receipt must be complete")
    nodes = [
        n for n in json.loads((run / "cohort.json").read_text())["cohort"] if n["role"] == "test"
    ]
    components = np.array([n["component"] for n in nodes])
    moments = {}
    reference_targets = None
    for arm in report["arms"]:
        with np.load(run / f"{arm['arm']}_selected_predictions.npz") as data:
            y, p = data["targets"], data["predictions"]
        if reference_targets is None:
            reference_targets = y
        elif not np.array_equal(y, reference_targets):
            raise ValueError("prediction exports do not align on selected targets")
        summaries, position = [], 0
        for node, seq in zip(nodes, arm["sequences"], strict=True):
            if node["row_group"] != seq["row_group"]:
                raise ValueError("sequence order changed between arms")
            end = position + seq["scored_rows"]
            summaries.append(
                np.stack(
                    [
                        _centered_moments(
                            np.clip(y[position:end, j], -2, 2).astype(np.float64),
                            np.clip(p[position:end, j], -2, 2).astype(np.float64),
                        )
                        for j in range(2)
                    ]
                )
            )
            position = end
        if position != len(y):
            raise ValueError("selected row counts do not match sequence boundaries")
        moments[arm["arm"]] = np.stack(summaries)
    intervals = {}
    for candidate, reference in COMPARISONS:
        intervals[f"{candidate}_minus_{reference}"] = paired_component_bootstrap(
            moments[candidate],
            moments[reference],
            components,
            n_boot=2000,
            seed=271828,
            centered=True,
        )
    (run / "paired_components.json").write_text(
        json.dumps(
            {
                "comparisons": intervals,
                "scope": "16 exposed components, conditional on one fitted seed; exploratory multiple comparisons, not fresh validation or training-seed uncertainty",
            },
            indent=2,
        )
    )
    eda = run.parent.parent / "eda" / run.name
    eda.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    arms = report["arms"]
    ax.barh(
        [a["arm"] for a in arms],
        [a["wp"] for a in arms],
        color=["#bf6b25" if a["arm"].endswith("spline") else "#3379a8" for a in arms],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Official pooled WP on 16 exposed diagnostic sequences")
    ax.set_title("Fixed one-epoch small-model screen; no public-score forecast")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(eda / "ten_arm_scores.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    spline = torch.load(run / "ssm_learned_spline.pt", map_location="cpu", weights_only=True)
    edges = spline["correction.edge_values"].numpy()
    knots = np.linspace(-1, 1, edges.shape[-1])
    for target, ax in enumerate(axes):
        selected = np.argsort(np.ptp(edges[target], axis=1))[-5:]
        for index in selected:
            ax.plot(knots, edges[target, index], label=f"latent coordinate {index}")
        ax.set_title(f"t{target}: five largest learned spline-edge ranges")
        ax.set_xlabel("Bounded latent state; no physical interpretation asserted")
        ax.set_ylabel("Additive prediction contribution")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(eda / "learned_spline_edges.png", dpi=160)
    plt.close(fig)
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    main()
