from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_common import OWN, phase, save_json

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Present fixed replay denominators and synthetic identification results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = json.loads((OWN / "raw_replay_results.json").read_text())
    synthetic = json.loads((OWN / "synthetic_results.json").read_text())
    origins = pd.read_csv(OWN / "all_origins_raw_replay.csv")
    for name, key in [
        ("component", "by_component"),
        ("class", "by_class"),
        ("component_class", "component_class"),
    ]:
        table = pd.DataFrame.from_dict(raw[key], orient="index")
        table.index.name = name
        table.to_csv(OWN / f"coverage_{name}.csv")
        LOGGER.info("%s\n%s", name.upper(), table.to_string())
    outside = origins[origins.outside & origins.mapped]
    censor = {}
    for law in ["fixed_extrema", "competing_barriers", "trailing_extrema"]:
        complete = outside[f"{law}_complete"]
        censor[law] = {
            "mapped_outside": len(outside),
            "complete_pair": int(complete.sum()),
            "unresolved_pair": int((~complete).sum()),
            "bound_violations_complete": int(
                outside.loc[complete, "raw_violation"].sum()
            ),
            "bound_violations_unresolved": int(
                outside.loc[~complete, "raw_violation"].sum()
            ),
            "reason_pairs_unresolved": outside.loc[~complete, f"{law}_reasons"]
            .value_counts()
            .to_dict(),
        }
    save_json("censoring_bound_crosscheck.json", censor)
    LOGGER.info("CENSORING %s", json.dumps(censor))
    LOGGER.info("FIELD_MISMATCHES %s", json.dumps(raw["recorded_field_mismatches"]))
    LOGGER.info(
        "NONCONSTANT_LABEL_SPANS %s", raw["nonconstant_represented_label_spans"]
    )
    LOGGER.info(
        "RAW_MARGINS %s %s",
        raw["raw_minimum_outside_bound_margin"],
        raw["raw_maximum_outside_bound_margin"],
    )
    table = pd.DataFrame.from_dict(raw["by_component"], orient="index")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(table))
    axes[0].bar(
        positions, table.raw_violations_outside, label="Mapped outside violates"
    )
    axes[0].bar(
        positions,
        table.missing_outside,
        bottom=table.raw_violations_outside,
        label="Outside map missing",
        color="tab:orange",
    )
    axes[0].set_xticks(positions, table.index, rotation=45)
    axes[0].set_xlabel("Supplied paired component (not independently rediscovered)")
    axes[0].set_ylabel("Origin count")
    axes[0].set_title("All 445 outside origins: 417 contradictions, 28 unknown")
    axes[0].legend(fontsize=8)
    for index, mode in enumerate(["changing_reference", "fixed_reference_control"]):
        counts = synthetic[mode]["counts"]
        axes[1].bar(
            index - 0.18,
            counts["label_difference_exact"],
            width=0.36,
            color="tab:blue",
            label="Exact label differences" if index == 0 else None,
        )
        axes[1].bar(
            index + 0.18,
            counts["true_quote_move_exact"],
            width=0.36,
            color="tab:green",
            label="Exact true quote moves" if index == 0 else None,
        )
    axes[1].set_xticks([0, 1], ["Changing future reference", "Fixed-reference control"])
    axes[1].set_ylabel("Exact held-out transitions / 252")
    axes[1].set_title("Interior exactness alone does not identify physical moves")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OWN / "audit_coverage_and_gauge.png", dpi=120)
    plt.close(fig)
    phase("results_summarized_figure_pending_inspection")


if __name__ == "__main__":
    main()
