from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS
from wnn.domain_stress import freeze_binary_panel
from wnn.splits import file_sha256, load_campaign_split


def freeze(root: Path, output: Path) -> None:
    """Freeze three input-only component panels before new candidate outcomes.

    Cutpoints use eligible supplied-validation fitting groups only. This avoids
    presenting the existing partial train summary as complete training coverage.
    Other required stress families remain explicit pending work.
    """
    if output.exists():
        raise FileExistsError(f"refusing to rewrite frozen panel: {output}")
    source = root / "cache/seqstats_valid.parquet"
    columns = ["row_group"] + [
        f"{name}_{suffix}" for suffix in ("chg", "nuniq") for name in FEATURE_COLUMNS
    ]
    table = pq.read_table(source, columns=columns)
    group = table["row_group"].to_numpy()
    if len(np.unique(group)) != len(group):
        raise ValueError("duplicate source-qualified summary group")
    change = np.column_stack([table[f"{name}_chg"].to_numpy() for name in FEATURE_COLUMNS])
    support = np.column_stack([table[f"{name}_nuniq"].to_numpy() for name in FEATURE_COLUMNS])
    if not np.isfinite(change).all() or np.any((change < 0) | (change > 1)):
        raise ValueError("invalid input change rates")
    if not np.isfinite(support).all() or np.any(support < 1):
        raise ValueError("invalid feature support counts")
    descriptors = {
        "activity": change.mean(axis=1),
        "instrument1_change_frequency": change[:, 52:104].mean(axis=1),
        "feature_support": np.log1p(support).mean(axis=1),
    }
    definitions = {
        "activity": "mean of 112 input-column change frequencies over whole sequence",
        "instrument1_change_frequency": "mean change frequency of documented input columns52:104",
        "feature_support": "mean log(1+unique input values) over112 feature columns",
    }
    folds = []
    for fold in range(4):
        split = load_campaign_split(
            root / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json", root
        )
        component_of = {g: c for s, g, c in split.components if s == "valid"}
        if set(group) != set(component_of):
            raise ValueError("valid summary coverage differs from frozen source membership")
        components = np.array([component_of[int(g)] for g in group])
        fitting = np.isin(group, split.valid_groups)
        evaluation = np.isin(group, split.evaluation_groups)
        panels = {}
        for name, values in descriptors.items():
            panels[name] = freeze_binary_panel(
                values[fitting], components[fitting], values[evaluation], components[evaluation]
            )
        folds.append(
            {
                "fold": fold,
                "split_sha256": split.sha256,
                "cutpoint_fit_source": "valid",
                "cutpoint_fit_row_groups": sorted(group[fitting].tolist()),
                "evaluation_source": "valid",
                "evaluation_row_groups": sorted(group[evaluation].tolist()),
                "exposure": "previously exposed; robustness only",
                "panels": panels,
            }
        )
    record = {
        "format": "wnn.strategy-domain-panels.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "coverage_item": "S30",
        "source": str(source.relative_to(root)),
        "source_sha256": file_sha256(source),
        "allowed_columns": columns,
        "descriptor_definitions": definitions,
        "source_code_sha256": {
            p: file_sha256(root / p)
            for p in (
                "scripts/freeze_strategy_domains.py",
                "src/wnn/domain_stress.py",
                "scripts/eda_seqstats.py",
            )
        },
        "mixture_high_fractions": [0.25, 0.75],
        "mixture_measure": "uniform component frequency within each domain; preserve complete selected rows and original-label magnitude weights",
        "mask": "need_prediction AND is_scored; alternate masks sensitivity only",
        "pending_panels": [
            "quiet/transition row states",
            "source similarity",
            "cross-fitted scoring propensity",
        ],
        "limitations": [
            "cutpoint fitting uses valid-fit source only",
            "descriptors use entire observed sequence for retrospective grouping, not prediction",
            "component descriptor averages eligible valid-source sequences; train-source sequences are not in this descriptor cache",
        ],
        "folds": folds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Create one immutable, input-only panel specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.output)


if __name__ == "__main__":
    main()
