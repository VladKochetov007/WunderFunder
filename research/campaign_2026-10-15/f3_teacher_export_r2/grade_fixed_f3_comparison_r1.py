"""Compare saved F3 endpoints and one fixed half blend with frozen metric helpers."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bandit_continuation import _official_mask, compare_arrays  # noqa: E402
from score_fold0_panel import _member  # noqa: E402

from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402


def main() -> None:
    """Reuse the generic member validator and paired scorer on the frozen F3 split."""
    output = Path(__file__).with_name("F3_FIXED_COMPARISON_R1.json")
    if output.exists():
        raise FileExistsError(output)
    split = load_campaign_split(
        ROOT / "research/campaign_2026-09-29/artifacts/splits/fold3.json", ROOT
    )
    if split.fold != 3 or len(split.evaluation_groups) != 505:
        raise ValueError("expected the fixed 505-group F3 evaluation panel")
    targets_path = ROOT / "cache/valid_targets.npz"
    with np.load(targets_path, allow_pickle=False) as cache:
        groups = list(map(int, cache["row_group"]))
        position = {group: index for index, group in enumerate(groups)}
        indices = np.array([position[group] for group in split.evaluation_groups])
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        scored = np.asarray(cache["mask"][indices], dtype=bool)
    mask = _official_mask(scored)
    component_of = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.array([component_of[group] for group in split.evaluation_groups])
    runs = {
        "teacher": "clean_t3_f3_muonall_h384_e3_s3_r5@ep2",
        "older_model": "c20261003_combo_s3_f3@ep3",
    }
    predictions = {}
    bindings = {}
    for label, run in runs.items():
        array, binding = _member(run, split, groups, ROOT)
        predictions[label] = np.asarray(array[indices], dtype=np.float32)
        bindings[label] = binding
    reference = predictions["older_model"]
    predictions["fixed_half"] = (
        np.float32(0.5) * predictions["teacher"] + np.float32(0.5) * reference
    )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "metric_id": METRIC_ID,
        "split_manifest_sha256": split.sha256,
        "targets_sha256": file_sha256(targets_path),
        "evaluation_groups": len(indices),
        "components": len(set(components.tolist())),
        "runs": runs,
        "bindings": bindings,
        "official": {
            label: global_wp(targets, values, mask) for label, values in predictions.items()
        },
        "paired_vs_older_model": {
            label: compare_arrays(
                targets,
                predictions[label],
                reference,
                scored,
                components,
                n_boot=2000,
                seed=271828,
            )
            for label in ("teacher", "fixed_half")
        },
        "scope": (
            "Exploratory exposed F3, one seed per recipe. Half weights fixed after "
            "seeing single scores; no weight search. Different training and "
            "unknown older-model inference precision prevent causal claims. "
            "Intervals condition on fitted predictions, not training variation."
        ),
        "sources_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                Path(__file__),
                ROOT / "scripts/score_fold0_panel.py",
                ROOT / "scripts/evaluate_bandit_continuation.py",
                ROOT / "src/wnn/gmetric.py",
            )
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["official"]))


if __name__ == "__main__":
    main()
