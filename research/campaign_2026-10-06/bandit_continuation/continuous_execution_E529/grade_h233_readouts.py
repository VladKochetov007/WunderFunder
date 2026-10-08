"""Evaluate fixed H233 readouts with source-qualified rows and audited masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREREG = HERE / "H233_FIXED_EP1_COMPARISON_PREREG_20261008_R1.json"
PREREG_SHA = "e59c74763f764b3b57893d8873ae64ea687d57c9c1579b5ad3450c8e0d0f1b5a"
CHECKPOINT_SHA = "11b3e72d6f61acd0a45b68fe7edf0b66ccea4fa39a1492e8101821b154ba86ee"
PARENT_SHA = "7d34ee5649e124948f0b651aa786083fa9ee66a01c439b034345a1d9df9ec3f0"
PARENT_PRED_SHA = "cb832b1397c75e318551fd6d6f0e22d09f3c4230c3c5ad56cab1a4c4e603c7e8"
TARGET_SHA = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
MODES = ("hidden_only", "additive", "multiplicative", "conditional_residual")
SHARED = HERE.parent / "attention_training_H218/grade_fold0_reference_blends.py"
PAIR = HERE.parent / "numerical_embedding_E540/grade_fixed_embedding_pair.py"
SCORER = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"


def load_module(name: str, path: Path) -> ModuleType:
    """Reuse the independently checked row, metric and paired-bootstrap helpers."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_masks(
    groups: list[int],
    evaluation: list[int],
    target_groups: list[int],
    original_mask: np.ndarray,
    selected_mask: np.ndarray,
    group_mask: np.ndarray,
) -> None:
    """Reject altered scored rows, warmup leakage and fitting-group inclusion."""
    if groups != target_groups or len(groups) != len(set(groups)):
        raise ValueError("Export row order must match the frozen full-valid target order")
    if len(evaluation) != len(set(evaluation)) or not set(evaluation).issubset(groups):
        raise ValueError("Evaluation membership is incomplete or duplicated")
    if (
        selected_mask.dtype != np.bool_
        or selected_mask.shape != original_mask.shape
        or group_mask.dtype != np.bool_
        or group_mask.shape != (len(groups),)
    ):
        raise ValueError("Evaluation masks have unexpected shapes or dtypes")
    expected_groups = np.isin(groups, evaluation)
    if not np.array_equal(group_mask, expected_groups):
        raise ValueError("Evaluation-group mask includes fitting groups or omits evaluation")
    after_warmup = np.arange(original_mask.shape[1]) >= 99
    for index, evaluated in enumerate(expected_groups):
        expected = original_mask[index].astype(bool) & after_warmup
        if not evaluated:
            expected = np.zeros_like(expected)
        if not np.array_equal(selected_mask[index], expected):
            raise ValueError(f"Scored mask differs at source-qualified valid group {groups[index]}")


def full_alignment(
    grade: ModuleType,
    evaluation: list[int],
    groups: list[int],
    reference_groups: list[int],
    target_groups: list[int],
) -> list[tuple[tuple[str, int], int, int, int]]:
    """Keep original full-export indices while joining only excluded groups."""
    index = grade.source_group_index("valid", groups)
    aligned = grade.align_evaluation_rows(evaluation, evaluation, reference_groups, target_groups)
    return [(key, index[key], ref_row, target_row) for key, _, ref_row, target_row in aligned]


def main() -> None:
    """Require root review of delivered fit/export artifacts before accessing scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier results; choose a new identity")
    grade = load_module("h233_shared_grade", SHARED)
    approval = json.loads(args.authorization.read_text())
    if (
        approval.get("schema") != "h233-fixed-readout-grade-admission-v1"
        or approval.get("approved_by") != "root"
        or approval.get("status") != "approved"
        or approval.get("fit_and_export_terminal_reviewed") is not True
        or approval.get("preregistration_sha256") != PREREG_SHA
        or grade.sha256_file(PREREG) != PREREG_SHA
        or approval.get("grader_sha256") != grade.sha256_file(Path(__file__))
    ):
        raise ValueError("Missing delivered-artifact root grading admission")
    required = {
        *MODES,
        "metadata",
        "selected_mask",
        "group_mask",
        "checkpoint",
        "provenance",
        "fit_terminal",
        "export_terminal",
        "source_manifest",
        "recipe",
        "split_manifest",
        "target_cache",
        "parent_prediction",
        "parent_sidecar",
        "parent_checkpoint",
        "parent_provenance",
        "shared_grader",
        "pair_scorer",
        "scorer",
        "gmetric",
    }
    if set(approval["artifacts"]) != required:
        raise ValueError("Incomplete or unexpected delivered artifact set")
    paths: dict[str, Path] = {}
    for name, entry in approval["artifacts"].items():
        path = (ROOT / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or grade.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Delivered artifact identity changed: {name}")
        paths[name] = path
    for name, expected in (
        ("shared_grader", SHARED),
        ("pair_scorer", PAIR),
        ("scorer", SCORER),
        ("gmetric", ROOT / "src/wnn/gmetric.py"),
    ):
        if paths[name] != expected.resolve():
            raise ValueError(f"Unexpected scoring helper: {name}")
    for name, digest in (
        ("checkpoint", CHECKPOINT_SHA),
        ("parent_checkpoint", PARENT_SHA),
        ("parent_prediction", PARENT_PRED_SHA),
        ("target_cache", TARGET_SHA),
    ):
        if approval["artifacts"][name]["sha256"] != digest:
            raise ValueError(f"Frozen checkpoint/data identity changed: {name}")
    prereg = json.loads(PREREG.read_text())
    if grade.sha256_file(paths["recipe"]) != prereg["recipe_sha256"]:
        raise ValueError("Frozen H233 recipe changed")
    recipe = json.loads(paths["recipe"].read_text())
    split = json.loads(paths["split_manifest"].read_text())
    if grade.sha256_file(paths["split_manifest"]) != recipe["validation"]["split_manifest_sha256"]:
        raise ValueError("Split differs from recipe")
    evaluation = split["evaluation"]["valid"]
    provenance = json.loads(paths["provenance"].read_text())
    membership = grade.validate_candidate_fit_membership(split, provenance, evaluation)
    parent_membership = grade.validate_candidate_fit_membership(
        split, json.loads(paths["parent_provenance"].read_text()), evaluation
    )
    if provenance.get("selected_epoch") != 1 or provenance.get("status") != "complete":
        raise ValueError("Fit has not completed the fixed epoch-1 endpoint")
    metadata = json.loads(paths["metadata"].read_text())
    if (
        metadata.get("recipe_id") != recipe["recipe_id"]
        or metadata.get("split") != "valid"
        or metadata.get("checkpoint_epoch") != 1
        or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA
        or metadata.get("parent_checkpoint_sha256") != PARENT_SHA
        or metadata.get("parent_anchor_parity", {}).get("passed") is not True
        or set(metadata.get("prediction_files", {})) != set(MODES)
        or len(evaluation) != 468
    ):
        raise ValueError("Export metadata does not bind the fixed recipe and parent")
    for mode in MODES:
        if metadata["prediction_files"][mode]["sha256"] != approval["artifacts"][mode]["sha256"]:
            raise ValueError(f"Export prediction binding differs: {mode}")
    for name, record in (
        ("selected_mask", metadata["selected_mask"]),
        ("group_mask", metadata["evaluation_group_mask"]),
    ):
        if record["sha256"] != approval["artifacts"][name]["sha256"]:
            raise ValueError(f"Export mask binding differs: {name}")
    scorer = load_module("h233_official_score", SCORER)
    pair = load_module("h233_pair_score", PAIR)
    if prereg["metric"] != scorer.METRIC_ID:
        raise ValueError("Official metric differs")
    targets = scorer.mmap_stored_npz(paths["target_cache"])
    selected = np.load(paths["selected_mask"], mmap_mode="r", allow_pickle=False)
    group_mask = np.load(paths["group_mask"], mmap_mode="r", allow_pickle=False)
    groups = metadata["row_groups"]
    target_groups = targets["row_group"].tolist()
    validate_masks(groups, evaluation, target_groups, targets["mask"], selected, group_mask)
    masked_targets = {**targets, "mask": selected}
    arrays = {mode: np.load(paths[mode], mmap_mode="r", allow_pickle=False) for mode in MODES}
    for values in arrays.values():
        grade.validate_prediction_array(values, len(groups))
    parent_sidecar = json.loads(paths["parent_sidecar"].read_text())
    if parent_sidecar.get("prediction_sha256") != PARENT_PRED_SHA:
        raise ValueError("Frozen parent sidecar differs")
    parent = np.load(paths["parent_prediction"], mmap_mode="r", allow_pickle=False)
    grade.validate_prediction_array(parent, len(parent_sidecar["row_groups"]))
    own_alignment = full_alignment(grade, evaluation, groups, groups, target_groups)
    parent_alignment = full_alignment(
        grade, evaluation, groups, parent_sidecar["row_groups"], target_groups
    )
    component = {row["row_group"]: row["component"] for row in split["sequences"]["valid"]}
    components = np.asarray([component[key[1]] for key, _, _, _ in own_alignment])
    primary = {
        mode: pair.score_pair(
            masked_targets,
            arrays[mode],
            arrays["hidden_only"],
            own_alignment,
            components,
            scorer,
            [0.0, 1.0],
            prereg["bootstrap"],
        )
        for mode in MODES
        if mode != "hidden_only"
    }
    secondary = {
        mode: pair.score_pair(
            masked_targets,
            arrays[mode],
            parent,
            parent_alignment,
            components,
            scorer,
            prereg["weights_on_readout"],
            prereg["bootstrap"],
        )
        for mode in MODES
    }
    for comparisons in (primary, secondary):
        for rows in comparisons.values():
            for row in rows:
                row["weight_on_readout"] = row.pop("weight_on_ple")
                row["delta_wp_vs_control"] = row.pop("delta_wp_vs_linear")
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "evaluated_exposed_discovery",
        "metric": scorer.METRIC_ID,
        "preregistration_sha256": PREREG_SHA,
        "authorization_sha256": grade.sha256_file(args.authorization),
        "grader_sha256": grade.sha256_file(Path(__file__)),
        "membership": membership,
        "parent_membership": parent_membership,
        "primary_vs_trained_hidden_only": primary,
        "secondary_vs_frozen_parent_all_weights": secondary,
        "group_count": len(evaluation),
        "parent_numerical_comparison_limit": prereg["parent_export_comparison_limit"],
        "public_forecast": None,
        "weight_selection": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({mode: rows[-1]["score"]["wp"] for mode, rows in secondary.items()})


if __name__ == "__main__":
    main()
