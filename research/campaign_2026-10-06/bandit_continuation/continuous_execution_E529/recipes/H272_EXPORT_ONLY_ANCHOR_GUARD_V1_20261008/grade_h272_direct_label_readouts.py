"""Grade H272 with the frozen H233 comparison math after root admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
PREREG = HERE / "H272_DIRECT_LABEL_COMPARISON_PREREG_V1_20261008.json"
PREREG_SHA = "63e71c18789a5e559a88fbd2cee68627986d778500773bdac8402556842bdbad"
RECIPE_SHA = "a5dd858598b6bf926410c80337aa8e725a809cbfee36af906d18c34d0859716f"
SOURCE_LOCK_SHA = "a8d56811e97783bcaa354a28beb992f8ce8acebc1b62f70f945c6697426aaa76"
PARENT_SHA = "7d34ee5649e124948f0b651aa786083fa9ee66a01c439b034345a1d9df9ec3f0"
PARENT_PRED_SHA = "cb832b1397c75e318551fd6d6f0e22d09f3c4230c3c5ad56cab1a4c4e603c7e8"
PARENT_SIDECAR_SHA = "bb335392edc599f181c0bbcb6523fd430b036647454bc0d76facf5b592211ad3"
PARENT_PROVENANCE_SHA = "60e8884d3d5886dc53dcb2b7d280fef284a702de8769ab333e07e1819d66f28a"
TARGET_SHA = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
BASE_GRADER = HERE.parents[1] / "grade_h233_readouts.py"
BASE_GRADER_SHA = "c084c12ea7ad3c9af3725a532c1aad3fd76e50a015b02931e2216cea53f7e19e"
SHARED = HERE.parents[2] / "attention_training_H218/grade_fold0_reference_blends.py"
SHARED_SHA = "88034869363f8475d8030f932e7bc24af0850241e3511d276b03ce7a53e5a4bd"
PAIR = HERE.parents[2] / "numerical_embedding_E540/grade_fixed_embedding_pair.py"
PAIR_SHA = "64d547a9f4f7861f5c104ebc95b6d9f8315f4d51554e092ec6cfd5076ccc5376"
SCORER = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
SCORER_SHA = "b6377a0e1d67030115c6d20a64ee00dc97ef15eaad7a806408a76ef70558ab39"
GMETRIC = ROOT / "src/wnn/gmetric.py"
GMETRIC_SHA = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
MODES = ("hidden_only", "additive", "multiplicative", "conditional_residual")
ARTIFACTS = {
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


def load_module(name: str, path: Path) -> ModuleType:
    """Import a pinned scoring helper without importing the training runtime."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_parent_sidecar(
    sidecar: dict[str, object],
    expected_groups: list[int],
    parent_checkpoint_sha256: str = PARENT_SHA,
    parent_prediction_sha256: str = PARENT_PRED_SHA,
) -> None:
    """Bind the C1 parent prediction metadata to its producer and full row order."""
    if (
        sidecar.get("run") != "bd26_c1v2_s3_f0"
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 1
        or sidecar.get("inference_precision") != "fp16"
        or sidecar.get("checkpoint_sha256") != parent_checkpoint_sha256
        or sidecar.get("prediction_sha256") != parent_prediction_sha256
        or sidecar.get("split_manifest_sha256") != SPLIT_SHA
        or sidecar.get("row_groups") != expected_groups
    ):
        raise ValueError("C1 parent sidecar differs from its frozen producer or valid row order")


def group_ids_sha256(groups: list[int]) -> str:
    """Hash ordered integer group IDs using the exporter’s canonical encoding."""
    canonical = np.asarray(groups, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _verify_frozen_scoring_sources(grade: ModuleType) -> dict[str, str]:
    """Fail closed if the preregistration or shared grading sources changed."""
    for path, expected, name in (
        (PREREG, PREREG_SHA, "H272 preregistration"),
        (BASE_GRADER, BASE_GRADER_SHA, "frozen H233 comparison grader"),
        (SHARED, SHARED_SHA, "shared fold-0 grader"),
        (PAIR, PAIR_SHA, "paired comparison scorer"),
        (SCORER, SCORER_SHA, "official campaign scorer"),
        (GMETRIC, GMETRIC_SHA, "official metric module"),
    ):
        if grade.sha256_file(path) != expected:
            raise ValueError(f"Pinned grading source changed: {name}")
    return {
        "h272_preregistration": PREREG_SHA,
        "h233_base_grader": BASE_GRADER_SHA,
        "shared_fold0_grader": SHARED_SHA,
        "paired_comparison_scorer": PAIR_SHA,
        "official_campaign_scorer": SCORER_SHA,
        "official_metric_module": GMETRIC_SHA,
        "h272_recipe": RECIPE_SHA,
        "h272_source_lock": SOURCE_LOCK_SHA,
    }


def main() -> None:
    """Grade only root-admitted H272 fit/export artifacts and fixed comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior results; choose a new output identity")

    grade = load_module("h272_shared_grade", SHARED)
    pinned_sources = _verify_frozen_scoring_sources(grade)
    admission = json.loads(args.authorization.read_text(encoding="utf-8"))
    if (
        admission.get("schema") != "h272-direct-label-readout-grade-admission-v1"
        or admission.get("approved_by") != "root"
        or admission.get("status") != "approved"
        or admission.get("fit_and_export_terminal_reviewed") is not True
        or admission.get("preregistration_sha256") != PREREG_SHA
        or admission.get("grader_sha256") != grade.sha256_file(Path(__file__))
    ):
        raise ValueError("Missing actual-artifact root H272 grading admission")
    if not isinstance(admission.get("artifacts"), dict) or set(admission["artifacts"]) != ARTIFACTS:
        raise ValueError("Incomplete or unexpected H272 delivered-artifact set")

    paths: dict[str, Path] = {}
    for name, entry in admission["artifacts"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"Malformed root artifact entry: {name}")
        path = (ROOT / entry["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or grade.sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Delivered artifact identity changed: {name}")
        paths[name] = path

    fixed_paths = {
        "metadata": ROOT
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/provenance.json",
        "selected_mask": ROOT
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/selected_mask.npy",
        "group_mask": ROOT
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1/evaluation_group_mask.npy",
        "checkpoint": ROOT / "artifacts/runs/bd26_h272_direct_label_c1s3_rank32_f0_v1/model_ep1.pt",
        "provenance": ROOT
        / "artifacts/runs/bd26_h272_direct_label_c1s3_rank32_f0_v1/provenance.json",
        "source_manifest": HERE.parents[1]
        / "recipes/H272_DIRECT_LABEL_FROZEN_RUNTIME_SOURCE_LOCK_V1_20261008.json",
        "recipe": HERE.parents[1] / "recipes/H272_C1S3_DIRECT_LABEL_READOUT_V1.json",
        "split_manifest": ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "target_cache": ROOT / "cache/valid_targets.npz",
        "parent_prediction": ROOT / "cache/pred_bd26_c1v2_s3_f0_valid.npy",
        "parent_sidecar": ROOT / "cache/pred_bd26_c1v2_s3_f0_valid.json",
        "parent_checkpoint": ROOT / "artifacts/runs/bd26_c1v2_s3_f0/model_ep1.pt",
        "parent_provenance": ROOT / "artifacts/runs/bd26_c1v2_s3_f0/provenance.json",
        "shared_grader": SHARED,
        "pair_scorer": PAIR,
        "scorer": SCORER,
        "gmetric": GMETRIC,
    }
    export_dir = (
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/exports/H272_C1S3_DIRECT_LABEL_E1_VALID_V1"
    )
    for mode in MODES:
        fixed_paths[mode] = export_dir / f"pred_{mode}.npy"
    for name, expected_path in fixed_paths.items():
        if paths[name] != expected_path.resolve():
            raise ValueError(f"Unexpected frozen artifact path: {name}")
    for name, expected_sha in (
        ("source_manifest", SOURCE_LOCK_SHA),
        ("recipe", RECIPE_SHA),
        ("split_manifest", SPLIT_SHA),
        ("target_cache", TARGET_SHA),
        ("parent_prediction", PARENT_PRED_SHA),
        ("parent_sidecar", PARENT_SIDECAR_SHA),
        ("parent_checkpoint", PARENT_SHA),
        ("parent_provenance", PARENT_PROVENANCE_SHA),
        ("shared_grader", SHARED_SHA),
        ("pair_scorer", PAIR_SHA),
        ("scorer", SCORER_SHA),
        ("gmetric", GMETRIC_SHA),
    ):
        if admission["artifacts"][name]["sha256"] != expected_sha:
            raise ValueError(f"Frozen source/data identity changed: {name}")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    recipe = json.loads(paths["recipe"].read_text(encoding="utf-8"))
    split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    evaluation = split["evaluation"]["valid"]
    if len(evaluation) != 468:
        raise ValueError("Frozen fold-0 evaluation membership is not 468 groups")
    if (
        prereg["recipe_sha256"] != RECIPE_SHA
        or recipe["recipe_id"] != "E529-H272-C1S3-DIRECT-LABEL-READOUT-V1"
    ):
        raise ValueError("H272 recipe does not match the preregistered direct-label arm")
    if (
        prereg["metric"] != "global_wp_clip2_v1"
        or prereg["modes"] != list(MODES)
        or prereg["weights_on_readout"] != [0, 0.1, 0.25, 0.5, 1]
        or prereg["bootstrap"]
        != {"unit": "paired frozen connected components", "replicates": 2000, "seed": 271828}
    ):
        raise ValueError("H272 fixed comparison contract differs from preregistration")

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    membership = grade.validate_candidate_fit_membership(split, provenance, evaluation)
    parent_provenance = json.loads(paths["parent_provenance"].read_text(encoding="utf-8"))
    parent_membership = grade.validate_candidate_fit_membership(
        split, parent_provenance, evaluation
    )
    if provenance.get("selected_epoch") != 1 or provenance.get("status") != "complete":
        raise ValueError("H272 fit has not completed its fixed epoch-1 endpoint")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    checkpoint_sha = admission["artifacts"]["checkpoint"]["sha256"]
    parent_sidecar = json.loads(paths["parent_sidecar"].read_text(encoding="utf-8"))
    parity = metadata.get("parent_anchor_parity", {})
    parity_error = parity.get("max_absolute_error")
    if (
        metadata.get("recipe_id") != recipe["recipe_id"]
        or metadata.get("split") != "valid"
        or metadata.get("checkpoint_epoch") != 1
        or metadata.get("checkpoint_sha256") != checkpoint_sha
        or metadata.get("parent_checkpoint_sha256") != PARENT_SHA
        or set(metadata.get("prediction_files", {})) != set(MODES)
        or parity.get("passed") is not True
        or parity.get("tolerance") != 1e-3
        or not isinstance(parity_error, (int, float))
        or not math.isfinite(float(parity_error))
        or parity_error > 1e-3
    ):
        raise ValueError("H272 export metadata does not bind the fixed recipe/checkpoint/parent")

    targets = grade.mmap_stored_npz(paths["target_cache"])
    target_groups = targets["row_group"].tolist()
    if metadata.get("row_groups") != target_groups or len(target_groups) != 1_873:
        raise ValueError("H272 export row order differs from the full frozen valid target order")
    validate_parent_sidecar(parent_sidecar, target_groups)
    if (
        metadata.get("parent_anchor_reference", {}).get("prediction_path")
        != "cache/pred_bd26_c1v2_s3_f0_valid.npy"
        or metadata.get("parent_anchor_reference", {}).get("prediction_sha256") != PARENT_PRED_SHA
        or metadata.get("parent_anchor_reference", {}).get("checkpoint_sha256") != PARENT_SHA
        or parity.get("reference") != metadata.get("parent_anchor_reference")
    ):
        raise ValueError("H272 exporter reference identity differs from the frozen C1 prediction")

    for mode in MODES:
        record = metadata["prediction_files"][mode]
        artifact = admission["artifacts"][mode]
        if (
            record.get("sha256") != artifact["sha256"]
            or record.get("path") != f"pred_{mode}.npy"
            or Path(artifact["path"]).name != record.get("path")
            or record.get("shape") != [1_873, 20_000, 2]
            or record.get("dtype") != "float32"
        ):
            raise ValueError(f"H272 prediction binding differs from export metadata: {mode}")
    for artifact_name, metadata_name in (
        ("selected_mask", "selected_mask"),
        ("group_mask", "evaluation_group_mask"),
    ):
        if metadata[metadata_name].get("sha256") != admission["artifacts"][artifact_name]["sha256"]:
            raise ValueError(f"H272 export mask binding differs: {artifact_name}")
    if (
        metadata["selected_mask"].get("evaluation_only") is not True
        or metadata["selected_mask"].get("warmup_rows_excluded") != 99
        or metadata["evaluation_group_mask"].get("evaluation_group_ids") != evaluation
        or metadata["evaluation_group_mask"].get("true_group_count") != 468
        or metadata["evaluation_group_mask"].get("row_groups_sha256")
        != group_ids_sha256(metadata["row_groups"])
        or metadata["evaluation_group_mask"].get("evaluation_group_ids_sha256")
        != group_ids_sha256(evaluation)
        or metadata["evaluation_group_mask"].get(
            "fit_development_group_count_exported_but_excluded"
        )
        != 1_405
        or metadata.get("grader_mask", {}).get("fit_development_groups_included") is not False
        or metadata.get("grader_mask", {}).get("row_groups_sha256")
        != group_ids_sha256(metadata["row_groups"])
        or metadata.get("grader_mask", {}).get("group_mask_path") != "evaluation_group_mask.npy"
    ):
        raise ValueError("H272 export does not declare the frozen excluded-group scoring mask")

    template = load_module("h272_frozen_h233_readout_grader", BASE_GRADER)
    pair = load_module("h272_fixed_pair_score", PAIR)
    scorer = load_module("h272_official_score", SCORER)
    if scorer.METRIC_ID != "global_wp_clip2_v1":
        raise ValueError("Official metric differs from the H272 preregistration")
    selected = np.load(paths["selected_mask"], mmap_mode="r", allow_pickle=False)
    group_mask = np.load(paths["group_mask"], mmap_mode="r", allow_pickle=False)
    groups = metadata["row_groups"]
    template.validate_masks(
        groups, evaluation, target_groups, targets["mask"], selected, group_mask
    )
    masked_targets = {**targets, "mask": selected}
    arrays = {mode: np.load(paths[mode], mmap_mode="r", allow_pickle=False) for mode in MODES}
    for values in arrays.values():
        grade.validate_prediction_array(values, len(groups))
    parent = np.load(paths["parent_prediction"], mmap_mode="r", allow_pickle=False)
    grade.validate_prediction_array(parent, len(parent_sidecar["row_groups"]))
    own_alignment = template.full_alignment(grade, evaluation, groups, groups, target_groups)
    parent_alignment = template.full_alignment(
        grade, evaluation, groups, parent_sidecar["row_groups"], target_groups
    )
    component = {row["row_group"]: row["component"] for row in split["sequences"]["valid"]}
    components = np.asarray([component[key[1]] for key, _, _, _ in own_alignment])
    bootstrap = prereg["bootstrap"]
    primary = {
        mode: pair.score_pair(
            masked_targets,
            arrays[mode],
            arrays["hidden_only"],
            own_alignment,
            components,
            scorer,
            [0.0, 1.0],
            bootstrap,
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
            bootstrap,
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
        "pinned_scoring_sources": pinned_sources,
        "membership": membership,
        "parent_membership": parent_membership,
        "primary_vs_trained_hidden_only": primary,
        "secondary_vs_frozen_parent_all_weights": secondary,
        "group_count": len(evaluation),
        "parent_numerical_comparison_limit": prereg["comparison_limit"],
        "public_forecast": None,
        "weight_selection": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({mode: rows[-1]["score"]["wp"] for mode, rows in secondary.items()})


if __name__ == "__main__":
    main()
