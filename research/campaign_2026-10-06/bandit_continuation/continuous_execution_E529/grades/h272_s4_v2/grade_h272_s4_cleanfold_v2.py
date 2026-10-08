from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
E529 = HERE.parents[1]
PREREG = E529 / "H272_S4_CLEANFOLD_V2_GRADE_PREREG_20261008_R1.json"
PREREG_SHA256 = "c72cfa84e7646771114966b2b238cb9cacadc9971a6b40483fbe2b3b7069a80b"
FIT_HELPERS = E529 / "grade_h233_readouts.py"
PAIR_HELPERS = E529.parent / "numerical_embedding_E540/grade_fixed_embedding_pair.py"
SCORER = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
GMETRIC = ROOT / "src/wnn/gmetric.py"
REPORT = E529 / "H272_S4_CLEANFOLD_V2_GRADE_RESULT_20261008_R1.json"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks before admitting it to the grade."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """Load a frozen grading helper without importing the training runtime."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load grading helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_file(relative_path: str, expected_sha256: str) -> Path:
    """Require a hash-bound file inside the project checkout."""
    path = (ROOT / relative_path).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Artifact is missing or outside the repository: {relative_path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Artifact SHA256 differs from the preregistration: {relative_path}")
    return path


def validate_fit_membership(
    split: dict[str, object],
    provenance: dict[str, object],
    evaluation_groups: list[int],
    expected_split_sha256: str,
) -> dict[str, object]:
    """Check the producer's source-qualified fit sets against the frozen split."""
    if provenance.get("split_manifest_sha256") != expected_split_sha256:
        raise ValueError("Fit provenance is bound to another split manifest")
    expected_fit = split.get("fit")
    actual_fit = provenance.get("fit_row_groups")
    if not isinstance(expected_fit, dict) or not isinstance(actual_fit, dict):
        raise ValueError("Fit provenance or split lacks source-qualified fit groups")
    if set(expected_fit) != {"train", "valid"} or set(actual_fit) != {"train", "valid"}:
        raise ValueError("Fit groups must distinguish train and valid source namespaces")

    membership_hashes = provenance.get("fit_membership_sha256")
    if not isinstance(membership_hashes, dict):
        membership_hashes = provenance.get("fit_membership_identity")
    if not isinstance(membership_hashes, dict):
        raise ValueError("Fit provenance lacks ordered source membership hashes")

    component_of = {
        (source, int(item["row_group"])): int(item["component"])
        for source, nodes in split["sequences"].items()
        for item in nodes
    }
    expected_keys: set[tuple[str, int]] = set()
    actual_keys: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for source in ("train", "valid"):
        expected_groups = list(map(int, expected_fit[source]))
        observed_groups = list(map(int, actual_fit[source]))
        if len(expected_groups) != len(set(expected_groups)):
            raise ValueError(f"Frozen split has duplicate {source} fit groups")
        if len(observed_groups) != len(set(observed_groups)):
            raise ValueError(f"Fit provenance has duplicate {source} groups")
        if set(observed_groups) != set(expected_groups):
            raise ValueError(f"Fit provenance differs from frozen {source} membership")

        record = membership_hashes[source]
        if isinstance(record, dict):
            expected_hash = record.get("ordered_int64_le_sha256")
            expected_count = record.get("count")
        else:
            expected_hash = record
            expected_count = len(observed_groups)
        observed_hash = hashlib.sha256(
            np.asarray(observed_groups, dtype="<i8").tobytes()
        ).hexdigest()
        if expected_hash != observed_hash or expected_count != len(observed_groups):
            raise ValueError(f"Fit provenance {source} membership digest differs")

        expected_keys.update((source, group) for group in expected_groups)
        actual_keys.update((source, group) for group in observed_groups)
        counts[source] = len(observed_groups)

    evaluation_keys = {("valid", int(group)) for group in evaluation_groups}
    if actual_keys != expected_keys or actual_keys & evaluation_keys:
        raise ValueError("Source-qualified evaluation groups overlap the fit or fit sets changed")
    if provenance.get("evaluation_row_groups") is not None:
        observed_evaluation = list(map(int, provenance["evaluation_row_groups"]))
        if set(observed_evaluation) != set(evaluation_groups):
            raise ValueError("Fit provenance evaluation membership differs from the split")
    elif provenance.get("exposed_fold0_overlap_groups") != 0:
        raise ValueError("Readout provenance does not attest zero exposed fold-0 overlap")

    missing_components = (actual_keys | evaluation_keys) - component_of.keys()
    if missing_components:
        raise ValueError("Frozen split omits component labels for fit or evaluation groups")
    fit_components = {component_of[key] for key in actual_keys}
    evaluation_components = {component_of[key] for key in evaluation_keys}
    if fit_components & evaluation_components:
        raise ValueError("Fit and evaluation connected components overlap")
    return {
        "fit_groups_by_source": counts,
        "evaluation_groups": len(evaluation_keys),
        "fit_evaluation_source_qualified_overlap": 0,
        "fit_evaluation_component_overlap": 0,
        "evaluation_components": len(evaluation_components),
    }


def main() -> None:
    """Validate and score the preregistered H272 S4 cleanfold comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior results: {args.output}")
    if sha256_file(PREREG) != PREREG_SHA256:
        raise ValueError("The frozen S4 cleanfold grade preregistration changed")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_prediction_scoring":
        raise ValueError("The comparison is not preregistered")

    fit = prereg["fit_and_parent"]
    export = prereg["export"]
    evaluation_data = prereg["evaluation_data"]
    sources = prereg["scoring_sources"]
    artifacts = {
        "recipe": repo_file(prereg["recipe"]["path"], prereg["recipe"]["sha256"]),
        "source_lock": repo_file(
            prereg["recipe"]["source_lock_path"], prereg["recipe"]["source_lock_sha256"]
        ),
        "fit_attempt": repo_file(fit["fit_attempt_path"], fit["fit_attempt_sha256"]),
        "fit_authorization": repo_file(
            fit["fit_authorization_path"], fit["fit_authorization_sha256"]
        ),
        "fit_terminal_review": repo_file(
            fit["fit_terminal_review_path"], fit["fit_terminal_review_sha256"]
        ),
        "checkpoint": repo_file(fit["readout_checkpoint_path"], fit["readout_checkpoint_sha256"]),
        "fit_provenance": repo_file(fit["fit_provenance_path"], fit["fit_provenance_sha256"]),
        "parent_checkpoint": repo_file(
            fit["parent_checkpoint_path"], fit["parent_checkpoint_sha256"]
        ),
        "parent_provenance": repo_file(
            fit["parent_provenance_path"], fit["parent_provenance_sha256"]
        ),
        "export_attempt": repo_file(export["export_attempt_path"], export["export_attempt_sha256"]),
        "export_authorization": repo_file(
            export["export_authorization_path"], export["export_authorization_sha256"]
        ),
        "export_receipt": repo_file(export["export_receipt_path"], export["export_receipt_sha256"]),
        "split_manifest": repo_file(
            evaluation_data["split_manifest_path"], evaluation_data["split_manifest_sha256"]
        ),
        "target_cache": repo_file(
            evaluation_data["target_cache_path"], evaluation_data["target_cache_sha256"]
        ),
        "fit_helpers": repo_file(
            "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/grade_h233_readouts.py",
            sources["h233_source_qualified_helpers_sha256"],
        ),
        "pair_helpers": repo_file(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/grade_fixed_embedding_pair.py",
            sources["paired_comparison_helper_sha256"],
        ),
        "shared_blend_helper": repo_file(
            "research/campaign_2026-10-06/bandit_continuation/attention_training_H218/grade_fold0_reference_blends.py",
            sources["shared_component_blend_helper_sha256"],
        ),
        "official_scorer": repo_file(
            "research/campaign_2026-09-30/audit/campaign_score.py",
            sources["official_campaign_scorer_sha256"],
        ),
        "official_metric": repo_file(
            "src/wnn/gmetric.py", sources["official_metric_module_sha256"]
        ),
    }

    export_dir = (ROOT / export["path"]).resolve(strict=True)
    if not export_dir.is_relative_to(ROOT):
        raise ValueError("Prediction export directory escaped the repository")
    for name, digest in export["prediction_sha256"].items():
        artifacts[name] = repo_file(
            f"{export['path']}/{name}.npy",
            digest,
        )
    for name, digest in (
        ("row_groups", export["row_groups_sha256"]),
        ("evaluation_group_mask", export["evaluation_group_mask_sha256"]),
        ("evaluation_selected_mask", export["evaluation_selected_mask_sha256"]),
        ("scored_mask", export["scored_mask_sha256"]),
    ):
        artifacts[name] = repo_file(f"{export['path']}/{name}.npy", digest)

    fit_attempt = json.loads(artifacts["fit_attempt"].read_text(encoding="utf-8"))
    export_attempt = json.loads(artifacts["export_attempt"].read_text(encoding="utf-8"))
    fit_authorization = json.loads(artifacts["fit_authorization"].read_text(encoding="utf-8"))
    export_authorization = json.loads(artifacts["export_authorization"].read_text(encoding="utf-8"))
    fit_review = json.loads(artifacts["fit_terminal_review"].read_text(encoding="utf-8"))
    export_receipt = json.loads(artifacts["export_receipt"].read_text(encoding="utf-8"))
    recipe = json.loads(artifacts["recipe"].read_text(encoding="utf-8"))
    split = json.loads(artifacts["split_manifest"].read_text(encoding="utf-8"))
    fit_provenance = json.loads(artifacts["fit_provenance"].read_text(encoding="utf-8"))
    parent_provenance = json.loads(artifacts["parent_provenance"].read_text(encoding="utf-8"))

    if (
        fit_attempt.get("id") != "h272-parent-v2-cleanfold-fit-r1-20261008"
        or fit_attempt.get("status") != "finished"
        or fit_attempt.get("exit_status") != 0
        or export_attempt.get("id") != "h272-parent-v2-cleanfold-export-r1-20261008"
        or export_attempt.get("status") != "finished"
        or export_attempt.get("exit_status") != 0
        or fit_authorization.get("fit_authorized") is not True
        or fit_authorization.get("recipe_sha256") != prereg["recipe"]["sha256"]
        or export_authorization.get("export_authorized") is not True
        or export_authorization.get("selected_epoch") != 1
        or export_authorization.get("selected_checkpoint_sha256")
        != fit["readout_checkpoint_sha256"]
        or fit_review.get("status") != "fixed_epoch_terminal_checkpoint_pass"
        or fit_review.get("checkpoint_sha256") != fit["readout_checkpoint_sha256"]
        or fit_review.get("quality_grade") is not None
    ):
        raise ValueError("Fit/export terminal approvals do not match the preregistered endpoint")

    if (
        recipe.get("recipe_id") != prereg["recipe"]["id"]
        or recipe.get("screen_and_grader", {}).get("metric") != prereg["comparison"]["metric"]
        or recipe.get("screen_and_grader", {}).get("fixed_readout_alphas")
        != prereg["comparison"]["readout_alphas"]
        or recipe.get("screen_and_grader", {}).get("prediction_shape")
        != prereg["export"]["prediction_shape"]
        or recipe.get("parent", {}).get("checkpoint_sha256") != fit["parent_checkpoint_sha256"]
        or recipe.get("split_manifest", {}).get("sha256")
        != evaluation_data["split_manifest_sha256"]
    ):
        raise ValueError("Frozen recipe differs from the preregistered grade contract")

    if (
        export_receipt.get("status") != "cleanfold_predictions_exported_ungraded"
        or export_receipt.get("grade_started") is not False
        or export_receipt.get("recipe_id") != prereg["recipe"]["id"]
        or export_receipt.get("authorization_sha256") != export["export_authorization_sha256"]
        or export_receipt.get("readout_checkpoint_sha256") != fit["readout_checkpoint_sha256"]
        or export_receipt.get("parent_checkpoint_sha256") != fit["parent_checkpoint_sha256"]
        or export_receipt.get("prediction_shape") != export["prediction_shape"]
        or export_receipt.get("prediction_dtype") != export["prediction_dtype"]
        or export_receipt.get("prediction_sha256") != export["prediction_sha256"]
        or export_receipt.get("evaluation_groups") != 468
        or export_receipt.get("evaluation_group_ids_sha256")
        != export["evaluation_group_ids_sha256"]
    ):
        raise ValueError("The ungraded producer receipt differs from the frozen export identity")

    fit_helpers = load_module("h272_s4_fit_helpers", artifacts["fit_helpers"])
    pair_helpers = load_module("h272_s4_pair_helpers", artifacts["pair_helpers"])
    scorer = load_module("h272_s4_official_scorer", artifacts["official_scorer"])
    blend_helpers = load_module("h272_s4_blend_helpers", artifacts["shared_blend_helper"])
    if (
        prereg["comparison"]["metric"] != scorer.METRIC_ID
        or tuple(blend_helpers.ALPHAS) != tuple(prereg["comparison"]["readout_alphas"])
        or prereg["comparison"]["bootstrap"]["replicates"] != blend_helpers.BOOTSTRAP_REPLICATES
        or prereg["comparison"]["bootstrap"]["seed"] != blend_helpers.BOOTSTRAP_SEED
    ):
        raise ValueError("Scoring helpers no longer implement the preregistered metric or ladder")

    evaluation = split["evaluation"]["valid"]
    if len(evaluation) != prereg["exposure"]["evaluation_groups"]:
        raise ValueError("Frozen evaluation group count differs from the preregistration")
    evaluation_hash = hashlib.sha256(
        np.asarray(evaluation, dtype="<i8").tobytes()
    ).hexdigest()
    if evaluation_hash != export["evaluation_group_ids_sha256"]:
        raise ValueError("Evaluation IDs differ from the exporter receipt")
    fit_membership = validate_fit_membership(
        split, fit_provenance, evaluation, evaluation_data["split_manifest_sha256"]
    )
    parent_membership = validate_fit_membership(
        split, parent_provenance, evaluation, evaluation_data["split_manifest_sha256"]
    )
    if (
        fit_provenance.get("selected_epoch") != 1
        or fit_provenance.get("status") != "complete"
        or fit_provenance.get("checkpoint_sha256", {}).get("model_ep1.pt")
        != fit["readout_checkpoint_sha256"]
        or fit_provenance.get("recipe_id") != prereg["recipe"]["id"]
        or fit_provenance.get("recipe_sha256") != prereg["recipe"]["sha256"]
        or fit_provenance.get("source_lock_sha256")
        != prereg["recipe"]["source_lock_sha256"]
        or fit_provenance.get("parent_run") != recipe["parent"]["run"]
        or fit_provenance.get("parent_checkpoint_sha256")
        != fit["parent_checkpoint_sha256"]
        or parent_provenance.get("selected_epoch") != recipe["parent"]["epoch"]
        or parent_provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
        != fit["parent_checkpoint_sha256"]
        or fit_membership["fit_evaluation_source_qualified_overlap"] != 0
        or fit_membership["fit_evaluation_component_overlap"] != 0
        or parent_membership["fit_evaluation_source_qualified_overlap"] != 0
        or parent_membership["fit_evaluation_component_overlap"] != 0
    ):
        raise ValueError("Source-qualified fit ancestry or fixed epoch-1 provenance failed")

    targets = scorer.mmap_stored_npz(artifacts["target_cache"])
    row_groups = np.load(artifacts["row_groups"], mmap_mode="r", allow_pickle=False)
    group_mask = np.load(artifacts["evaluation_group_mask"], mmap_mode="r", allow_pickle=False)
    selected_mask = np.load(
        artifacts["evaluation_selected_mask"], mmap_mode="r", allow_pickle=False
    )
    scored_mask = np.load(artifacts["scored_mask"], mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(export["prediction_shape"])
    if (
        row_groups.shape != (expected_shape[0],)
        or row_groups.dtype != np.int64
        or group_mask.shape != (expected_shape[0],)
        or group_mask.dtype != np.bool_
        or selected_mask.shape != expected_shape[:2]
        or selected_mask.dtype != np.bool_
        or scored_mask.shape != expected_shape[:2]
        or scored_mask.dtype != np.bool_
    ):
        raise ValueError("Exported row index or mask shape/dtype differs from the contract")
    groups = row_groups.tolist()
    target_groups = targets["row_group"].tolist()
    if groups != list(range(1873)) or target_groups != groups:
        raise ValueError("Valid-source row groups are incomplete or reordered")
    expected_group_mask = np.isin(row_groups, np.asarray(evaluation, dtype=np.int64))
    if not np.array_equal(group_mask, expected_group_mask):
        raise ValueError("Evaluation group mask differs from the frozen valid-source groups")
    if not np.array_equal(scored_mask, targets["mask"]):
        raise ValueError("Exporter scored mask differs from the official valid target mask")
    expected_selected = scored_mask & group_mask[:, None]
    if not np.array_equal(selected_mask, expected_selected):
        raise ValueError("Evaluation selected mask differs from scored rows and frozen groups")

    arrays = {
        name: np.load(artifacts[name], mmap_mode="r", allow_pickle=False)
        for name in ("parent", "hidden_only", "multiplicative")
    }
    for values in arrays.values():
        blend_helpers.validate_prediction_array(values, expected_shape[0], expected_shape[1])

    alignment = fit_helpers.full_alignment(
        blend_helpers,
        evaluation,
        groups,
        groups,
        target_groups,
    )
    valid_components = {
        ("valid", item["row_group"]): item["component"] for item in split["sequences"]["valid"]
    }
    components = np.asarray([valid_components[row[0]] for row in alignment], dtype=np.int64)
    if len(np.unique(components)) != prereg["exposure"]["evaluation_components"]:
        raise ValueError("Evaluation component count differs from the frozen split")

    score_targets = {**targets, "mask": selected_mask}
    bootstrap = {
        "replicates": prereg["comparison"]["bootstrap"]["replicates"],
        "seed": prereg["comparison"]["bootstrap"]["seed"],
    }
    fixed_results = {}
    for mode in prereg["comparison"]["candidate_modes"]:
        rows = pair_helpers.score_pair(
            score_targets,
            arrays[mode],
            arrays["parent"],
            alignment,
            components,
            scorer,
            prereg["comparison"]["readout_alphas"],
            bootstrap,
        )
        fixed_results[mode] = []
        for row in rows:
            fixed_results[mode].append(
                {
                    "alpha_on_readout": row["weight_on_ple"],
                    "score": row["score"],
                    "delta_vs_parent": row["delta_wp_vs_linear"],
                    "paired_component_bootstrap_vs_parent": row["paired_component_bootstrap"],
                }
            )

    if fixed_results["hidden_only"][0]["score"] != fixed_results["multiplicative"][0]["score"]:
        raise ValueError("The fixed alpha-zero parent baseline differs across readout modes")
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "evaluated_exposed_discovery",
        "metric": scorer.METRIC_ID,
        "preregistration_sha256": PREREG_SHA256,
        "grader_sha256": sha256_file(Path(__file__).resolve()),
        "fit_membership": fit_membership,
        "parent_membership": parent_membership,
        "evaluation_group_count": len(evaluation),
        "evaluation_component_count": len(np.unique(components)),
        "official_selected_row_count": int(np.sum(selected_mask[:, 99:] & group_mask[:, None])),
        "fixed_comparisons_all_reported": fixed_results,
        "public_forecast": None,
        "weight_selection": False,
        "interpretation_limits": prereg["limits"],
        "artifact_sha256": {name: sha256_file(path) for name, path in artifacts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
