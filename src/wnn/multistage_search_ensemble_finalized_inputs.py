from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_ensemble_descendants import (
    read_candidate_request,
    validate_candidate_ancestry,
)
from wnn.multistage_search_ensemble_inputs import _validate_inner_prediction
from wnn.splits import CampaignSplit
from wnn.training.sequence_monitor_identity import membership_sha256

MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def regular_campaign_file(path: Path, label: str, limit: int = 16_000_000) -> Path:
    """Require a bounded, non-symlink campaign file before parsing its contents."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"{label} must be a bounded regular file")
    return path.resolve(strict=True)


def read_bound_campaign_reference(record: object, label: str) -> tuple[Path, dict[str, object]]:
    """Load and SHA-verify a prepared campaign path reference."""
    if not isinstance(record, Mapping):
        raise ValueError(f"campaign manifest omits its {label} reference")
    path = regular_campaign_file(Path(str(record.get("path"))), label)
    value = read_json(path)
    if record.get("sha256") != file_sha256(path):
        raise ValueError(f"campaign {label} bytes changed after preparation")
    if not isinstance(value, dict):
        raise ValueError(f"campaign {label} must contain a JSON object")
    return path, value


def load_finalized_candidates(
    run_dir: Path,
    references: Sequence[str],
    auxiliary_sha: str,
    primary_sha: str,
    split: CampaignSplit,
    expected_fit: list[list[object]],
    expected_control: list[list[object]],
) -> list[dict[str, object]]:
    """Validate finalizer sidecars and fixed-final checkpoints for Stage 4 use."""
    if (
        len(expected_fit) != 8_260
        or len(expected_control) != 2_065
        or not 2 <= len(references) <= 32
        or len(references) != len(set(references))
    ):
        raise ValueError("stage 4 requires 2-32 unique candidate references")
    run_root = run_dir.resolve(strict=True)
    names: set[str] = set()
    checkpoints: set[str] = set()
    candidates: list[dict[str, object]] = []
    for reference in references:
        name, separator, result_text = reference.partition("=")
        if not separator or not name or not result_text or name in names:
            raise ValueError("candidate reference must be unique NAME=trial_result.json")
        names.add(name)
        candidates.append(
            _load_finalized_candidate(
                run_root,
                name,
                Path(result_text),
                auxiliary_sha,
                primary_sha,
                split,
                expected_fit,
                expected_control,
                checkpoints,
            )
        )
    return candidates


def verify_finalized_candidate_inputs_unchanged(candidates: Sequence[Mapping[str, object]]) -> None:
    """Reject a result if any bound parent or artifact changed during the Optuna search."""
    for candidate in candidates:
        for path_key, sha_key in (
            ("trial_result_path_object", "trial_result_sha256"),
            ("trial_request_path_object", "trial_request_sha256"),
            ("config_path_object", "config_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
            ("prediction_manifest_path", "prediction_manifest_sha256"),
            ("inner_objective_path", "inner_objective_sha256"),
        ):
            path = Path(str(candidate[path_key]))
            if file_sha256(path) != candidate[sha_key]:
                raise ValueError(f"Stage 4 candidate input changed during search: {path}")


def _load_finalized_candidate(
    run_root: Path,
    name: str,
    result_path: Path,
    auxiliary_sha: str,
    primary_sha: str,
    split: CampaignSplit,
    expected_fit: list[list[object]],
    expected_control: list[list[object]],
    checkpoints: set[str],
) -> dict[str, object]:
    """Validate one COMPLETE fixed-final result and its auxiliary prediction artifacts."""
    result_path = regular_campaign_file(result_path, f"{name} trial result")
    if not result_path.is_relative_to(run_root):
        raise ValueError(f"candidate trial result escapes the campaign: {name}")
    result = read_json(result_path)
    if not isinstance(result, Mapping):
        raise ValueError(f"candidate {name} trial result must be a JSON object")
    selection = result.get("selected_endpoint")
    if (
        result.get("status") != "COMPLETE"
        or type(result.get("stage")) is not int
        or result["stage"] not in (1, 2, 3)
        or not isinstance(selection, Mapping)
        or selection.get("selection_role")
        != ("fixed_final" if result.get("stage") == 1 else "configured_final")
    ):
        raise ValueError(f"candidate {name} is not a completed fixed-final fit")
    fit_dir = Path(str(result.get("fit_output"))).resolve(strict=True)
    checkpoint = regular_campaign_file(
        Path(str(selection.get("checkpoint_path"))),
        f"{name} model",
        limit=MAX_CHECKPOINT_BYTES,
    )
    if checkpoint != (fit_dir / "model.pt").resolve(strict=True):
        raise ValueError(f"candidate {name} fixed-final checkpoint differs from its fit output")
    model_sha = file_sha256(checkpoint)
    if (
        result.get("fit_model_sha256") != model_sha
        or selection.get("checkpoint_sha256") != model_sha
        or model_sha in checkpoints
    ):
        raise ValueError(f"candidate {name} checkpoint hash is inconsistent or duplicated")
    checkpoints.add(model_sha)

    postprocess_dir = (
        result_path.parent if result["stage"] == 1 else fit_dir / "fixed_final_postprocess_r2"
    )
    objective_path = regular_campaign_file(
        postprocess_dir / "inner_objective.json", f"{name} inner objective"
    )
    prediction_dir = postprocess_dir / (
        "fixed_final_control_predictions" if result["stage"] == 1 else "control_predictions"
    )
    if prediction_dir.is_symlink() or not prediction_dir.is_dir():
        raise ValueError(f"candidate {name} control prediction directory is missing")
    prediction_path = regular_campaign_file(
        prediction_dir / "manifest.json", f"{name} prediction manifest"
    )
    objective, prediction = read_json(objective_path), read_json(prediction_path)
    if not isinstance(objective, Mapping) or not isinstance(prediction, Mapping):
        raise ValueError(f"candidate {name} objective and prediction manifest must be JSON objects")
    if file_sha256(prediction_path) != objective.get("prediction_manifest_sha256"):
        raise ValueError(f"candidate {name} prediction manifest differs from its objective")
    objective_sha = file_sha256(objective_path)
    identity, validated_model_sha = _validate_inner_prediction(
        objective,
        prediction,
        auxiliary_sha,
        auxiliary_sha,
        primary_sha,
        expected_fit,
        expected_control,
    )
    request_path, request, request_sha, config = read_candidate_request(
        run_root, result_path, result, name
    )
    validate_candidate_ancestry(
        request,
        result,
        identity,
        objective["prediction_producer"],
        config,
        split,
        expected_fit,
        expected_control,
        auxiliary_sha,
        primary_sha,
    )
    provenance = objective["producer"]["provenance"]
    identity_sha = canonical_json_sha256(identity)
    evaluation = objective.get("evaluation")
    result_objective = result.get("objective")
    metric = objective.get("objective")
    if (
        validated_model_sha != model_sha
        or provenance.get("identity_sha256") != identity_sha
        or result.get("profile_identity_sha256") != identity_sha
        or selection.get("identity_sha256") != identity_sha
        or not isinstance(evaluation, Mapping)
        or evaluation.get("members") != expected_control
        or objective.get("fit_members_sha256")
        != membership_sha256([tuple(member) for member in expected_fit])
        or objective.get("evaluation_members_sha256")
        != membership_sha256([tuple(member) for member in expected_control])
        or objective.get("selected_rows") != len(expected_control) * (SEQUENCE_LENGTH - WARMUP)
        or not isinstance(result_objective, Mapping)
        or result_objective.get("model_sha256") != model_sha
        or not isinstance(metric, Mapping)
        or result_objective.get("objective") != metric
        or selection.get("wp") != metric.get("value")
        or result.get("score_receipt_sha256") != objective_sha
    ):
        raise ValueError(f"candidate {name} result does not bind its exact inner control endpoint")
    if result.get("prediction_manifest_sha256") != file_sha256(prediction_path):
        raise ValueError(f"candidate {name} result prediction-manifest SHA differs")
    return {
        "name": name,
        "stage": result["stage"],
        "trial_number": result["trial_number"],
        "family": result["family"],
        "config_sha256": result["config_sha256"],
        "config_identity_sha256": request["config_identity_sha256"],
        "config_path": request["config_path"],
        "config_path_object": Path(str(request["config_path"])).resolve(strict=True),
        "trial_result_path": str(result_path),
        "trial_result_sha256": file_sha256(result_path),
        "trial_result_path_object": result_path,
        "trial_request_path": str(request_path),
        "trial_request_sha256": request_sha,
        "trial_request_path_object": request_path,
        "checkpoint_sha256": model_sha,
        "checkpoint_path": checkpoint,
        "prediction_manifest_sha256": file_sha256(prediction_path),
        "prediction_manifest_path": prediction_path,
        "inner_objective_sha256": objective_sha,
        "inner_objective_path": objective_path,
        "producer_identity_sha256": identity_sha,
        "objective": dict(metric),
        "predictions": {"directory": prediction_dir, "entries": prediction["sequences"]},
    }
