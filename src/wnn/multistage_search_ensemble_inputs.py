from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json


def load_qualified_candidates(
    run_dir: Path,
    references: Sequence[str],
    auxiliary_sha: str,
    split_sha: str,
    primary_sha: str,
    expected_fit: list[list[object]],
    expected_control: list[list[object]],
) -> list[dict[str, object]]:
    """Load unique fitted candidates whose sidecars prove exact inner-control ancestry."""
    if len(references) != len(set(references)):
        raise ValueError("stage 4 candidate references must be unique")
    candidates: list[dict[str, object]] = []
    checkpoint_hashes: set[str] = set()
    for reference in references:
        name, separator, path_text = reference.partition("=")
        if not separator or not name or not path_text:
            raise ValueError("candidate reference must use NAME=trial_result.json")
        result_path = Path(path_text).resolve(strict=True)
        if not result_path.is_relative_to(run_dir.resolve()):
            raise ValueError("stage 4 candidates must be produced inside this campaign")
        result = read_json(result_path)
        fit_dir = Path(str(result["fit_output"])).resolve(strict=True)
        if not fit_dir.is_relative_to(run_dir.resolve()):
            raise ValueError("stage 4 fit output escapes the campaign directory")
        objective_path = fit_dir / "inner_objective.json"
        prediction_dir = fit_dir / "selected_control_predictions"
        prediction_path = prediction_dir / "manifest.json"
        objective, prediction = read_json(objective_path), read_json(prediction_path)
        identity, checkpoint_sha = _validate_inner_prediction(
            objective,
            prediction,
            auxiliary_sha,
            split_sha,
            primary_sha,
            expected_fit,
            expected_control,
        )
        if file_sha256(prediction_path) != objective.get("prediction_manifest_sha256"):
            raise ValueError(f"candidate {name} prediction sidecar hash changed")
        selection = result.get("selected_endpoint")
        if not isinstance(selection, Mapping):
            raise ValueError(f"candidate {name} trial result omits its selected endpoint")
        checkpoint_path = Path(str(selection.get("checkpoint_path"))).resolve(strict=True)
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"candidate {name} selected checkpoint identity changed")
        if checkpoint_sha in checkpoint_hashes:
            raise ValueError("stage 4 pool contains duplicate checkpoint identities")
        checkpoint_hashes.add(checkpoint_sha)
        candidates.append(
            {
                "name": name,
                "trial_result_path": str(result_path),
                "trial_result_sha256": file_sha256(result_path),
                "checkpoint_sha256": checkpoint_sha,
                "prediction_manifest_sha256": file_sha256(prediction_path),
                "producer_identity_sha256": canonical_json_sha256(identity),
                "objective": objective["objective"],
                "predictions": {"directory": prediction_dir, "entries": prediction["sequences"]},
            }
        )
    return candidates


def _validate_inner_prediction(
    objective: Mapping[str, object],
    prediction: Mapping[str, object],
    auxiliary_sha: str,
    split_sha: str,
    primary_sha: str | None,
    expected_fit: list[list[object]],
    expected_control: list[list[object]],
) -> tuple[dict[str, object], str]:
    evaluation = objective.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("candidate objective evaluation metadata is malformed")
    if (
        objective.get("schema") != "wnn.multistage-inner-objective.v1"
        or objective.get("status") != "complete"
        or objective.get("scope") not in (None, "inner_train_proxy")
        or objective.get("split_sha256") != auxiliary_sha
        or objective.get("primary_split_sha256") != primary_sha
        or evaluation.get("scope") != "inner_train_proxy"
        or evaluation.get("source") != "train"
        or evaluation.get("row_mask") != "all_after_warmup AND row_index>=99"
        or evaluation.get("is_scored_used") is not False
    ):
        raise ValueError("candidate objective is not the required source-qualified inner proxy")
    producer, prediction_producer = objective.get("producer"), objective.get("prediction_producer")
    if not isinstance(producer, Mapping) or not isinstance(prediction_producer, Mapping):
        raise ValueError("candidate objective omits its prediction producer identity")
    provenance = producer.get("provenance")
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    if not isinstance(identity, dict):
        raise ValueError("candidate producer identity is malformed")
    identity_sha = canonical_json_sha256(identity)
    data = identity.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("candidate producer data identity is malformed")
    if (
        provenance.get("identity_sha256") != identity_sha
        or data.get("split_sha256") != auxiliary_sha
        or data.get("primary_split_sha256") != primary_sha
        or data.get("fit_members") != expected_fit
        or data.get("evaluation_members") != expected_control
        or data.get("evaluation_source") != "train"
        or producer.get("checkpoint_sha256") != objective.get("model_sha256")
        or producer.get("split_sha256") != auxiliary_sha
        or producer.get("fit_members_sha256") != objective.get("fit_members_sha256")
    ):
        raise ValueError("candidate run ancestry includes held-out or nontrain members")
    if (
        prediction.get("schema") != "sequence-predictions-v1"
        or prediction.get("complete") is not True
        or prediction.get("membership") != "evaluation"
        or prediction.get("split_sha256") != auxiliary_sha
        or prediction.get("producer") != prediction_producer
        or prediction_producer.get("checkpoint_sha256") != objective.get("model_sha256")
        or prediction_producer.get("fit_members") != expected_fit
    ):
        raise ValueError("prediction export does not match its qualified objective producer")
    entries = prediction.get("sequences")
    observed = (
        [[entry.get("split"), entry.get("row_group")] for entry in entries]
        if isinstance(entries, list) and all(isinstance(entry, Mapping) for entry in entries)
        else []
    )
    if observed != expected_control:
        raise ValueError("candidate prediction membership does not cover the inner control")
    checkpoint_sha = objective.get("model_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("candidate checkpoint SHA256 is malformed")
    return identity, checkpoint_sha
