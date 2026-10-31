from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_reviewed_successor_validation import (
    validate_initializer,
    validate_profile_members,
    validate_stage3_teacher,
)
from wnn.multistage_search_successor_fit_exit import validate_fit_exit
from wnn.multistage_search_successor_review import validate_successor_review
from wnn.splits import load_campaign_split
from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members
from wnn.training.sequence_epoch_prediction import validate_index
from wnn.training.sequence_provenance import validate_profile_receipt


def validate_successor_fit(
    run_dir: Path,
    trial_dir: Path,
    fit_dir: Path,
    request: Mapping[str, object],
    config: Mapping[str, object],
    split_path: Path,
    data_root: Path,
    stage: int,
    trial_number: int,
) -> dict[str, object]:
    """Verify review, ancestry, configured-final checkpoint, and exact fit exit."""
    import torch

    profile_dir = trial_dir / "profile"
    approval = read_json(profile_dir / "approval_record.json")
    approved_path = Path(str(approval["approved_profile_path"])).resolve(strict=True)
    native_path = Path(str(request["profile_receipt_path"])).resolve(strict=True)
    review_path = Path(str(approval["review_receipt_path"])).resolve(strict=True)
    request_path = trial_dir / "trial_request.json"
    native, approved, review = (
        read_json(native_path),
        read_json(approved_path),
        read_json(review_path),
    )
    identity = native.get("identity")
    external_review = approved.get("external_review")
    native_sha, approved_sha, review_sha = (
        file_sha256(native_path),
        file_sha256(approved_path),
        file_sha256(review_path),
    )
    review_proof = validate_successor_review(
        run_dir, trial_dir, stage, trial_number, request, review_path
    )
    if (
        not isinstance(identity, Mapping)
        or native.get("approved") is not False
        or approved.get("approved") is not True
        or approved.get("identity") != identity
        or approval.get("stage") != stage
        or approval.get("trial_number") != trial_number
        or approval.get("native_profile_sha256") != native_sha
        or approval.get("approved_profile_sha256") != approved_sha
        or approval.get("review_receipt_path") != str(review_path)
        or approval.get("review_receipt_sha256") != review_sha
        or approval.get("review_revision") != review_proof["revision"]
        or approval.get("review_policy_sha256") != review_proof["policy_sha256"]
        or approval.get("reviewer_script_sha256") != review_proof["checker_sha256"]
        or request.get("profile_receipt_sha256") != native_sha
        or request.get("profile_identity_sha256") != canonical_json_sha256(identity)
        or not isinstance(external_review, Mapping)
        or external_review.get("native_profile_sha256") != native_sha
        or external_review.get("review_receipt_path") != str(review_path)
        or external_review.get("review_receipt_sha256") != review_sha
        or external_review.get("review_revision") != review_proof["revision"]
        or external_review.get("review_policy_sha256") != review_proof["policy_sha256"]
        or external_review.get("reviewer_script_sha256") != review_proof["checker_sha256"]
        or review.get("status") != "approved"
        or review.get("stage") != stage
        or review.get("trial_number") != trial_number
        or review.get("native_profile_sha256") != native_sha
        or review.get("trial_request_sha256") != file_sha256(request_path)
        or identity.get("config_sha256") != request.get("config_identity_sha256")
        or identity.get("source_sha256") != request.get("source_sha256")
    ):
        raise ValueError("successor approval/review no longer binds this exact native profile")
    validate_profile_receipt(approved_path, dict(identity))
    split = load_campaign_split(split_path, data_root)
    fit_members, control_members = auxiliary_members(split)
    data = identity.get("data")
    validate_profile_members(data, split, fit_members, control_members)
    if stage == 2:
        validate_initializer(request, config, identity, data, fit_members, control_members)
    else:
        validate_stage3_teacher(request, config, identity, data, split, fit_members)
    checkpoint = fit_dir / "model.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError("successor final checkpoint is missing or unsafe")
    terminal = read_json(fit_dir / "terminal_status.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    fit_model_sha = file_sha256(checkpoint)
    profile_ref = provenance.get("profile_receipt") if isinstance(provenance, Mapping) else None
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("config") != dict(config)
        or payload.get("data") != data
        or not isinstance(provenance, Mapping)
        or provenance.get("identity") != identity
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
        or provenance.get("completed_epochs") != config.get("epochs")
        or provenance.get("requested_epochs") != config.get("epochs")
        or not isinstance(profile_ref, Mapping)
        or profile_ref.get("sha256") != approved_sha
        or payload.get("completed_epochs") != config.get("epochs")
        or payload.get("epoch") != int(config["epochs"]) - 1
        or terminal.get("status") != "completed"
        or terminal.get("completion_kind") != "max_epochs"
        or terminal.get("completed_epochs") != config.get("epochs")
        or terminal.get("requested_epochs") != config.get("epochs")
        or terminal.get("model_sha256") != fit_model_sha
        or terminal.get("identity") != identity
    ):
        raise ValueError("successor checkpoint is not the reviewed configured-final fit")
    index_path = fit_dir / "epoch_index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise FileNotFoundError("successor final epoch index is missing or unsafe")
    index = read_json(index_path)
    index_sha = index.get("index_sha256")
    if not isinstance(index_sha, str):
        raise ValueError("successor fixed-final checkpoint lacks its native epoch index")
    validate_index(index, index_sha)
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("successor native epoch index contains no retained checkpoints")
    last_epoch_payload: Mapping[str, object] | None = None
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("successor native epoch index entry is malformed")
        relative = Path(str(entry["checkpoint_path"]))
        epoch_checkpoint = fit_dir / relative
        if epoch_checkpoint.is_symlink() or not epoch_checkpoint.is_file():
            raise FileNotFoundError("successor indexed epoch checkpoint is missing or unsafe")
        if not epoch_checkpoint.resolve(strict=True).is_relative_to(fit_dir.resolve()):
            raise ValueError("successor indexed epoch checkpoint escapes its fit directory")
        if file_sha256(epoch_checkpoint) != entry.get("checkpoint_sha256"):
            raise ValueError(
                "successor indexed epoch checkpoint bytes differ from the native index"
            )
        epoch_payload = torch.load(epoch_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(epoch_payload, Mapping):
            raise ValueError("successor indexed epoch checkpoint payload is malformed")
        epoch_provenance = epoch_payload.get("provenance")
        if (
            epoch_payload.get("checkpoint_schema") != "wnn.sequence-epoch.v1"
            or epoch_payload.get("identity") != identity
            or epoch_payload.get("config") != dict(config)
            or epoch_payload.get("data") != data
            or epoch_payload.get("epoch") != entry.get("epoch")
            or epoch_payload.get("completed_epochs") != entry.get("completed_epochs")
            or epoch_payload.get("global_step") != entry.get("global_step")
            or not isinstance(epoch_provenance, Mapping)
            or epoch_provenance.get("identity") != identity
            or epoch_provenance.get("normalization") != provenance.get("normalization")
            or epoch_provenance.get("completed_epochs") != entry.get("completed_epochs")
            or epoch_provenance.get("global_updates") != entry.get("global_step")
            or not isinstance(epoch_provenance.get("profile_receipt"), Mapping)
            or epoch_provenance["profile_receipt"].get("sha256") != approved_sha
        ):
            raise ValueError(
                "successor indexed epoch payload differs from its exact native boundary"
            )
        last_epoch_payload = epoch_payload
    final_model, epoch_model = payload.get("model"), last_epoch_payload.get("model")
    if (
        not isinstance(final_model, Mapping)
        or not isinstance(epoch_model, Mapping)
        or set(final_model) != set(epoch_model)
        or any(
            not isinstance(final_model[name], torch.Tensor)
            or not isinstance(epoch_model[name], torch.Tensor)
            or final_model[name].dtype != epoch_model[name].dtype
            or final_model[name].shape != epoch_model[name].shape
            or not torch.equal(final_model[name], epoch_model[name])
            for name in final_model
        )
    ):
        raise ValueError("configured-final model weights differ from the last indexed epoch state")
    if (
        index.get("identity") != identity
        or index.get("completed_epochs") != config.get("epochs")
        or index.get("global_step") != provenance.get("global_updates")
        or provenance.get("epoch_index_sha256") != index_sha
        or terminal.get("global_updates") != provenance.get("global_updates")
        or terminal.get("epoch_index_sha256") != index_sha
        or entries[-1].get("epoch") != payload.get("epoch")
        or entries[-1].get("completed_epochs") != payload.get("completed_epochs")
        or entries[-1].get("global_step") != provenance.get("global_updates")
    ):
        raise ValueError("successor native terminal/index misses the configured final boundary")
    for name in ("monitor_events.jsonl", "monitor_evaluations.jsonl", "metrics.jsonl"):
        path = fit_dir / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"successor fit is missing native {name}")
    fit_receipt = validate_fit_exit(
        run_dir,
        trial_dir,
        fit_dir,
        request,
        approved_path,
        review_sha,
        f"stage{stage}-trial-{trial_number:06d}",
        stage,
        trial_number,
    )
    return {
        "identity": dict(identity),
        "payload": dict(payload),
        "native_profile_sha256": native_sha,
        "approved_profile_sha256": approved_sha,
        "review": review_proof,
        "fit_receipt": fit_receipt,
        "epoch_index_sha256": index_sha,
        "epoch_index_file_sha256": file_sha256(index_path),
    }


def validate_terminal_event(
    fit_dir: Path,
    payload: Mapping[str, object],
    objective: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    """Match the Source6 fixed-final objective to the terminal native train-control event."""
    import json

    events = [
        json.loads(line)
        for line in (fit_dir / "monitor_evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("kind") == "evaluation"
    ]
    if not matches:
        raise ValueError("successor fit has no native control evaluation curve")
    final = matches[-1]
    context, metrics = final.get("context"), final.get("metrics")
    control = context.get("control") if isinstance(context, Mapping) else None
    value = objective.get("objective")
    data = identity.get("data")
    provenance = payload.get("provenance")
    if (
        not isinstance(control, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(value, Mapping)
    ):
        raise ValueError("terminal control evaluation is malformed")
    if (
        final.get("epoch") != payload.get("completed_epochs")
        or final.get("step")
        != (provenance.get("global_updates") if isinstance(provenance, Mapping) else None)
        or control.get("role") != "control"
        or control.get("scope") != "inner_train_proxy"
        or control.get("source") != "train"
        or control.get("split_sha256") != data.get("split_sha256")
        or control.get("primary_split_sha256") != data.get("primary_split_sha256")
        or control.get("row_mask") != "all_after_warmup"
        or control.get("mask_definition")
        != "all rows after row 99 warmup; actual fitting objective mask"
        or control.get("member_count") != objective.get("sequence_count")
        or control.get("member_sha256") != objective.get("evaluation_members_sha256")
        or control.get("selected_rows") != objective.get("selected_rows")
        or control.get("is_scored_used") not in (None, False)
    ):
        raise ValueError("terminal native control context differs from the profiled fit")
    for key, name in (("control/wp", "value"), ("control/wp/t0", "t0"), ("control/wp/t1", "t1")):
        if abs(float(metrics[key]) - float(value[name])) > 1e-7:
            raise ValueError("standalone Source6 score differs from terminal native control event")
