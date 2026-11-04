from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    canonical_json_sha256,
    file_sha256,
    read_json,
)


def admission_attempt(admission: Mapping[str, object], attempt_id: str) -> Mapping[str, object]:
    """Find the exact pre-registered stage-one attempt this external fit may satisfy."""
    rows = admission.get("preexisting_fit_attempts")
    matches = (
        [row for row in rows if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1 or matches[0].get("stage") != 1:
        raise ValueError("adoption requires one exact pre-registered stage-one fit attempt")
    return matches[0]


def validate_terminal_owner(
    attempt_id: str,
    attempt: Mapping[str, object],
    owner: Mapping[str, object],
    terminal: Mapping[str, object],
    owner_path: Path,
    output_dir: Path,
    deadline: Mapping[str, object],
) -> None:
    """Bind terminal evidence to the preserved original PID/birth/PGID receipt."""
    command = owner.get("command")
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    pid, birth, pgid = owner.get("pid"), owner.get("birth"), owner.get("pgid")
    started, owner_deadline = owner.get("started_at_epoch"), owner.get("deadline_epoch")
    finished, reserve = terminal.get("finished_at_epoch"), deadline.get("reserve_start_epoch")
    from wnn.multistage_search_runtime import process_identity

    live = process_identity(pid) if type(pid) is int else None
    if (
        owner.get("schema") != "wnn.multistage-external-worker.v1"
        or owner.get("attempt_id") != attempt_id
        or owner.get("stage") != 1
        or owner.get("mode") != "fit"
        or owner.get("output_dir") != str(output_dir)
        or not isinstance(command, list)
        or owner.get("command_sha256") != command_sha
        or type(pid) is not int
        or not isinstance(birth, str)
        or type(pgid) is not int
        or live == {"pid": pid, "birth": birth, "pgid": pgid}
        or terminal.get("schema") != "wnn.multistage-fit-terminal.v1"
        or terminal.get("status") != "artifacts_complete_exit_unknown"
        or terminal.get("attempt_id") != attempt_id
        or terminal.get("owner_receipt_sha256") != file_sha256(owner_path)
        or terminal.get("exit_code") is not None
        or terminal.get("exit_observation") != "outer_wrapper_returncode_not_persisted"
        or terminal.get("failure_logs_empty") is not True
        or not isinstance(started, (int, float))
        or not isinstance(owner_deadline, (int, float))
        or not isinstance(finished, (int, float))
        or not isinstance(reserve, (int, float))
        or finished < started
        or finished > owner_deadline
        or finished > reserve
        or finished - started > 7200
        or terminal.get("output_dir") != str(output_dir)
        or attempt.get("attempt_id") != attempt_id
        or attempt.get("stage") != 1
        or attempt.get("output_dir") != str(output_dir)
    ):
        raise ValueError(
            "terminal fit receipt does not prove the admitted original worker completed"
        )


def validate_profile_and_fit(
    attempt: Mapping[str, object],
    config: Mapping[str, object],
    config_path: Path,
    native: Mapping[str, object],
    native_path: Path,
    native_sha: str,
    approved: Mapping[str, object],
    approved_path: Path,
    output_dir: Path,
    admission: Mapping[str, object],
    terminal: Mapping[str, object],
) -> None:
    """Verify exact profile/config/data identity and the fixed-final model/index artifacts."""
    identity = native.get("identity")
    profile_data = identity.get("data") if isinstance(identity, Mapping) else None
    if not isinstance(identity, Mapping) or not isinstance(profile_data, Mapping):
        raise ValueError("native profile identity is incomplete")
    expected_raw_config = attempt.get("config_sha256")
    review = approved.get("external_review")
    review_sha = attempt.get("profile_review_sha256")
    review_path_value = (
        review.get("review_receipt_path", review.get("path"))
        if isinstance(review, Mapping)
        else None
    )
    review_sha_value = (
        review.get("review_receipt_sha256", review.get("sha256"))
        if isinstance(review, Mapping)
        else None
    )
    review_path = Path(str(review_path_value)) if isinstance(review_path_value, str) else Path(".")
    if (
        (isinstance(expected_raw_config, str) and file_sha256(config_path) != expected_raw_config)
        or canonical_json_sha256(config) != attempt.get("config_identity_sha256")
        or identity.get("config_sha256") != canonical_json_sha256(config)
        or identity.get("source_sha256") != attempt.get("training_source_sha256")
        or identity.get("data_sha256") != canonical_json_sha256(profile_data)
        or approved.get("approved") is not True
        or approved.get("identity") != identity
        or approved.get("receipt_schema") != "wnn.sequence-profile.v1"
        or native.get("approved") is not False
        or native.get("status") != "profile_complete"
        or not isinstance(review, Mapping)
        or review.get("native_profile_sha256") != native_sha
        or not isinstance(review_sha, str)
        or review_sha_value != review_sha
        or review_path.is_symlink()
        or not review_path.is_file()
        or file_sha256(review_path) != review_sha
        or file_sha256(native_path) != native_sha
    ):
        raise ValueError("preexisting profile, config, or source identity is not exact")
    from wnn.training.sequence_provenance import validate_profile_receipt

    validate_profile_receipt(approved_path, dict(identity))
    auxiliary = admission["data"]["auxiliary_manifest"]
    if profile_data.get("split_sha256") != auxiliary.get("sha256"):
        raise ValueError("preexisting profile used a different auxiliary split")
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members

    split = load_campaign_split(
        Path(str(auxiliary["path"])), Path(str(admission["data"]["data_root"]))
    )
    fit_members, control_members = auxiliary_members(split)
    runtime = identity.get("runtime")
    precision = runtime.get("precision") if isinstance(runtime, Mapping) else None
    if (
        profile_data.get("fit_members") != [list(item) for item in fit_members]
        or profile_data.get("evaluation_members") != [list(item) for item in control_members]
        or profile_data.get("primary_split_sha256") != admission["data"]["primary_split_sha256"]
        or profile_data.get("evaluation_role") != "control"
        or profile_data.get("evaluation_source") != "train"
        or profile_data.get("evaluation_mask") != "all_after_warmup"
        or profile_data.get("training_mask") != "all_after_warmup"
        or profile_data.get("teacher") is not None
        or profile_data.get("initializer") is not None
        or config.get("precision_mode") != "fp32"
        or config.get("epochs") != 4
        or not isinstance(runtime, Mapping)
        or runtime.get("world_size") != 1
        or runtime.get("matmul_precision") != "highest"
        or runtime.get("matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("autocast") is not False
        or runtime.get("amp_scaler") is not False
        or not isinstance(precision, Mapping)
        or precision.get("mode") != "fp32"
    ):
        raise ValueError(
            "preexisting fit is not the exact label-only strict-FP32 train/control recipe"
        )

    model_path, index_path, status_path = (
        output_dir / "model.pt",
        output_dir / "epoch_index.json",
        output_dir / "terminal_status.json",
    )
    if any(path.is_symlink() for path in (model_path, index_path, status_path)):
        raise ValueError("completed fit model, index, and status must be regular files")
    status, index = read_json(status_path), read_json(index_path)
    model_sha, index_file_sha = file_sha256(model_path), file_sha256(index_path)
    index_sha = index.get("index_sha256")
    from wnn.training.sequence_epoch_prediction import validate_index

    if not isinstance(index_sha, str):
        raise ValueError("completed fit epoch index has no internal digest")
    validate_index(index, index_sha)
    import torch

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    provenance_profile = (
        provenance.get("profile_receipt") if isinstance(provenance, Mapping) else None
    )
    if (
        status.get("status") != "completed"
        or status.get("completion_kind") != "max_epochs"
        or status.get("model_sha256") != model_sha
        or status.get("epoch_index_sha256") != index_sha
        or status.get("identity") != identity
        or index.get("identity") != identity
        or index.get("completed_epochs") != 4
        or index.get("global_step") != 2640
        or payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("config") != dict(config)
        or payload.get("data") != profile_data
        or payload.get("completed_epochs") != 4
        or payload.get("epoch") != 3
        or not isinstance(provenance, Mapping)
        or provenance.get("identity") != identity
        or not isinstance(provenance_profile, Mapping)
        or provenance_profile.get("sha256") != file_sha256(approved_path)
        or provenance.get("global_updates") != 2640
        or status.get("global_updates") != 2640
        or terminal.get("global_updates") != 2640
        or terminal.get("model_sha256") != model_sha
        or terminal.get("epoch_index_sha256") != index_sha
        or terminal.get("epoch_index_file_sha256") != index_file_sha
        or terminal.get("terminal_status_sha256") != file_sha256(status_path)
    ):
        raise ValueError("fixed-final checkpoint, terminal status, and epoch index do not agree")
