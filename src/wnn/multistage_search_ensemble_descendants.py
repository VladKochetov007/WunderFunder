from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.splits import CampaignSplit


def read_candidate_request(
    run_root: Path, result_path: Path, result: Mapping[str, object], name: str
) -> tuple[Path, dict[str, object], str, dict[str, object]]:
    """Bind a finalized result to its immutable request and normalized model config."""
    request_ref = result.get("trial_request_path")
    request_path = (
        Path(str(request_ref))
        if isinstance(request_ref, str)
        else result_path.parent / "trial_request.json"
    )
    request_path = _regular_file(request_path, f"{name} trial request", 8_000_000)
    if not request_path.is_relative_to(run_root):
        raise ValueError(f"candidate {name} request escapes the campaign")
    request = read_json(request_path)
    objective = result.get("objective")
    if not isinstance(request, Mapping) or not isinstance(objective, Mapping):
        raise ValueError(f"candidate {name} request or result objective is malformed")
    request_sha = file_sha256(request_path)
    identity_sha = result.get("profile_identity_sha256")
    if (
        result.get("trial_request_sha256") is not None
        and result.get("trial_request_sha256") != request_sha
    ):
        raise ValueError(f"candidate {name} request SHA differs from its result")
    if (
        request.get("stage") != result.get("stage")
        or request.get("trial_number") != result.get("trial_number")
        or request.get("family") != result.get("family")
        or request.get("config_sha256") != result.get("config_sha256")
        or request.get("profile_identity_sha256") != identity_sha
        or request.get("primary_split_sha256") != objective.get("primary_split_sha256")
        or request.get("auxiliary_split_sha256") != objective.get("split_sha256")
        or request.get("objective_scope") != "inner_train_proxy"
        or request.get("primary_valid_used") is not False
    ):
        raise ValueError(f"candidate {name} request does not bind its result and inner scope")
    from wnn.training.sequence_config import load_config

    config_path = _regular_file(Path(str(request.get("config_path"))), f"{name} config", 4_000_000)
    if not config_path.is_relative_to(run_root) or file_sha256(config_path) != request.get(
        "config_sha256"
    ):
        raise ValueError(f"candidate {name} config differs from its campaign request")
    config = load_config(config_path)
    producer = objective.get("prediction_producer")
    provenance = producer.get("provenance") if isinstance(producer, Mapping) else None
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    expected_config = producer.get("config") if isinstance(producer, Mapping) else None
    scored_producer = objective.get("producer")
    scored_provenance = (
        scored_producer.get("provenance") if isinstance(scored_producer, Mapping) else None
    )
    scored_identity = (
        scored_provenance.get("identity") if isinstance(scored_provenance, Mapping) else None
    )
    if (
        not isinstance(identity, Mapping)
        or not isinstance(scored_identity, Mapping)
        or not isinstance(expected_config, Mapping)
        or identity != scored_identity
        or config != expected_config
        or not isinstance(config.get("model"), Mapping)
        or config["model"].get("architecture") != request.get("family")
        or request.get("config_identity_sha256") != canonical_json_sha256(config)
        or identity.get("config_sha256") != canonical_json_sha256(config)
    ):
        raise ValueError(f"candidate {name} normalized config differs from producer identity")
    return request_path, dict(request), request_sha, config


def validate_candidate_ancestry(
    request: Mapping[str, object],
    result: Mapping[str, object],
    identity: Mapping[str, object],
    producer: Mapping[str, object],
    config: Mapping[str, object],
    split: CampaignSplit,
    expected_fit: list[list[object]],
    expected_control: list[list[object]],
    auxiliary_sha: str,
    primary_sha: str,
) -> None:
    """Allow only direct Stage1 fits or ancestry-validated auxiliary Stage2/3 descendants."""
    data = identity.get("data")
    if (
        not isinstance(data, Mapping)
        or len(expected_fit) != 8_260
        or len(expected_control) != 2_065
    ):
        raise ValueError("candidate ancestry does not use the frozen 8260/2065 cohorts")
    if (
        data.get("fit_members") != expected_fit
        or data.get("evaluation_members") != expected_control
        or data.get("split_sha256") != auxiliary_sha
        or data.get("primary_split_sha256") != primary_sha
        or identity.get("config_sha256") != canonical_json_sha256(config)
        or producer.get("teacher") != data.get("teacher")
    ):
        raise ValueError("candidate fit/control membership differs from the frozen cohorts")
    fit_members = [tuple(member) for member in expected_fit]
    control_members = [tuple(member) for member in expected_control]
    stage = result.get("stage")
    if stage == 1:
        if (
            data.get("teacher") is not None
            or data.get("initializer") is not None
            or producer.get("teacher") is not None
            or config.get("kd_weight") != 0
            or config.get("hint_weight") != 0
            or not isinstance(config.get("model"), Mapping)
            or config["model"].get("hint_dim") != 0
        ):
            raise ValueError("stage1 ensemble parents must be direct label-only fixed-final fits")
    elif stage == 2:
        if data.get("teacher") is not None:
            raise ValueError("Stage2 ensemble parents cannot carry unbound teacher ancestry")
        from wnn.multistage_search_reviewed_successor_validation import validate_initializer

        validate_initializer(request, config, identity, data, fit_members, control_members)
    elif stage == 3:
        if data.get("initializer") is not None:
            raise ValueError("Stage3 pair recipe does not admit an initializer")
        # This recursively verifies the canonical teacher cache and every weighted-pool parent.
        from wnn.multistage_search_reviewed_successor_validation import validate_stage3_teacher

        validate_stage3_teacher(request, config, identity, data, split, fit_members)
    else:
        raise ValueError("candidate stage is outside the frozen Stage1-3 ancestry graph")


def _regular_file(path: Path, label: str, limit: int) -> Path:
    """Resolve a bounded regular request/config file without following its final symlink."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"{label} must be a bounded regular file")
    return path.resolve(strict=True)
