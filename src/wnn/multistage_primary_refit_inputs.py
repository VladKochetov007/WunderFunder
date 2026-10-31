from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import file_sha256, read_json
from wnn.multistage_search_ensemble_finalized_inputs import (
    load_finalized_candidates,
    read_bound_campaign_reference,
    regular_campaign_file,
)
from wnn.splits import CampaignSplit
from wnn.training.sequence_cache_precision import validate_precision_contract
from wnn.training.sequence_config import load_config


def _strict_fp32(
    config: Mapping[str, object], provenance: Mapping[str, object], runtime: Mapping[str, object]
) -> bool:
    try:
        validate_precision_contract(dict(config), dict(provenance), dict(runtime))
    except ValueError:
        return False
    return (
        config.get("precision_mode") == "fp32"
        and runtime.get("matmul_precision") == "highest"
        and runtime.get("matmul_tf32") is False
        and runtime.get("cudnn_tf32") is False
        and runtime.get("autocast") is False
        and runtime.get("amp_scaler") is False
    )


def _load_raw_and_normalized(
    candidate: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(str(candidate["config_path_object"]))
    raw = read_json(path)
    normalized = load_config(path)
    objective = read_json(Path(str(candidate["inner_objective_path"])))
    result = read_json(Path(str(candidate["trial_result_path"])))
    producer = objective.get("producer")
    provenance = producer.get("provenance") if isinstance(producer, Mapping) else None
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    data = identity.get("data") if isinstance(identity, Mapping) else None
    runtime = identity.get("runtime") if isinstance(identity, Mapping) else None
    model = normalized.get("model")
    if (
        result.get("stage") != 1
        or result.get("selected_endpoint", {}).get("selection_role") != "fixed_final"
        or not isinstance(identity, Mapping)
        or not isinstance(data, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(runtime, Mapping)
        or data.get("teacher") is not None
        or data.get("initializer") is not None
        or not isinstance(producer, Mapping)
        or producer.get("teacher") is not None
        or normalized.get("kd_weight") != 0
        or normalized.get("hint_weight") != 0
        or not isinstance(model, Mapping)
        or model.get("hint_dim") != 0
        or not _strict_fp32(normalized, provenance, runtime)
        or not isinstance(objective.get("objective"), Mapping)
        or objective["objective"].get("name") != "wp"
    ):
        raise ValueError("parent is not a direct strict-FP32 fixed-final Stage1 label fit")
    return raw, normalized


def validate_and_rank_stage1(
    run_dir: Path,
    stage4_path: Path,
    primary_sha: str,
    auxiliary_sha: str,
    split: CampaignSplit,
) -> tuple[list[dict[str, object]], str]:
    """Validate all direct Stage1 inputs and order them by fixed-final inner WP only."""
    result = read_json(regular_campaign_file(stage4_path, "Stage4 result"))
    request_path = regular_campaign_file(
        stage4_path.parent / "runner_request.json", "Stage4 request"
    )
    request = read_json(request_path)
    run_path = regular_campaign_file(run_dir / "run.json", "campaign run manifest")
    run = read_json(run_path)
    _, admission = read_bound_campaign_reference(run.get("admission"), "admission")
    _, contract = read_bound_campaign_reference(run.get("contract"), "contract")
    if (
        result.get("status") != "complete"
        or result.get("scope") != "inner_train_proxy"
        or result.get("primary_split_sha256") != primary_sha
        or result.get("auxiliary_split_sha256") != auxiliary_sha
        or result.get("evaluation_source") != "train"
        or request.get("run_manifest_sha256") != file_sha256(run_path)
        or request.get("admission_sha256") != run.get("admission", {}).get("sha256")
        or request.get("contract_sha256") != run.get("contract", {}).get("sha256")
        or request.get("primary_split_sha256") != primary_sha
        or request.get("auxiliary_split_sha256") != auxiliary_sha
        or request.get("used_primary_valid") is not False
        or not isinstance(admission.get("data"), Mapping)
        or admission["data"].get("primary_split_sha256") != primary_sha
        or not isinstance(contract, Mapping)
    ):
        raise ValueError("Stage4 result/request is not bound to the frozen train-only campaign")
    inventory = request.get("candidate_trial_results")
    members = result.get("ensemble", {}).get("members")
    if not isinstance(inventory, list) or not isinstance(members, list):
        raise ValueError("Stage4 candidate inventory is malformed")
    member_by_path = {
        str(item.get("trial_result_path")): item for item in members if isinstance(item, Mapping)
    }
    if len(member_by_path) != len(members) or len(inventory) != len(members):
        raise ValueError("Stage4 result does not retain one member per candidate input")
    stage1: list[dict[str, object]] = []
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Stage4 candidate reference is malformed")
        path, digest = item.get("path"), item.get("sha256")
        bound = member_by_path.get(str(path))
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(bound, Mapping)
            or bound.get("trial_result_sha256") != digest
            or any(item.get(key) != bound.get(key) for key in ("stage", "trial_number", "family"))
        ):
            raise ValueError("Stage4 candidate/result identities disagree")
        if item.get("stage") == 1:
            stage1.append(item)
    if not 2 <= len(stage1) <= 32:
        raise ValueError("Stage4 has fewer than two direct Stage1 candidates")
    references = [f"{item['name']}={item['path']}" for item in stage1]
    fit = [["train", group] for group in split.train_groups]
    control = [["train", group] for group in split.evaluation_groups]
    candidates = load_finalized_candidates(
        run_dir, references, auxiliary_sha, primary_sha, split, fit, control
    )
    by_path = {
        str(Path(str(item["trial_result_path"])).resolve(strict=True)): item for item in candidates
    }
    ranked: list[dict[str, object]] = []
    for item in stage1:
        candidate = by_path.get(str(Path(str(item["path"])).resolve(strict=True)))
        if candidate is None:
            raise ValueError("validated Stage1 candidate lost its result path")
        raw, normalized = _load_raw_and_normalized(candidate)
        metric = candidate.get("objective")
        value = metric.get("value") if isinstance(metric, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("candidate fixed-final inner WP must be finite")
        ranked.append(
            {
                "candidate": candidate,
                "raw_config": raw,
                "normalized_config": normalized,
                "inner_wp": float(value),
            }
        )
    ranked.sort(
        key=lambda entry: (
            -float(entry["inner_wp"]),
            str(entry["candidate"]["family"]),
            int(entry["candidate"]["trial_number"]),
        )
    )
    return ranked, file_sha256(request_path)
