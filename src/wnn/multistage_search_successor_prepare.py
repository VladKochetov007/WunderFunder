from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_sources import training_source_for_stage
from wnn.multistage_search_study import open_study
from wnn.multistage_search_successor_parent import parent_identity_sha, qualified_parent
from wnn.multistage_search_successor_proposal_cap import resolve_stage2_proposal_limit
from wnn.splits import load_campaign_split
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members
from wnn.training.sequence_config import load_config


def prepare_stage2_trial(
    run_dir: Path,
    parent_result_path: Path,
    *,
    lr_multiplier: float | None = None,
) -> dict[str, object]:
    """Ask TPE for one lower-LR, two-epoch fresh-initializer continuation."""
    manifest, admission = _campaign_inputs(run_dir)
    parent, checkpoint, parent_config = qualified_parent(parent_result_path, allowed={1})
    objective = parent.get("objective")
    auxiliary_ref = admission["data"].get("auxiliary_manifest")
    if not isinstance(auxiliary_ref, Mapping) or not isinstance(objective, Mapping):
        raise ValueError("stage2 parent or admission is missing its source-qualified split")
    if objective.get("split_sha256") != auxiliary_ref.get("sha256") or objective.get(
        "primary_split_sha256"
    ) != admission["data"].get("primary_split_sha256"):
        raise ValueError("stage2 parent must match this campaign's frozen auxiliary train split")
    if type(parent_config.get("lr")) not in (int, float) or parent_config["lr"] <= 0:
        raise ValueError("stage2 parent config needs a positive base learning rate")
    study = open_study(run_dir, "stage2-proposer", stage=2)
    proposal_limit = resolve_stage2_proposal_limit(run_dir, manifest, admission, study)
    existing_trials = study.get_trials(deepcopy=False)
    if len(existing_trials) >= proposal_limit.max_trials:
        raise RuntimeError("stage2 Optuna trial cap is exhausted")
    parent_identity_sha256 = parent_identity_sha(parent_result_path)
    if lr_multiplier is not None:
        if type(lr_multiplier) not in (int, float) or lr_multiplier not in (0.1, 0.25, 0.5):
            raise ValueError("stage2 LR multiplier must be one of 0.1, 0.25, or 0.5")
        study.enqueue_trial(
            {"lr_multiplier": float(lr_multiplier)},
            user_attrs={
                "stage2": {"status": "prepared_anchor", "lr_multiplier": float(lr_multiplier)}
            },
        )
    trial = study.ask()
    number = _trial_number(trial)
    if (
        proposal_limit.permitted_next_trial_number is not None
        and number != proposal_limit.permitted_next_trial_number
    ):
        raise ValueError("approved Stage2 cap permits proposal 6 only")
    multiplier = trial.suggest_categorical("lr_multiplier", [0.1, 0.25, 0.5])
    config = dict(parent_config)
    config["lr"] = float(parent_config["lr"]) * float(multiplier)
    config["epochs"] = 2
    definition = {
        "stage": 2,
        "trial_number": number,
        "family": str(parent.get("family", "unknown")),
        "params": {"lr_multiplier": multiplier, "lr": config["lr"], "epochs": 2},
        "pair_id": None,
        "pair_role": "finetune",
        "parent_result_path": str(parent_result_path.resolve(strict=True)),
        "parent_result_sha256": file_sha256(parent_result_path),
        "initializer_path": str(checkpoint),
        "initializer_sha256": file_sha256(checkpoint),
        "parent_identity_sha256": parent_identity_sha256,
        "reused_skip_control": {
            "result_path": str(parent_result_path.resolve()),
            "objective": parent["objective"],
            "fit_attempt_added": False,
        },
        "config": config,
    }
    if proposal_limit.amendment_path is not None:
        definition["proposal_cap_amendment"] = {
            "path": str(proposal_limit.amendment_path),
            "sha256": proposal_limit.amendment_sha256,
            "review_path": str(proposal_limit.review_path),
            "review_sha256": proposal_limit.review_sha256,
            "base_cap": 6,
            "derived_cap": 7,
            "permitted_trial_number": 6,
        }
    result = _write_trial(run_dir, manifest, admission, trial, definition)
    trial.set_user_attr("stage2", {"trial_dir": result["trial_dir"], "status": "prepared"})
    return result


def prepare_stage3_pair(
    run_dir: Path,
    base_config_path: Path,
    teacher_cache_path: Path,
    *,
    family: str,
) -> dict[str, object]:
    """Ask TPE for one KD treatment and create its exact label-only matched control."""
    manifest, admission = _campaign_inputs(run_dir)
    data = admission["data"]
    auxiliary_ref = data["auxiliary_manifest"]
    if not isinstance(auxiliary_ref, Mapping):
        raise ValueError("stage3 admission has no frozen auxiliary manifest reference")
    auxiliary = Path(str(auxiliary_ref["path"])).resolve(strict=True)
    data_root = Path(str(data["data_root"])).resolve(strict=True)
    split = load_campaign_split(auxiliary, data_root)
    fit_members, _ = auxiliary_members(split)
    cache = TeacherCache(teacher_cache_path.resolve(strict=True), split, fit_members)
    pool_producer = cache.metadata.get("producer")
    if not isinstance(pool_producer, Mapping) or pool_producer.get("kind") != (
        "weighted_teacher_pool_v1"
    ):
        raise ValueError("stage3 requires a validated weighted auxiliary-fit teacher pool")
    base_path = base_config_path.resolve(strict=True)
    base_sha = file_sha256(base_path)
    base = load_config(base_path)
    stage3_policy = admission.get("stage3")
    expected_base = (
        stage3_policy.get("student_base_config") if isinstance(stage3_policy, Mapping) else None
    )
    expected_family = (
        stage3_policy.get("student_family") if isinstance(stage3_policy, Mapping) else None
    )
    if (
        not isinstance(expected_base, Mapping)
        or family != expected_family
        or base_sha != expected_base.get("sha256")
    ):
        raise ValueError("stage3 student family/config must match the fixed admitted base")
    _validate_student_config(base)
    study = open_study(run_dir, "stage3-proposer", stage=3)
    max_trials = admission.get("stage3", {}).get("max_trials")
    existing_trials = study.get_trials(deepcopy=False)
    waiting = sum(
        getattr(getattr(row, "state", None), "name", None) == "WAITING" for row in existing_trials
    )
    if waiting > 1:
        raise ValueError("stage3 supports at most one queued treatment anchor per pair")
    needed_new_trials = 2 - waiting
    if type(max_trials) is not int or len(existing_trials) + needed_new_trials > max_trials:
        raise RuntimeError("stage3 does not have capacity for another matched two-fit pair")
    kd_trial = study.ask()
    kd_number = _trial_number(kd_trial)
    kd_weight = kd_trial.suggest_categorical("kd_weight", [0.05, 0.1, 0.25, 0.5, 1.0])
    seed_base = base.get("seed")
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError("stage3 student base config needs a nonnegative seed")
    pair_seed = seed_base + kd_number
    control_trial = study.ask()
    control_number = _trial_number(control_trial)
    pair_id = f"pair-{kd_number:06d}-{control_number:06d}"
    pair_dir = run_dir / "stage3" / "pairs" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=False)
    variants: list[dict[str, object]] = []
    for trial, number, role, weight in (
        (control_trial, control_number, "label_only_control", 0.0),
        (kd_trial, kd_number, "kd_student", float(kd_weight)),
    ):
        config = dict(base)
        config["epochs"] = 2
        config["seed"] = pair_seed
        config["kd_weight"] = weight
        definition = {
            "stage": 3,
            "trial_number": number,
            "family": family,
            "params": {"kd_weight": weight, "epochs": 2, "seed": pair_seed},
            "pair_id": pair_id,
            "pair_role": role,
            "pair_mate_trial_number": kd_number if number == control_number else control_number,
            "base_config_path": str(base_path),
            "base_config_sha256": base_sha,
            "teacher_cache_path": str(teacher_cache_path.resolve(strict=True))
            if weight > 0
            else None,
            "teacher_cache_manifest_sha256": file_sha256(teacher_cache_path / "manifest.json")
            if weight > 0
            else None,
            "teacher_pool_id": pool_producer.get("pool_id"),
            "config": config,
        }
        item = _write_trial(run_dir, manifest, admission, trial, definition)
        trial.set_user_attr(
            "stage3",
            {"trial_dir": item["trial_dir"], "status": "prepared", "pair_id": pair_id},
        )
        variants.append(item)
    pair = {
        "schema": "wnn.multistage-stage3-pair.v1",
        "pair_id": pair_id,
        "teacher_cache_path": str(teacher_cache_path.resolve()),
        "teacher_cache_manifest_sha256": cache.identity["manifest_sha256"],
        "teacher_pool_id": pool_producer.get("pool_id"),
        "base_config_sha256": base_sha,
        "variants": variants,
        "created_at_utc": utc_now(),
    }
    pair_path = pair_dir / "pair.json"
    atomic_write_json(pair_path, pair, replace=False)
    return pair | {"pair_path": str(pair_path.resolve()), "pair_sha256": file_sha256(pair_path)}


def _write_trial(
    run_dir: Path,
    manifest: Mapping[str, object],
    admission: Mapping[str, object],
    trial: object,
    definition: Mapping[str, object],
) -> dict[str, object]:
    """Write an immutable config and successor definition before any profile is launched."""
    stage = int(definition["stage"])
    number = _trial_number(trial)
    if number != definition["trial_number"]:
        raise ValueError("successor definition trial number differs from Optuna")
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{number:06d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    config_path = trial_dir / "config.json"
    config = definition.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("successor config is malformed")
    raw_config = _serialize_runtime_config(config)
    atomic_write_json(config_path, raw_config, replace=False)
    normalized_config = load_config(config_path)
    if normalized_config != dict(config):
        raise ValueError("serialized successor config differs from its normalized recipe")
    config_sha = file_sha256(config_path)
    auxiliary = admission["data"]["auxiliary_manifest"]
    aux_path = Path(str(auxiliary["path"])).resolve(strict=True)
    if file_sha256(aux_path) != auxiliary["sha256"]:
        raise ValueError("successor auxiliary manifest changed after admission")
    source_root, source_revision, expected_source_sha, expected_files_sha = (
        training_source_for_stage(manifest, stage)
    )
    from wnn.training.sequence_provenance import source_identity

    source = source_identity(source_root, source_root / "scripts" / "train_sequence.py")
    if (
        source["sha256"] != expected_source_sha
        or canonical_json_sha256(source["files"]) != expected_files_sha
    ):
        raise ValueError("successor training source differs from the frozen source snapshot")
    value = dict(definition) | {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "config_identity_sha256": canonical_json_sha256(normalized_config),
        "primary_split_sha256": admission["data"]["primary_split_sha256"],
        "auxiliary_split_manifest": str(aux_path),
        "auxiliary_split_sha256": auxiliary["sha256"],
        "data_root": str(Path(str(admission["data"]["data_root"])).resolve(strict=True)),
        "training_source_root": str(source_root),
        "source_revision": source_revision,
        "source_sha256": source["sha256"],
        "source_files_sha256": canonical_json_sha256(source["files"]),
        "objective_scope": "inner_train_proxy",
        "primary_valid_used": False,
        "created_at_utc": utc_now(),
    }
    value.pop("config", None)
    definition_path = trial_dir / "trial_definition.json"
    atomic_write_json(definition_path, value, replace=False)
    return {
        "stage": stage,
        "trial_number": number,
        "family": value["family"],
        "params": value["params"],
        "trial_dir": str(trial_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "definition_path": str(definition_path.resolve()),
        "definition_sha256": file_sha256(definition_path),
        "pair_id": value.get("pair_id"),
        "pair_role": value.get("pair_role"),
    }


def _serialize_runtime_config(config: Mapping[str, object]) -> dict[str, object]:
    """Write native raw JSON while keeping its normalized identity unchanged."""
    serialized = dict(config)
    model = serialized.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("successor config model is malformed")
    raw_model = dict(model)
    architecture = raw_model.get("architecture")
    top_precision = serialized.get("precision_mode")
    model_precision = raw_model.get("precision_mode")
    if architecture != "galam" and model_precision is not None:
        if model_precision != top_precision:
            raise ValueError("runtime-only model precision differs from top-level precision")
        raw_model.pop("precision_mode")
    serialized["model"] = raw_model
    return serialized


def _campaign_inputs(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(manifest["admission"]["path"])))
    limits = admission.get("stage_fit_caps")
    if not isinstance(limits, Mapping):
        raise ValueError("admission has no four-stage fit caps")
    for stage in (2, 3):
        if type(limits.get(str(stage))) is not int or limits[str(stage)] <= 0:
            raise ValueError(f"stage{stage} has no admitted scientific fit capacity")
    return manifest, admission


def _validate_student_config(config: Mapping[str, object]) -> None:
    """Require an FP32 label-only student config with no auxiliary output heads."""
    model = config.get("model")
    if (
        config.get("precision_mode") != "fp32"
        or config.get("hint_weight") != 0
        or not isinstance(model, Mapping)
        or model.get("hint_dim") != 0
    ):
        raise ValueError("stage3 student base must be strict-FP32 with hint_dim=0")


def _trial_number(trial: object) -> int:
    """Return a well-formed durable Optuna trial number."""
    value = getattr(trial, "number", None)
    if type(value) is not int or value < 0:
        raise ValueError("Optuna trial number is invalid")
    return value
