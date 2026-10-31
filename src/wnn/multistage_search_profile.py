from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
    validate_admission,
)
from wnn.multistage_search_profile_execute import _execute_profile
from wnn.multistage_search_runtime import base_environment, current_process_owner, utc_now
from wnn.multistage_search_sources import training_source_for_stage
from wnn.multistage_search_space import apply_stage1_config, suggest_stage1
from wnn.multistage_search_study import open_study
from wnn.multistage_search_worker_support import (
    _device_torch_total_memory,
    _device_total_memory,
    _device_uuid,
    _find_anchor,
    _process_tuple,
)
from wnn.multistage_search_worker_types import ProfileJob


def run_profile_trial(
    run_dir: Path,
    admission: Mapping[str, object],
    worker_id: str,
    gpu_key: str,
    device: Mapping[str, object],
    *,
    prepared_trial_number: int | None = None,
) -> dict[str, object]:
    """Ask one TPE config, freeze it, and run only its bounded real-data profile."""
    run_manifest = read_json(run_dir / "run.json")
    validated = validate_admission(
        read_json(Path(str(run_manifest["contract"]["path"]))),
        admission,
        str(run_manifest["contract"]["sha256"]),
    )
    study = open_study(run_dir, worker_id)
    prepared_snapshot = None
    if prepared_trial_number is None:
        trial = study.ask()
        trial.set_user_attr("worker_id", worker_id)
    else:
        trial, prepared_snapshot = _prepared_running_trial(
            study, prepared_trial_number, worker_id
        )
    trial_number = int(trial.number)
    stage_attrs = getattr(trial, "user_attrs", {}).get("stage1", {})
    starting_attrs = dict(stage_attrs) if isinstance(stage_attrs, Mapping) else {}
    starting_attrs.update({"status": "profile_starting", "trial_number": trial_number})
    trial.set_user_attr("stage1", starting_attrs)
    family, params = suggest_stage1(trial, validated["families"])
    if prepared_snapshot is not None:
        _require_frozen_stage1_params(study, prepared_snapshot, trial_number)
    family_spec = validated["families"][family]
    if not isinstance(family_spec, Mapping):
        raise ValueError("admitted stage1 family definition is malformed")
    base_config = validated["family_configs"][family]
    base_config_sha = validated["family_config_hashes"][family]
    if not isinstance(base_config, Mapping) or not isinstance(base_config_sha, str):
        raise ValueError("admitted family base config identity is incomplete")
    config = apply_stage1_config(base_config, family, params, family_spec)
    trial_attrs = getattr(trial, "user_attrs", {})
    anchor_id = trial_attrs.get("anchor_id") if isinstance(trial_attrs, Mapping) else None
    config_identity_sha256 = canonical_json_sha256(config)
    preprofiled = admission.get("preprofiled_config_identity_sha256s", [])
    if not isinstance(preprofiled, list) or any(
        not isinstance(value, str) for value in preprofiled
    ):
        raise ValueError("preprofiled config identities must be a list of SHA256 values")
    prior_trial_identities = {
        value
        for existing in study.get_trials(deepcopy=False)
        if isinstance(getattr(existing, "user_attrs", None), Mapping)
        for value in (
            existing.user_attrs.get("config_identity_sha256"),
            existing.user_attrs.get("stage1", {}).get("config_identity_sha256")
            if isinstance(existing.user_attrs.get("stage1"), Mapping)
            else None,
        )
        if isinstance(value, str)
    }
    if config_identity_sha256 in preprofiled or config_identity_sha256 in prior_trial_identities:
        import optuna

        trial.set_user_attr("config_identity_sha256", config_identity_sha256)
        trial.set_user_attr("terminal_reason", "exact config already has a reviewed profile")
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "preprofiled_duplicate_skipped",
                "stage": 1,
                "trial_number": trial_number,
                "family": family,
                "config_identity_sha256": config_identity_sha256,
                "at_utc": utc_now(),
            },
        )
        return {
            "status": "preprofiled_duplicate_skipped",
            "trial_number": trial_number,
            "config_identity_sha256": config_identity_sha256,
        }
    stage_dir = run_dir / "stage1"
    trial_dir = stage_dir / "trials" / f"{trial_number:06d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    config_path = trial_dir / "config.json"
    config_bytes = None
    if anchor_id is not None:
        anchor = _find_anchor(validated["anchors"], anchor_id)
        anchor_family, anchor_params = anchor.get("family"), anchor.get("params")
        if anchor_family != family or anchor_params != params:
            raise ValueError("queued anchor parameters differ from its immutable anchor definition")
        anchor_config_ref = anchor.get("config")
        if not isinstance(anchor_config_ref, Mapping):
            raise ValueError("queued anchor lacks an exact config path and digest")
        anchor_config_path = Path(str(anchor_config_ref["path"])).resolve(strict=True)
        anchor_config_sha = file_sha256(anchor_config_path)
        if anchor_config_sha != anchor_config_ref.get("sha256"):
            raise ValueError("queued anchor config bytes differ from their frozen digest")
        anchor_config = read_json(anchor_config_path)
        if anchor_config != config:
            raise ValueError("generated anchor config differs from the exact R3 anchor config")
        config_bytes = anchor_config_path.read_bytes()
    if config_bytes is None:
        atomic_write_json(config_path, config, replace=False)
    else:
        config_path.write_bytes(config_bytes)
    config_sha = file_sha256(config_path)
    repo_root, revision, source_sha256, source_files_sha256 = training_source_for_stage(
        run_manifest, 1
    )

    train_entry = repo_root / "scripts" / "train_sequence.py"
    from wnn.training.sequence_provenance import source_identity

    source = source_identity(repo_root, train_entry)
    if (
        source.get("sha256") != source_sha256
        or canonical_json_sha256(source.get("files")) != source_files_sha256
    ):
        raise ValueError("pinned training source changed after campaign preparation")
    data = admission["data"]
    profile_policy = admission["profile_policy"]
    auxiliary = data["auxiliary_manifest"]
    if not isinstance(auxiliary, Mapping):
        raise ValueError("auxiliary manifest identity is malformed")
    aux_path, aux_sha = Path(str(auxiliary["path"])).resolve(strict=True), str(auxiliary["sha256"])
    if file_sha256(aux_path) != aux_sha:
        raise ValueError("auxiliary split manifest changed after admission")
    worker_owner_path = run_dir / "workers" / f"{worker_id}.launch_receipt.json"
    worker_owner = read_json(worker_owner_path)
    worker_command_sha = worker_owner.get("command_sha256")
    if not isinstance(worker_command_sha, str) or len(worker_command_sha) != 64:
        raise ValueError("detached worker owner receipt lacks its exact command hash")
    if _process_tuple(worker_owner) != _process_tuple(current_process_owner()):
        raise ValueError(
            "detached worker process identity differs from its immutable owner receipt"
        )
    profile_updates = int(profile_policy["profile_updates"])
    output_dir = trial_dir / "profile" / "training_output"
    profile_dir = output_dir.parent
    profile_dir.mkdir(parents=True)
    native_profile_path = output_dir / "profile_receipt.json"
    worker_receipt_path = profile_dir / "worker_receipt.json"
    plan_path = trial_dir / "trial_plan.json"
    proposal_key = canonical_json_sha256(
        {
            "config_sha256": config_sha,
            "auxiliary_split_sha256": aux_sha,
            "primary_split_sha256": data["primary_split_sha256"],
            "trainer_source_sha256": source["sha256"],
            "runtime_policy_sha256": canonical_json_sha256(profile_policy),
            "device": dict(device),
        }
    )
    worker_env = base_environment(str(device["cuda_visible_devices"]), source_root=repo_root)
    thread_environment = {
        key: worker_env[key]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        if key in worker_env
    }
    cpu_thread_env = {key: worker_env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    request = {
        "schema": "wnn.multistage-trial-plan.v1",
        "experiment_id": run_manifest["experiment_id"],
        "stage": 1,
        "trial_number": trial_number,
        "worker_id": worker_id,
        "gpu_key": gpu_key,
        "family": family,
        "params": params,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "base_config_sha256": base_config_sha,
        "primary_split_sha256": data["primary_split_sha256"],
        "auxiliary_split_manifest": str(aux_path),
        "auxiliary_split_sha256": aux_sha,
        "data_root": str(Path(str(data["data_root"])).resolve(strict=True)),
        "objective_scope": "inner_train_proxy",
        "primary_valid_used": False,
        "profile_output_path": str(output_dir.resolve()),
        "profile_receipt_path": str(native_profile_path.resolve()),
        "worker_receipt_path": str(worker_receipt_path.resolve()),
        "requested_updates": profile_updates,
        "requested_profile_updates": profile_updates,
        "requested_world_size": profile_policy["world_size"],
        "world_size": profile_policy["world_size"],
        "requested_device_set": [device["cuda_visible_devices"]],
        "expected_device_uuid_set": [_device_uuid(device)],
        "expected_device_total_memory_bytes": _device_total_memory(device),
        "expected_device_torch_total_memory_bytes": _device_torch_total_memory(device),
        "anchor_id": anchor_id,
        "effective_thread_environment": thread_environment,
        "cpu_thread_env": cpu_thread_env,
        "worker_owned_process": _process_tuple(current_process_owner()),
        "worker_owned_process_command_sha256": worker_command_sha,
        "worker_owner_receipt_path": str(worker_owner_path.resolve()),
        "worker_owner_receipt_sha256": file_sha256(worker_owner_path),
        "source_revision": revision,
        "source_sha256": source["sha256"],
        "source_script_sha256": source["sha256"],
        "source_files_sha256": canonical_json_sha256(source["files"]),
        "runtime_policy_sha256": canonical_json_sha256(profile_policy),
        "profile_checker_policy_sha256": run_manifest["profile_checker_policy"]["sha256"],
        "profile_policy_sha256": run_manifest["profile_checker_policy"]["sha256"],
        "reviewer_script_sha256": run_manifest["profile_reviewer_script"]["sha256"],
        "proposal_key": proposal_key,
        "created_at_utc": utc_now(),
    }
    trial.set_user_attr("config_identity_sha256", config_identity_sha256)
    trial.set_user_attr(
        "stage1",
        {
            "trial_dir": str(trial_dir),
            "proposal_key": proposal_key,
            "config_identity_sha256": config_identity_sha256,
        },
    )
    atomic_write_json(plan_path, request, replace=False)
    return _execute_profile(
        ProfileJob(
            run_dir=run_dir,
            run_manifest=run_manifest,
            worker_id=worker_id,
            gpu_key=gpu_key,
            device=device,
            study=study,
            trial=trial,
            trial_number=trial_number,
            family=family,
            config_path=config_path,
            config_sha256=config_sha,
            repo_root=repo_root,
            source_sha256=str(source["sha256"]),
            auxiliary_path=aux_path,
            profile_updates=profile_updates,
            output_dir=output_dir,
            profile_dir=profile_dir,
            native_profile_path=native_profile_path,
            worker_receipt_path=worker_receipt_path,
            plan_path=plan_path,
            proposal_key=proposal_key,
            request=request,
            worker_env=worker_env,
        )
    )


def _prepared_running_trial(
    study: object, trial_number: int, worker_id: str
) -> tuple[object, object]:
    """Reconstruct one exact assigned RUNNING Optuna trial without asking another trial."""
    if type(trial_number) is not int or trial_number < 0:
        raise ValueError("prepared stage1 trial number must be a nonnegative integer")
    import optuna

    matches = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if getattr(trial, "number", None) == trial_number
    ]
    if len(matches) != 1:
        raise ValueError("prepared trial number is not unique in this exact stage1 study")
    frozen = matches[0]
    if (
        frozen.state is not optuna.trial.TrialState.RUNNING
        or not isinstance(getattr(frozen, "user_attrs", None), Mapping)
        or frozen.user_attrs.get("worker_id") != worker_id
    ):
        raise ValueError("prepared trial must be RUNNING and assigned to this exact worker")
    trial_id = getattr(frozen, "_trial_id", None)
    if type(trial_id) is not int:
        raise ValueError("prepared Optuna trial omits its storage trial id")
    live = optuna.trial.Trial(study, trial_id)
    if live.number != trial_number:
        raise ValueError("reconstructed Optuna trial id differs from its requested number")
    return live, frozen


def _require_frozen_stage1_params(study: object, frozen: object, trial_number: int) -> None:
    """Reject prepared trials if deterministic suggestions add or alter frozen dimensions."""
    refreshed = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if getattr(trial, "number", None) == trial_number
    ]
    if len(refreshed) != 1:
        raise ValueError("prepared trial disappeared from its stage1 study")
    after = refreshed[0]
    if (
        after.state is not frozen.state
        or after.state.name != "RUNNING"
        or after.params != frozen.params
        or after.distributions != frozen.distributions
        or after.user_attrs.get("worker_id") != frozen.user_attrs.get("worker_id")
    ):
        raise ValueError("prepared stage1 parameters were not fully frozen before profiling")
