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
from wnn.multistage_search_profile_execute import _execute_profile
from wnn.multistage_search_runtime import base_environment, current_process_owner
from wnn.multistage_search_study import open_study
from wnn.multistage_search_worker_support import (
    _device_torch_total_memory,
    _device_total_memory,
    _device_uuid,
    _process_tuple,
)
from wnn.multistage_search_worker_types import ProfileJob


def run_successor_profile(
    run_dir: Path,
    *,
    stage: int,
    trial_number: int,
    worker_id: str,
    gpu_key: str,
    device: Mapping[str, object],
) -> dict[str, object]:
    """Profile one prepared stage2/3 config against its exact initializer or teacher cache."""
    if stage not in {2, 3}:
        raise ValueError("successor profile workers support stage2 and stage3")
    run_manifest = read_json(run_dir / "run.json")
    admission = read_json(Path(str(run_manifest["admission"]["path"])))
    study = open_study(run_dir, worker_id, stage=stage)
    frozen_trial = next(
        (
            item
            for item in study.get_trials(deepcopy=False)
            if getattr(item, "number", None) == trial_number
            and getattr(getattr(item, "state", None), "name", None) == "RUNNING"
        ),
        None,
    )
    if frozen_trial is None:
        raise ValueError("successor profile must belong to one running Optuna trial")
    trial_id = getattr(frozen_trial, "_trial_id", None)
    if type(trial_id) is not int:
        raise ValueError("running successor proposal omits its durable Optuna trial id")
    import optuna

    optuna_trial = optuna.trial.Trial(study, trial_id)
    if optuna_trial.number != trial_number:
        raise ValueError("reconstructed successor Trial differs from the requested number")
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{trial_number:06d}"
    definition_path = trial_dir / "trial_definition.json"
    definition = read_json(definition_path)
    if definition.get("stage") != stage or definition.get("trial_number") != trial_number:
        raise ValueError("successor definition does not match the requested trial")
    config_path = Path(str(definition["config_path"])).resolve(strict=True)
    if file_sha256(config_path) != definition.get("config_sha256"):
        raise ValueError("successor config changed after proposal")
    profile_dir = trial_dir / "profile"
    output_dir = profile_dir / "training_output"
    profile_dir.mkdir(parents=True, exist_ok=False)
    native_path = output_dir / "profile_receipt.json"
    worker_receipt_path = profile_dir / "worker_receipt.json"
    plan_path = trial_dir / "trial_plan.json"
    source_root = Path(str(definition["training_source_root"])).resolve(strict=True)
    from wnn.training.sequence_provenance import source_identity

    source = source_identity(source_root, source_root / "scripts" / "train_sequence.py")
    if source.get("sha256") != definition.get("source_sha256") or canonical_json_sha256(
        source.get("files")
    ) != definition.get("source_files_sha256"):
        raise ValueError("successor trainer source changed after its config was prepared")
    auxiliary = Path(str(definition["auxiliary_split_manifest"])).resolve(strict=True)
    if file_sha256(auxiliary) != definition.get("auxiliary_split_sha256"):
        raise ValueError("successor auxiliary split changed after its config was prepared")
    owner_path = run_dir / "workers" / f"{worker_id}.launch_receipt.json"
    owner = read_json(owner_path)
    current_owner = current_process_owner()
    if _process_tuple(owner) != _process_tuple(current_owner):
        raise ValueError("successor profile worker differs from its immutable launch owner")
    owner_command_sha = owner.get("command_sha256")
    if not isinstance(owner_command_sha, str) or len(owner_command_sha) != 64:
        raise ValueError("successor worker owner receipt has no command hash")
    profile_policy = admission["profile_policy"]
    profile_updates = int(profile_policy["profile_updates"])
    worker_env = base_environment(str(device["cuda_visible_devices"]), source_root=source_root)
    thread_environment = {
        name: worker_env[name]
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        if name in worker_env
    }
    cpu_thread_env = {name: worker_env[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    proposal_key = canonical_json_sha256(
        {
            "stage": stage,
            "config_sha256": definition["config_sha256"],
            "auxiliary_split_sha256": definition["auxiliary_split_sha256"],
            "source_sha256": source["sha256"],
            "runtime_policy": profile_policy,
            "initializer_sha256": definition.get("initializer_sha256"),
            "teacher_cache_manifest_sha256": definition.get("teacher_cache_manifest_sha256"),
            "device_uuid": _device_uuid(device),
        }
    )
    request = {
        "schema": "wnn.multistage-trial-plan.v1",
        "experiment_id": run_manifest["experiment_id"],
        "stage": stage,
        "trial_number": trial_number,
        "worker_id": worker_id,
        "gpu_key": gpu_key,
        "family": definition["family"],
        "params": definition["params"],
        "pair_id": definition.get("pair_id"),
        "pair_role": definition.get("pair_role"),
        "pair_mate_trial_number": definition.get("pair_mate_trial_number"),
        "config_path": str(config_path),
        "config_sha256": definition["config_sha256"],
        "config_identity_sha256": definition["config_identity_sha256"],
        "primary_split_sha256": definition["primary_split_sha256"],
        "auxiliary_split_manifest": str(auxiliary),
        "auxiliary_split_sha256": definition["auxiliary_split_sha256"],
        "data_root": definition["data_root"],
        "objective_scope": "inner_train_proxy",
        "primary_valid_used": False,
        "profile_output_path": str(output_dir.resolve()),
        "profile_receipt_path": str(native_path.resolve()),
        "worker_receipt_path": str(worker_receipt_path.resolve()),
        "requested_updates": profile_updates,
        "requested_profile_updates": profile_updates,
        "requested_world_size": profile_policy["world_size"],
        "world_size": profile_policy["world_size"],
        "requested_device_set": [device["cuda_visible_devices"]],
        "expected_device_uuid_set": [_device_uuid(device)],
        "expected_device_total_memory_bytes": _device_total_memory(device),
        "expected_device_torch_total_memory_bytes": _device_torch_total_memory(device),
        "initializer_parent_trial_result_sha256": definition.get("parent_result_sha256"),
        "anchor_id": None,
        "cpu_thread_env": cpu_thread_env,
        "effective_thread_environment": thread_environment,
        "worker_owned_process": _process_tuple(current_owner),
        "worker_owned_process_command_sha256": owner_command_sha,
        "worker_owner_receipt_path": str(owner_path.resolve()),
        "worker_owner_receipt_sha256": file_sha256(owner_path),
        "source_revision": definition["source_revision"],
        "source_sha256": source["sha256"],
        "source_script_sha256": source["sha256"],
        "source_files_sha256": definition["source_files_sha256"],
        "runtime_policy_sha256": canonical_json_sha256(profile_policy),
        "profile_checker_policy_sha256": run_manifest["profile_checker_policy"]["sha256"],
        "profile_policy_sha256": run_manifest["profile_checker_policy"]["sha256"],
        "proposal_key": proposal_key,
        "created_at_utc": utc_now(),
    }
    for name in (
        "initializer_path",
        "initializer_sha256",
        "teacher_cache_path",
        "teacher_cache_manifest_sha256",
        "parent_result_path",
        "parent_result_sha256",
        "parent_identity_sha256",
        "teacher_pool_id",
    ):
        if definition.get(name) is not None:
            request[name] = definition[name]
    atomic_write_json(plan_path, request, replace=False)
    optuna_trial.set_user_attr(
        f"stage{stage}", {"trial_dir": str(trial_dir.resolve()), "status": "profile_running"}
    )
    optuna_trial.set_user_attr("worker_id", worker_id)
    profile_result = _execute_profile(
        ProfileJob(
            run_dir=run_dir,
            run_manifest=run_manifest,
            worker_id=worker_id,
            gpu_key=gpu_key,
            device=device,
            study=study,
            trial=optuna_trial,
            trial_number=trial_number,
            family=str(definition["family"]),
            config_path=config_path,
            config_sha256=str(definition["config_sha256"]),
            repo_root=source_root,
            source_sha256=str(source["sha256"]),
            auxiliary_path=auxiliary,
            profile_updates=profile_updates,
            output_dir=output_dir,
            profile_dir=profile_dir,
            native_profile_path=native_path,
            worker_receipt_path=worker_receipt_path,
            plan_path=plan_path,
            proposal_key=proposal_key,
            request=request,
            worker_env=worker_env,
            stage=stage,
        )
    )
    if profile_result.get("status") == "profile_complete_waiting_review":
        from wnn.multistage_search_review import run_profile_reviewer

        review = run_profile_reviewer(run_dir, trial_number, stage=stage)
        review_path = profile_dir / "successor_profile_review_receipt.json"
        return profile_result | {
            "review_receipt_path": str(review_path.resolve()),
            "review_receipt_sha256": review["review_receipt_sha256"],
            "review_status": review["status"],
        }
    return profile_result
