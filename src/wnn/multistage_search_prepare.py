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
    verify_file_ref,
)
from wnn.multistage_search_budget import initialize_budget, utc_now
from wnn.multistage_search_study import enqueue_anchors, open_study


def prepare_run(
    contract_path: Path,
    admission_path: Path,
    run_dir: Path,
    training_source: Path | None = None,
    training_source_revision: str | None = None,
    evaluation_source: Path | None = None,
    evaluation_source_revision: str | None = None,
    stage_training_sources: Mapping[str, tuple[Path, str]] | None = None,
) -> dict[str, object]:
    """Freeze contract/admission identities and prepare the durable four-stage study."""
    contract_path, admission_path = (
        contract_path.resolve(strict=True),
        admission_path.resolve(strict=True),
    )
    contract, admission = read_json(contract_path), read_json(admission_path)
    contract_sha, admission_sha = file_sha256(contract_path), file_sha256(admission_path)
    validated = validate_admission(contract, admission, contract_sha)
    limits = validated["limits"]
    if not isinstance(limits, Mapping):
        raise TypeError("validated contract limits are malformed")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    frozen = run_dir / "frozen"
    frozen.mkdir()
    frozen_contract, frozen_admission = frozen / "CONTRACT.json", frozen / "ADMISSION.json"
    frozen_contract.write_bytes(contract_path.read_bytes())
    frozen_admission.write_bytes(admission_path.read_bytes())
    aux_manifest_path, aux_manifest_sha = verify_file_ref(
        admission["data"]["auxiliary_manifest"], "auxiliary manifest"
    )
    profile_updates = admission["profile_policy"].get("profile_updates")
    profile_wall = admission["profile_policy"].get("profile_wall_seconds")
    if type(profile_updates) is not int or profile_updates < 1:
        raise ValueError("profile policy must request at least one real update")
    if type(profile_wall) is not int or not 1 <= profile_wall <= limits["max_profile_wall_seconds"]:
        raise ValueError("profile wall timeout exceeds its contract bound")
    profile_started = admission.get("first_gpu_profile_epoch")
    if profile_started is not None and not isinstance(profile_started, (int, float)):
        raise ValueError("first_gpu_profile_epoch must be a numeric UTC epoch")
    stage_caps = validated["stage_fit_caps"]
    source_root = Path(__file__).resolve().parents[2]
    configured_training_source = training_source or Path(
        str(admission.get("training_source_root", source_root))
    )
    training_root = configured_training_source.resolve(strict=True)
    if not (training_root / "scripts" / "train_sequence.py").is_file():
        raise FileNotFoundError("pinned training source has no canonical sequence trainer")
    source_revision = training_source_revision or admission.get("training_source_revision")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("prepare requires an immutable training source revision identifier")
    configured_evaluation_source = evaluation_source or Path(
        str(admission.get("evaluation_source_root", training_root))
    )
    evaluation_root = configured_evaluation_source.resolve(strict=True)
    for entry in ("scripts/predict_sequence.py", "scripts/score_sequence.py"):
        if not (evaluation_root / entry).is_file():
            raise FileNotFoundError(f"pinned evaluation source is missing {entry}")
    evaluation_revision = evaluation_source_revision or admission.get(
        "evaluation_source_revision", source_revision
    )
    if not isinstance(evaluation_revision, str) or not evaluation_revision.strip():
        raise ValueError("prepare requires an immutable evaluation source revision")
    from wnn.training.sequence_provenance import source_identity

    trainer_source = source_identity(training_root, training_root / "scripts" / "train_sequence.py")
    expected_training_sha = admission.get("training_source_identity_sha256")
    if expected_training_sha is not None and trainer_source.get("sha256") != expected_training_sha:
        raise ValueError("training source identity differs from the independently pinned admission")
    stage_source_specs = {
        "1": (training_root, source_revision),
        "2": (training_root, source_revision),
        "3": (training_root, source_revision),
    }
    if stage_training_sources is not None:
        for stage_key, spec in stage_training_sources.items():
            if stage_key not in {"1", "2", "3"} or not isinstance(spec, tuple) or len(spec) != 2:
                raise ValueError("stage training sources must map stages 1-3 to (root, revision)")
            stage_root, stage_revision = spec
            if not isinstance(stage_root, Path) or not isinstance(stage_revision, str):
                raise ValueError("stage training source root and revision are malformed")
            stage_root = stage_root.resolve(strict=True)
            stage_revision = stage_revision.strip()
            if not stage_revision or not (stage_root / "scripts" / "train_sequence.py").is_file():
                raise ValueError(f"stage {stage_key} source lacks its canonical sequence trainer")
            stage_source_specs[stage_key] = (stage_root, stage_revision)
    stage_source_records: dict[str, dict[str, str]] = {}
    for stage_key, (stage_root, stage_revision) in stage_source_specs.items():
        stage_identity = (
            trainer_source
            if stage_root == training_root
            else source_identity(stage_root, stage_root / "scripts" / "train_sequence.py")
        )
        stage_sha = stage_identity.get("sha256")
        stage_files_sha = canonical_json_sha256(stage_identity.get("files"))
        expected_stage = admission.get("stage_training_source_identities", {})
        expected_record = (
            expected_stage.get(stage_key) if isinstance(expected_stage, Mapping) else None
        )
        if isinstance(expected_record, Mapping) and (
            expected_record.get("sha256") != stage_sha
            or expected_record.get("files_sha256") != stage_files_sha
        ):
            raise ValueError(f"stage {stage_key} source differs from its admitted identity")
        stage_source_records[stage_key] = {
            "root": str(stage_root),
            "revision": stage_revision,
            "sha256": str(stage_sha),
            "files_sha256": stage_files_sha,
        }
    predict_source = source_identity(
        evaluation_root, evaluation_root / "scripts" / "predict_sequence.py"
    )
    score_source = source_identity(
        evaluation_root, evaluation_root / "scripts" / "score_sequence.py"
    )
    policy_path = Path(str(validated["profile_checker_policy_path"]))
    profile_policy = validated["profile_policy"]
    reviewer_ref = profile_policy.get("reviewer_script")
    if not isinstance(reviewer_ref, Mapping):
        raise ValueError("admission must pin its exact independent profile reviewer")
    reviewer_path = Path(str(reviewer_ref.get("path"))).resolve(strict=True)
    reviewer_sha = reviewer_ref.get("sha256")
    reviewer_policy = read_json(policy_path)
    if (
        reviewer_path.is_symlink()
        or not isinstance(reviewer_sha, str)
        or file_sha256(reviewer_path) != reviewer_sha
        or reviewer_policy.get("reviewer_script_sha256") != reviewer_sha
    ):
        raise ValueError("admitted profile reviewer differs from the frozen policy bytes")
    run_manifest: dict[str, object] = {
        "schema": "wnn.multistage-run.v1",
        "experiment_id": contract["experiment_id"],
        "status": "prepared",
        "created_at_utc": utc_now(),
        "repo_root": str(source_root),
        "training_source_root": str(training_root),
        "training_source_revision": source_revision,
        "training_source_sha256": trainer_source["sha256"],
        "training_source_files_sha256": canonical_json_sha256(trainer_source["files"]),
        "stage_training_sources": stage_source_records,
        "evaluation_source_root": str(evaluation_root),
        "evaluation_source_revision": evaluation_revision,
        "evaluation_source_sha256": predict_source["sha256"],
        "evaluation_source_files_sha256": canonical_json_sha256(predict_source["files"]),
        "evaluation_score_script_sha256": score_source["files"]["scripts/score_sequence.py"],
        "contract": {"path": str(frozen_contract), "sha256": contract_sha},
        "admission": {"path": str(frozen_admission), "sha256": admission_sha},
        "auxiliary_manifest": {"path": str(aux_manifest_path), "sha256": aux_manifest_sha},
        "profile_checker_policy": {
            "path": validated["profile_checker_policy_path"],
            "sha256": validated["profile_checker_policy_sha256"],
        },
        "profile_reviewer_script": {
            "path": str(reviewer_path.resolve()),
            "sha256": reviewer_sha,
        },
        "limits": limits,
        "stage_fit_caps": stage_caps,
        "stage1_family_extension": {
            "path": validated["family_extension_path"],
            "sha256": validated["family_extension_sha256"],
        }
        if validated["family_extension_path"] is not None
        else None,
        "profile_updates": profile_updates,
        "profile_wall_seconds": profile_wall,
        "tpe_startup_trials": int(admission.get("tpe_startup_trials", 5)),
        "sampler_seed": int(admission.get("sampler_seed", 20261031)),
    }
    if run_manifest["tpe_startup_trials"] < 1:
        raise ValueError("TPE startup trial count must be positive")
    atomic_write_json(run_dir / "run.json", run_manifest, replace=False)
    atomic_write_json(
        run_dir / "cache.json",
        {"schema": "wnn.multistage-result-cache.v1", "entries": {}},
        replace=False,
    )
    preexisting = admission.get("preexisting_fit_attempts", [])
    initial_profiles = admission.get("preexisting_profile_attempts", 0)
    if not isinstance(preexisting, list) or type(initial_profiles) is not int:
        raise ValueError("preexisting attempt counters are malformed")
    initialize_budget(
        run_dir,
        max_fit_attempts=limits["max_fit_attempts"],
        max_profile_attempts=limits["max_profile_attempts"],
        stage_fit_caps=stage_caps,
        prior_fit_attempts=preexisting,
        prior_profile_attempts=initial_profiles,
        first_profile_epoch=None if profile_started is None else float(profile_started),
        campaign_wall_seconds=limits["campaign_wall_seconds"],
        collection_reserve_seconds=limits["collection_reserve_seconds"],
    )
    study = open_study(run_dir, "prepare")
    anchor_count = enqueue_anchors(study, validated["anchors"], validated["families"])
    run_manifest["queued_anchor_count"] = anchor_count
    atomic_write_json(run_dir / "run.json", run_manifest)
    for path in (run_dir / "stage1", run_dir / "stage2", run_dir / "stage3", run_dir / "stage4"):
        path.mkdir()
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "prepared",
            "at_utc": utc_now(),
            "contract_sha256": contract_sha,
            "admission_sha256": admission_sha,
            "queued_anchor_count": anchor_count,
            "status": "prepared",
        },
    )
    return run_manifest
