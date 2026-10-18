"""F1 plan, checkpoint-receipt, export and grade validators."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from result_io import (
    MAX_EVIDENCE_BYTES,
    hash_stable,
    membership_sha256,
    project_path,
    read_json,
    require,
)

Json = dict[str, Any]  # Producer JSON contains nested heterogeneous values.
BASE_SHA = "96cfb67b3447866651ae04638166f17dccb9729b3d4a9340f1b72b8d7b089f96"
RECOVERY_SHA = "f1aa897f0be902e114935f0ac9044b0616e2f3e3a889991651d028823938a111"
FOLD_SHA = "d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9"
V2_SUPPORT_RELATIVE = Path(
    "research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints_v2/PREPARATION_MANIFEST_V2_R2.json"
)
V2_SUPPORT_SHA = "bbebbed8c061776eba1732147ea6084057ed4d9cb249c8d11c1df0ea479eb2fd"
V4_SUPPORT_RELATIVE = Path(
    "research/campaign_2026-10-18/priority25_execution_r4/f1_recovery_r1/checkpoints_v4/PREPARATION_MANIFEST_V4.json"
)
V4_SUPPORT_SHA = "0492d5f569c46cbf498511dce62a6a8d33fec7ed986cae93f369b5ffe0a0970c"
TRAINING_STATE_RELATIVE = Path("source-a266b036/src/wnn/training_state.py")
LOCAL_TRAINING_STATE_RELATIVE = Path("src/wnn/training_state.py")
TERMINAL_ARTIFACTS = {
    "model_ep5.pt",
    "model.pt",
    "provenance.json",
    "training_state.pt",
    "history.json",
}
STEP_IDS = {
    "stage1_wide",
    "stage1_deep",
    "continuation_wide",
    "continuation_deep",
    "export_wide",
    "export_deep",
    "grade",
}


def sha_text(value: object, label: str) -> str:
    """Require a canonical lowercase SHA256 hex string."""
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        f"{label} is not a SHA256",
    )
    return value


def step_map(plan: Json) -> dict[str, Json]:
    """Index the exact seven-step frozen F1 execution sequence."""
    steps = plan.get("steps")
    require(isinstance(steps, list), "plan steps are missing")
    mapped = {step.get("step_id"): step for step in steps if isinstance(step, dict)}
    require(set(mapped) == STEP_IDS, "F1 plan step IDs do not match the frozen sequence")
    return mapped


def validate_plans(
    base: Json, recovery: Json, fold: Json, base_sha: str, recovery_sha: str, fold_sha: str
) -> tuple[dict[str, Json], dict[str, Json], list[int], int, dict[str, int]]:
    """Validate immutable F1 identities and the narrowly approved retry diff."""
    require(
        base_sha == BASE_SHA and recovery_sha == RECOVERY_SHA and fold_sha == FOLD_SHA,
        "base, recovery or fold manifest differs from the frozen F1 packet",
    )
    base_steps, recovery_steps = step_map(base), step_map(recovery)
    cell = base.get("cell", {})
    require(
        base.get("rank") == 1 and cell.get("fold") == 1 and cell.get("seed") == 1001,
        "base plan is not rank1 F1/seed1001",
    )
    require(
        recovery.get("experiment_id") == base.get("experiment_id")
        and recovery.get("hypothesis_id") == base.get("hypothesis_id"),
        "recovery identity differs from base",
    )
    require(
        recovery.get("source_revision") == base.get("source_revision")
        and recovery.get("grader_revision") == base.get("grader_revision"),
        "recovery plan changed source revisions",
    )
    require(
        fold.get("fold") == 1 and fold.get("n_folds") == 4, "fold manifest is not F1 of four folds"
    )
    require(
        base.get("source_bindings", {}).get("fold1_manifest_sha256") == fold_sha
        and recovery.get("source_bindings", {}).get("fold1_manifest_sha256") == fold_sha,
        "plan fold hash mismatch",
    )
    groups = fold.get("evaluation", {}).get("valid")
    require(
        isinstance(groups, list) and groups and all(type(x) is int and x >= 0 for x in groups),
        "fold evaluation membership is malformed",
    )
    require(len(groups) == len(set(groups)), "fold evaluation membership contains duplicates")
    fit_train, fit_valid = fold.get("fit", {}).get("train"), fold.get("fit", {}).get("valid")
    require(
        isinstance(fit_train, list)
        and isinstance(fit_valid, list)
        and all(type(x) is int and x >= 0 for x in fit_train + fit_valid),
        "fold fit membership is malformed",
    )
    require(
        fit_train
        and fit_valid
        and len(set(fit_train)) == len(fit_train)
        and len(set(fit_valid)) == len(fit_valid),
        "fold fit membership is empty or duplicated",
    )
    sequence_map = fold.get("sequences", {}).get("valid")
    require(isinstance(sequence_map, list), "fold valid sequence component map is missing")
    component_by_group = {
        item.get("row_group"): item.get("component")
        for item in sequence_map
        if isinstance(item, dict)
    }
    require(
        all(
            group in component_by_group and type(component_by_group[group]) is int
            for group in groups
        ),
        "fold evaluation group lacks a component identity",
    )
    components = len({component_by_group[group] for group in groups})
    schedule = base.get("derived_f1_training_schedule", {})
    per_epoch = schedule.get("expected_updates_per_epoch")
    stage1_updates = schedule.get("expected_stage1_updates")
    total_updates = schedule.get("expected_six_epoch_schedule_updates")
    batches = schedule.get("batches_per_epoch")
    require(
        all(type(x) is int and x > 0 for x in (per_epoch, stage1_updates, total_updates, batches)),
        "static F1 schedule is malformed",
    )
    expected = {
        "stage1_updates": stage1_updates,
        "stage1_batches": batches * 2,
        "terminal_updates": total_updates,
        "terminal_batches": batches * 6,
        "fit_train_groups": len(fit_train),
        "fit_valid_groups": len(fit_valid),
        "evaluation_groups": len(groups),
        "evaluation_components": components,
    }
    require(
        stage1_updates == per_epoch * 2 and total_updates == per_epoch * 6,
        "F1 update schedule fields are internally inconsistent",
    )

    restored = copy.deepcopy(recovery)
    original, retry = base_steps["continuation_wide"], step_map(restored)["continuation_wide"]
    recovery_attempt = retry.get("attempt")
    require(
        recovery_attempt == "computer-vision-homework-r01-wide-f1s1001-ep25-v2",
        "recovery wide attempt ID is unexpected",
    )
    original_attempt = original.get("attempt")
    command, original_command = retry.get("launch_command"), original.get("launch_command")
    require(
        isinstance(command, list) and isinstance(original_command, list),
        "wide continuation vectors are malformed",
    )
    index = command.index("--attempt") if "--attempt" in command else -1
    original_index = original_command.index("--attempt") if "--attempt" in original_command else -1
    require(index >= 0 and original_index >= 0, "wide continuation launch lacks --attempt")
    command[index + 1] = original_command[original_index + 1]
    retry["attempt"] = original_attempt
    require(
        restored == base, "recovery plan differs beyond its approved wide attempt substitutions"
    )
    require(
        recovery_attempt != original_attempt
        and _attempt_arg(recovery_steps["continuation_wide"]["launch_command"]) == recovery_attempt,
        "wide recovery command and attempt differ",
    )
    return base_steps, recovery_steps, groups, components, expected


def _attempt_arg(command: list[str]) -> str:
    """Return a launch vector's attempt argument."""
    require("--attempt" in command, "launch vector lacks --attempt")
    index = command.index("--attempt")
    require(index + 1 < len(command), "launch vector has a missing --attempt value")
    return command[index + 1]


def validate_attempt_record(
    record_path: Path,
    expected_id: str,
    expected_command: list[str],
    expected_sha: str | None = None,
) -> tuple[Json, str]:
    """Bind a successful attempt record to its exact ID and command."""
    record, digest, _ = read_json(record_path)
    if expected_sha is not None:
        require(
            digest == expected_sha,
            f"attempt record hash differs from its collection receipt: {record_path}",
        )
    require(
        record.get("id") == expected_id
        and record.get("status") == "finished"
        and record.get("exit_status") == 0,
        f"attempt did not finish successfully: {expected_id}",
    )
    require(
        record.get("command") == expected_command,
        f"attempt command differs from frozen plan: {expected_id}",
    )
    source_map = record.get("source_sha256")
    require(
        isinstance(source_map, dict) and source_map,
        f"attempt source manifest is missing: {expected_id}",
    )
    require(
        all(
            isinstance(k, str)
            and isinstance(v, str)
            and len(v) == 64
            and all(c in "0123456789abcdef" for c in v)
            for k, v in source_map.items()
        ),
        f"attempt source manifest is malformed: {expected_id}",
    )
    source_digest = hashlib.sha256(
        json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        record.get("source_manifest_sha256") == source_digest,
        f"attempt source manifest digest mismatch: {expected_id}",
    )
    return record, digest


def _artifact(receipt: Json, name: str) -> Json:
    """Get one named artifact entry from a producer receipt."""
    artifacts = receipt.get("artifacts")
    require(
        isinstance(artifacts, dict) and isinstance(artifacts.get(name), dict),
        f"receipt lacks artifact {name}",
    )
    return artifacts[name]


def _read_support_package(
    project: Path,
    relative: Path,
    expected_sha: str,
    manifest_key: str,
    expected_plan_keys: dict[str, str],
) -> tuple[Json, dict[str, str], dict[str, Path], str]:
    """Read and byte-check one exact helper package manifest and its files."""
    manifest_path = project_path(project / relative, project)
    manifest, manifest_sha, _ = read_json(manifest_path)
    require(manifest_sha == expected_sha, f"checkpoint support manifest changed: {relative}")
    for key, value in expected_plan_keys.items():
        require(
            manifest.get(key) == value, f"checkpoint support manifest {relative} has wrong {key}"
        )
    files = manifest.get(manifest_key)
    require(
        isinstance(files, dict) and files, f"checkpoint support file map is missing: {relative}"
    )
    package_files: dict[str, str] = {}
    package_paths: dict[str, Path] = {}
    for name, expected_file_sha in files.items():
        member = Path(name) if isinstance(name, str) else Path("/")
        require(
            isinstance(name, str)
            and name
            and not member.is_absolute()
            and ".." not in member.parts,
            "checkpoint support manifest has unsafe file path",
        )
        sha_text(expected_file_sha, f"checkpoint support file {name}")
        path = project_path(manifest_path.parent / member, project)
        actual_sha, _ = hash_stable(path, MAX_EVIDENCE_BYTES)
        require(
            actual_sha == expected_file_sha,
            f"checkpoint support file differs from its manifest: {name}",
        )
        package_files[name] = actual_sha
        package_paths[name] = path
    return manifest, package_files, package_paths, manifest_sha


def validate_helper_profiles(
    project: Path,
    terminal_meta: dict[str, Json],
    source_revision: str,
) -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
    """Require V2 stage1/deep and the V4 wide terminal helper sets."""
    _, v2_files, v2_paths, v2_sha = _read_support_package(
        project,
        V2_SUPPORT_RELATIVE,
        V2_SUPPORT_SHA,
        "helper_files_sha256",
        {
            "schema": "priority25.f1_checkpoint_support_preparation.v2",
            "plan_sha256": BASE_SHA,
            "canonical_trainer_revision": source_revision,
        },
    )
    _, v4_files, v4_paths, v4_sha = _read_support_package(
        project,
        V4_SUPPORT_RELATIVE,
        V4_SUPPORT_SHA,
        "support_files_sha256",
        {
            "schema": "priority25.f1_recovery_checkpoint_support_preparation.v4",
            "base_plan_sha256": BASE_SHA,
            "recovery_plan_sha256": RECOVERY_SHA,
        },
    )
    source_candidates = (project / TRAINING_STATE_RELATIVE, project / LOCAL_TRAINING_STATE_RELATIVE)
    existing_sources = [path for path in source_candidates if path.exists()]
    require(
        len(existing_sources) == 1,
        "expected exactly one pinned trainer source tree for training_state.py",
    )
    training_state = project_path(existing_sources[0], project)
    training_state_sha, _ = hash_stable(training_state, MAX_EVIDENCE_BYTES)
    v2_dependency_names = (
        "f1_checkpoint_common.py",
        "f1_checkpoint_evidence.py",
        "f1_checkpoint_lineage.py",
        "f1_checkpoint_tensors.py",
        "collect_f1_checkpoint.py",
    )
    v2_profile = {
        "collector_sha256": v2_files["collect_f1_checkpoint.py"],
        "collector_dependencies": {"f1_checkpoint_common.py": v2_files["f1_checkpoint_common.py"]},
        "verification_helper_sha256": v2_files["verify_f1_checkpoint.py"],
        "verification_dependencies": {name: v2_files[name] for name in v2_dependency_names}
        | {"wnn/training_state.py": training_state_sha},
    }
    v4_profile = {
        "collector_sha256": v4_files["collect_f1_checkpoint.py"],
        "collector_dependencies": {"f1_checkpoint_common.py": v4_files["f1_checkpoint_common.py"]},
        "verification_helper_sha256": v4_files["verify_f1_checkpoint.py"],
        "verification_dependencies": {name: v4_files[name] for name in v2_dependency_names}
        | {"wnn/training_state.py": training_state_sha},
    }
    for arm in ("wide", "deep"):
        meta = terminal_meta[arm]
        phases = (
            (meta["stage1_collection"], meta["stage1_verification"], v2_profile),
            (meta["collection"], meta["verification"], v4_profile if arm == "wide" else v2_profile),
        )
        for collection, verification, profile in phases:
            require(
                collection.get("collector_sha256") == profile["collector_sha256"]
                and collection.get("collector_dependencies") == profile["collector_dependencies"]
                and verification.get("verification_helper_sha256")
                == profile["verification_helper_sha256"]
                and verification.get("verification_dependencies")
                == profile["verification_dependencies"],
                f"{arm} checkpoint helper profile differs from its pinned support package",
            )
    paths: dict[str, Path] = {
        "support/v2/PREPARATION_MANIFEST_V2_R2.json": project_path(
            project / V2_SUPPORT_RELATIVE, project
        ),
        "support/v4/PREPARATION_MANIFEST_V4.json": project_path(
            project / V4_SUPPORT_RELATIVE, project
        ),
        "code/training_state.py": training_state,
    }
    hashes = {
        "support/v2/PREPARATION_MANIFEST_V2_R2.json": v2_sha,
        "support/v4/PREPARATION_MANIFEST_V4.json": v4_sha,
        "code/training_state.py": training_state_sha,
    }
    paths.update({f"support/v2/{name}": path for name, path in v2_paths.items()})
    hashes.update({f"support/v2/{name}": digest for name, digest in v2_files.items()})
    paths.update({f"support/v4/{name}": path for name, path in v4_paths.items()})
    hashes.update({f"support/v4/{name}": digest for name, digest in v4_files.items()})
    return paths, hashes, {"v2": v2_sha, "wide_terminal_v4": v4_sha}


def validate_terminal(
    project: Path,
    arm: str,
    base_steps: dict[str, Json],
    recovery_steps: dict[str, Json],
    base_sha: str,
    recovery_sha: str,
    fold_sha: str,
    source_revision: str,
    registry_sha: str,
    terminal_collection_path: Path,
    terminal_verification_path: Path,
    stage_collection_path: Path,
    stage_verification_path: Path,
    expected: dict[str, int],
) -> tuple[Json, Json, Json, Json, Json, Json, dict[str, str], dict[str, int]]:
    """Bind terminal and stage1 receipt identities, counters and same-arm ancestry."""
    collection, csha, _ = read_json(terminal_collection_path)
    verification, vsha, _ = read_json(terminal_verification_path)
    stage_collection, scsha, _ = read_json(stage_collection_path)
    stage_verification, svsha, _ = read_json(stage_verification_path)
    step = (
        recovery_steps[f"continuation_{arm}"]
        if arm == "wide"
        else base_steps[f"continuation_{arm}"]
    )
    stage_step = base_steps[f"stage1_{arm}"]
    run, stage_run, attempt_id, stage_attempt = (
        step["run_name"],
        stage_step["run_name"],
        step["attempt"],
        stage_step["attempt"],
    )
    require(
        verification.get("status") == "passed"
        and verification.get("phase") == "terminal"
        and verification.get("arm") == arm
        and verification.get("run") == run,
        f"{arm} terminal verification identity/status mismatch",
    )
    require(
        verification.get("plan_sha256") == base_sha
        and verification.get("fold1_manifest_sha256") == fold_sha
        and verification.get("next_epoch") == 6,
        f"{arm} terminal plan/fold/epoch mismatch",
    )
    require(
        verification.get("source_revision") == source_revision, f"{arm} source revision mismatch"
    )
    require(
        verification.get("portable_registry_sha256") == registry_sha,
        f"{arm} portable registry mismatch",
    )
    evidence = verification.get("attempt")
    require(
        isinstance(evidence, dict)
        and evidence.get("id") == attempt_id
        and evidence.get("status") == "finished"
        and evidence.get("exit_status") == 0,
        f"{arm} terminal attempt mismatch",
    )
    require(
        collection.get("phase") == "terminal"
        and collection.get("arm") == arm
        and collection.get("run") == run
        and collection.get("attempt") == attempt_id,
        f"{arm} terminal collection identity mismatch",
    )
    require(
        collection.get("plan_sha256") == base_sha
        and collection.get("source_snapshot_hash_match") is True
        and collection.get("producer_model_hash_matches") is True,
        f"{arm} terminal source/artifact integrity mismatch",
    )
    require(
        collection.get("source_stats_before") == collection.get("source_stats_after"),
        f"{arm} terminal source changed during collection",
    )
    require(
        collection.get("attempt_evidence", {}).get("sha256") == evidence.get("record_sha256"),
        f"{arm} terminal attempt hash mismatch",
    )
    require(
        set(collection.get("artifacts", {})) == TERMINAL_ARTIFACTS,
        f"{arm} terminal artifact set differs from the producer contract",
    )
    for receipt, label in (
        (collection, "terminal collection"),
        (verification, "terminal verification"),
    ):
        for name in TERMINAL_ARTIFACTS:
            artifact = _artifact(receipt, name)
            sha_text(artifact.get("sha256"), f"{arm} {label} {name}")
            require(
                type(artifact.get("bytes")) is int and artifact["bytes"] > 0,
                f"{arm} {label} {name} byte count is invalid",
            )
    require(
        verification.get("collection_receipt_sha256") == csha,
        f"{arm} verifier binds another collection receipt",
    )
    observed = verification.get("completed_update_counts")
    require(isinstance(observed, dict), f"{arm} terminal update counters missing")
    terminal_counts = {
        "optimizer_attempts": expected["terminal_updates"],
        "optimizer_completed_updates": expected["terminal_updates"],
        "amp_skipped_updates": 0,
        "ema_updates": 0,
        "completed_batches": expected["terminal_batches"],
    }
    require(
        all(observed.get(key) == value for key, value in terminal_counts.items()),
        f"{arm} terminal counters mismatch",
    )
    require(
        verification.get("planned_schedule_updates") == expected["terminal_updates"]
        and verification.get("planned_schedule_batches") == expected["terminal_batches"],
        f"{arm} terminal planned schedule mismatch",
    )
    require(
        verification.get("scheduler_last_epoch") == expected["terminal_updates"]
        and verification.get("scheduler_step_count") == expected["terminal_updates"] + 1
        and verification.get("history_epochs_verified") == 6,
        f"{arm} terminal scheduler/history mismatch",
    )
    require(
        stage_verification.get("status") == "passed"
        and stage_verification.get("phase") == "stage1"
        and stage_verification.get("arm") == arm
        and stage_verification.get("run") == stage_run,
        f"{arm} stage1 verification mismatch",
    )
    require(
        stage_verification.get("plan_sha256") == base_sha
        and stage_verification.get("fold1_manifest_sha256") == fold_sha
        and stage_verification.get("next_epoch") == 2
        and stage_verification.get("source_revision") == source_revision
        and stage_verification.get("portable_registry_sha256") == registry_sha,
        f"{arm} stage1 plan/fold/epoch/source mismatch",
    )
    require(
        stage_verification.get("attempt", {}).get("id") == stage_attempt
        and stage_verification.get("attempt", {}).get("status") == "finished"
        and stage_verification.get("attempt", {}).get("exit_status") == 0,
        f"{arm} stage1 attempt mismatch",
    )
    require(
        stage_verification.get("collection_receipt_sha256") == scsha,
        f"{arm} stage1 verifier binds another collection receipt",
    )
    require(
        stage_collection.get("phase") == "stage1"
        and stage_collection.get("arm") == arm
        and stage_collection.get("run") == stage_run
        and stage_collection.get("attempt") == stage_attempt
        and stage_collection.get("plan_sha256") == base_sha,
        f"{arm} stage1 collection mismatch",
    )
    require(
        stage_collection.get("source_stats_before") == stage_collection.get("source_stats_after")
        and stage_collection.get("source_snapshot_hash_match") is True,
        f"{arm} stage1 collection integrity mismatch",
    )
    stage_record_ref = stage_collection.get("attempt_evidence", {})
    stage_record_path = project_path(Path(stage_record_ref.get("path", "")), project)
    stage_record, stage_record_sha = validate_attempt_record(
        stage_record_path,
        stage_attempt,
        stage_step["expected_command"],
        stage_record_ref.get("sha256"),
    )
    require(
        stage_record_sha == stage_verification.get("attempt", {}).get("record_sha256"),
        f"{arm} stage1 attempt record differs from verifier",
    )
    stage_counts = stage_verification.get("completed_update_counts", {})
    require(
        stage_counts.get("optimizer_attempts") == expected["stage1_updates"]
        and stage_counts.get("optimizer_completed_updates") == expected["stage1_updates"]
        and stage_counts.get("amp_skipped_updates") == 0
        and stage_counts.get("ema_updates") == 0
        and stage_counts.get("completed_batches") == expected["stage1_batches"],
        f"{arm} stage1 counters mismatch",
    )
    require(
        stage_verification.get("scheduler_last_epoch") == expected["stage1_updates"]
        and stage_verification.get("scheduler_step_count") == expected["stage1_updates"] + 1
        and stage_verification.get("history_epochs_verified") == 2,
        f"{arm} stage1 scheduler/history mismatch",
    )
    state_sha = _artifact(stage_verification, "training_state.pt").get("sha256")
    require(
        _artifact(stage_collection, "training_state.pt").get("sha256") == state_sha
        and verification.get("stage1_parent_training_state_sha256") == state_sha,
        f"{arm} terminal ancestry state mismatch",
    )
    parent = collection.get("resumed_from", {})
    require(
        parent.get("sha256") == state_sha and parent.get("next_epoch") == 2,
        f"{arm} continuation did not resume the exact stage1 checkpoint",
    )
    expected_parent_path = Path(stage_step["run_output_directory"]) / "training_state.pt"
    require(
        parent.get("path") == str(expected_parent_path),
        f"{arm} continuation parent path differs from its stage1 run",
    )
    record_ref = collection.get("attempt_evidence", {})
    record_path = project_path(Path(record_ref.get("path", "")), project)
    record, record_sha = validate_attempt_record(
        record_path, attempt_id, step["expected_command"], record_ref.get("sha256")
    )
    require(
        record_sha == evidence.get("record_sha256"),
        f"{arm} terminal attempt record differs from verifier",
    )
    require(
        verification.get("source_archive_manifest_sha256")
        == stage_verification.get("source_archive_manifest_sha256")
        and verification.get("portable_registry_sha256")
        == stage_verification.get("portable_registry_sha256"),
        f"{arm} source/registry ancestry mismatch",
    )
    if arm == "wide":
        binding = verification.get("recovery_binding")
        rejected = base_steps["continuation_wide"].get("attempt")
        require(
            isinstance(binding, dict)
            and binding.get("base_plan_sha256") == base_sha
            and binding.get("recovery_plan_sha256") == recovery_sha
            and binding.get("recovery_attempt_id") == attempt_id,
            "wide recovery binding mismatch",
        )
        require(
            binding.get("rejected_attempt_id") == rejected
            and binding.get("rejected_attempt_status") == "not_started"
            and binding.get("rejected_attempt_phase") == "admission_failed",
            "wide rejected-attempt binding mismatch",
        )
        require(
            collection.get("recovery_binding") == binding,
            "wide collection and verification recovery bindings differ",
        )
    for name in TERMINAL_ARTIFACTS:
        artifact, verified = _artifact(collection, name), _artifact(verification, name)
        require(
            artifact.get("sha256") == verified.get("sha256")
            and artifact.get("bytes") == verified.get("bytes"),
            f"{arm} terminal artifact receipts differ: {name}",
        )
    hashes = {
        "collection": csha,
        "verification": vsha,
        "stage1_collection": scsha,
        "stage1_verification": svsha,
        "collection_helper": collection.get("collector_sha256"),
        "verification_helper": verification.get("verification_helper_sha256"),
        "stage1_collection_helper": stage_collection.get("collector_sha256"),
        "stage1_verification_helper": stage_verification.get("verification_helper_sha256"),
        "stage1_attempt_record": stage_record_sha,
        "terminal_attempt_record": record_sha,
        "stage1_state": state_sha,
        "terminal_state": _artifact(verification, "training_state.pt").get("sha256"),
        "checkpoint": _artifact(verification, "model_ep5.pt").get("sha256"),
        "provenance": _artifact(verification, "provenance.json").get("sha256"),
        "source_archive": verification.get("source_archive_manifest_sha256"),
        "portable_registry": verification.get("portable_registry_sha256"),
    }
    for key, digest in hashes.items():
        sha_text(digest, f"{arm} {key}")
    return (
        collection,
        verification,
        stage_collection,
        stage_verification,
        csha,
        vsha,
        hashes,
        terminal_counts,
    )


def _helper_identity(collection: Json, verification: Json) -> Json:
    """Return the source-qualified helper identity recorded by two receipts."""
    collection_sha = sha_text(collection.get("collector_sha256"), "checkpoint collector helper")
    verification_sha = sha_text(
        verification.get("verification_helper_sha256"), "checkpoint verifier helper"
    )
    collection_dependencies = collection.get("collector_dependencies")
    verification_dependencies = verification.get("verification_dependencies")
    for value, label in (
        (collection_dependencies, "collector dependencies"),
        (verification_dependencies, "verifier dependencies"),
    ):
        require(isinstance(value, dict) and value, f"{label} are missing")
        for name, digest in value.items():
            require(isinstance(name, str) and name, f"{label} contain an invalid filename")
            sha_text(digest, f"{label} {name}")
    return {
        "collection_helper_sha256": collection_sha,
        "collection_dependencies": collection_dependencies,
        "verification_helper_sha256": verification_sha,
        "verification_dependencies": verification_dependencies,
    }


def _sealed_phase(
    collection: Json, verification: Json, collection_sha: str, verification_sha: str
) -> Json:
    """Build the exact receipt/helper binding that an independent seal reviews."""
    attempt = verification.get("attempt")
    require(
        isinstance(attempt, dict) and isinstance(attempt.get("id"), str),
        "checkpoint verifier attempt identity is missing",
    )
    helpers = _helper_identity(collection, verification)
    return {
        "attempt_id": attempt["id"],
        "collection_receipt_sha256": collection_sha,
        "verification_receipt_sha256": verification_sha,
        "collection_schema": collection.get("schema"),
        "verification_schema": verification.get("schema"),
        **helpers,
    }


def validate_terminal_seal(
    seal: Json,
    seal_sha: str,
    base_sha: str,
    recovery_sha: str,
    fold_sha: str,
    arms: dict[str, tuple[Json, Json, str, str, dict[str, str]]],
    terminal_meta: dict[str, Json],
    review_receipts: dict[str, tuple[Json, str]],
    collector_support_sha256: dict[str, str],
    checkpoint_support_manifests_sha256: dict[str, str],
) -> None:
    """Require an arrival-time review seal for all receipts and helper bytes."""
    require(
        seal.get("schema") == "priority25.f1_terminal_evidence_seal.v2"
        and seal.get("status") == "sealed"
        and seal.get("review_status") == "passed",
        "terminal evidence seal is not reviewed and sealed",
    )
    require(
        seal.get("base_plan_sha256") == base_sha
        and seal.get("recovery_plan_sha256") == recovery_sha
        and seal.get("fold1_manifest_sha256") == fold_sha,
        "terminal seal plan/fold identity mismatch",
    )
    require(
        seal.get("collector_support_sha256") == collector_support_sha256,
        "result collector support-file binding mismatch",
    )
    require(
        seal.get("checkpoint_support_manifests_sha256") == checkpoint_support_manifests_sha256,
        "checkpoint support-manifest binding mismatch",
    )
    reviews = seal.get("review_receipts")
    require(
        isinstance(reviews, dict) and set(reviews) == {"wide", "deep"},
        "terminal seal lacks both independent review receipts",
    )
    for arm in ("wide", "deep"):
        collection, verification, csha, vsha, _ = arms[arm]
        meta = terminal_meta[arm]
        entry = seal.get("arms", {}).get(arm)
        require(isinstance(entry, dict), f"terminal seal lacks {arm} entry")
        expected_phases = {
            "stage1": _sealed_phase(
                meta["stage1_collection"],
                meta["stage1_verification"],
                meta["hashes"]["stage1_collection"],
                meta["hashes"]["stage1_verification"],
            ),
            "terminal": _sealed_phase(collection, verification, csha, vsha),
        }
        require(
            entry.get("receipts") == expected_phases,
            f"terminal seal {arm} receipt/helper bindings mismatch",
        )
        review, review_sha = review_receipts[arm]
        expected_review = {
            "schema": "priority25.f1_terminal_result_review.v1",
            "status": "passed",
            "arm": arm,
            "base_plan_sha256": base_sha,
            "recovery_plan_sha256": recovery_sha,
            "fold1_manifest_sha256": fold_sha,
            "receipts": expected_phases,
            "collector_support_sha256": collector_support_sha256,
            "checkpoint_support_manifests_sha256": checkpoint_support_manifests_sha256,
        }
        require(
            all(review.get(key) == value for key, value in expected_review.items()),
            f"{arm} review does not bind the exact result and support files",
        )
        review_ref = reviews[arm]
        require(
            isinstance(review_ref, dict)
            and review_ref.get("sha256") == review_sha
            and review_ref.get("status") == "passed",
            f"terminal seal {arm} review receipt binding mismatch",
        )
        review_status = review.get("status")
        review_passed = isinstance(review_status, str) and (
            review_status.lower() == "passed" or review_status.startswith("PASS")
        )
        require(
            review_passed or review.get("review_status") == "passed",
            f"{arm} independent review receipt is not passed",
        )
        sha_text(review_sha, f"{arm} independent review")
    require(seal_sha != "", "terminal evidence seal hash is empty")


def validate_sidecar(
    sidecar: Json,
    run: str,
    groups: list[int],
    fold_sha: str,
    checkpoint_sha: str,
    prediction_sha: str,
) -> None:
    """Require the canonical feature-only FP32 evaluation export manifest."""
    expected = {
        "run": run,
        "split": "valid",
        "epoch": 5,
        "split_manifest_sha256": fold_sha,
        "campaign_split_manifest_sha256": fold_sha,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": membership_sha256(groups),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
    }
    for key, value in expected.items():
        require(sidecar.get(key) == value, f"export sidecar {run} field {key} mismatch")


def validate_grade(
    grade: Json,
    groups: list[int],
    components: int,
    fold_sha: str,
    arms: dict[str, Json],
    prediction_shas: dict[str, str],
    sidecar_shas: dict[str, str],
    bootstrap_count: int,
) -> None:
    """Validate the official paired grade schema without predicting its score."""
    require(
        grade.get("schema") == "priority25.endpoint_grade.v1"
        and grade.get("fold") == 1
        and grade.get("n_folds") == 4,
        "grade schema/fold mismatch",
    )
    require(
        grade.get("split_manifest_sha256") == fold_sha
        and grade.get("evaluation_row_groups") == groups
        and grade.get("evaluation_components") == components
        and grade.get("metric_id") == "global_wp_clip2_v1",
        "grade membership/component/metric mismatch",
    )
    target_sha = grade.get("target_cache_sha256")
    require(
        isinstance(target_sha, str)
        and len(target_sha) == 64
        and all(c in "0123456789abcdef" for c in target_sha),
        "grade target-cache hash is malformed",
    )
    scores: dict[str, Json] = {}
    rows: list[int] = []
    for field, arm in (("endpoint", "wide"), ("reference", "deep")):
        endpoint = grade.get(field)
        require(
            isinstance(endpoint, dict)
            and endpoint.get("run") == arms[arm]["run"]
            and endpoint.get("epoch") == 5,
            f"grade {field} identity mismatch",
        )
        require(
            endpoint.get("checkpoint_sha256") == arms[arm]["checkpoint_sha256"]
            and endpoint.get("provenance_sha256") == arms[arm]["provenance_sha256"],
            f"grade {field} checkpoint mismatch",
        )
        require(
            endpoint.get("prediction_sha256") == prediction_shas[arm]
            and endpoint.get("sidecar_sha256") == sidecar_shas[arm],
            f"grade {field} export mismatch",
        )
        score = endpoint.get("score")
        require(
            isinstance(score, dict) and score.get("metric_id") == "global_wp_clip2_v1",
            f"grade {field} score missing",
        )
        require(
            all(
                type(score.get(key)) in (int, float)
                and math.isfinite(score[key])
                and 0 <= score[key] <= 1
                for key in ("wp", "t0", "t1")
            ),
            f"grade {field} score is invalid",
        )
        require(
            math.isclose(score["wp"], (score["t0"] + score["t1"]) / 2, rel_tol=0, abs_tol=1e-12),
            f"grade {field} score aggregates inconsistently",
        )
        selected = score.get("selected_rows")
        require(type(selected) is int and selected > 0, f"grade {field} selected_rows is invalid")
        rows.append(selected)
        scores[field] = score
    require(rows[0] == rows[1], "grade endpoint/reference selected-row counts differ")
    paired = grade.get("paired_component_bootstrap")
    require(
        isinstance(paired, dict)
        and paired.get("resamples") == bootstrap_count
        and paired.get("n_boot") == bootstrap_count
        and paired.get("seed") == 271828
        and paired.get("components") == components
        and paired.get("n_components") == components
        and paired.get("n_sequences") == len(groups)
        and paired.get("centered_moments") is True
        and paired.get("metric_id") == "global_wp_clip2_v1"
        and paired.get("method") == "paired whole-component bootstrap using centered moments",
        "paired bootstrap configuration mismatch",
    )
    numeric = ("mean_diff", "bootstrap_sd", "p_le_0", "candidate_wp", "reference_wp")
    require(
        all(
            type(paired.get(key)) in (int, float) and math.isfinite(paired[key]) for key in numeric
        ),
        "paired bootstrap contains nonfinite summary",
    )
    require(
        paired["bootstrap_sd"] >= 0
        and 0 <= paired["p_le_0"] <= 1
        and math.isclose(paired["candidate_wp"], scores["endpoint"]["wp"], rel_tol=0, abs_tol=1e-12)
        and math.isclose(
            paired["reference_wp"], scores["reference"]["wp"], rel_tol=0, abs_tol=1e-12
        )
        and math.isclose(
            paired["mean_diff"],
            paired["candidate_wp"] - paired["reference_wp"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "paired bootstrap summaries disagree with endpoint scores",
    )
    interval = paired.get("ci95")
    require(
        isinstance(interval, list)
        and len(interval) == 2
        and all(type(x) in (int, float) and math.isfinite(x) for x in interval)
        and interval[0] <= interval[1]
        and -1 <= interval[0] <= 1
        and -1 <= interval[1] <= 1,
        "paired bootstrap interval is invalid",
    )
