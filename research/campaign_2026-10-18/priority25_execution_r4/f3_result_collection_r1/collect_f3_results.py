#!/usr/bin/env python3
"""Capture the existing F3/seed1001 verified exports and official paired grade."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import f3_result_checks as checks
from result_io import (
    MAX_EVIDENCE_BYTES,
    MAX_JSON_BYTES,
    MAX_PREDICTION_BYTES,
    MAX_SIDECAR_BYTES,
    copy_stable,
    hash_stable,
    inspect_npy,
    project_path,
    read_json,
    require,
)

Json = dict[str, Any]
REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
SCHEMA = "priority25.f3_result_collection.v1"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
SUPPORT_REL = Path("research/campaign_2026-10-18/priority25_execution_r4/f3_result_collection_r1")


def _limit(name: str) -> int:
    if name.endswith(".npy"):
        return MAX_PREDICTION_BYTES
    if name.startswith("predictions/") and name.endswith(".json"):
        return MAX_SIDECAR_BYTES
    if name.endswith(".log"):
        return MAX_EVIDENCE_BYTES
    return MAX_JSON_BYTES if name.endswith(".json") else MAX_EVIDENCE_BYTES


def _support(
    project: Path,
) -> tuple[
    dict[str, Path],
    dict[str, str],
    dict[str, str],
    dict[Path, str],
    dict[str, str],
    dict[str, Path],
]:
    packet = project_path(project / SUPPORT_REL, project)
    manifest_path = project_path(project / checks.CHECKPOINT_MANIFEST_REL, project)
    manifest, manifest_sha, _ = read_json(manifest_path)
    require(
        manifest_sha == checks.CHECKPOINT_MANIFEST_SHA, "F3 checkpoint support manifest SHA differs"
    )
    helper_hashes: dict[str, str] = {}
    helper_paths: dict[str, Path] = {}
    helper_source_hashes: dict[Path, str] = {}
    for name in checks.COLLECTOR_FILES:
        path = project_path(packet / name, project)
        digest, _ = hash_stable(path, MAX_EVIDENCE_BYTES)
        helper_hashes[name] = digest
        helper_paths[name] = path
        helper_source_hashes[path] = digest
    for name in ("README.md", "PREPARATION_MANIFEST.json", "TEST_RECEIPT.json"):
        path = project_path(packet / name, project)
        digest, _ = hash_stable(path, MAX_EVIDENCE_BYTES)
        helper_paths[name] = path
        helper_source_hashes[path] = digest
    checkpoint_artifacts = manifest.get("artifacts")
    require(isinstance(checkpoint_artifacts, dict), "checkpoint support artifact map missing")
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    checkpoint_source_hashes: dict[Path, str] = {}
    for name, entry in checkpoint_artifacts.items():
        require(
            isinstance(entry, dict) and isinstance(entry.get("sha256"), str),
            "checkpoint support entry malformed",
        )
        path = project_path(manifest_path.parent / name, project)
        digest, _ = hash_stable(path, MAX_EVIDENCE_BYTES)
        require(digest == entry["sha256"], f"checkpoint support source differs: {name}")
        checkpoint_hashes[name] = digest
        checkpoint_paths[name] = path
        checkpoint_source_hashes[path] = digest
    short_names = [Path(name).name for name in checkpoint_hashes]
    require(
        len(set(short_names)) == len(short_names), "checkpoint helper artifact basenames collide"
    )
    support_hashes_by_name = {Path(name).name: digest for name, digest in checkpoint_hashes.items()}
    return (
        helper_paths,
        helper_hashes,
        {"f3_checkpoint_support_r1": manifest_sha},
        helper_source_hashes | checkpoint_source_hashes,
        support_hashes_by_name,
        checkpoint_paths,
    )


def _checkpoint_inputs(
    project: Path,
    plan: Json,
    steps: dict[str, Json],
    arm: str,
    phase: str,
    support_files: dict[str, str],
    expected_membership: Json,
) -> tuple[Json, Json, str, str, dict[str, Json]]:
    step = steps[checks.CHECKPOINT_STEPS[(arm, phase)]]
    attempt_path = project_path(project / "runtime/attempts" / f"{step['attempt']}.json", project)
    attempt, attempt_sha, _ = read_json(attempt_path)
    checks.validate_attempt(attempt_path, step, project, project / checks.PLAN_REL, checks.PLAN_SHA)
    collection_path = project_path(Path(step["collection_receipt_path"]), project)
    verification_path = project_path(Path(step["verification_receipt_path"]), project)
    collection, verification, csha, vsha, artifacts = checks.validate_checkpoint_phase(
        project,
        plan,
        steps,
        arm,
        phase,
        collection_path,
        verification_path,
        attempt,
        attempt_sha,
        support_files,
        expected_membership,
    )
    return (
        collection,
        verification,
        csha,
        vsha,
        {
            "collection_path": collection_path,
            "collection_bytes": collection_path.stat().st_size,
            "collection_sha": csha,
            "verification_path": verification_path,
            "verification_bytes": verification_path.stat().st_size,
            "verification_sha": vsha,
            "snapshot_path": Path(step["archive_directory"]),
            "artifacts": artifacts,
        },
    )


def collect(args: argparse.Namespace) -> Json:
    """Validate all sources before create-only publication of a result snapshot."""
    project = args.project_root.resolve(strict=True)
    require(project == REMOTE_ROOT, "collector must run on the exact Beleriand campaign root")
    (
        helper_paths,
        helper_hashes,
        checkpoint_manifests,
        support_source_hashes,
        support_files_by_name,
        checkpoint_artifact_paths,
    ) = _support(project)
    plan_path = project_path(args.plan, project)
    plan, plan_sha, steps, fold, groups, components = checks.validate_plan(project, plan_path)
    expected_membership = checks.expected_fold_membership(fold)
    snapshot = project_path(args.snapshot_dir, project)
    receipt_path = project_path(args.receipt, project)
    require(not snapshot.exists() and not snapshot.is_symlink(), "snapshot path is create-only")
    require(
        not receipt_path.exists() and not receipt_path.is_symlink(),
        "result receipt path is create-only",
    )
    require(
        snapshot != receipt_path
        and not snapshot.is_relative_to(receipt_path)
        and not receipt_path.is_relative_to(snapshot),
        "output paths overlap",
    )

    expected_contract = project / checks.PLAN_REL
    arms: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for arm in ("wide", "deep"):
        checkpoints[arm] = {}
        for phase in ("stage1", "terminal"):
            collection, verification, csha, vsha, meta = _checkpoint_inputs(
                project,
                plan,
                steps,
                arm,
                phase,
                support_files=support_files_by_name,
                expected_membership=expected_membership,
            )
            # Receipts and reviews are captured only after the files above have
            # been read from their exact plan-declared paths.
            meta.update({"collection": collection, "verification": verification})
            checkpoints[arm][phase] = meta
        stage, terminal = checkpoints[arm]["stage1"], checkpoints[arm]["terminal"]
        stage_verify = stage["verification"]
        terminal_verify = terminal["verification"]
        require(
            stage_verify.get("fold3_membership") == terminal_verify.get("fold3_membership")
            and stage_verify.get("fit_membership_identity")
            == terminal_verify.get("fit_membership_identity")
            and stage_verify.get("normalization_identity")
            == terminal_verify.get("normalization_identity"),
            f"{arm} stage1 and terminal fit/fold identities differ",
        )
        state_sha = stage["artifacts"]["training_state.pt"]["sha256"]
        require(
            terminal_verify.get("stage1_parent_training_state_sha256") == state_sha
            and terminal["collection"].get("resumed_from")
            == {
                "path": str(
                    Path(steps[f"stage1_{arm}"]["run_output_directory"]) / "training_state.pt"
                ),
                "sha256": state_sha,
                "next_epoch": 2,
            },
            f"{arm} terminal resume ancestry differs",
        )
        arms[arm] = {"stage1": stage, "terminal": terminal}
    require(
        checkpoints["wide"]["terminal"]["verification"].get("fold3_membership")
        == checkpoints["deep"]["terminal"]["verification"].get("fold3_membership")
        and checkpoints["wide"]["terminal"]["verification"].get("fit_membership_identity")
        == checkpoints["deep"]["terminal"]["verification"].get("fit_membership_identity")
        and checkpoints["wide"]["terminal"]["verification"].get("normalization_identity")
        == checkpoints["deep"]["terminal"]["verification"].get("normalization_identity"),
        "wide/deep terminal fitting or fold identities differ",
    )

    seal_path = project_path(args.terminal_seal, project)
    seal, seal_sha, _ = read_json(seal_path)
    seal_arms = {}
    for arm in ("wide", "deep"):
        seal_arms[arm] = {}
        for phase in ("stage1", "terminal"):
            meta = checkpoints[arm][phase]
            seal_arms[arm][phase] = meta
            for kind in ("collection", "verification"):
                path = meta[f"{kind}_path"]
                meta[f"{kind}_bytes"] = path.stat().st_size
    review_bindings = checks.validate_seal(project, seal_path, seal_sha, seal_arms, helper_hashes)

    prediction_info: dict[str, Json] = {}
    prediction_paths: dict[str, Path] = {}
    sidecar_paths: dict[str, Path] = {}
    prediction_shas: dict[str, str] = {}
    sidecar_shas: dict[str, str] = {}
    prediction_stats: dict[str, dict[str, int]] = {}
    grade_arms: dict[str, Json] = {}
    for arm in ("wide", "deep"):
        step = steps[f"export_{arm}"]
        pred_path = project_path(Path(step["prediction_path"]), project)
        sidecar_path = project_path(Path(step["sidecar_path"]), project)
        require(sidecar_path == pred_path.with_suffix(".json"), f"{arm} sidecar path differs")
        pred_header, pred_sha, pred_stat = inspect_npy(pred_path, [505, 20_000, 2])
        sidecar, side_sha, _ = read_json(sidecar_path, MAX_SIDECAR_BYTES)
        terminal_verify = checkpoints[arm]["terminal"]["verification"]
        checkpoint_sha = terminal_verify["artifacts"]["model_ep5.pt"]["sha256"]
        checks.validate_sidecar(sidecar, str(step["run_name"]), groups, checkpoint_sha, pred_sha)
        prediction_paths[arm], sidecar_paths[arm] = pred_path, sidecar_path
        prediction_shas[arm], sidecar_shas[arm] = pred_sha, side_sha
        prediction_stats[arm] = pred_stat
        prediction_info[arm] = {"path": str(pred_path), "sha256": pred_sha, "header": pred_header}
        grade_arms[arm] = {
            "run": step["run_name"],
            "checkpoint_sha256": checkpoint_sha,
            "provenance_sha256": terminal_verify["artifacts"]["provenance.json"]["sha256"],
        }

    grade_step = steps["grade"]
    grade_path = project_path(Path(grade_step["grade_output"]), project)
    grade, grade_sha, grade_identity = read_json(grade_path)
    grade_command = grade_step["expected_command"]
    bootstrap = int(grade_command[grade_command.index("--bootstrap") + 1])
    checks.validate_grade(
        grade, groups, components, grade_arms, prediction_shas, sidecar_shas, bootstrap
    )

    attempts: dict[str, Json] = {}
    source_paths: dict[str, Path] = {
        "evidence/F3_BELERIAND_PLAN.json": plan_path,
        "evidence/fold3.json": project / checks.FOLD_REL,
        "evidence/F3_TERMINAL_EVIDENCE_SEAL.json": seal_path,
        "evidence/grade.json": grade_path,
        **{f"collector/{name}": path for name, path in helper_paths.items()},
        **{f"checkpoint_support/{name}": path for name, path in checkpoint_artifact_paths.items()},
    }
    expected_shas: dict[Path, str] = {
        plan_path: plan_sha,
        seal_path: seal_sha,
        grade_path: grade_sha,
        project / checks.FOLD_REL: checks.FOLD_SHA,
    }
    expected_shas.update(support_source_hashes)
    # Archive the exact split metadata receipt that justifies the frozen fold
    # identity without copying the multi-gigabyte data/cache payloads.
    split_metadata = project_path(
        project / "research/campaign_2026-10-18/priority25_execution_r4/"
        "f3_beleriand_preparation_r1/F3_SPLIT_METADATA_V1.json",
        project,
    )
    source_paths["evidence/F3_SPLIT_METADATA_V1.json"] = split_metadata
    expected_shas[split_metadata] = checks.SPLIT_METADATA_SHA
    checkpoint_manifest_path = project_path(project / checks.CHECKPOINT_MANIFEST_REL, project)
    source_paths["evidence/F3_CHECKPOINT_SUPPORT_MANIFEST.json"] = checkpoint_manifest_path
    expected_shas[checkpoint_manifest_path] = checks.CHECKPOINT_MANIFEST_SHA
    for arm in ("wide", "deep"):
        for phase in ("stage1", "terminal"):
            meta = checkpoints[arm][phase]
            source_paths[f"evidence/{arm}_{phase}_collection.json"] = meta["collection_path"]
            source_paths[f"evidence/{arm}_{phase}_verification.json"] = meta["verification_path"]
            expected_shas[meta["collection_path"]] = meta["collection_sha"]
            expected_shas[meta["verification_path"]] = meta["verification_sha"]
        review_path, review_sha = review_bindings[arm]
        source_paths[f"reviews/{arm}.json"] = review_path
        expected_shas[review_path] = review_sha
        step = steps[f"export_{arm}"]
        source_paths[f"predictions/{arm}.npy"] = prediction_paths[arm]
        source_paths[f"predictions/{arm}.json"] = sidecar_paths[arm]
        expected_shas[prediction_paths[arm]] = prediction_shas[arm]
        expected_shas[sidecar_paths[arm]] = sidecar_shas[arm]
        predictor = project / "source-a266b036/scripts/predict_run.py"
        source_paths[f"code/predict_run_{arm}.py"] = predictor
        expected_shas[predictor] = checks.PREDICTOR_SHA

    grader_arg = next(
        value
        for value in grade_command
        if isinstance(value, str) and value.endswith("/grade_endpoint.py")
    )
    grader_path = project_path(Path(grader_arg), project)
    source_paths["code/grade_endpoint.py"] = grader_path
    expected_shas[grader_path] = checks.GRADER_SHA
    for step_id, step in steps.items():
        attempt_path = project_path(
            project / "runtime/attempts" / f"{step['attempt']}.json", project
        )
        record, digest, source_map = checks.validate_attempt(
            attempt_path, step, project, expected_contract, checks.PLAN_SHA
        )
        log_path = project_path(project / "runtime/logs" / f"{step['attempt']}.log", project)
        require(
            project_path(Path(record.get("log", "")), project) == log_path,
            f"{step_id} log path differs from plan attempt",
        )
        source_paths[f"attempts/{step_id}.json"] = attempt_path
        source_paths[f"logs/{step_id}.log"] = log_path
        expected_shas[attempt_path] = digest
        attempts[step_id] = {
            "id": step["attempt"],
            "record_sha256": digest,
            "source_manifest_sha256": record["source_manifest_sha256"],
            "source_sha256": source_map,
            "command_matches_plan": True,
        }
        if step_id.startswith(("stage1_", "continuation_")):
            arm, phase = (
                step_id.split("_")[-1],
                "stage1" if step_id.startswith("stage1_") else "terminal",
            )
            meta = checkpoints[arm][phase]
            require(
                meta["collection"].get("attempt_record_sha256") == digest,
                f"{step_id} receipt and raw record differ",
            )

    identities: dict[str, dict[str, int]] = {}
    source_hashes: dict[Path, str] = {}
    total_bytes = 0
    for relative, source in source_paths.items():
        digest, identity = hash_stable(source, _limit(relative))
        if source in expected_shas:
            require(digest == expected_shas[source], f"source changed after validation: {relative}")
        source_hashes[source] = digest
        if relative.startswith("predictions/") and relative.endswith(".npy"):
            arm = relative.split("/")[1].split(".")[0]
            require(
                identity == prediction_stats[arm] and digest == prediction_shas[arm],
                "prediction changed after scan",
            )
        if relative == "evidence/grade.json":
            require(
                digest == grade_sha and identity == grade_identity, "grade changed after validation"
            )
        identities[relative] = identity
        total_bytes += identity["size"]
    require(total_bytes <= MAX_ARCHIVE_BYTES, "result evidence exceeds 256 MiB cap")
    require(not snapshot.exists() and not receipt_path.exists(), "output appeared during preflight")

    snapshot.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.staging-", dir=snapshot.parent))
    created = False
    receipt_temp: Path | None = None
    try:
        copied = {
            relative: copy_stable(
                source,
                stage / relative,
                source_hashes[source],
                identities[relative],
            )
            for relative, source in source_paths.items()
        }
        snapshot.mkdir(mode=0o700)
        created = True
        for item in sorted(stage.rglob("*")):
            destination = snapshot / item.relative_to(stage)
            if item.is_dir():
                destination.mkdir(mode=0o700)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.link(item, destination, follow_symlinks=False)
                item.unlink()
        shutil.rmtree(stage)
        result: Json = {
            "schema": SCHEMA,
            "status": "collection_passed",
            "project_root": str(project),
            "snapshot_directory": str(snapshot),
            "plan_bindings": {
                "plan_sha256": plan_sha,
                "fold3_manifest_sha256": checks.FOLD_SHA,
                "terminal_seal_sha256": seal_sha,
            },
            "evaluation": {
                "fold": 3,
                "n_folds": 4,
                "evaluation_groups": 505,
                "evaluation_components": 145,
                "metric_id": "global_wp_clip2_v1",
                "bootstrap_resamples": bootstrap,
            },
            "arms": {
                arm: {
                    "run": grade_arms[arm]["run"],
                    "stage1_collection_sha256": checkpoints[arm]["stage1"]["collection_sha"],
                    "stage1_verification_sha256": checkpoints[arm]["stage1"]["verification_sha"],
                    "terminal_collection_sha256": checkpoints[arm]["terminal"]["collection_sha"],
                    "terminal_verification_sha256": checkpoints[arm]["terminal"][
                        "verification_sha"
                    ],
                    "stage1_snapshot_path": str(checkpoints[arm]["stage1"]["snapshot_path"]),
                    "terminal_snapshot_path": str(checkpoints[arm]["terminal"]["snapshot_path"]),
                    "stage1_state_sha256": checkpoints[arm]["stage1"]["artifacts"][
                        "training_state.pt"
                    ]["sha256"],
                    "terminal_state_sha256": checkpoints[arm]["terminal"]["artifacts"][
                        "training_state.pt"
                    ]["sha256"],
                    "epoch5_checkpoint_sha256": grade_arms[arm]["checkpoint_sha256"],
                    "provenance_sha256": grade_arms[arm]["provenance_sha256"],
                    "cumulative_updates": 89_280,
                }
                for arm in ("wide", "deep")
            },
            "predictions": {
                arm: prediction_info[arm] | {"sidecar_sha256": sidecar_shas[arm]}
                for arm in ("wide", "deep")
            },
            "grade": {
                "path": str(grade_path),
                "sha256": grade_sha,
                "wide_wp": grade["endpoint"]["score"]["wp"],
                "deep_wp": grade["reference"]["score"]["wp"],
                "delta_wide_minus_deep": grade["paired_component_bootstrap"]["mean_diff"],
                "ci95": grade["paired_component_bootstrap"]["ci95"],
                "p_le_0": grade["paired_component_bootstrap"]["p_le_0"],
            },
            "attempts": attempts,
            "files": copied,
            "collector_support_sha256": helper_hashes,
            "checkpoint_support_manifests_sha256": checkpoint_manifests,
            "scope": {
                "prediction_payload_sha256_checked": True,
                "prediction_payload_finiteness_stream_checked": True,
                "prediction_payload_deserialized": False,
                "checkpoint_payload_opened": False,
                "grade_rerun": False,
                "training_or_export_run": False,
            },
            "limitations": [
                "One exposed F3/seed1001 pair only; not a clean holdout or CP30 claim.",
                "The collector archives existing predictions and the official grade. It does not train, export, or grade.",
                "Checkpoint tensor bytes are bound to the separately reviewed support receipts; this collector does not open checkpoint payloads.",
            ],
        }
        receipt_temp = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
        payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with receipt_temp.open("xb") as stream:
            os.chmod(receipt_temp, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(receipt_temp, receipt_path, follow_symlinks=False)
        receipt_temp.unlink()
        receipt_temp = None
        return result
    except Exception:
        if created:
            shutil.rmtree(snapshot, ignore_errors=True)
        raise
    finally:
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--terminal-seal", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), sort_keys=True))
