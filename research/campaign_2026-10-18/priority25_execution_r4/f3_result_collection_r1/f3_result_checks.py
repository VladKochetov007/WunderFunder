"""F3/seed1001 result, terminal seal, export and paired-grade checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from result_io import hash_stable, membership_sha256, project_path, read_json, require

Json = dict[str, Any]
PLAN_SHA = "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
FOLD_SHA = "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f"
CHECKPOINT_MANIFEST_SHA = "7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc"
SPLIT_METADATA_SHA = "5facd5e512dd8782009ea6630bc42a560ed3f974b513d419b7fa9f0257fc99a9"
PREDICTOR_SHA = "750b9e77aa55673f61a4dd84d809d91dd315bf86eb36c36b065bc67a7d0f4b65"
GRADER_SHA = "cb90c0bce31cf6aa2b8e4f6a7536c30972b14c6995fb0db1e9f72aa867c30ec0"
SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
ARCHIVE_SHA = "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341"
CHECKPOINT_MANIFEST_REL = Path(
    "research/campaign_2026-10-18/priority25_execution_r4/"
    "f3_checkpoint_support_r1/PREPARATION_MANIFEST.json"
)
PLAN_REL = Path(
    "research/campaign_2026-10-18/priority25_execution_r4/"
    "f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
)
FOLD_REL = Path("splits/fold3.json")
SPLIT_METADATA_REL = Path(
    "research/campaign_2026-10-18/priority25_execution_r4/"
    "f3_beleriand_preparation_r1/F3_SPLIT_METADATA_V1.json"
)
COLLECTOR_FILES = ("result_io.py", "f3_result_checks.py", "collect_f3_results.py")
CHECKPOINT_STEPS = {
    (arm, phase): f"{'stage1' if phase == 'stage1' else 'continuation'}_{arm}"
    for arm in ("wide", "deep")
    for phase in ("stage1", "terminal")
}


def sha_text(value: object, label: str) -> str:
    """Require lowercase SHA-256 text."""
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{label} must be a SHA256",
    )
    return value


def step_map(plan: Json) -> dict[str, Json]:
    """Index the exact seven original F3/seed1001 plan steps."""
    rows = plan.get("steps")
    require(isinstance(rows, list), "F3 plan steps missing")
    mapped = {row.get("step_id"): row for row in rows if isinstance(row, dict)}
    require(
        set(mapped)
        == {
            "stage1_wide",
            "stage1_deep",
            "continuation_wide",
            "continuation_deep",
            "export_wide",
            "export_deep",
            "grade",
        },
        "F3 plan step set differs",
    )
    require(
        all(str(mapped[key].get("attempt", "")).endswith("-v1") for key in mapped),
        "only original F3 v1 attempts are permitted",
    )
    return mapped


def validate_plan(
    project: Path, plan_path: Path
) -> tuple[Json, str, dict[str, Json], Json, list[int], int]:
    """Bind the unmodified plan, exact fold3 JSON and evaluation membership."""
    plan_path = project_path(plan_path, project)
    plan, plan_sha, _ = read_json(plan_path)
    require(plan_sha == PLAN_SHA, "F3 plan SHA differs")
    require(plan.get("state") == "UNADMITTED", "F3 plan is not the frozen unadmitted plan")
    require(plan.get("launch_command_executed") is False, "plan execution metadata differs")
    require(plan_path == project / PLAN_REL, "F3 plan path differs")
    steps = step_map(plan)
    fold_path = project_path(project / FOLD_REL, project)
    fold, fold_sha, _ = read_json(fold_path, 2_000_000)
    require(fold_sha == FOLD_SHA, "fold3 SHA differs")
    metadata_path = project_path(project / SPLIT_METADATA_REL, project)
    metadata, metadata_sha, _ = read_json(metadata_path, 128 * 1024)
    require(
        metadata_sha == SPLIT_METADATA_SHA
        and metadata.get("schema") == "priority25.f3_split_metadata.v1"
        and metadata.get("status") == "PASS_READONLY_METADATA"
        and metadata.get("sha256") == FOLD_SHA
        and metadata.get("bytes") == 1_389_282
        and metadata.get("counts")
        == {
            "evaluation_components": 145,
            "evaluation_valid": 505,
            "excluded_train": 67,
            "fit_train": 10_539,
            "fit_valid": 1_368,
            "total_components": 3_289,
        }
        and metadata.get("scope")
        == {
            "cache_payloads_read": False,
            "numpy_or_torch_imported": False,
            "remote_mutation_or_job": False,
            "training_or_inference": False,
            "whole_dataset_payloads_read": False,
        }
        and metadata.get("membership")
        == {
            "evaluation.valid": {
                "count": 505,
                "ordered_json_sha256": "58a558a53b7a9d7d9034776e03e37f29b8f07867d3c5e2f1f5957fd41f1422f2",
            },
            "excluded.corrupt_train": {
                "count": 1,
                "ordered_json_sha256": "6874a9a9eb52b9c2224ab6eec9754d6d819a95a7dcfc3e1ceb0051409d64bd93",
            },
            "excluded.train": {
                "count": 67,
                "ordered_json_sha256": "b7d7aac747cf344e50e1bbaeeff55d190f39c604f4b1a871b9be4e0ab8c59d76",
            },
            "excluded.valid": {
                "count": 505,
                "ordered_json_sha256": "58a558a53b7a9d7d9034776e03e37f29b8f07867d3c5e2f1f5957fd41f1422f2",
            },
            "fit.train": {
                "count": 10_539,
                "ordered_json_sha256": "9f211ee90812dcdd4005c058b5c94656f509cf8183d10d7e676eb4e0cde7d5e1",
            },
            "fit.valid": {
                "count": 1_368,
                "ordered_json_sha256": "bbebeec6b40238872b68cdc7eac2c8911183306b8d07471a11daefd430b30de1",
            },
            "sequences.valid": {
                "count": 1_873,
                "row_group_component_pairs_sha256": "b5fa8a8302dc763dde101ce4c76274458325338cd11bd89132778072a1deb1ba",
            },
        },
        "F3 split metadata receipt differs",
    )
    require(
        (fold.get("version"), fold.get("fold"), fold.get("n_folds")) == (1, 3, 4), "fold identity"
    )
    groups = fold.get("evaluation", {}).get("valid")
    require(
        isinstance(groups, list)
        and len(groups) == 505
        and all(type(value) is int and value >= 0 for value in groups)
        and len(set(groups)) == len(groups),
        "fold3 evaluation membership differs",
    )
    sequences = fold.get("sequences", {}).get("valid")
    require(isinstance(sequences, list), "fold3 valid source-qualified sequence map missing")
    component_map = {
        row.get("row_group"): row.get("component") for row in sequences if isinstance(row, dict)
    }
    require(all(value in component_map for value in groups), "fold3 component identity missing")
    components = len({component_map[value] for value in groups})
    require(components == 145, "fold3 evaluation component count differs")
    return plan, plan_sha, steps, fold, groups, components


def validate_attempt(
    path: Path, step: Json, project: Path, expected_contract: Path, expected_sha: str
) -> tuple[Json, str, dict[str, str]]:
    """Bind an original successful runner record to its plan command and source map."""
    path = project_path(path, project)
    record, digest, _ = read_json(path)
    attempt = str(step["attempt"])
    require(
        path == project / "runtime/attempts" / f"{attempt}.json", f"{attempt} record path differs"
    )
    require(
        record.get("id") == attempt
        and record.get("command") == step.get("expected_command")
        and record.get("unit") == f"wnn-bandit-{attempt}.service"
        and record.get("status") == "finished"
        and record.get("exit_status") == 0,
        f"{attempt} did not finish with the frozen command",
    )
    require(
        record.get("contract") == str(expected_contract)
        and record.get("contract_sha256") == expected_sha,
        f"{attempt} does not bind the exact F3 plan contract",
    )
    source_map = record.get("source_sha256")
    require(isinstance(source_map, dict) and bool(source_map), f"{attempt} source map missing")
    require(
        all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for key, value in source_map.items()
        ),
        f"{attempt} source map malformed",
    )
    source_digest = hashlib.sha256(
        json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        record.get("source_manifest_sha256") == source_digest,
        f"{attempt} source manifest digest differs",
    )
    if step["step_id"].startswith("export_"):
        require(
            source_map.get("scripts/predict_run.py") == PREDICTOR_SHA,
            "export predictor source pin differs",
        )
        keys = [key for key in source_map if Path(key).name == "predict_run.py"]
        require(keys == ["scripts/predict_run.py"], "export predictor source key differs")
        predictor = project / "source-a266b036/scripts/predict_run.py"
        actual, _ = hash_stable(predictor, 16 * 1024 * 1024)
        require(actual == PREDICTOR_SHA, "frozen predictor bytes differ")
    if step["step_id"] == "grade":
        command = step["expected_command"]
        grader_paths = [
            value
            for value in command
            if isinstance(value, str) and value.endswith("/grade_endpoint.py")
        ]
        require(
            len(grader_paths) == 1 and source_map.get(grader_paths[0]) == GRADER_SHA,
            "grade source map differs",
        )
        grader_path = project_path(Path(grader_paths[0]), project)
        actual, _ = hash_stable(grader_path, 16 * 1024 * 1024)
        require(actual == GRADER_SHA, "frozen grader bytes differ")
    return record, digest, dict(source_map)


def _artifact_map(
    receipt: Json, snapshot: Path, names: set[str], project: Path, label: str
) -> dict[str, Json]:
    artifacts = receipt.get("artifacts")
    require(
        isinstance(artifacts, dict) and set(artifacts) == names, f"{label} artifact set differs"
    )
    result = {}
    for name in sorted(names):
        entry = artifacts[name]
        require(isinstance(entry, dict), f"{label} artifact {name} entry malformed")
        path = project_path(snapshot / name, project)
        digest, identity = hash_stable(path, 100_000_000)
        require(
            entry.get("path") == str(path)
            and entry.get("bytes") == identity["size"]
            and entry.get("sha256") == digest,
            f"{label} artifact {name} differs from snapshot bytes",
        )
        result[name] = {"path": str(path), "bytes": identity["size"], "sha256": digest}
    return result


def validate_checkpoint_phase(
    project: Path,
    plan: Json,
    steps: dict[str, Json],
    arm: str,
    phase: str,
    collection_path: Path,
    verification_path: Path,
    attempt: Json,
    attempt_sha: str,
    support_files: dict[str, str],
    expected_membership: Json,
) -> tuple[Json, Json, str, str, dict[str, Json]]:
    """Validate one actual checkpoint collection/verifier pair and final snapshot path."""
    step = steps[CHECKPOINT_STEPS[(arm, phase)]]
    collection, csha, _ = read_json(project_path(collection_path, project))
    verification, vsha, _ = read_json(project_path(verification_path, project))
    expected_phase = "stage1" if phase == "stage1" else "terminal"
    run = step["run_name"]
    snapshot = Path(step["archive_directory"])
    attempt_id = step["attempt"]
    names = {
        "stage1": {
            "model_ep0.pt",
            "model_ep1.pt",
            "provenance.json",
            "training_state.pt",
            "history.json",
        },
        "terminal": {
            "model_ep5.pt",
            "model.pt",
            "provenance.json",
            "training_state.pt",
            "history.json",
        },
    }[phase]
    require(
        collection.get("schema") == "priority25.f3_checkpoint_collection.v1"
        and (
            collection.get("arm"),
            collection.get("phase"),
            collection.get("run"),
            collection.get("attempt"),
        )
        == (arm, expected_phase, run, attempt_id)
        and collection.get("plan_sha256") == PLAN_SHA
        and collection.get("snapshot_directory") == str(snapshot)
        and collection.get("source_directory") == str(project / "data/artifacts/runs" / str(run)),
        f"{arm}/{phase} collection identity differs",
    )
    require(
        collection.get("source_stats_before") == collection.get("source_stats_after")
        and collection.get("source_snapshot_hash_match") is True
        and collection.get("producer_model_hash_matches") is True,
        f"{arm}/{phase} source stability/hash claim missing",
    )
    require(
        collection.get("attempt_record_sha256") == attempt_sha,
        f"{arm}/{phase} collection does not bind raw attempt SHA",
    )
    require(
        collection.get("collector_sha256") == support_files["collect_f3_checkpoint.py"]
        and collection.get("collector_dependencies")
        == {"f3_checkpoint_common.py": support_files["f3_checkpoint_common.py"]},
        f"{arm}/{phase} collection helper profile differs",
    )
    collector_attempt = collection.get("attempt_evidence")
    require(isinstance(collector_attempt, dict), f"{arm}/{phase} copied attempt evidence missing")
    copied_attempt = project_path(snapshot / "attempt.json", project)
    attempt_digest, attempt_identity = hash_stable(copied_attempt, 4 * 1024 * 1024)
    require(
        collector_attempt.get("path") == str(copied_attempt)
        and collector_attempt.get("bytes") == attempt_identity["size"]
        and collector_attempt.get("sha256") == attempt_digest
        and attempt_digest == attempt_sha,
        f"{arm}/{phase} copied attempt bytes differ",
    )
    require(
        collector_attempt.get("id") == attempt_id
        and collector_attempt.get("status") == "finished"
        and collector_attempt.get("exit_status") == 0
        and collector_attempt.get("contract_sha256") == PLAN_SHA
        and collector_attempt.get("source_manifest_sha256") == attempt.get("source_manifest_sha256")
        and collector_attempt.get("source_sha256") == attempt.get("source_sha256"),
        f"{arm}/{phase} copied attempt metadata differs",
    )
    artifact_values = _artifact_map(
        collection, snapshot, names, project, f"{arm}/{phase} collection"
    )
    expected_count = 29_760 if phase == "stage1" else 89_280
    expected_batches = 2_976 if phase == "stage1" else 8_928
    require(
        verification.get("schema") == "priority25.f3_checkpoint_verification.v1"
        and verification.get("status") == "passed"
        and (verification.get("arm"), verification.get("phase"), verification.get("run"))
        == (arm, expected_phase, run)
        and verification.get("next_epoch") == (2 if phase == "stage1" else 6)
        and verification.get("plan_sha256") == PLAN_SHA
        and verification.get("fold3_manifest_sha256") == FOLD_SHA
        and verification.get("source_revision") == SOURCE_REVISION
        and verification.get("source_archive_manifest_sha256") == ARCHIVE_SHA
        and verification.get("portable_registry_sha256") == REGISTRY_SHA,
        f"{arm}/{phase} verification identity differs",
    )
    require(
        verification.get("collection_receipt_sha256") == csha,
        f"{arm}/{phase} verifier binds another collection",
    )
    verified_attempt = verification.get("attempt")
    require(
        isinstance(verified_attempt, dict)
        and verified_attempt.get("id") == attempt_id
        and verified_attempt.get("record_sha256") == attempt_sha
        and verified_attempt.get("command") == step["expected_command"]
        and verified_attempt.get("status") == "finished"
        and verified_attempt.get("exit_status") == 0,
        f"{arm}/{phase} verifier attempt identity differs",
    )
    counts = verification.get("completed_update_counts")
    require(
        isinstance(counts, dict)
        and counts.get("optimizer_attempts") == expected_count
        and counts.get("optimizer_completed_updates") == expected_count
        and counts.get("amp_skipped_updates") == 0
        and counts.get("ema_updates") == 0
        and counts.get("completed_batches") == expected_batches
        and verification.get("planned_schedule_updates") == 89_280
        and verification.get("planned_schedule_batches") == 8_928
        and verification.get("scheduler_last_epoch") == expected_count
        and verification.get("scheduler_step_count") == expected_count + 1
        and verification.get("history_epochs_verified") == (2 if phase == "stage1" else 6),
        f"{arm}/{phase} counters/scheduler differ",
    )
    membership = verification.get("fold3_membership")
    require(
        membership == expected_membership,
        f"{arm}/{phase} fold3 membership differs",
    )
    verified_artifacts = _artifact_map(
        verification, snapshot, names, project, f"{arm}/{phase} verification"
    )
    require(
        all(artifact_values[name] == verified_artifacts[name] for name in names),
        f"{arm}/{phase} collection and verifier artifact hashes differ",
    )
    dependency_names = {
        "f3_checkpoint_common.py",
        "f3_checkpoint_evidence.py",
        "f3_checkpoint_lineage.py",
        "f3_checkpoint_tensors.py",
        "f3_checkpoint_membership.py",
        "f3_checkpoint_semantics.py",
        "collect_f3_checkpoint.py",
        "wnn/training_state.py",
    }
    deps = verification.get("verification_dependencies")
    require(
        isinstance(deps, dict) and set(deps) == dependency_names,
        f"{arm}/{phase} verifier dependencies differ",
    )
    expected_deps = {
        key: support_files[key] for key in dependency_names if key != "wnn/training_state.py"
    }
    actual_state_source, _ = hash_stable(
        project / "source-a266b036/src/wnn/training_state.py", 16 * 1024 * 1024
    )
    expected_deps["wnn/training_state.py"] = actual_state_source
    require(deps == expected_deps, f"{arm}/{phase} verifier dependency hashes differ")
    require(
        verification.get("verification_helper_sha256") == support_files["verify_f3_checkpoint.py"],
        f"{arm}/{phase} verifier helper SHA differs",
    )
    return collection, verification, csha, vsha, artifact_values


def expected_fold_membership(fold: Json) -> Json:
    """Derive ordered fit/evaluation membership digests from exact fold lists."""
    import numpy as np

    fit = fold.get("fit")
    evaluation = fold.get("evaluation")
    require(
        isinstance(fit, dict) and isinstance(evaluation, dict), "fold membership sections missing"
    )
    train, valid, eval_valid = fit.get("train"), fit.get("valid"), evaluation.get("valid")
    require(
        isinstance(train, list)
        and isinstance(valid, list)
        and isinstance(eval_valid, list)
        and len(train) == 10_539
        and len(valid) == 1_368
        and len(eval_valid) == 505,
        "fold membership lists differ",
    )
    train_order = np.random.default_rng(1001).permutation(np.asarray(train, dtype="<i8"))[:10_607]
    valid_order = np.asarray(valid, dtype="<i8")
    eval_order = np.random.default_rng(12345).permutation(np.asarray(eval_valid, dtype="<i8"))

    def identity(values: Any) -> Json:
        return {
            "count": len(values),
            "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    return {
        "manifest_sha256": FOLD_SHA,
        "fit_train_groups": len(train),
        "fit_valid_groups": len(valid),
        "evaluation_groups": len(eval_valid),
        "evaluation_components": 145,
        "fit_membership_identity": {
            "train": identity(train_order),
            "valid": identity(valid_order),
            "valid_repeat": 1,
            "dataset_length": 11_907,
        },
        "ordered_evaluation_membership_sha256": identity(eval_order)["ordered_int64_le_sha256"],
    }


def validate_seal(
    project: Path,
    seal_path: Path,
    seal_sha: str,
    arms: dict[str, dict[str, Any]],
    collector_hashes: dict[str, str],
) -> dict[str, tuple[Path, str]]:
    """Require arrival-time receipt-file inventory and independent reviewer binding."""
    seal, actual_seal_sha, _ = read_json(project_path(seal_path, project))
    require(actual_seal_sha == seal_sha, "terminal seal changed after read")
    require(
        seal.get("schema") == "priority25.f3_terminal_evidence_seal.v1"
        and seal.get("status") == "sealed"
        and seal.get("review_status") == "passed"
        and seal.get("plan_sha256") == PLAN_SHA
        and seal.get("fold3_manifest_sha256") == FOLD_SHA
        and seal.get("checkpoint_support_manifest_sha256") == CHECKPOINT_MANIFEST_SHA
        and seal.get("collector_support_sha256") == collector_hashes,
        "F3 terminal seal identity or helper bindings differ",
    )
    sealed_arms = seal.get("arms")
    require(
        isinstance(sealed_arms, dict) and set(sealed_arms) == {"wide", "deep"}, "seal arms differ"
    )
    review_bindings: dict[str, tuple[Path, str]] = {}
    for arm in ("wide", "deep"):
        entry = sealed_arms[arm]
        require(isinstance(entry, dict), f"{arm} seal entry malformed")
        expected_receipts = {}
        for phase in ("stage1", "terminal"):
            meta = arms[arm][phase]
            expected_receipts[phase] = {
                "collection": {
                    "path": str(meta["collection_path"]),
                    "bytes": meta["collection_bytes"],
                    "sha256": meta["collection_sha"],
                },
                "verification": {
                    "path": str(meta["verification_path"]),
                    "bytes": meta["verification_bytes"],
                    "sha256": meta["verification_sha"],
                },
            }
        require(entry.get("receipts") == expected_receipts, f"{arm} seal receipt inventory differs")
        require(
            entry.get("snapshot_paths")
            == {phase: str(arms[arm][phase]["snapshot_path"]) for phase in ("stage1", "terminal")},
            f"{arm} seal snapshot paths differ",
        )
        review_ref = entry.get("review_receipt")
        require(isinstance(review_ref, dict), f"{arm} review receipt reference missing")
        review_path = project_path(Path(review_ref.get("path", "")), project)
        review, review_sha, review_identity = read_json(review_path)
        require(
            review_ref.get("bytes") == review_identity["size"]
            and review_ref.get("sha256") == review_sha
            and review_ref.get("status") == "passed",
            f"{arm} review receipt file binding differs",
        )
        expected_review = {
            "schema": "priority25.f3_terminal_result_review.v1",
            "status": "passed",
            "arm": arm,
            "plan_sha256": PLAN_SHA,
            "fold3_manifest_sha256": FOLD_SHA,
            "checkpoint_support_manifest_sha256": CHECKPOINT_MANIFEST_SHA,
            "collector_support_sha256": collector_hashes,
            "receipts": expected_receipts,
            "snapshot_paths": entry["snapshot_paths"],
        }
        require(
            all(review.get(key) == value for key, value in expected_review.items()),
            f"{arm} review does not bind exact receipts/support",
        )
        review_bindings[arm] = (review_path, review_sha)
    return review_bindings


def validate_sidecar(
    sidecar: Json, run: str, groups: list[int], checkpoint_sha: str, prediction_sha: str
) -> None:
    """Validate the frozen feature-only FP32 evaluation sidecar fields."""
    expected = {
        "run": run,
        "split": "valid",
        "epoch": 5,
        "split_manifest_sha256": FOLD_SHA,
        "campaign_split_manifest_sha256": FOLD_SHA,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": membership_sha256(groups),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
    }
    require(
        all(sidecar.get(key) == value for key, value in expected.items()),
        f"{run} sidecar fields differ",
    )


def validate_grade(
    grade: Json,
    groups: list[int],
    components: int,
    arms: dict[str, Json],
    predictions: dict[str, str],
    sidecars: dict[str, str],
    bootstrap_count: int,
) -> None:
    """Check the official paired grade schema and its references to exact exports."""
    require(
        grade.get("schema") == "priority25.endpoint_grade.v1"
        and grade.get("fold") == 3
        and grade.get("n_folds") == 4
        and grade.get("split_manifest_sha256") == FOLD_SHA
        and grade.get("evaluation_row_groups") == groups
        and grade.get("evaluation_components") == components
        and grade.get("metric_id") == "global_wp_clip2_v1",
        "official grade fold/membership/metric identity differs",
    )
    for field, arm in (("endpoint", "wide"), ("reference", "deep")):
        endpoint = grade.get(field)
        require(
            isinstance(endpoint, dict)
            and endpoint.get("run") == arms[arm]["run"]
            and endpoint.get("epoch") == 5
            and endpoint.get("checkpoint_sha256") == arms[arm]["checkpoint_sha256"]
            and endpoint.get("provenance_sha256") == arms[arm]["provenance_sha256"]
            and endpoint.get("prediction_sha256") == predictions[arm]
            and endpoint.get("sidecar_sha256") == sidecars[arm],
            f"grade {field} does not bind the pinned F3 endpoint",
        )
        score = endpoint.get("score")
        require(
            isinstance(score, dict) and score.get("metric_id") == "global_wp_clip2_v1",
            f"grade {field} score missing",
        )
        for key in ("wp", "t0", "t1"):
            require(
                type(score.get(key)) in (int, float)
                and math.isfinite(score[key])
                and 0 <= score[key] <= 1,
                f"grade {field} {key} invalid",
            )
        require(
            math.isclose(score["wp"], (score["t0"] + score["t1"]) / 2, abs_tol=1e-12, rel_tol=0),
            f"grade {field} aggregate differs",
        )
        require(
            type(score.get("selected_rows")) is int and score["selected_rows"] > 0,
            f"grade {field} selected rows invalid",
        )
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
        "paired whole-component bootstrap configuration differs",
    )
    for key in ("mean_diff", "bootstrap_sd", "p_le_0", "candidate_wp", "reference_wp"):
        require(
            type(paired.get(key)) in (int, float) and math.isfinite(paired[key]),
            f"paired grade {key} invalid",
        )
    require(
        paired["bootstrap_sd"] >= 0
        and 0 <= paired["p_le_0"] <= 1
        and math.isclose(
            paired["candidate_wp"], grade["endpoint"]["score"]["wp"], abs_tol=1e-12, rel_tol=0
        )
        and math.isclose(
            paired["reference_wp"], grade["reference"]["score"]["wp"], abs_tol=1e-12, rel_tol=0
        )
        and math.isclose(
            paired["mean_diff"],
            paired["candidate_wp"] - paired["reference_wp"],
            abs_tol=1e-12,
            rel_tol=0,
        ),
        "paired grade summaries disagree",
    )
    interval = paired.get("ci95")
    require(
        isinstance(interval, list)
        and len(interval) == 2
        and all(type(value) in (int, float) and math.isfinite(value) for value in interval)
        and interval[0] <= interval[1],
        "paired grade CI invalid",
    )
