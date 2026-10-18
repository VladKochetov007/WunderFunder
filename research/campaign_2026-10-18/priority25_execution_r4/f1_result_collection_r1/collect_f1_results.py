"""Collect existing verified F1 exports and grade; never trains or regrades."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import result_checks
import result_io
from result_checks import (
    sha_text,
    validate_attempt_record,
    validate_grade,
    validate_helper_profiles,
    validate_plans,
    validate_sidecar,
    validate_terminal,
    validate_terminal_seal,
)
from result_io import (
    MAX_EVIDENCE_BYTES,
    MAX_JSON_BYTES,
    MAX_PREDICTION_BYTES,
    MAX_SIDECAR_BYTES,
    CollectionError,
    copy_stable,
    hash_stable,
    inspect_npy,
    project_path,
    read_json,
    require,
)

Json = dict[str, Any]  # Producer JSON uses heterogeneous nested values.
SCHEMA = "priority25.f1_result_collection.v1"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _arg_path(value: Path, project: Path) -> Path:
    """Resolve an input lexically under the project without following symlinks."""
    candidate = value if value.is_absolute() else project / value
    return project_path(candidate, project)


def _collector_support(project: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Hash the exact imported collector modules that the review seal binds."""
    module_paths = {
        "collect_f1_results.py": Path(__file__),
        "result_checks.py": Path(result_checks.__file__),
        "result_io.py": Path(result_io.__file__),
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, module_path in module_paths.items():
        path = _arg_path(module_path, project)
        digest, _ = hash_stable(path, MAX_EVIDENCE_BYTES)
        paths[name], hashes[name] = path, digest
    return paths, hashes


def _input_limit(name: str) -> int:
    """Return the bounded streaming cap for one archive input."""
    if name.endswith(".npy"):
        return MAX_PREDICTION_BYTES
    if name.startswith("predictions/") and name.endswith(".json"):
        return MAX_SIDECAR_BYTES
    if name.startswith("logs/"):
        return MAX_EVIDENCE_BYTES
    return MAX_JSON_BYTES if name.endswith(".json") else MAX_EVIDENCE_BYTES


def _attempt_from_plan(project: Path, step: Json) -> tuple[Json, Path, str]:
    """Load the runner record by attempt ID, never the documentary receipt_path."""
    attempt_id = str(step["attempt"])
    record_path = project / "runtime" / "attempts" / f"{attempt_id}.json"
    record, digest = validate_attempt_record(record_path, attempt_id, step["expected_command"])
    log_path = project_path(Path(record.get("log", "")), project)
    expected_log = project / "runtime" / "logs" / f"{attempt_id}.log"
    require(log_path == expected_log, f"attempt log path mismatch for {attempt_id}")
    return record, log_path, digest


def _seal_reviews(seal: Json, project: Path) -> tuple[dict[str, tuple[Json, str]], dict[str, Path]]:
    """Read both independently reviewed terminal receipts named in the seal."""
    review_values: dict[str, tuple[Json, str]] = {}
    review_paths: dict[str, Path] = {}
    refs = seal.get("review_receipts")
    require(
        isinstance(refs, dict) and set(refs) == {"wide", "deep"},
        "seal must name wide and deep review receipts",
    )
    for arm in ("wide", "deep"):
        ref = refs[arm]
        require(
            isinstance(ref, dict) and isinstance(ref.get("path"), str),
            f"seal {arm} review path is missing",
        )
        path = _arg_path(Path(ref["path"]), project)
        review, digest, _ = read_json(path)
        require(ref.get("sha256") == digest, f"seal {arm} review receipt hash mismatch")
        review_values[arm], review_paths[arm] = (review, digest), path
    return review_values, review_paths


def collect(args: argparse.Namespace) -> Json:
    """Validate every source first, then publish a create-only result snapshot."""
    project = args.project_root.resolve(strict=True)
    support_paths, support_hashes = _collector_support(project)
    inputs = {
        name: _arg_path(getattr(args, name), project)
        for name in (
            "base_plan",
            "recovery_plan",
            "fold_manifest",
            "terminal_seal",
            "wide_stage1_collection",
            "wide_stage1_verification",
            "wide_terminal_collection",
            "wide_terminal_verification",
            "deep_stage1_collection",
            "deep_stage1_verification",
            "deep_terminal_collection",
            "deep_terminal_verification",
        )
    }
    snapshot = _arg_path(args.snapshot_dir, project)
    receipt_path = _arg_path(args.receipt, project)
    require(
        not snapshot.exists() and not receipt_path.exists(),
        "snapshot and receipt paths are create-only",
    )
    base, base_sha, _ = read_json(inputs["base_plan"])
    recovery, recovery_sha, _ = read_json(inputs["recovery_plan"])
    fold, fold_sha, _ = read_json(inputs["fold_manifest"])
    expected_hashes: dict[Path, str] = {
        inputs["base_plan"]: base_sha,
        inputs["recovery_plan"]: recovery_sha,
        inputs["fold_manifest"]: fold_sha,
        **{support_paths[name]: support_hashes[name] for name in support_paths},
    }
    base_steps, recovery_steps, groups, components, schedule = validate_plans(
        base, recovery, fold, base_sha, recovery_sha, fold_sha
    )
    require(
        len(groups) == 457 and components == 149,
        "fold1 evaluation group/component counts differ from the frozen F1 source",
    )
    source_revision = base.get("source_bindings", {}).get("trainer_commit")
    registry_sha = base.get("source_bindings", {}).get("portable_registry_sha256")
    require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(c in "0123456789abcdef" for c in source_revision),
        "trainer source revision is not a Git commit SHA",
    )
    sha_text(registry_sha, "portable registry")
    arms: dict[str, tuple[Json, Json, str, str, dict[str, str]]] = {}
    terminal_meta: dict[str, Json] = {}
    for arm in ("wide", "deep"):
        c, v, sc, sv, csha, vsha, hashes, counts = validate_terminal(
            project,
            arm,
            base_steps,
            recovery_steps,
            base_sha,
            recovery_sha,
            fold_sha,
            source_revision,
            registry_sha,
            inputs[f"{arm}_terminal_collection"],
            inputs[f"{arm}_terminal_verification"],
            inputs[f"{arm}_stage1_collection"],
            inputs[f"{arm}_stage1_verification"],
            schedule,
        )
        fold_binding = v.get("fold1_membership", {})
        require(
            fold_binding.get("manifest_sha256") == fold_sha
            and fold_binding.get("evaluation_groups") == len(groups)
            and fold_binding.get("evaluation_components") == components,
            f"{arm} terminal fold membership mismatch",
        )
        stage_fold = sv.get("fold1_membership", {})
        require(stage_fold == fold_binding, f"{arm} stage1/terminal fold membership differs")
        require(
            v.get("fit_membership_identity") == sv.get("fit_membership_identity")
            and v.get("normalization_identity") == sv.get("normalization_identity"),
            f"{arm} stage1/terminal fit contract differs",
        )
        arms[arm] = (c, v, csha, vsha, hashes)
        expected_hashes[inputs[f"{arm}_terminal_collection"]] = csha
        expected_hashes[inputs[f"{arm}_terminal_verification"]] = vsha
        expected_hashes[inputs[f"{arm}_stage1_collection"]] = hashes["stage1_collection"]
        expected_hashes[inputs[f"{arm}_stage1_verification"]] = hashes["stage1_verification"]
        terminal_meta[arm] = {
            "collection": c,
            "verification": v,
            "stage1_collection": sc,
            "stage1_verification": sv,
            "hashes": hashes,
            "counts": counts,
        }
    checkpoint_support_paths, checkpoint_support_hashes, checkpoint_support_manifests = (
        validate_helper_profiles(project, terminal_meta, source_revision)
    )
    for relative, path in checkpoint_support_paths.items():
        expected_hashes[path] = checkpoint_support_hashes[relative]
    for key in (
        "fit_membership_identity",
        "normalization_identity",
        "fold1_membership",
        "source_archive_manifest_sha256",
        "portable_registry_sha256",
    ):
        require(arms["wide"][1].get(key) == arms["deep"][1].get(key), f"F1 arms differ on {key}")
    seal, seal_sha, _ = read_json(inputs["terminal_seal"])
    review_values, review_paths = _seal_reviews(seal, project)
    expected_hashes[inputs["terminal_seal"]] = seal_sha
    for arm, (_, digest) in review_values.items():
        expected_hashes[review_paths[arm]] = digest
    validate_terminal_seal(
        seal,
        seal_sha,
        base_sha,
        recovery_sha,
        fold_sha,
        arms,
        terminal_meta,
        review_values,
        support_hashes,
        checkpoint_support_manifests,
    )

    grade_step = base_steps["grade"]
    grade_command = grade_step["expected_command"]
    fold_path = inputs["fold_manifest"]
    require(
        Path(grade_command[grade_command.index("--split") + 1]) == fold_path,
        "grade vector does not bind the supplied fold1 manifest",
    )
    grade_output = Path(grade_command[grade_command.index("--output") + 1])
    require(
        grade_output == project / "receipts/RANK01_F1_S1001_GRADE.json"
        and Path(grade_step["grade_output"]) == grade_output,
        "grade output differs from frozen F1 paths",
    )
    grade, grade_sha, grade_identity = read_json(grade_output)
    expected_hashes[grade_output] = grade_sha
    bootstrap = int(grade_command[grade_command.index("--bootstrap") + 1])
    prediction_paths: dict[str, Path] = {}
    sidecar_paths: dict[str, Path] = {}
    prediction_hashes: dict[str, str] = {}
    sidecar_hashes: dict[str, str] = {}
    prediction_headers: dict[str, Json] = {}
    prediction_stats: dict[str, dict[str, int]] = {}
    for arm in ("wide", "deep"):
        step = base_steps[f"export_{arm}"]
        command = step["expected_command"]
        require(
            Path(command[command.index("--campaign-split-manifest") + 1]) == fold_path
            and "--campaign-evaluation-only" in command
            and command[command.index("--inference-precision") + 1] == "fp32",
            f"{arm} export vector is not fold1 evaluation-only FP32",
        )
        prediction = _arg_path(Path(step["prediction_path"]), project)
        sidecar = _arg_path(Path(step["sidecar_path"]), project)
        require(
            prediction.parent == project / "data/cache"
            and sidecar == prediction.with_suffix(".json"),
            f"{arm} prediction output path differs from the frozen cache path",
        )
        header, pred_sha, pred_stat = inspect_npy(prediction, [len(groups), 20_000, 2])
        sidecar_data, side_sha, _ = read_json(sidecar, MAX_SIDECAR_BYTES)
        checkpoint_sha = arms[arm][1]["artifacts"]["model_ep5.pt"]["sha256"]
        validate_sidecar(
            sidecar_data,
            base_steps[f"continuation_{arm}"]["run_name"],
            groups,
            fold_sha,
            checkpoint_sha,
            pred_sha,
        )
        prediction_paths[arm], sidecar_paths[arm] = prediction, sidecar
        prediction_hashes[arm], sidecar_hashes[arm] = pred_sha, side_sha
        prediction_headers[arm], prediction_stats[arm] = header, pred_stat
        expected_hashes[prediction] = pred_sha
        expected_hashes[sidecar] = side_sha
    grade_arms = {
        arm: {
            "run": base_steps[f"continuation_{arm}"]["run_name"],
            "checkpoint_sha256": arms[arm][1]["artifacts"]["model_ep5.pt"]["sha256"],
            "provenance_sha256": arms[arm][1]["artifacts"]["provenance.json"]["sha256"],
        }
        for arm in ("wide", "deep")
    }
    validate_grade(
        grade,
        groups,
        components,
        fold_sha,
        grade_arms,
        prediction_hashes,
        sidecar_hashes,
        bootstrap,
    )

    attempt_records: dict[str, Json] = {}
    attempt_logs: dict[str, Path] = {}
    attempt_hashes: dict[str, str] = {}
    steps = {**base_steps, "continuation_wide": recovery_steps["continuation_wide"]}
    for name in ("export_wide", "export_deep", "grade"):
        record, log, digest = _attempt_from_plan(project, steps[name])
        attempt_records[name], attempt_logs[name], attempt_hashes[name] = record, log, digest
    grader_path = next(
        (token for token in grade_command if token.endswith("/grade_endpoint.py")), None
    )
    require(grader_path is not None, "grade vector has no unique grader entrypoint")
    grader_sha = base.get("source_bindings", {}).get("grader_endpoint_source_sha256")
    require(
        attempt_records["grade"]["source_sha256"].get(grader_path) == grader_sha,
        "grade attempt source map does not bind the frozen grader",
    )
    code_sources: dict[str, Path] = {}
    for arm in ("wide", "deep"):
        predict = [
            token
            for token in steps[f"export_{arm}"]["expected_command"]
            if token.endswith("/scripts/predict_run.py")
        ]
        require(
            len(predict) == 1 and predict[0] in attempt_records[f"export_{arm}"]["source_sha256"],
            f"{arm} attempt source map lacks canonical exporter",
        )
        predict_path = _arg_path(Path(predict[0]), project)
        predict_sha, _ = hash_stable(predict_path, MAX_EVIDENCE_BYTES)
        require(
            attempt_records[f"export_{arm}"]["source_sha256"][predict[0]] == predict_sha,
            f"{arm} export source changed after its attempt",
        )
        code_sources[f"code/predict_run_{arm}.py"] = predict_path
        expected_hashes[predict_path] = predict_sha
    grader_source = _arg_path(Path(grader_path), project)
    grader_file_sha, _ = hash_stable(grader_source, MAX_EVIDENCE_BYTES)
    require(grader_file_sha == grader_sha, "grader file differs from the frozen source binding")
    code_sources["code/grade_endpoint.py"] = grader_source
    expected_hashes[grader_source] = grader_file_sha

    sources: dict[str, Path] = {
        "evidence/base_plan.json": inputs["base_plan"],
        "evidence/recovery_plan.json": inputs["recovery_plan"],
        "evidence/fold1.json": fold_path,
        "evidence/terminal_seal.json": inputs["terminal_seal"],
        "evidence/grade.json": grade_output,
        "reviews/wide.json": review_paths["wide"],
        "reviews/deep.json": review_paths["deep"],
    }
    sources.update(code_sources)
    sources.update({f"collector/{name}": path for name, path in support_paths.items()})
    sources.update(checkpoint_support_paths)
    for arm in ("wide", "deep"):
        sources[f"evidence/{arm}_stage1_collection.json"] = inputs[f"{arm}_stage1_collection"]
        sources[f"evidence/{arm}_stage1_verification.json"] = inputs[f"{arm}_stage1_verification"]
        sources[f"evidence/{arm}_terminal_collection.json"] = inputs[f"{arm}_terminal_collection"]
        sources[f"evidence/{arm}_terminal_verification.json"] = inputs[
            f"{arm}_terminal_verification"
        ]
        sources[f"predictions/{arm}.npy"] = prediction_paths[arm]
        sources[f"predictions/{arm}.json"] = sidecar_paths[arm]
        for phase, key in (("stage1", "stage1_collection"), ("terminal", "collection")):
            evidence = terminal_meta[arm][key].get("attempt_evidence", {})
            attempt_path = _arg_path(Path(evidence.get("path", "")), project)
            record_id = (
                base_steps[f"stage1_{arm}"]["attempt"]
                if phase == "stage1"
                else steps[f"continuation_{arm}"]["attempt"]
            )
            plan_step = (
                base_steps[f"stage1_{arm}"] if phase == "stage1" else steps[f"continuation_{arm}"]
            )
            _, record_sha = validate_attempt_record(
                attempt_path, record_id, plan_step["expected_command"], evidence.get("sha256")
            )
            expected_hashes[attempt_path] = record_sha
            sources[f"attempts/{arm}_{phase}.json"] = attempt_path
    attempt_manifest: dict[str, Json] = {}
    for name in ("export_wide", "export_deep", "grade"):
        attempt_id = steps[name]["attempt"]
        sources[f"attempts/{attempt_id}.json"] = (
            project / "runtime" / "attempts" / f"{attempt_id}.json"
        )
        sources[f"logs/{attempt_id}.log"] = attempt_logs[name]
        expected_hashes[sources[f"attempts/{attempt_id}.json"]] = attempt_hashes[name]
        attempt_manifest[name] = {
            "id": attempt_id,
            "record_sha256": attempt_hashes[name],
            "source_manifest_sha256": attempt_records[name]["source_manifest_sha256"],
            "command_matches_plan": True,
        }

    preflight: dict[str, tuple[str, dict[str, int]]] = {}
    total_bytes = 0
    for relative, source in sources.items():
        digest, stat_value = hash_stable(source, _input_limit(relative))
        expected_digest = expected_hashes.get(source)
        if expected_digest is not None:
            require(digest == expected_digest, f"source changed after validation: {relative}")
        if relative.startswith("predictions/") and relative.endswith(".npy"):
            require(
                digest == prediction_hashes[relative.split("/")[1].split(".")[0]],
                "prediction changed after finite-value scan",
            )
            require(
                stat_value == prediction_stats[relative.split("/")[1].split(".")[0]],
                "prediction identity changed after scan",
            )
        if relative == "evidence/grade.json":
            require(
                stat_value == grade_identity and digest == grade_sha,
                "grade output changed after validation",
            )
        total_bytes += stat_value["size"]
        preflight[relative] = (digest, stat_value)
    require(total_bytes <= MAX_ARCHIVE_BYTES, f"archive inputs exceed {MAX_ARCHIVE_BYTES} bytes")
    require(
        not snapshot.exists() and not receipt_path.exists(),
        "create-only output path appeared during preflight",
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _arg_path(snapshot.parent, project)
    _arg_path(receipt_path.parent, project)
    stage = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.staging-", dir=snapshot.parent))
    receipt_temp: Path | None = None
    snapshot_created = False
    try:
        copied: dict[str, Json] = {}
        for relative, source in sources.items():
            digest, stat_value = preflight[relative]
            copied[relative] = copy_stable(source, stage / relative, digest, stat_value)
        snapshot.mkdir(mode=0o700)
        snapshot_created = True
        for source in sorted(stage.rglob("*")):
            relative = source.relative_to(stage)
            destination = snapshot / relative
            if source.is_dir():
                destination.mkdir(mode=0o700)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.link(source, destination, follow_symlinks=False)
                source.unlink()
        shutil.rmtree(stage)
        result: Json = {
            "schema": SCHEMA,
            "status": "collection_passed",
            "created_unix": time.time(),
            "project_root": str(project),
            "snapshot_directory": str(snapshot),
            "plan_bindings": {
                "base_plan_sha256": base_sha,
                "recovery_plan_sha256": recovery_sha,
                "fold1_manifest_sha256": fold_sha,
                "wide_recovery_attempt": recovery_steps["continuation_wide"]["attempt"],
                "terminal_seal_sha256": seal_sha,
            },
            "evaluation": {
                "fold": 1,
                "n_folds": 4,
                "evaluation_row_groups": groups,
                "evaluation_components": components,
                "metric_id": "global_wp_clip2_v1",
                "bootstrap_resamples": bootstrap,
            },
            "terminals": {
                arm: {
                    "run": grade_arms[arm]["run"],
                    "attempt": arms[arm][1]["attempt"]["id"],
                    "terminal_collection_sha256": arms[arm][2],
                    "terminal_verification_sha256": arms[arm][3],
                    "stage1_state_sha256": arms[arm][4]["stage1_state"],
                    "terminal_state_sha256": arms[arm][4]["terminal_state"],
                    "checkpoint_sha256": arms[arm][4]["checkpoint"],
                    "provenance_sha256": arms[arm][4]["provenance"],
                    "collection_helper_sha256": arms[arm][4]["collection_helper"],
                    "verification_helper_sha256": arms[arm][4]["verification_helper"],
                    "counts": terminal_meta[arm]["counts"],
                }
                for arm in ("wide", "deep")
            },
            "predictions": {
                arm: {
                    "path": str(prediction_paths[arm]),
                    "sha256": prediction_hashes[arm],
                    "sidecar_sha256": sidecar_hashes[arm],
                    "npy_header": prediction_headers[arm],
                }
                for arm in ("wide", "deep")
            },
            "grade": {
                "path": str(grade_output),
                "sha256": grade_sha,
                "target_cache_sha256": grade["target_cache_sha256"],
                "wide_wp": grade["endpoint"]["score"]["wp"],
                "deep_wp": grade["reference"]["score"]["wp"],
                "delta_wide_minus_deep": grade["paired_component_bootstrap"]["mean_diff"],
                "ci95": grade["paired_component_bootstrap"]["ci95"],
                "p_le_0": grade["paired_component_bootstrap"]["p_le_0"],
            },
            "attempts": attempt_manifest,
            "files": copied,
            "collector_support_sha256": support_hashes,
            "checkpoint_support_manifests_sha256": checkpoint_support_manifests,
            "scope": {
                "prediction_payload_sha256_checked": True,
                "prediction_payload_finiteness_stream_checked": True,
                "prediction_payload_deserialized": False,
                "checkpoint_payload_opened": False,
                "grade_rerun": False,
                "target_or_parquet_opened": False,
            },
            "limitations": [
                "This binds one previously exposed fold1/seed1001 pair; it is screening evidence, not held-out confirmation.",
                "The helper archives existing predictions and official grade; it does not run export, training or grading.",
                "Checkpoint tensors, parquet targets and full trainer source are not opened here; terminal helper review and receipts are bound by exact hashes.",
                "This result does not establish CP30, convergence, resumed/uninterrupted parity or public transfer.",
            ],
        }
        payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        receipt_temp = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
        with receipt_temp.open("xb") as stream:
            os.chmod(receipt_temp, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(receipt_temp, receipt_path, follow_symlinks=False)
        receipt_temp.unlink()
        return result
    except (OSError, CollectionError, ValueError) as exc:
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)
        if snapshot_created:
            shutil.rmtree(snapshot, ignore_errors=True)
        raise CollectionError(
            f"F1 result collection failed before receipt publication: {exc}"
        ) from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    """Parse one project-local F1 evidence collection request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--terminal-seal", type=Path, required=True)
    for arm in ("wide", "deep"):
        for phase in ("stage1", "terminal"):
            parser.add_argument(f"--{arm}-{phase}-collection", type=Path, required=True)
            parser.add_argument(f"--{arm}-{phase}-verification", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the bounded collector and emit a compact status line."""
    result = collect(parse_args())
    print(
        json.dumps(
            {"status": result["status"], "receipt": result["snapshot_directory"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
