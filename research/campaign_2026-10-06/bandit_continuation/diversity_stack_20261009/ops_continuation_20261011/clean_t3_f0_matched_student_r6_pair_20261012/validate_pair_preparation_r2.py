"""Verify bound cache receipts and staged pair readiness without reading arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET_DIR = Path(__file__).resolve().parent
PACKET_REPO_DIR = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012"
)
PLAN_PATH = PACKET_DIR / "PAIR_PREPARATION_R2.json"


class ValidationError(ValueError):
    """Raised when a current source, cache, or authorization binding differs."""


def sha256_file(path: Path) -> str:
    """Hash bounded-size evidence files; never open cache or prediction arrays."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_bindings(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from iter_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_bindings(child)


def resolve_bound_path(project_root: Path, raw: str) -> Path:
    """Resolve project bindings or this packet's local evidence copy."""
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        root_candidate = (project_root / candidate).resolve()
        if root_candidate.is_file():
            resolved = root_candidate
        else:
            try:
                suffix = candidate.relative_to(PACKET_REPO_DIR)
            except ValueError as exc:
                raise ValidationError(f"bound path is absent: {raw}") from exc
            resolved = (PACKET_DIR / suffix).resolve(strict=True)
    if not resolved.is_relative_to(project_root.resolve(strict=True)) and not resolved.is_relative_to(PACKET_DIR.resolve(strict=True)):
        raise ValidationError(f"bound path escapes project and packet roots: {raw}")
    return resolved


def verify_bindings(plan: dict[str, Any], project_root: Path) -> list[str]:
    verified: list[str] = []
    seen: set[tuple[str, str]] = set()
    for binding in iter_bindings(plan["immutable_bindings"]):
        raw = str(binding["path"])
        expected_sha = str(binding["sha256"])
        key = (raw, expected_sha)
        if key in seen:
            continue
        seen.add(key)
        path = resolve_bound_path(project_root, raw)
        if path.stat().st_size != binding["bytes"]:
            raise ValidationError(f"bound file byte count differs: {raw}")
        if sha256_file(path) != expected_sha:
            raise ValidationError(f"bound file SHA256 differs: {raw}")
        verified.append(raw)
    return verified


def load_bound_json(plan: dict[str, Any], key: str, project_root: Path) -> dict[str, Any]:
    binding = plan["immutable_bindings"][key]
    path = resolve_bound_path(project_root, binding["path"])
    if path.stat().st_size != binding["bytes"] or sha256_file(path) != binding["sha256"]:
        raise ValidationError(f"receipt binding differs: {key}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValidationError(f"receipt is not an object: {key}")
    return result


def verify_receipt_documents(
    plan: dict[str, Any],
    input_receipt: dict[str, Any],
    cache_receipt: dict[str, Any],
    root_release: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Cross-check source ancestry, exclusion, target formula, and cache outputs."""
    if input_receipt.get("status") != "PASS_ALL_TEACHER_INPUTS_READY":
        raise ValidationError("teacher input receipt is not passing")
    if input_receipt.get("fit_train_predictions_pending") != []:
        raise ValidationError("teacher FIT_TRAIN inputs remain pending")
    if set(input_receipt.get("fit_train_predictions_ready", [])) != {
        f"clean_t3_f0_muonall_h384_e3_s{seed}_r5" for seed in (3, 4, 5)
    }:
        raise ValidationError("teacher input receipt does not bind exactly seeds 3/4/5")
    if cache_receipt.get("status") != "PASS_FROZEN_CONSUMER_AND_ACTUAL_OUTPUT_VALIDATION":
        raise ValidationError("cache receipt is not a frozen-consumer/output PASS")
    if cache_receipt.get("student_fit_or_profile_launched") is not False:
        raise ValidationError("cache receipt unexpectedly claims student execution")
    release_binding = plan["immutable_bindings"]["root_cache_materialization_release"]
    if cache_receipt.get("root_release_sha256") != release_binding["sha256"]:
        raise ValidationError("cache receipt is not bound to the plan's root release")
    if root_release.get("cache_materialization_authorized") is not True or root_release.get("status") != "conditional_serial_after_s5_terminal_validity":
        raise ValidationError("root receipt does not authorize the recorded cache-only materialization")

    expected_membership = plan["fold_membership"]
    observed_membership = cache_receipt.get("source_qualified_membership", {})
    membership_fields = {
        "fold": expected_membership["fold"],
        "split_manifest_sha256": expected_membership["split_manifest_sha256"],
        "fit_train_count": expected_membership["fit_train_count"],
        "fit_train_order_sha256": expected_membership["fit_train_set_sha256"],
        "fit_valid_count": expected_membership["fit_valid_count"],
        "fit_valid_order_sha256": expected_membership["fit_valid_set_sha256"],
        "evaluation_valid_count": expected_membership["evaluation_valid_count"],
        "fit_valid_evaluation_overlap": expected_membership["fit_valid_evaluation_overlap"],
    }
    for key, value in membership_fields.items():
        if observed_membership.get(key) != value:
            raise ValidationError(f"source-qualified fold membership differs at {key}")
    if expected_membership["fit_valid_evaluation_overlap"] != 0:
        raise ValidationError("FIT_VALID contains exposed evaluation groups")

    if plan["r2_corrections"] != [{
        "field": "fold_membership.fit_valid_set_sha256",
        "r1_value": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a179529858023c9568d75",
        "r2_value": observed_membership["fit_valid_order_sha256"],
        "basis": "bound cache materialization validation receipt source_qualified_membership.fit_valid_order_sha256",
        "reason": "R1 value was truncated to 62 hex characters; R2 preserves R1 and corrects the successor binding to the authoritative 64-character fingerprint.",
    }]:
        raise ValidationError("R1 FIT_VALID fingerprint erratum is missing or differs")

    target = plan["teacher_target"]
    target_contract = cache_receipt.get("target_contract", {})
    if target_contract.get("homogeneous_scale") != 0.8125 or target_contract.get("teacher_mean") != "unweighted arithmetic mean in float32" or target_contract.get("clipping") != "none during cache construction":
        raise ValidationError("cache target formula or clipping semantics differ")
    if target["formula"] != "T=float32(0.8125*float32_mean(seed3_ep2,seed4_ep2,seed5_ep2)); student target=0.1*y+0.9*T; no cache-time clipping.":
        raise ValidationError("paired student target formula differs from the validated cache")

    inputs_by_seed = input_receipt["teacher_records"]
    ancestry_by_seed = cache_receipt["teacher_ancestry"]
    teacher_plans = target["teachers"]
    if len(inputs_by_seed) != 3 or [teacher["seed"] for teacher in teacher_plans] != [3, 4, 5]:
        raise ValidationError("expected exactly three ordered F0 teacher records")
    expected_arrays = {
        "train": {"key": "main_homogeneous_train", "shape": [10535, 20000, 2], "bytes": 1685600128},
        "valid": {"key": "main_homogeneous_valid", "shape": [1405, 20000, 2], "bytes": 224800128},
    }
    current_cache_files: list[dict[str, Any]] = []
    for scope, spec in expected_arrays.items():
        output = cache_receipt["output_files"][spec["key"]]
        target_output = target["expected_outputs"]["train" if scope == "train" else "fit_valid"]
        for key in ("shape", "bytes", "dtype", "sha256"):
            if output.get(key) != target_output.get(key):
                raise ValidationError(f"cache {scope} output differs from plan at {key}")
        expected_path = (project_root / target_output["path"]).resolve()
        if Path(output["path"]).resolve() != expected_path:
            raise ValidationError(f"cache {scope} output path differs from plan")
        if output.get("shape") != spec["shape"] or output.get("bytes") != spec["bytes"] or output.get("dtype") != "float32":
            raise ValidationError(f"cache {scope} output shape/dtype/bytes differ")
        if output.get("finite_validation") != "PASS_ALL_VALUES_BY_FROZEN_CONSUMER":
            raise ValidationError(f"cache {scope} output lacks complete finite validation")
        array_path = project_root / output["path"]
        if not array_path.is_relative_to(project_root) or not array_path.is_file() or array_path.stat().st_size != output["bytes"]:
            raise ValidationError(f"cache {scope} current path/size differs")
        current_cache_files.append({
            "path": output["path"],
            "bytes": array_path.stat().st_size,
            "sha256_from_validation_receipt": output["sha256"],
            "array_rehashed_by_preflight": False,
        })

    observed_teachers: list[dict[str, Any]] = []
    for seed, teacher_plan, input_teacher, ancestry in zip(
        (3, 4, 5), teacher_plans, inputs_by_seed, ancestry_by_seed, strict=True
    ):
        if teacher_plan["seed"] != seed or ancestry.get("seed") != seed:
            raise ValidationError("teacher seed ordering differs")
        if ancestry.get("run") != teacher_plan["run"] or ancestry.get("epoch") != 2:
            raise ValidationError(f"teacher seed {seed} run/epoch ancestry differs")
        if ancestry.get("checkpoint_sha256") != teacher_plan["checkpoint_sha256"]:
            raise ValidationError(f"teacher seed {seed} checkpoint ancestry differs")
        if ancestry.get("provenance_sha256") != teacher_plan["provenance_sha256"]:
            raise ValidationError(f"teacher seed {seed} provenance ancestry differs")
        input_predictions = input_teacher["predictions"]
        for source_scope, teacher_key, cache_key in (
            ("train", "fit_train_prediction", "train_prediction_sha256"),
            ("valid", "fit_valid_full_valid_projection_source", "valid_prediction_sha256"),
        ):
            source = input_predictions[source_scope]
            source_array = source["array"]
            teacher_target = teacher_plan[teacher_key]
            if f"_s{seed}_r5_" not in source["path"]:
                raise ValidationError(f"teacher seed {seed} input path is misordered")
            if source.get("prediction_sha256") != ancestry.get(cache_key) or source_array.get("sha256") != ancestry.get(cache_key):
                raise ValidationError(f"teacher seed {seed} {source_scope} input hash differs")
            if source.get("sidecar_sha256") != teacher_target.get("sidecar_sha256"):
                raise ValidationError(f"teacher seed {seed} {source_scope} sidecar differs")
            if teacher_target.get("array_sha256") != ancestry.get(cache_key):
                raise ValidationError(f"teacher seed {seed} {source_scope} plan hash differs")
        observed_teachers.append({
            "seed": seed,
            "run": ancestry["run"],
            "checkpoint_sha256": ancestry["checkpoint_sha256"],
            "provenance_sha256": ancestry["provenance_sha256"],
            "train_prediction_sha256": ancestry["train_prediction_sha256"],
            "valid_prediction_sha256": ancestry["valid_prediction_sha256"],
            "input_order_sha256": input_predictions["train"]["ordered_row_groups_sha256"],
        })

    if plan["authorization"]["gpu_profile"] or plan["authorization"]["student_fit"] or plan["authorization"]["provider_upload_or_push"]:
        raise ValidationError("preparation manifest incorrectly grants execution authority")
    if plan["kaggle_delivery"].get("status") != "CACHE_VALIDATED_BUT_NOT_UPLOAD_READY":
        raise ValidationError("Kaggle delivery status is not pending source/cache assets and root release")

    return {
        "status": "CACHE_VALIDATED_PENDING_PROFILE_DELIVERY_AND_ROOT_RELEASE",
        "teacher_ancestry": observed_teachers,
        "source_qualified_membership": membership_fields,
        "cache_outputs": current_cache_files,
        "cache_arrays_rehashed_by_preflight": False,
        "profile_authorized": False,
        "student_fit_authorized": False,
        "provider_action": False,
    }


def inspect_preparation(project_root: Path) -> dict[str, Any]:
    if "torch" in sys.modules:
        raise ValidationError("static cache preparation check must run before torch import")
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != "wnn.clean-t3-f0-matched-student-pair-preparation.r1":
        raise ValidationError("unexpected preparation manifest schema")
    if plan.get("status") != "PREPARATION_ONLY_CACHE_VALIDATED_PENDING_PROFILE_DELIVERY_AND_ROOT_RELEASE":
        raise ValidationError("preparation status changed")
    verified = verify_bindings(plan, project_root)

    binding_root = plan["immutable_bindings"]
    input_receipt = load_bound_json(plan, "actual_teacher_input_validation_receipt_r2", project_root)
    cache_receipt = load_bound_json(plan, "cache_materialization_validation_receipt", project_root)
    root_release = load_bound_json(plan, "root_cache_materialization_release", project_root)
    result = verify_receipt_documents(plan, input_receipt, cache_receipt, root_release, project_root)

    recipe_binding = binding_root["base_recipe"]
    recipe_path = resolve_bound_path(project_root, recipe_binding["path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    r1_validator_path = PACKET_DIR / "validate_pair_preparation_r1.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("pair_r1_validator", r1_validator_path)
    if spec is None or spec.loader is None:
        raise ValidationError("cannot load preserved R1 command validator")
    r1_validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r1_validator)
    r1_validator.verify_command_derivation(plan, recipe)

    blockers = list(plan["blocking_dependencies"])
    return {
        "schema": "wnn.clean-t3-f0-matched-student-pair-cache-validated-preflight.r2",
        "status": result["status"],
        "created_utc": datetime.now(UTC).isoformat(),
        "preparation_id": plan["preparation_id"],
        "preparation_manifest_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "project_root": str(project_root.resolve(strict=True)),
        "static_binding_paths_verified": len(verified),
        "cache_materialization_validation_receipt_sha256": binding_root["cache_materialization_validation_receipt"]["sha256"],
        "actual_teacher_input_validation_receipt_sha256": binding_root["actual_teacher_input_validation_receipt_r2"]["sha256"],
        "root_cache_materialization_release_sha256": binding_root["root_cache_materialization_release"]["sha256"],
        **result,
        "blockers": blockers,
        "launch_allowed": False,
        "torch_imported": False,
        "prediction_array_values_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = inspect_preparation(args.project_root.resolve(strict=True))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
