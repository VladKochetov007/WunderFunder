"""Create the cache-validated successor manifest without reading large arrays."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET_DIR = Path(__file__).resolve().parent
WORKTREE_ROOT = Path("/home/vlad/development/wunder/wnn_clean_f0_paired_students_20261012")
PACKET_RELATIVE_DIR = PACKET_DIR.relative_to(WORKTREE_ROOT)
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R1_PATH = PACKET_DIR / "PAIR_PREPARATION_R1.json"
R2_PATH = PACKET_DIR / "PAIR_PREPARATION_R2.json"
EVIDENCE_DIR = PACKET_DIR / "cache_evidence_r2"

PINS = {
    "r1_manifest": "77c82a31b8763faecd5988ac1b5ab34d3bfc3efa548857dada5790db3a950574",
    "input_validation": "3e5c09731a1dbf83b34de06f34fdf76c54ec3ceb7ea53376d63d46c798441910",
    "cache_validation": "f3213306dd4ec17137286721cbdf21a6f50a62a1527763cb8cf63105398b50c8",
    "root_materialization_release": "707c38a9a19c1c87103393ac8dd17fb2cd61af6b9f0f84daafd56820861691a0",
    "metadata": "9d587f359cf9acf1b3cc8abeefc8e19e38fc2f42212d932409f52f8e933872b3",
}


class PreparationError(ValueError):
    """Raised when an immutable producer or cache receipt does not match."""


def sha256_file(path: Path) -> str:
    """Hash bounded-size control files only; cache arrays are never opened."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"expected JSON object: {path}")
    return value


def binding(path: Path, display_path: str) -> dict[str, Any]:
    return {"path": display_path, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def receipt_path(name: str) -> Path:
    return EVIDENCE_DIR / name


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate receipt identities, scope, membership, and cache output descriptors."""
    if sha256_file(R1_PATH) != PINS["r1_manifest"]:
        raise PreparationError("preserved R1 manifest bytes changed")

    input_path = receipt_path("ACTUAL_INPUT_VALIDATION_R2.json")
    cache_path = receipt_path("CACHE_MATERIALIZATION_VALIDATION_R1.json")
    release_path = receipt_path("ROOT_CONDITIONAL_CACHE_MATERIALIZATION_RELEASE_R1.json")
    input_receipt = load_json(input_path)
    cache_receipt = load_json(cache_path)
    release = load_json(release_path)

    expected_hashes = {
        input_path: PINS["input_validation"],
        cache_path: PINS["cache_validation"],
        release_path: PINS["root_materialization_release"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise PreparationError(f"evidence receipt SHA differs: {path.name}")

    if input_receipt.get("status") != "PASS_ALL_TEACHER_INPUTS_READY":
        raise PreparationError("teacher input validation did not pass")
    if input_receipt.get("fit_train_predictions_pending") != []:
        raise PreparationError("teacher FIT_TRAIN predictions are still pending")
    if set(input_receipt.get("fit_train_predictions_ready", [])) != {
        "clean_t3_f0_muonall_h384_e3_s3_r5",
        "clean_t3_f0_muonall_h384_e3_s4_r5",
        "clean_t3_f0_muonall_h384_e3_s5_r5",
    }:
        raise PreparationError("input receipt does not bind exactly teacher seeds 3/4/5")

    if cache_receipt.get("status") != "PASS_FROZEN_CONSUMER_AND_ACTUAL_OUTPUT_VALIDATION":
        raise PreparationError("frozen cache consumer or actual output validation did not pass")
    if cache_receipt.get("student_fit_or_profile_launched") is not False:
        raise PreparationError("cache producer receipt unexpectedly records student execution")
    if cache_receipt.get("root_release_sha256") != PINS["root_materialization_release"]:
        raise PreparationError("cache receipt is not bound to the expected root materialization release")
    if release.get("cache_materialization_authorized") is not True:
        raise PreparationError("root receipt does not authorize cache materialization")
    if release.get("status") != "conditional_serial_after_s5_terminal_validity":
        raise PreparationError("root materialization release scope/status differs")

    membership = cache_receipt.get("source_qualified_membership", {})
    expected_membership = {
        "evaluation_valid_count": 468,
        "fit_train_count": 10535,
        "fit_train_order_sha256": "8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda",
        "fit_train_source": "datasets/train.parquet",
        "fit_valid_count": 1405,
        "fit_valid_evaluation_overlap": 0,
        "fit_valid_order_sha256": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75",
        "fit_valid_source": "datasets/valid.parquet",
        "fold": 0,
        "split_manifest_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "target_order_source": "train_targets.npz and valid_targets.npz row_group members only",
    }
    for key, expected_value in expected_membership.items():
        if membership.get(key) != expected_value:
            raise PreparationError(f"source-qualified F0 membership differs at {key}")

    ancestry = cache_receipt.get("teacher_ancestry", [])
    if [item.get("seed") for item in ancestry] != [3, 4, 5]:
        raise PreparationError("teacher ancestry is not ordered seeds 3/4/5")
    for teacher in ancestry:
        if teacher.get("epoch") != 2 or teacher.get("run") != (
            f"clean_t3_f0_muonall_h384_e3_s{teacher['seed']}_r5"
        ):
            raise PreparationError("teacher cache ancestry differs from frozen EP2 sources")

    input_teachers = input_receipt.get("teacher_records", [])
    if len(input_teachers) != 3:
        raise PreparationError("input validation does not contain exactly three teacher records")
    for seed, input_teacher, ancestry_teacher in zip([3, 4, 5], input_teachers, ancestry, strict=True):
        for scope, expected_hash, expected_rows, expected_order in (
            ("train", ancestry_teacher["train_prediction_sha256"], 10535,
             "8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda"),
            ("valid", ancestry_teacher["valid_prediction_sha256"], 1873,
             "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7"),
        ):
            observed = input_teacher.get("predictions", {}).get(scope, {})
            array = observed.get("array", {})
            if f"_s{seed}_r5_" not in observed.get("path", ""):
                raise PreparationError(f"input teacher prediction order/path does not match seed {seed}")
            if observed.get("prediction_sha256") != expected_hash or array.get("sha256") != expected_hash:
                raise PreparationError(f"seed {seed} {scope} source/cache prediction hashes differ")
            if observed.get("row_group_count") != expected_rows or observed.get("ordered_row_groups_sha256") != expected_order:
                raise PreparationError(f"seed {seed} {scope} source row-group order differs")
            if not observed.get("ready") or not observed.get("sidecar_sha256"):
                raise PreparationError(f"seed {seed} {scope} output/sidecar is not validated")

    outputs = cache_receipt.get("output_files", {})
    expected = {
        "main_homogeneous_train": {
            "shape": [10535, 20000, 2],
            "bytes": 1685600128,
            "dtype": "float32",
            "sha256": "6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3",
            "ordered_row_groups_sha256": "8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda",
        },
        "main_homogeneous_valid": {
            "shape": [1405, 20000, 2],
            "bytes": 224800128,
            "dtype": "float32",
            "sha256": "d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2",
            "ordered_row_groups_sha256": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75",
        },
    }
    for name, expected_fields in expected.items():
        record = outputs.get(name, {})
        for key, value in expected_fields.items():
            if record.get(key) != value:
                raise PreparationError(f"cache output {name} field {key} differs")
        if record.get("finite_validation") != "PASS_ALL_VALUES_BY_FROZEN_CONSUMER":
            raise PreparationError(f"cache output {name} lacks full finite-value validation")

    target_contract = cache_receipt.get("target_contract", {})
    if target_contract.get("homogeneous_scale") != 0.8125:
        raise PreparationError("homogeneous target scale differs")
    if target_contract.get("teacher_mean") != "unweighted arithmetic mean in float32":
        raise PreparationError("teacher mean semantics differ")
    if target_contract.get("clipping") != "none during cache construction":
        raise PreparationError("cache clipping semantics differ")

    metadata_path = PROJECT_ROOT / "cache/clean_t3_f0_h390_t3r5_a9_r1/metadata.json"
    if metadata_path.stat().st_size != 323260 or sha256_file(metadata_path) != PINS["metadata"]:
        raise PreparationError("current cache metadata differs from the validated receipt")
    for record in outputs.values():
        array_path = Path(record["path"])
        if not array_path.is_file() or array_path.stat().st_size != record["bytes"]:
            raise PreparationError(f"cache array path/size differs: {array_path}")

    return input_receipt, cache_receipt, release, outputs


def build_manifest() -> dict[str, Any]:
    """Derive R2 from R1 and attach the completed cache evidence, create-only."""
    input_receipt, cache_receipt, release, outputs = validate_inputs()
    plan = load_json(R1_PATH)
    plan = copy.deepcopy(plan)
    plan["preparation_id"] = "clean-t3-f0-matched-student-pair-r6-20261012-cache-validated-r2"
    plan["created_utc"] = datetime.now(UTC).isoformat()
    plan["status"] = "PREPARATION_ONLY_CACHE_VALIDATED_PENDING_PROFILE_DELIVERY_AND_ROOT_RELEASE"
    old_fit_valid_hash = plan["fold_membership"]["fit_valid_set_sha256"]
    plan["fold_membership"]["fit_valid_set_sha256"] = cache_receipt[
        "source_qualified_membership"
    ]["fit_valid_order_sha256"]
    plan["r2_corrections"] = [{
        "field": "fold_membership.fit_valid_set_sha256",
        "r1_value": old_fit_valid_hash,
        "r2_value": plan["fold_membership"]["fit_valid_set_sha256"],
        "basis": "bound cache materialization validation receipt source_qualified_membership.fit_valid_order_sha256",
        "reason": "R1 value was truncated to 62 hex characters; R2 preserves R1 and corrects the successor binding to the authoritative 64-character fingerprint.",
    }]

    rel_root = PROJECT_ROOT.resolve(strict=True)
    evidence_bindings: dict[str, dict[str, Any]] = {}
    for key, filename in (
        ("cache_materialization_validation_receipt", "CACHE_MATERIALIZATION_VALIDATION_R1.json"),
        ("actual_teacher_input_validation_receipt_r2", "ACTUAL_INPUT_VALIDATION_R2.json"),
        ("root_cache_materialization_release", "ROOT_CONDITIONAL_CACHE_MATERIALIZATION_RELEASE_R1.json"),
    ):
        path = receipt_path(filename)
        display_path = (PACKET_RELATIVE_DIR / "cache_evidence_r2" / filename).as_posix()
        evidence_bindings[key] = binding(path, display_path)
    r1_binding_path = (PACKET_RELATIVE_DIR / "PAIR_PREPARATION_R1.json").as_posix()
    plan["immutable_bindings"]["historical_r1_manifest"] = binding(R1_PATH, r1_binding_path)
    plan["immutable_bindings"].pop("actual_teacher_input_validation_receipt", None)
    plan["immutable_bindings"].update(evidence_bindings)

    metadata_path = PROJECT_ROOT / "cache/clean_t3_f0_h390_t3r5_a9_r1/metadata.json"
    plan["immutable_bindings"]["validated_cache_metadata"] = binding(
        metadata_path,
        metadata_path.relative_to(rel_root).as_posix(),
    )

    plan["teacher_target"]["materialization_receipt"] = {
        **evidence_bindings["cache_materialization_validation_receipt"],
        "status": cache_receipt["status"],
    }
    plan["teacher_target"]["root_cache_release"] = {
        **evidence_bindings["root_cache_materialization_release"],
        "status": release["status"],
        "scope": "conditional cache materialization only; no profile or student fit authorization",
    }
    plan["teacher_target"]["actual_teacher_input_validation_receipt"] = {
        **evidence_bindings["actual_teacher_input_validation_receipt_r2"],
        "status": input_receipt["status"],
    }
    plan["teacher_target"]["status"] = cache_receipt["status"]
    plan["teacher_target"]["validation_summary"] = {
        "source_qualified_fold": 0,
        "teacher_seeds": [3, 4, 5],
        "fit_train_groups": 10535,
        "fit_valid_groups": 1405,
        "excluded_evaluation_groups": 468,
        "fit_valid_evaluation_overlap": 0,
        "full_array_hashes_and_finite_scan": "PASS in bound frozen-consumer receipt; not repeated by this metadata-only builder",
        "target_formula": "float32(0.8125 * float32_mean(seed3_ep2, seed4_ep2, seed5_ep2))",
    }
    plan["teacher_target"]["expected_outputs"]["train"]["sha256"] = outputs["main_homogeneous_train"]["sha256"]
    plan["teacher_target"]["expected_outputs"]["fit_valid"]["sha256"] = outputs["main_homogeneous_valid"]["sha256"]

    input_teachers = input_receipt["teacher_records"]
    ancestry = {record["seed"]: record for record in cache_receipt["teacher_ancestry"]}
    if len(input_teachers) != 3:
        raise PreparationError("expected exactly three input-validation teacher records")
    for teacher_plan, input_teacher in zip(plan["teacher_target"]["teachers"], input_teachers, strict=True):
        seed = teacher_plan["seed"]
        if seed not in ancestry:
            raise PreparationError(f"seed {seed} missing from cache ancestry")
        if ancestry[seed]["checkpoint_sha256"] != teacher_plan["checkpoint_sha256"]:
            raise PreparationError(f"seed {seed} checkpoint ancestry differs")
        if ancestry[seed]["provenance_sha256"] != teacher_plan["provenance_sha256"]:
            raise PreparationError(f"seed {seed} provenance ancestry differs")
        for scope, field in (("train", "fit_train_prediction"), ("valid", "fit_valid_full_valid_projection_source")):
            observed = input_teacher["predictions"][scope]
            array = observed["array"]
            expected_seed_path = f"_s{seed}_r5_"
            if expected_seed_path not in observed["path"]:
                raise PreparationError(f"input teacher order/path does not match seed {seed}")
            expected_hash = ancestry[seed]["train_prediction_sha256" if scope == "train" else "valid_prediction_sha256"]
            if observed["prediction_sha256"] != expected_hash or array["sha256"] != expected_hash:
                raise PreparationError(f"seed {seed} {scope} input/cache prediction hashes disagree")
            teacher_plan[field].update({
                "array_sha256": expected_hash,
                "bytes": array["bytes"],
                "dtype": array["dtype"],
                "path": Path(array["path"]).relative_to(rel_root).as_posix(),
                "shape": array["shape"],
                "sidecar_path": Path(observed["sidecar_path"]).relative_to(rel_root).as_posix(),
                "sidecar_sha256": observed["sidecar_sha256"],
                "ordered_row_groups_sha256": observed["ordered_row_groups_sha256"],
                "status": "PASS_ACTUAL_INPUT_R2_AND_CACHE_VALIDATION_R1",
            })
            teacher_plan[field]["input_validation_receipt"] = evidence_bindings[
                "actual_teacher_input_validation_receipt_r2"
            ]
            if scope == "valid":
                teacher_plan[field]["projection_count"] = 1405
                teacher_plan[field]["excluded_evaluation_count"] = 468
                teacher_plan[field]["row_group_count"] = 1873
        teacher_plan["fit_train_prediction"].pop("contract", None)
        teacher_plan["fit_train_prediction"].pop("status_at_capture", None)
        teacher_plan["fit_train_prediction"]["receipt"] = evidence_bindings[
            "actual_teacher_input_validation_receipt_r2"
        ]
        teacher_plan["fit_train_prediction"]["cache_validation_receipt"] = evidence_bindings[
            "cache_materialization_validation_receipt"
        ]

    plan["blocking_dependencies"] = [
        "Bind exact private Kaggle source and validated cache dataset versions, READY receipt, nested mount paths, and project-root aliases.",
        "Run and review the required full-cache H256/B16/C2000 FP16 profile on one actual Kaggle device, serially; no profile result is claimed.",
        "Root must refresh quota/resources, check output collisions, and separately authorize the profile and both six-epoch student fits.",
        "Stage and reverify both own parent checkpoints/provenance under the frozen trainer aliases before any fit admission.",
    ]
    plan["kaggle_delivery"]["status"] = "CACHE_VALIDATED_BUT_NOT_UPLOAD_READY"
    return plan


def main() -> int:
    """Create one new R2 manifest while preserving all R1 bytes."""
    if R2_PATH.exists():
        raise FileExistsError(R2_PATH)
    plan = build_manifest()
    rendered = json.dumps(plan, sort_keys=True, indent=2) + "\n"
    with R2_PATH.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(json.dumps({"path": str(R2_PATH), "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                      "bytes": len(rendered.encode()), "status": plan["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
