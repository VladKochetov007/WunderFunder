#!/usr/bin/env python3
"""Independently audit the additive clean-T3 cache metadata compatibility repair."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OWNER = Path("/home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile")
PKT = OWNER / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1"
)
CACHE_R1 = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r1"
CACHE_R2 = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r2"
SOURCE = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot"
)
ADAPTER_CONTRACT = PKT / "cache_adapter_r2/CACHE_METADATA_COMPATIBILITY_ADAPTER_R2.json"
ADAPTER_SCRIPT = PKT / "cache_adapter_r2/build_cache_metadata_compatibility_r2.py"
VALIDATION = PKT / "cache_adapter_r2/CACHE_METADATA_COMPATIBILITY_VALIDATION_R2.json"
PROFILE_R1 = PKT / "student_resource_profile_r2/PROFILE_CONTRACT_R1.json"
PROFILE_R2 = PKT / "student_resource_profile_r2/PROFILE_CONTRACT_R2.json"
LAUNCH_R2 = PKT / "student_resource_profile_r2/PROFILE_LAUNCH_MANIFEST_R2.json"
TRAINER = SOURCE / "scripts/train_rnn.py"
CACHE_PRODUCER = SOURCE / "scripts/build_cleanfold_t3_cache.py"
EXPECTED_EPOCH = {
    "student_epochs": 6,
    "student_checkpoint_policy": "fixed",
    "student_checkpoint_epoch": 5,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract_literal_epoch_convention(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
            if key_node is not None and ast.literal_eval(key_node) == "epoch_convention":
                value = ast.literal_eval(value_node)
                if isinstance(value, dict):
                    found.append(value)
    require(len(found) == 1, "frozen producer must contain one literal metadata epoch_convention")
    return found[0]


def actual_trainer_provenance_block(path: Path) -> tuple[ast.If, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (
            isinstance(test.left, ast.Name)
            and test.left.id == "distill_cache_validation"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        has_consumer = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "provenance"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "distillation_cache"
                for target in child.targets
            )
            for child in ast.walk(node)
        )
        if has_consumer:
            matches.append(node)
    require(len(matches) == 1, "frozen trainer must have one guarded distillation provenance block")
    block = matches[0]
    keys: list[str] = []
    for child in ast.walk(block):
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name)
            and child.value.id == "cache_metadata"
            and isinstance(child.slice, ast.Constant)
            and isinstance(child.slice.value, str)
        ):
            keys.append(child.slice.value)
    require(len(keys) == len(set(keys)), "trainer provenance block repeats a cache metadata key")
    return block, sorted(keys)


def execute_actual_provenance_block(
    block: ast.If, metadata: dict[str, Any], validator_output: dict[str, Any]
) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "args": SimpleNamespace(
            distill_cache_root=str(CACHE_R2),
            distill_cache_variant="homogeneous",
        ),
        "distill_cache_validation": validator_output,
        "provenance": {},
    }
    module = ast.Module(body=[block], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), filename=str(TRAINER), mode="exec")
    # The validator return shape is the actual trainer interface. Its metadata
    # object is taken from the already-run full validator receipt and current file.
    require(
        validator_output.get("metadata", metadata) is metadata
        or validator_output.get("metadata") == metadata,
        "validator interface object does not carry the current actual metadata",
    )
    exec(code, namespace)
    return namespace["provenance"]["distillation_cache"]


def normalize_profile_argv(argv: list[str]) -> list[str]:
    """Ignore only the R1→R2 output name and corrected cache-root path."""
    normalized = list(argv)
    for option, marker in (
        ("--name", "<PROFILE_NAME>"),
        ("--distill-cache-root", "<CACHE_ROOT>"),
    ):
        positions = [index for index, value in enumerate(normalized) if value == option]
        require(len(positions) == 1, f"profile argv must contain one {option}")
        value_index = positions[0] + 1
        require(value_index < len(normalized), f"profile argv has no value for {option}")
        normalized[value_index] = marker
    return normalized


def main() -> None:
    receipt_path = Path(__file__).with_name("INDEPENDENT_REVIEW_R2.json")
    require(CACHE_R1.is_dir() and CACHE_R2.is_dir(), "R1 and R2 cache roots must exist")
    require(all(p.is_file() for p in (
        ADAPTER_CONTRACT, ADAPTER_SCRIPT, VALIDATION, PROFILE_R1, PROFILE_R2, LAUNCH_R2,
        TRAINER, CACHE_PRODUCER,
    )), "a pinned review input is missing")

    adapter = read_json(ADAPTER_CONTRACT)
    validation = read_json(VALIDATION)
    profile1 = read_json(PROFILE_R1)
    profile2 = read_json(PROFILE_R2)
    launch = read_json(LAUNCH_R2)
    metadata1_path = CACHE_R1 / "metadata.json"
    metadata2_path = CACHE_R2 / "metadata.json"
    metadata1 = read_json(metadata1_path)
    metadata2 = read_json(metadata2_path)

    expected_pins = {
        "adapter_contract_sha256": (
            ADAPTER_CONTRACT,
            "c04fc22b78be9869c4e447316a5bd84a100bd4ef469bb244ce6f591974d9c033",
        ),
        "adapter_script_sha256": (
            ADAPTER_SCRIPT,
            "2e486fd26bacfcdf3cf752c765f8e9b7815027f3a0f451586ca1c209c87b21f2",
        ),
        "validation_receipt_sha256": (
            VALIDATION,
            "90528f221fdca114f506871fed06372bb7580a9d2700632abd156aa03830bedf",
        ),
        "profile_contract_r2_sha256": (
            PROFILE_R2,
            "cae424ce880651db4b618a757ae9f2979fca40a45b28273e3d2b413b32ed2e76",
        ),
        "profile_contract_r1_sha256": (
            PROFILE_R1,
            "a72c86609903fcccf76e565a97b94f00edcdd593f804a89297598aeb9640d9bc",
        ),
        "launch_manifest_r2_sha256": (
            LAUNCH_R2,
            "02481fe96b3b968f8e145e6f26a94d3ad63ab02cedd929a83f9b013366f22132",
        ),
        "r1_metadata_sha256": (
            metadata1_path,
            "9d587f359cf9acf1b3cc8abeefc8e19e38fc2f42212d932409f52f8e933872b3",
        ),
        "r2_metadata_sha256": (
            metadata2_path,
            "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8",
        ),
        "actual_trainer_sha256": (
            TRAINER,
            "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6",
        ),
        "actual_producer_sha256": (
            CACHE_PRODUCER,
            "62aea91099b14cdd5ef987dcbd60675dbb650b152bdfe2a27aeda3ed24e3d12e",
        ),
    }
    for label, (path, expected) in expected_pins.items():
        actual = sha256_file(path)
        require(actual == expected, f"{label} changed: {actual}")
    require(validation.get("status") == "PASS_ACTUAL_FROZEN_CACHE_AND_PROVENANCE_CONSUMER",
            "owner's full-validator receipt is not PASS")
    require(validation.get("student_profile_authorized") is False, "validator authorized a profile")
    require(validation.get("student_fit_authorized") is False, "validator authorized a fit")
    require(validation.get("profile_fit_or_grade") is False, "R2 cache step includes profile/fit/grade")

    require(metadata1.get("epoch_convention") is None, "preserved R1 no longer reproduces the failure")
    require(metadata2.get("epoch_convention") == EXPECTED_EPOCH, "R2 epoch convention is incorrect")
    require(metadata2.get("student_contract", {}).get("epochs") == 6, "student contract epoch count changed")
    require(metadata2.get("student_contract", {}).get("checkpoint_policy") == "fixed",
            "student checkpoint policy changed")
    require(metadata2.get("student_contract", {}).get("checkpoint_epoch") == 5,
            "student checkpoint epoch changed")
    actual_epoch = extract_literal_epoch_convention(CACHE_PRODUCER)
    require(actual_epoch == EXPECTED_EPOCH, "R2 field differs from the frozen cache producer's literal")

    allowed_metadata_differences = {
        "adapter_contract", "adapter_contract_sha256", "epoch_convention"
    }
    changed = {
        key for key in set(metadata1) | set(metadata2)
        if metadata1.get(key) != metadata2.get(key)
    }
    require(changed == allowed_metadata_differences,
            f"R2 metadata changed unexpected fields: {sorted(changed)}")
    unchanged_fields = (
        "format", "version", "cache_profile", "fold", "split_manifest",
        "split_manifest_sha256", "fit_groups", "evaluation_valid_groups", "family_order",
        "homogeneous_family", "source_coefficients", "coefficient_sum", "homogeneous_scale",
        "source_qualification", "sources", "target_contract", "student_contract",
        "stored_targets", "files", "output_profile",
    )
    for key in unchanged_fields:
        require(metadata1.get(key) == metadata2.get(key), f"cache semantic field changed: {key}")

    require(profile1["predeclared_recipe"] == profile2["predeclared_recipe"],
            "R2 profile changed the predeclared scientific recipe")
    require(profile1["membership"] == profile2["membership"],
            "R2 profile changed fitting/evaluation membership")
    require(profile1["input_source_artifacts"] == profile2["input_source_artifacts"],
            "R2 profile changed source checkpoint/data bindings")
    require(profile1["resource_request"] == profile2["resource_request"],
            "R2 profile changed resource limits")
    argv_r1 = profile1["runnable"]["argv"]
    argv_r2 = profile2["runnable"]["argv"]
    require(normalize_profile_argv(argv_r1) == normalize_profile_argv(argv_r2),
            "R2 argv changes exceed unique output name and corrected cache root")

    require(len(metadata2["fit_groups"]["train"]) == 10535, "train fit membership count changed")
    require(len(metadata2["fit_groups"]["valid"]) == 1405, "valid fit membership count changed")
    require(len(metadata2["evaluation_valid_groups"]) == 468, "outer evaluation count changed")
    require(set(metadata2["fit_groups"]["valid"]).isdisjoint(metadata2["evaluation_valid_groups"]),
            "valid fit and outer evaluation groups overlap")
    require(metadata2["source_qualification"]["evaluation_valid_groups_in_cache"] == 0,
            "metadata admits evaluation rows into fitting cache")
    require(metadata2["source_qualification"]["prediction_arrays_joined_by_row_group_id"] is True,
            "teacher targets are not recorded as row-group joined")
    sources = metadata2["sources"]
    require([source["seed"] for source in sources] == [3, 4, 5], "teacher seed order changed")
    require(all(source["epoch"] == 2 for source in sources), "teacher checkpoint epoch changed")
    for source in sources:
        fit = source["fit_group_memberships"]
        require(fit["train"]["count"] == 10535 and fit["train"]["member_set_matches_f0"] is True,
                f"teacher train ancestry mismatch: {source['run']}")
        require(fit["valid"]["count"] == 1405 and fit["valid"]["member_set_matches_f0"] is True,
                f"teacher valid ancestry mismatch: {source['run']}")
        require(fit["valid"]["ordered_sha256"] ==
                "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75",
                f"teacher valid ancestry hash mismatch: {source['run']}")

    trainer_declared = adapter["frozen_consumer"]
    require((MAIN / trainer_declared["trainer_path"]).resolve() == TRAINER.resolve(),
            "adapter points at a different trainer")
    require(trainer_declared["trainer_sha256"] == sha256_file(TRAINER), "adapter trainer pin differs")
    validation_consumer = validation["consumer_validation"]
    require(validation_consumer["actual_trainer_path"] == str(TRAINER),
            "actual validator used a different trainer")
    require(validation_consumer["actual_trainer_sha256"] == sha256_file(TRAINER),
            "actual validator trainer hash differs")
    require(validation_consumer["validator_metadata_sha256"] == sha256_file(metadata2_path),
            "actual validator receipt is for different metadata")
    verified_files = validation_consumer["verified_files"]
    require(
        verified_files == {
            "main_homogeneous_train": metadata2["files"]["main_homogeneous_train"]["sha256"],
            "main_homogeneous_valid": metadata2["files"]["main_homogeneous_valid"]["sha256"],
        },
        "actual frozen validator receipt does not bind both target arrays",
    )

    block, consumer_keys = actual_trainer_provenance_block(TRAINER)
    require(consumer_keys == [
        "coefficient_sum", "epoch_convention", "family_order", "homogeneous_family",
        "source_coefficients",
    ], f"actual trainer provenance fields changed: {consumer_keys}")
    # This is the actual frozen validator return interface, populated from the
    # owner's completed full validation receipt; arrays are not rescanned here.
    trainer_validator_output = {
        "metadata": metadata2,
        "metadata_sha256": validation_consumer["validator_metadata_sha256"],
        "verified_files": verified_files,
    }
    observed_provenance = execute_actual_provenance_block(block, metadata2, trainer_validator_output)
    require(observed_provenance["epoch_convention"] == EXPECTED_EPOCH,
            "actual trainer provenance block failed to carry the repaired field")
    # Run the same actual source block with preserved R1 metadata to verify it
    # reproduces the diagnosed pre-CUDA KeyError.
    old_validator_output = {
        "metadata": metadata1,
        "metadata_sha256": sha256_file(metadata1_path),
        "verified_files": {
            "main_homogeneous_train": metadata1["files"]["main_homogeneous_train"]["sha256"],
            "main_homogeneous_valid": metadata1["files"]["main_homogeneous_valid"]["sha256"],
        },
    }
    try:
        execute_actual_provenance_block(block, metadata1, old_validator_output)
    except KeyError as error:
        require(error.args == ("epoch_convention",), f"unexpected R1 failure: {error!r}")
        r1_reproduces_keyerror = True
    else:
        raise AssertionError("actual trainer block did not reproduce R1 epoch_convention KeyError")

    array_identity: dict[str, Any] = {}
    for key, entry in metadata2["files"].items():
        r1_path = CACHE_R1 / entry["path"]
        r2_path = CACHE_R2 / entry["path"]
        s1 = r1_path.stat()
        s2 = r2_path.stat()
        require(os.path.samefile(r1_path, r2_path), f"{key} is not hard-linked to R1")
        require(s1.st_dev == s2.st_dev and s1.st_ino == s2.st_ino,
                f"{key} hardlink device/inode differs")
        require(s2.st_size == entry["bytes"], f"{key} size differs from metadata")
        owner_file = validation["output_arrays"][key]
        require(owner_file["same_inode_as_r1"] is True, f"owner did not verify {key} inode")
        require(owner_file["sha256"] == entry["sha256"], f"owner array digest differs for {key}")
        require(s2.st_nlink >= 2, f"{key} link count does not show shared R1/R2 inode")
        array_identity[key] = {
            "path_r1": str(r1_path),
            "path_r2": str(r2_path),
            "bytes": s2.st_size,
            "sha256": entry["sha256"],
            "same_device_inode": True,
            "link_count": s2.st_nlink,
            "read_only_consumer": "frozen validator mmap_mode='r'; no array write in adapter",
        }

    require(profile2.get("profile_only") is True, "profile contract is not profile-only")
    require(profile2.get("launch_authorized") is False, "profile contract authorizes launch")
    require(profile2.get("student_fit_authorized") is False, "profile contract authorizes student fit")
    require(profile2.get("gpu_fit_authorized") is False, "profile contract authorizes GPU fit")
    require(profile2.get("cache_metadata_compatibility", {}).get("output_metadata_sha256")
            == sha256_file(metadata2_path), "profile contract pins different R2 metadata")
    require(profile2.get("cache_metadata_compatibility", {}).get("source_metadata_sha256")
            == sha256_file(metadata1_path), "profile contract lost R1 source metadata pin")
    require(launch.get("profile_only") is True and launch.get("gpu_fit_authorized") is False
            and launch.get("student_fit_authorized") is False,
            "launch manifest authorizes work beyond profile-only")
    require(launch.get("profile_contract_sha256") == sha256_file(PROFILE_R2),
            "launch manifest profile contract pin differs")
    require(launch.get("cache_metadata_sha256") == sha256_file(metadata2_path),
            "launch manifest cache metadata pin differs")
    controller = launch["controller"]
    require(controller.get("shared_gpu_lock_required") is True, "launch omits shared GPU lock")
    require(controller.get("fresh_admission_required") is True, "launch omits fresh admission")
    require(controller.get("worker_cpu_cores") == 1 and controller.get("worker_ram_bytes") == 5_500_000_000,
            "declared worker resources changed")
    require(controller.get("allocator_bytes") == 4_300_000_000 and
            controller.get("observed_vram_ceiling_bytes") == 5_000_000_000,
            "declared GPU memory limits changed")
    output_dir = Path(launch["expected_output_directory"])
    require(not output_dir.exists(), "R2 profile output directory already exists")
    r2_attempt_files = list((PKT / "student_resource_profile_r2").glob("PROFILE_ATTEMPT_R2*"))
    require(not r2_attempt_files, "an R2 profile attempt file exists; task is no longer preparation-only")

    worktree_head = subprocess.check_output(
        ["git", "-C", str(OWNER), "rev-parse", "HEAD"], text=True
    ).strip()
    worktree_status = subprocess.check_output(
        ["git", "-C", str(OWNER), "status", "--porcelain"], text=True
    ).splitlines()
    require(worktree_head == "b3313fe1f40bcefcfa48c9cc59d49384b48a3925",
            f"actor packet commit changed: {worktree_head}")
    receipt = {
        "schema": "wnn.independent-clean-t3-cache-metadata-r2-review.r1",
        "status": "PASS_METADATA_CONSUMER_REPAIR_ONLY",
        "review_scope": "Actual cache metadata/consumer compatibility, lineage continuity, linked-array identity and pending profile contract; no fit or profile execution.",
        "actor_worktree": str(OWNER),
        "actor_commit": worktree_head,
        "actor_worktree_clean": not worktree_status,
        "pins": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, (path, _) in expected_pins.items()
        },
        "r2_validation_receipt_status": validation["status"],
        "r2_actual_consumer_fields": consumer_keys,
        "r2_actual_provenance_block_executed": True,
        "r2_provenance_epoch_convention": observed_provenance["epoch_convention"],
        "r1_actual_provenance_block_reproduced_keyerror": r1_reproduces_keyerror,
        "metadata_diff_allowed_fields": sorted(allowed_metadata_differences),
        "metadata_diff_observed_fields": sorted(changed),
        "profile_scientific_recipe_unchanged": True,
        "profile_membership_and_source_bindings_unchanged": True,
        "profile_resource_request_unchanged": True,
        "profile_argv_allowed_deltas": ["--name output identifier", "--distill-cache-root R2 cache path"],
        "teacher_seeds_epochs": [[source["seed"], source["epoch"]] for source in sources],
        "fit_valid_groups": len(metadata2["fit_groups"]["valid"]),
        "excluded_outer_evaluation_groups": len(metadata2["evaluation_valid_groups"]),
        "source_qualification": metadata2["source_qualification"],
        "array_identity": array_identity,
        "profile": {
            "contract_path": str(PROFILE_R2),
            "contract_sha256": sha256_file(PROFILE_R2),
            "launch_manifest_path": str(LAUNCH_R2),
            "launch_manifest_sha256": sha256_file(LAUNCH_R2),
            "profile_only": True,
            "launch_authorized": False,
            "student_fit_authorized": False,
            "output_directory_absent": True,
            "r2_attempt_files_absent": True,
            "fresh_root_resource_release_still_required": True,
        },
        "limitations": [
            "The full trainer cache validator and its array SHA/finite scans are taken from the actor's R2 validation receipt; independent review did not reread the 1.91 GB arrays.",
            "The independent point-of-use test executed the exact provenance AST block against that actual validator output and current metadata; it did not import Torch, initialize CUDA, run a profile, or fit a model.",
            "Hardlinked arrays share writable inodes; integrity relies on the adapter's create-only/no-write behavior and the trainer/validator's read-only consumption.",
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "actor_commit": worktree_head,
        "r1_keyerror_reproduced": r1_reproduces_keyerror,
        "r2_actual_consumer_pass": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

