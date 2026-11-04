"""Check bounded engineering evidence for a fresh primary-train refit profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from primary_refit_profile_review_support import (
    argv_value,
    canonical,
    check_budget_deadline,
    check_memory,
    check_owned_worker,
    mapping,
    rank0_ok,
    read,
    sha,
    verify_source,
)

POLICY_NAME = "PRIMARY_REFIT_PROFILE_POLICY_R4.json"
SUPPORT_NAME = "primary_refit_profile_review_support.py"
POLICY_CORE_SHA256 = "c7d0880db7b0c126cea080be1732aa10e83d6039d2dc3f8265188c1981882b76"


def review(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = args.policy.resolve(strict=True)
    policy = read(policy_path)
    script_sha = sha(Path(__file__).resolve())
    policy_core = dict(policy)
    policy_core.pop("reviewer_script_sha256", None)
    support_path = Path(__file__).with_name(SUPPORT_NAME).resolve(strict=True)
    if (
        policy_path.name != POLICY_NAME
        or policy.get("schema") != "wnn.primary-refit-profile-policy.v1"
        or policy.get("status") != "frozen"
        or policy.get("reviewer_script_sha256") != script_sha
        or policy.get("reviewer_support_sha256") != sha(support_path)
        or canonical(policy_core) != POLICY_CORE_SHA256
    ):
        raise ValueError("frozen Stage 4 profile policy/support does not pin this reviewer")

    paths = {
        name: getattr(args, name).resolve(strict=True)
        for name in (
            "request",
            "selection",
            "profile",
            "worker",
            "attempt",
            "budget",
            "deadline",
            "device_inventory",
        )
    }
    request, selection, profile = (
        read(paths["request"]),
        read(paths["selection"]),
        read(paths["profile"]),
    )
    worker, attempt, budget = read(paths["worker"]), read(paths["attempt"]), read(paths["budget"])
    deadline, inventory = read(paths["deadline"]), read(paths["device_inventory"])
    request_sha, profile_sha, worker_sha = (
        sha(paths["request"]),
        sha(paths["profile"]),
        sha(paths["worker"]),
    )
    expected, limits, bounds = map(
        mapping, (policy.get("identity"), policy.get("profile"), policy.get("campaign_bounds"))
    )
    checks: dict[str, bool] = {}

    training_source = mapping(request.get("training_source"))
    selected_by = mapping(request.get("selected_by"))
    config_ref, primary_ref = mapping(request.get("config")), mapping(request.get("primary_split"))
    profile_plan, fit_plan = mapping(request.get("profile")), mapping(request.get("fit"))
    checks["selection_request"] = (
        selection.get("schema") == "wnn.multistage-primary-refit-selection.v1"
        and selection.get("selection_used_primary_valid") is False
        and selection.get("auxiliary_split_sha256") == expected.get("auxiliary_selection_sha256")
        and selected_by.get("stage") == 1
        and type(selected_by.get("trial_number")) is int
        and isinstance(selected_by.get("family"), str)
        and isinstance(selected_by.get("fixed_final_inner_wp"), (int, float))
        and math.isfinite(selected_by["fixed_final_inner_wp"])
        and any(
            isinstance(row, dict)
            and row.get("request_path") == str(paths["request"])
            and row.get("request_sha256") == request_sha
            and row.get("config_sha256") == config_ref.get("sha256")
            and row.get("fit_id") == request.get("fit_id")
            for row in selection.get("selected", [])
        )
        and str(request.get("fit_id", "")).endswith(
            f"s1t{selected_by.get('trial_number', -1):06d}-{selected_by.get('family', '')}"
        )
        and any(
            isinstance(row, dict)
            and row.get("trial_number") == selected_by.get("trial_number")
            and row.get("family") == selected_by.get("family")
            and row.get("fixed_final_inner_wp") == selected_by.get("fixed_final_inner_wp")
            for row in selection.get("ranked_direct_stage1", [])
        )
    )

    command = profile_plan.get("argv")
    if not isinstance(command, list) or len(command) < 2:
        raise ValueError("prepared profile argv is malformed")
    trainer = Path(str(command[1])).resolve(strict=True)
    source_root = trainer.parent.parent.resolve(strict=True)
    checks["source4_inventory"] = (
        trainer.name == "train_sequence.py"
        and trainer.parent.name == "scripts"
        and training_source.get("sha256") == expected.get("source_sha256")
        and verify_source(
            source_root, training_source.get("files"), str(expected.get("source_sha256"))
        )
        and worker.get("source_root") == str(source_root)
    )
    if not checks["source4_inventory"]:
        raise ValueError(
            "profile request/worker does not bind the complete pinned Source4 inventory"
        )
    if mapping(profile.get("identity")).get("source_sha256") != expected.get(
        "source_sha256"
    ) or mapping(profile.get("identity")).get("source_files") != training_source.get("files"):
        raise ValueError(
            "native profile source identity differs from the prepared Source4 inventory"
        )
    source_path = str(source_root / "src")
    sys.path[:] = [
        item for item in sys.path if Path(item or ".").resolve() != Path(source_path).resolve()
    ]
    sys.path.insert(0, source_path)
    for name in tuple(sys.modules):
        if name == "wnn" or name.startswith("wnn."):
            del sys.modules[name]
    from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split
    from wnn.training.sequence_cache_precision import validate_precision_contract
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_features import sequence_input_schema, sequence_input_width
    from wnn.training.sequence_normalization import (
        build_normalization_identity,
        normalization_identity_sha256,
    )

    config_path = Path(str(config_ref.get("path"))).resolve(strict=True)
    config = load_config(config_path)
    split_path = Path(str(primary_ref.get("path"))).resolve(strict=True)
    data_root_text = argv_value(command, "--data-root")
    if not isinstance(data_root_text, str):
        raise ValueError("prepared profile command has no data root")
    data_root = Path(data_root_text).resolve(strict=True)
    expected_profile_argv = [
        command[0],
        str(trainer),
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split_path),
        "--output",
        str(Path(str(profile_plan.get("output"))).resolve()),
        "--profile-updates",
        str(limits.get("updates")),
    ]
    checks["prepared_command_config"] = (
        request.get("schema") == "wnn.multistage-primary-refit-request.v1"
        and command == expected_profile_argv
        and argv_value(command, "--config") == str(config_path)
        and argv_value(command, "--split-manifest") == str(split_path)
        and argv_value(command, "--output") == profile_plan.get("output")
        and argv_value(command, "--profile-updates") == str(limits.get("updates"))
        and not any(token in command for token in ("--resume", "--initializer", "--teacher-cache"))
        and sha(config_path) == config_ref.get("sha256")
        and canonical(config) == config_ref.get("normalized_identity_sha256")
        and config.get("epochs") == config_ref.get("epochs_preserved")
        and config.get("seed") == config_ref.get("seed_preserved")
        and config.get("precision_mode") == "fp32"
        and config.get("kd_weight") == config.get("hint_weight") == 0
        and mapping(config.get("model")).get("hint_dim") == 0
    )
    fit_argv = fit_plan.get("argv_after_approval_and_reservation")
    expected_fit_argv = [
        command[0],
        command[1],
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split_path),
        "--output",
        str(Path(str(fit_plan.get("output"))).resolve()),
        "--profile-receipt",
        str(Path(str(profile_plan.get("output"))).resolve() / "approved_profile_receipt.json"),
    ]
    checks["fresh_label_only_fit"] = (
        fit_plan.get("resume") is False
        and fit_plan.get("initializer") is None
        and fit_plan.get("teacher_or_cache") is None
        and fit_plan.get("transferred_auxiliary_weights") is False
        and fit_argv == expected_fit_argv
    )

    env = mapping(worker.get("child_environment"))
    registry_path = Path(str(env.get("WNN_PORTABLE_DATA_MANIFEST"))).resolve(strict=True)
    registry_sha = mapping(policy.get("portable_data_registry")).get("sha256")
    if sha(registry_path) != registry_sha:
        raise ValueError("worker portable registry differs from the frozen data registry")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry_path)
    split = load_campaign_split(split_path, data_root)
    checks["official_primary_membership"] = (
        split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        and split.sha256 == expected.get("primary_split_sha256") == primary_ref.get("sha256")
        and len(split.train_groups) == expected.get("primary_fit_train_sequences")
        and len(split.valid_groups) == expected.get("primary_fit_valid_sequences") == 0
        and len(split.evaluation_groups) == expected.get("primary_eval_valid_sequences")
        and split.evaluation_source == split.evaluation_role == "valid"
        and primary_ref.get("fit_train_sequences") == expected.get("primary_fit_train_sequences")
        and primary_ref.get("fit_valid_sequences") == 0
        and primary_ref.get("evaluation_sequences") == expected.get("primary_eval_valid_sequences")
        and profile_plan.get("primary_valid_score_used_for_admission") is False
    )
    identity = mapping(profile.get("identity"))
    data, runtime = mapping(identity.get("data")), mapping(identity.get("runtime"))
    fit_members = [["train", group] for group in split.train_groups]
    eval_members = [["valid", group] for group in split.evaluation_groups]
    normalization = mapping(data.get("normalization"))
    norm_sources = mapping(normalization.get("source_fingerprints"))
    train_fingerprint = read(split_path).get("sources", {}).get("datasets/train.parquet")
    normal_members = [("train", group) for group in split.train_groups[:32]]
    model = mapping(config.get("model"))
    input_mode = str(model.get("input_feature_mode", "raw112"))
    norm_identity = build_normalization_identity(
        norm_sources,
        split.sha256,
        normal_members,
        "prefix",
        sequence_input_width(input_mode),
        input_schema=sequence_input_schema(input_mode),
    )
    checks["fit_only_normalization"] = (
        data.get("fit_members") == fit_members
        and data.get("evaluation_members") == eval_members
        and data.get("training_mask") == config.get("training_mask")
        and data.get("primary_split_sha256") == data.get("split_sha256") == split.sha256
        and data.get("teacher") is None
        and data.get("initializer") is None
        and normalization.get("scope") == "prefix"
        and normalization.get("fit_prefix_sequences") == normalization.get("sequence_count") == 32
        and norm_sources == {"datasets/train.parquet": train_fingerprint}
        and normalization.get("identity_sha256") == normalization_identity_sha256(norm_identity)
        and isinstance(normalization.get("moments_sha256"), str)
        and len(normalization["moments_sha256"]) == 64
    )
    try:
        validate_precision_contract(config, runtime, runtime)
        strict_fp32 = True
    except (TypeError, ValueError):
        strict_fp32 = False
    checks["strict_fp32_world1"] = (
        strict_fp32
        and identity.get("world_size") == runtime.get("world_size") == 1
        and runtime.get("matmul_precision") == "highest"
        and runtime.get("matmul_tf32") is False
        and runtime.get("cudnn_tf32") is False
        and runtime.get("autocast") is False
        and runtime.get("amp_scaler") is False
    )

    profile_output = Path(str(profile_plan.get("output"))).resolve(strict=True)
    losses = mapping(profile.get("losses"))
    vals = [losses.get(key) for key in ("label", "distillation", "hint", "total")]
    finite = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals
    )
    loss_ok = (
        finite
        and math.isclose(float(losses["total"]), float(losses["label"]), rel_tol=1e-6, abs_tol=1e-7)
        and float(losses["distillation"]) >= 0
        and float(losses["hint"]) >= 0
    )
    checks["native_profile"] = (
        profile.get("receipt_schema") == "wnn.sequence-profile.v1"
        and profile.get("status") == "profile_complete"
        and profile.get("approved") is False
        and profile.get("updates") == limits.get("updates") == 50
        and profile.get("errors") == []
        and profile.get("error_count") == 0
        and profile.get("scientific_checkpoint_written") is False
        and identity.get("config_sha256") == canonical(config)
        and profile_plan.get("primary_valid_score_used_for_admission") is False
        and mapping(profile.get("evaluation_annotation")).get("no_eval_during_fit") is True
        and type(profile.get("elapsed_seconds")) in (int, float)
        and 0 < profile["elapsed_seconds"] <= limits.get("max_wall_seconds", 0)
        and type(profile.get("real_rows")) is int
        and profile["real_rows"] > 0
        and type(profile.get("forwarded_rows")) is int
        and profile["forwarded_rows"] >= profile["real_rows"]
        and type(profile.get("valid_rows")) is int
        and profile["valid_rows"] > 0
        and loss_ok
    )
    rank_path = profile_output / "rank0.jsonl"
    checks["finite_rank0_updates"] = rank0_ok(rank_path) and attempt.get("rank0_log_sha256") == sha(
        rank_path
    )

    checks["device_memory"] = check_memory(
        policy, limits, paths["device_inventory"], inventory, attempt, worker, profile, runtime, env
    )
    checks["owned_worker"] = check_owned_worker(
        worker,
        attempt,
        request,
        request_sha,
        command,
        mapping(limits.get("effective_thread_environment")),
        mapping(profile_plan.get("environment_template")),
        data_root,
    ) and env.get("WNN_PORTABLE_DATA_MANIFEST") == str(registry_path)
    checks["budget_deadline"] = (
        sha(paths["deadline"]) == bounds.get("deadline_receipt_sha256")
        and check_budget_deadline(
            policy,
            limits,
            attempt,
            budget,
            deadline,
            request_sha,
            paths["profile"],
            profile_sha,
            paths["worker"],
            worker_sha,
            paths["budget"],
            paths["deadline"],
            paths["device_inventory"],
            sha(paths["device_inventory"]),
            rank_path,
        )
        and attempt.get("fit_id") == request.get("fit_id")
    )
    passed = all(checks.values())
    return {
        "schema": "wnn.primary-refit-profile-engineering-review.v1",
        "status": "engineering_checks_passed" if passed else "engineering_checks_failed",
        "approved": False,
        "scientific_candidate_promotion": False,
        "primary_score_used": False,
        "reviewer_script_sha256": script_sha,
        "policy_sha256": sha(policy_path),
        "request_sha256": request_sha,
        "selection_receipt_sha256": sha(paths["selection"]),
        "native_profile_sha256": profile_sha,
        "worker_receipt_sha256": worker_sha,
        "attempt_receipt_sha256": sha(paths["attempt"]),
        "config_sha256": config_ref.get("sha256"),
        "source_sha256": expected.get("source_sha256"),
        "primary_split_sha256": expected.get("primary_split_sha256"),
        "checks": checks,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "request",
        "selection",
        "profile",
        "worker",
        "attempt",
        "budget",
        "deadline",
        "device-inventory",
        "receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name(POLICY_NAME))
    args = parser.parse_args()
    result = review(args)
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("review output must be fresh; preserve prior reviews")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.receipt.write_text(encoded, encoding="utf-8")
    print(encoded)
    if result["status"] != "engineering_checks_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
