"""Admit one full-data refit from a separately root-approved exact profile."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

from primary_refit_actor_common_r4 import (
    _admission,
    _campaign_inputs,
    _packet,
    _request,
    _runtime_paths,
    _source_and_config,
)
from primary_refit_actor_support_r4 import (
    INVENTORY_SHA256,
    POLICY_SHA256,
    PRIMARY_SHA256,
    REVIEW_SUPPORT_SHA256,
    REVIEWER_SHA256,
    SOURCE4_SHA256,
    load_device_inventory,
    profile_environment,
    read_json,
    reserve_stage4_fit,
    resolve_device,
    sha256,
    write_new,
)

APPROVAL_SCHEMA = "wnn.primary-refit-root-native-approval.v1"


def _primary_profile_data(data: Mapping[str, object]) -> bool:
    """Accept the native missing alias only with exact official primary membership."""
    fit, evaluation = data.get("fit_members"), data.get("evaluation_members")
    return (
        data.get("split_sha256") == PRIMARY_SHA256
        and data.get("primary_split_sha256") in (None, PRIMARY_SHA256)
        and data.get("evaluation_role") == data.get("evaluation_source") == "valid"
        and data.get("teacher") is None
        and data.get("initializer") is None
        and isinstance(fit, list)
        and len(fit) == 10325
        and isinstance(evaluation, list)
        and len(evaluation) == 1873
        and all(
            isinstance(row, list)
            and len(row) == 2
            and row[0] == "train"
            and type(row[1]) is int
            and row[1] >= 0
            for row in fit
        )
        and all(
            isinstance(row, list)
            and len(row) == 2
            and row[0] == "valid"
            and type(row[1]) is int
            and row[1] >= 0
            for row in evaluation
        )
        and len({row[1] for row in fit}) == 10325
        and len({row[1] for row in evaluation}) == 1873
    )


def _validate_root_approval(
    approval: Mapping[str, object],
    request: Mapping[str, object],
    request_sha: str,
    profile_path: Path,
    review_path: Path,
) -> None:
    config, source, split = request["config"], request["training_source"], request["primary_split"]
    required = {
        "fit_id": request["fit_id"],
        "request_sha256": request_sha,
        "native_profile_sha256": sha256(profile_path),
        "engineering_review_sha256": sha256(review_path),
        "config_sha256": config["sha256"],
        "source_sha256": source["sha256"],
        "primary_split_sha256": split["sha256"],
        "policy_sha256": POLICY_SHA256,
        "reviewer_script_sha256": REVIEWER_SHA256,
        "reviewer_support_sha256": REVIEW_SUPPORT_SHA256,
        "device_inventory_sha256": INVENTORY_SHA256,
    }
    if (
        approval.get("schema") != APPROVAL_SCHEMA
        or approval.get("status") != "approved"
        or approval.get("approved") is not True
        or approval.get("approver_id") != "root"
        or approval.get("native_fit_authorized") is not True
        or approval.get("scientific_candidate_promotion") is not False
        or approval.get("primary_valid_used_for_selection") is not False
        or any(approval.get(key) != value for key, value in required.items())
    ):
        raise ValueError(
            "explicit root native approval is absent or does not bind exact refit inputs"
        )


def run_fit(args: argparse.Namespace) -> dict[str, object]:
    operator_root = args.operator_root.resolve(strict=True)
    packet = _packet(operator_root)
    prepared_root, runtime_root, run_dir = (
        path.resolve(strict=True) for path in (args.prepared_root, args.runtime_root, args.run_dir)
    )
    selection_path, _, request, request_sha = _request(prepared_root, args.fit_id)
    _, deadline = _campaign_inputs(run_dir, runtime_root)
    admission = _admission(runtime_root, run_dir)
    _, inventory = load_device_inventory(runtime_root, admission, args.device_inventory)
    source_root, config_path, _ = _source_and_config(request)
    fit_plan = request["fit"]
    profile_plan = request["profile"]
    output = Path(str(fit_plan["output"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError("prepared scientific output already exists")
    native_path = Path(str(profile_plan["output"])) / "profile_receipt.json"
    review_path = native_path.parent / "engineering_review.json"
    attempt_path = native_path.parent / "profile_attempt_receipt.json"
    native, review, attempt = (
        read_json(path, label)
        for path, label in (
            (native_path, "native profile"),
            (review_path, "engineering review"),
            (attempt_path, "profile attempt"),
        )
    )
    identity = native.get("identity")
    profile_data = identity.get("data") if isinstance(identity, dict) else None
    if (
        review.get("schema") != "wnn.primary-refit-profile-engineering-review.v1"
        or review.get("status") != "engineering_checks_passed"
        or review.get("approved") is not False
        or review.get("scientific_candidate_promotion") is not False
        or review.get("primary_score_used") is not False
        or not isinstance(review.get("checks"), dict)
        or not review["checks"]
        or not all(value is True for value in review["checks"].values())
        or review.get("request_sha256") != request_sha
        or review.get("selection_receipt_sha256") != sha256(selection_path)
        or review.get("native_profile_sha256") != sha256(native_path)
        or review.get("config_sha256") != request["config"]["sha256"]
        or review.get("source_sha256") != SOURCE4_SHA256
        or review.get("primary_split_sha256") != PRIMARY_SHA256
        or review.get("policy_sha256") != POLICY_SHA256
        or review.get("reviewer_script_sha256") != REVIEWER_SHA256
        or attempt.get("profile_sha256") != sha256(native_path)
        or attempt.get("fit_id") != args.fit_id
        or review.get("attempt_receipt_sha256") != sha256(attempt_path)
        or review.get("worker_receipt_sha256") != attempt.get("worker_receipt_sha256")
        or attempt.get("device_inventory_sha256") != INVENTORY_SHA256
        or attempt.get("request_sha256") != request_sha
        or attempt.get("status") != "completed"
        or native.get("approved") is not False
        or native.get("status") != "profile_complete"
        or not isinstance(identity, dict)
        or identity.get("config_sha256") != request["config"]["normalized_identity_sha256"]
        or identity.get("source_sha256") != SOURCE4_SHA256
        or identity.get("source_files") != request["training_source"].get("files")
        or not isinstance(profile_data, dict)
        or not _primary_profile_data(profile_data)
    ):
        raise ValueError(
            "native profile and independent engineering review are not an accepted pair"
        )
    approval_path = args.native_approval.resolve(strict=True)
    approval = read_json(approval_path, "root native approval")
    _validate_root_approval(approval, request, request_sha, native_path, review_path)
    profile_devices = native["identity"]["runtime"]["devices"]
    if not isinstance(profile_devices, list) or len(profile_devices) != 1:
        raise ValueError("profile is not a one-device strict-FP32 identity")
    uuid = str(profile_devices[0].get("uuid"))
    keys = admission.get("profile_policy", {}).get("devices", {})
    matches = [
        str(key)
        for key, spec in keys.items()
        if isinstance(spec, Mapping)
        and str(spec.get("uuid", "")).casefold().removeprefix("gpu-")
        == uuid.casefold().removeprefix("gpu-")
    ]
    if len(matches) != 1:
        raise ValueError("profile GPU does not map uniquely to the current Stage 4 admission")
    device = resolve_device(admission, inventory, matches[0])
    if str(device["cuda_visible_devices"]).casefold().removeprefix(
        "gpu-"
    ) != uuid.casefold().removeprefix("gpu-"):
        raise ValueError("fit GPU identity must exactly match the approved fresh profile")
    approved_path = native_path.parent / "approved_profile_receipt.json"
    command = fit_plan.get("argv_after_approval_and_reservation")
    expected = [
        str(profile_plan["argv"][0]),
        str(source_root / "scripts/train_sequence.py"),
        "--config",
        str(config_path),
        "--data-root",
        str(Path(profile_plan["argv"][profile_plan["argv"].index("--data-root") + 1]).resolve()),
        "--split-manifest",
        str(
            Path(profile_plan["argv"][profile_plan["argv"].index("--split-manifest") + 1]).resolve()
        ),
        "--output",
        str(output.resolve()),
        "--profile-receipt",
        str(approved_path.resolve()),
    ]
    if (
        command != expected
        or request["primary_split"].get("path") != expected[expected.index("--split-manifest") + 1]
        or any(flag in command for flag in ("--resume", "--initializer", "--teacher-cache"))
    ):
        raise ValueError(
            "prepared full-fit command differs from a fresh label-only primary-train refit"
        )
    approved = native | {
        "approved": True,
        "external_review": {
            "path": str(review_path.resolve()),
            "sha256": sha256(review_path),
            "policy_sha256": POLICY_SHA256,
            "native_profile_sha256": sha256(native_path),
        },
        "root_native_approval": {"path": str(approval_path), "sha256": sha256(approval_path)},
    }
    attempt_id = f"stage4-primary-refit-{args.fit_id}"
    from wnn.multistage_search_budget_fit import finish_budget_attempt

    reservation, wall = reserve_stage4_fit(
        run_dir, attempt_id, int(request["selected_by"]["trial_number"]), matches[0]
    )
    paths = _runtime_paths(runtime_root, args.fit_id)
    env = profile_environment(request, device)
    child_environment = {
        key: env[key]
        for key in set(profile_plan["environment_template"]) | {"CUBLAS_EMULATE_SINGLE_PRECISION"}
        if key in env
    }
    fit_request_path = paths["requests"] / f"{attempt_id}.json"
    fit_request = {
        "command": command,
        "source_root": str(source_root),
        "stage": 4,
        "trial_number": int(request["selected_by"]["trial_number"]),
        "gpu_key": matches[0],
        "output_dir": str(output.resolve()),
        "timeout_seconds": wall,
        "reserve_start_epoch": deadline["reserve_start_epoch"],
        "log_path": str((paths["logs"] / f"{attempt_id}.fit.log").resolve()),
        "child_receipt_path": str((paths["receipts"] / f"{attempt_id}.child.json").resolve()),
        "budget_run_dir": str(run_dir),
        "fit_attempt_id": attempt_id,
        "child_environment": child_environment,
        "profile_sha256": sha256(native_path),
        "engineering_review_sha256": sha256(review_path),
        "root_native_approval_sha256": sha256(approval_path),
        "config_sha256": sha256(config_path),
        "selection_receipt_sha256": sha256(selection_path),
        "source_sha256": SOURCE4_SHA256,
        "primary_split_sha256": PRIMARY_SHA256,
    }
    try:
        write_new(approved_path, approved)
        write_new(fit_request_path, fit_request)
        guardian = packet / "primary_refit_fit_guardian_r4.py"
        if not guardian.is_file() or guardian.is_symlink():
            raise FileNotFoundError("bounded native fit guardian is missing")
        guardian_env = os.environ.copy()
        guardian_env.update(child_environment)
        guardian_env["PYTHONPATH"] = os.pathsep.join(
            (str(operator_root / "src"), child_environment["PYTHONPATH"])
        )
        from wnn.multistage_search_runtime import launch_detached_worker

        owner_path = paths["receipts"] / f"{attempt_id}.owner.json"
        owner = launch_detached_worker(
            [
                str(profile_plan["argv"][0]),
                str(guardian),
                "--request",
                str(fit_request_path.resolve()),
            ],
            cwd=source_root,
            env=guardian_env,
            receipt_path=owner_path,
            metadata={
                "fit_id": args.fit_id,
                "stage": 4,
                "trial_number": fit_request["trial_number"],
                "gpu_key": matches[0],
                "profile_sha256": fit_request["profile_sha256"],
                "approval_sha256": fit_request["root_native_approval_sha256"],
                "budget_attempt_id": attempt_id,
            },
        )
    except Exception:
        finish_budget_attempt(run_dir, attempt_id, "launch_failed", None)
        raise
    return {
        "status": "fit_launched",
        "fit_id": args.fit_id,
        "attempt_id": attempt_id,
        "profile_sha256": fit_request["profile_sha256"],
        "config_sha256": fit_request["config_sha256"],
        "primary_split_sha256": PRIMARY_SHA256,
        "wall_limit_seconds": wall,
        "reservation": reservation,
        "fit_request": str(fit_request_path),
        "owner": owner,
    }
