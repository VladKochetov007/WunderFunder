"""Advance the fixed 18-fit campaign through profile, review, and fit waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import final_submit_chain as director
import final_submit_chain_contract as contract

from wnn.final_submit_manifest import FinalFitMembership, validate_final_fit_membership
from wnn.multistage_search import atomic_write_json, file_lock
from wnn.multistage_search_owned_command import run_owned_command
from wnn.multistage_search_runtime import base_environment, hold_gpu_slot, process_identity
from wnn.training.environment import TrainingEnvironment
from wnn.training.final_submit_bridge_teachers import audit_final_fit_ancestor_closure
from wnn.training.source_identity import training_source_identity

WRAPPER_SHA256 = "8ccc3c5bfa1c69e814391cf2d0a22105fc22d9991d907ced8c97f3150db8af41"
ROOT_PROFILE_NODES = {
    "fmu128_s3",
    "frawW_s3",
    "fin256_s4",
    "fmu128_s4",
    "frawW_s4",
    "orgbase",
}
HISTORICAL_PROFILE_COUNT = 4
PROFILE_KEY_PREFIX = "legacy-"
EARLY_SUCCESSOR_IDS = ("ffull_s3", "ffull_s4", "frawW_s3ft", "frawW_s4ft")


def _profile_key(node_id: str) -> str:
    return f"{PROFILE_KEY_PREFIX}{node_id}-r3"


def _write_create_only(path: Path, value: dict[str, Any]) -> None:
    """Write immutable requests and receipts without overwriting prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_paths(args: argparse.Namespace) -> None:
    """Resolve campaign paths while preserving the requested interpreter symlink."""
    args.runtime_root = args.runtime_root.resolve()
    args.source_root = (args.source_root or args.runtime_root / "source").resolve()
    args.data_root = (args.data_root or args.runtime_root / "data").resolve()
    args.plan = (args.plan or args.runtime_root / "inputs/final-plan-r2.json").resolve()
    args.membership = (
        args.membership or args.runtime_root / "inputs/final-fit-membership-r1.json"
    ).resolve()
    args.source_audit = (
        args.source_audit or args.runtime_root / "inputs/SOURCE_VALIDATION_R1.json"
    ).resolve()
    args.source_capsule = (
        args.source_capsule or args.runtime_root / "inputs/final-submit-source-39d83f739.tar.gz"
    ).resolve()
    args.native_config = (
        args.native_config or args.runtime_root / "inputs/LSTM_CONFIG_R2.json"
    ).resolve()
    args.controller_root = (
        args.controller_root or args.runtime_root / "control/final-submit-chain"
    ).resolve()
    args.root_admission = (
        args.root_admission or args.runtime_root / "ROOT_EXECUTION_ADMISSION_R1.json"
    ).resolve()
    args.grader = (
        args.grader
        or args.runtime_root / "control-grader-r2/scripts/review_final_submit_profile.py"
    ).resolve()
    args.profile_panel = (
        args.profile_panel or args.runtime_root / "receipts/root-profile-panel-r3.approved.json"
    ).resolve()
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.director_script = (
        args.director_script or args.runtime_root / "control-code-r1/scripts/final_submit_chain.py"
    ).resolve()
    args.wrapper = (args.wrapper or args.runtime_root / "profile_wrapper_r2.py").resolve()
    args.state_path = (args.state_path or args.controller_root / "campaign-state.json").resolve()
    args.adopt_director_receipt = (
        args.adopt_director_receipt or args.runtime_root / "receipts/director-wave1-r1.launch.json"
    ).resolve()
    args.r2_root_admission = (
        args.r2_root_admission or args.runtime_root / "ROOT_EXECUTION_ADMISSION_R2.json"
    ).resolve()
    args.r2_director_script = (
        args.r2_director_script
        or args.runtime_root / "control-code-r2/scripts/final_submit_chain.py"
    ).resolve()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError(f"runtime interpreter is not executable: {args.python}")


def _verify_root(args: argparse.Namespace) -> dict[str, Any]:
    """Require the frozen root admission before any profile or fit stage."""
    return contract.verify_root_admission(
        admission_path=args.root_admission,
        grader_path=args.grader,
        director_path=args.director_script,
        helper_path=args.director_script.with_name("final_submit_chain_contract.py"),
        runtime_helper_path=args.director_script.with_name("final_submit_chain_runtime.py"),
        execution_helper_path=args.director_script.with_name("final_submit_chain_execution.py"),
        plan_sha256=contract.PLAN_SHA256,
        source_capsule_sha256=contract.SOURCE_CAPSULE_SHA256,
        now_epoch=time.time(),
    )


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck plan, extracted source bytes, data sources, wrapper, and config pins."""
    plan = contract.read_json(args.plan)
    nodes = contract.verify_plan(plan, args.plan)
    if contract.sha256_file(args.source_capsule) != contract.SOURCE_CAPSULE_SHA256:
        raise ValueError("source capsule bytes differ from the pinned archive")
    if contract.sha256_file(args.wrapper) != WRAPPER_SHA256:
        raise ValueError("profile wrapper bytes differ from the reviewed R2 wrapper")
    director._verify_source_tree(args.source_root, args.source_capsule)
    membership = contract.verify_sources(
        data_root=args.data_root,
        membership_path=args.membership,
        audit_path=args.source_audit,
        audit_sha256=contract.SOURCE_AUDIT_SHA256,
    )
    if (
        contract.sha256_file(args.native_config)
        != plan["optional_source4_lstm"]["config_file_sha256"]
    ):
        raise ValueError("native raw config bytes differ from the amended frozen plan")
    return plan, {"nodes": nodes, "membership": membership}


def _load_pinned_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate small pinned inputs without rehashing the large parquet files while adopting."""
    plan = contract.read_json(args.plan)
    nodes = contract.verify_plan(plan, args.plan)
    if contract.sha256_file(args.source_capsule) != contract.SOURCE_CAPSULE_SHA256:
        raise ValueError("source capsule bytes differ from the pinned archive")
    if contract.sha256_file(args.wrapper) != WRAPPER_SHA256:
        raise ValueError("profile wrapper bytes differ from the reviewed R2 wrapper")
    if (
        contract.sha256_file(args.native_config)
        != plan["optional_source4_lstm"]["config_file_sha256"]
    ):
        raise ValueError("native raw config bytes differ from the amended frozen plan")
    if contract.sha256_file(args.source_audit) != contract.SOURCE_AUDIT_SHA256:
        raise ValueError("receiver source-validation receipt bytes differ")
    raw = contract.read_json(args.membership, 2_000_000)
    sources = raw.get("sources", {})
    train_sha = sources.get("train", {}).get("sha256")
    valid_sha = sources.get("valid", {}).get("sha256")
    validate_final_fit_membership(raw, train_sha, valid_sha, contract.SOURCE_AUDIT_SHA256)
    membership = FinalFitMembership(
        path=args.membership,
        manifest_sha256=contract.sha256_file(args.membership),
        fit_members_sha256=str(raw["fit_members_sha256"]),
        train_groups=tuple(raw["fit_members"]["train"]),
        valid_groups=tuple(raw["fit_members"]["valid"]),
        train_sha256=str(train_sha),
        valid_sha256=str(valid_sha),
        source_validation_sha256=contract.SOURCE_AUDIT_SHA256,
    )
    return plan, {"nodes": nodes, "membership": membership}


def _validate_approval(
    node_id: str,
    approval_path: Path,
    expected_sha256: str,
    request_path: Path,
    expected_request_sha256: str | None,
    plan: dict[str, Any],
    membership: Any,
    mode: str = "legacy_tl3",
) -> dict[str, Any]:
    """Bind an independent approval to the exact node request and review evidence."""
    approval = contract.validate_profile_approval(
        node_id=node_id,
        approval_path=approval_path,
        expected_sha256=expected_sha256,
        plan_sha256=contract.PLAN_SHA256,
        membership=membership,
        mode=mode,
    )
    request = contract.read_json(request_path)
    request_sha256 = contract.sha256_file(request_path)
    review = contract.read_json(approval_path.parent / "profile_review.json")
    evidence = review.get("evidence_sha256", {})
    node_args = next(
        (node["args"] for node in plan.get("nodes", []) if node.get("id") == node_id), None
    )
    if (
        request.get("schema") != "wnn.final-submit-profile-request.v1"
        or request.get("node_id") != node_id
        or request.get("plan_sha256") != contract.PLAN_SHA256
        or request.get("mode") != mode
        or request.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or request.get("final_fit_manifest_sha256") != membership.manifest_sha256
        or request.get("source_audit_sha256") != contract.SOURCE_AUDIT_SHA256
        or request.get("fit_members_sha256") != membership.fit_members_sha256
        or request.get("fit_member_count") != contract.FIT_MEMBERS
        or request.get("evaluation_member_count") != 0
        or request.get("no_eval_during_fit") is not True
        or request.get("independent_score_claim") != "forbidden"
        or request_sha256 != expected_request_sha256
        or evidence.get("request") != request_sha256
        or (mode == "legacy_tl3" and approval.get("request_sha256") != request_sha256)
        or (mode == "legacy_tl3" and request.get("recipe_argv") != node_args)
        or (
            mode == "legacy_tl3"
            and request.get("recipe_sha256")
            != contract.canonical_sha256(request.get("recipe_argv"))
        )
    ):
        raise ValueError(f"profile request/review/approval identity differs for {node_id}")
    return {
        "path": approval_path,
        "sha256": expected_sha256,
        "gpu_uuid": str(request["gpu_uuid"]),
        "request_path": request_path,
        "request_sha256": request_sha256,
    }


def _panel_entries(path: Path) -> list[dict[str, Any]]:
    """Read the root-issued six-profile panel receipt."""
    panel = contract.read_json(path)
    entries = panel.get("profiles", panel.get("entries", panel.get("approved")))
    if isinstance(entries, dict):
        entries = [dict(value, node_id=key) for key, value in entries.items()]
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError("root profile panel must contain a profiles, entries or approved list")
    return entries


def _load_profiles(
    args: argparse.Namespace, plan: dict[str, Any], membership: Any
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load frozen R2 approvals, root panel approvals, and this campaign's reviewed profiles."""
    result: dict[str, dict[str, Any]] = {}
    review_root = args.runtime_root / "profile-reviews"
    defaults = {
        "source4_lstm192x1": (
            review_root / "native-lstm-r2-final/approved_profile_receipt.json",
            contract.NATIVE_PROFILE_SHA256,
            args.runtime_root / "inputs/native-lstm-r2.request.json",
        ),
        "fin256_s3": (
            review_root / "legacy-fin256-r2-final/approved_profile_receipt.json",
            contract.FIN256_PROFILE_SHA256,
            args.runtime_root / "inputs/legacy-fin256-r2.request.json",
        ),
    }
    for node_id, (approval_path, digest, request_path) in defaults.items():
        if approval_path.is_file():
            mode = "native_source4_lstm" if node_id == "source4_lstm192x1" else "legacy_tl3"
            result[node_id] = _validate_approval(
                node_id,
                approval_path,
                digest,
                request_path,
                contract.sha256_file(request_path),
                plan,
                membership,
                mode,
            )
    seen_panel: set[str] = set()
    if args.profile_panel.is_file():
        for item in _panel_entries(args.profile_panel):
            node_id = str(item.get("node_id", ""))
            if node_id not in ROOT_PROFILE_NODES or node_id in seen_panel:
                raise ValueError(f"root panel contains an unknown or duplicate node: {node_id}")
            seen_panel.add(node_id)
            approval_path = Path(str(item["path"]))
            request_path = Path(str(item["request_path"]))
            if not approval_path.is_absolute():
                approval_path = args.runtime_root / approval_path
            if not request_path.is_absolute():
                request_path = args.runtime_root / request_path
            entry = _validate_approval(
                node_id,
                approval_path,
                str(item["sha256"]),
                request_path,
                str(item["request_sha256"]),
                plan,
                membership,
            )
            if entry["gpu_uuid"] != item.get("gpu_uuid"):
                raise ValueError(f"root panel GPU identity differs for {node_id}")
            if node_id in result:
                raise ValueError(f"root panel repeats an already pinned profile: {node_id}")
            result[node_id] = entry
    # Deterministic own-review paths support safe restart after grading completed.
    for node_id in contract.TL3_IDS:
        if node_id in result:
            continue
        key = _profile_key(node_id)
        request_path = args.runtime_root / "inputs" / f"{key}.request.json"
        approval_path = review_root / f"{key}-review2/approved_profile_receipt.json"
        if approval_path.is_file():
            result[node_id] = _validate_approval(
                node_id,
                approval_path,
                contract.sha256_file(approval_path),
                request_path,
                contract.sha256_file(request_path),
                plan,
                membership,
            )
    return result, seen_panel


def _profile_request(
    args: argparse.Namespace,
    node: dict[str, Any],
    source_identity: dict[str, Any],
    gpu_uuid: str,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    """Build a create-only engineering profile request from one frozen recipe."""
    node_id = str(node["id"])
    key = _profile_key(node_id)
    request_path = args.runtime_root / "inputs" / f"{key}.request.json"
    log_path = args.runtime_root / "logs" / f"{key}.log"
    process_path = args.runtime_root / "receipts" / f"{key}.child.json"
    output_path = args.runtime_root / "profiles" / key
    review_path = args.runtime_root / "profile-reviews" / f"{key}-review2"
    recipe_argv = list(node["args"])
    if recipe_argv.count("--save-training-state") != 1:
        raise ValueError(f"frozen recipe has an unexpected training-state flag: {node_id}")
    recipe_argv.remove("--save-training-state")
    name_values = contract.option_values(recipe_argv, "--name")
    chunks = contract.option_values(recipe_argv, "--chunk")
    if len(name_values) != 1 or len(chunks) != 1:
        raise ValueError(f"profile recipe has ambiguous name or chunk values: {node_id}")
    name_index = recipe_argv.index("--name") + 1
    recipe_argv[name_index] += "_engineering_r2"
    chunk = int(chunks[0])
    if chunk <= 0 or 20_000 % chunk:
        raise ValueError(f"profile chunk does not divide sequence length: {node_id}")
    profile_batches = math.ceil(50 / (20_000 / chunk))
    recipe_argv.extend(("--profile-steps", str(profile_batches), "--profile-warmup", "1"))
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "590s",
        str(args.python),
        str(args.wrapper),
        str(log_path),
        *recipe_argv,
    ]
    request = {
        "schema": "wnn.final-submit-profile-request.v1",
        "mode": "legacy_tl3",
        "node_id": node_id,
        "plan_sha256": contract.PLAN_SHA256,
        "recipe_sha256": contract.canonical_sha256(node["args"]),
        "recipe_argv": node["args"],
        "argv": command,
        "argv_sha256": contract.canonical_sha256(command),
        "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
        "full_source_code_sha256": source_identity["sha256"],
        "final_fit_manifest_sha256": contract.sha256_file(args.membership),
        "source_audit_sha256": contract.SOURCE_AUDIT_SHA256,
        "fit_members_sha256": contract.read_json(args.membership)["fit_members_sha256"],
        "fit_member_count": contract.FIT_MEMBERS,
        "evaluation_member_count": 0,
        "no_eval_during_fit": True,
        "independent_score_claim": "forbidden",
        "gpu_uuid": gpu_uuid,
        "cpu_threads": 4,
        "profile_output": str(output_path),
        "log_path": str(log_path),
        "process_receipt": str(process_path),
        "profile_updates": None,
        "profile_batches": profile_batches,
        "warmup_batches": 1,
        "config_file_sha256": None,
        "config_identity_sha256": None,
        "request_created_epoch": time.time(),
        "expected_source": source_identity,
    }
    return request, request_path, log_path, process_path, review_path


def _run_profile(
    args: argparse.Namespace,
    node: dict[str, Any],
    gpu_uuid: str,
    state: dict[str, Any],
) -> None:
    """Run one owned profile, grade it independently, and persist its approval identity."""
    node_id = str(node["id"])
    source_identity = training_source_identity(
        TrainingEnvironment(source_root=args.source_root, root=args.data_root)
    )
    request, request_path, log_path, receipt_path, review_path = _profile_request(
        args, node, source_identity, gpu_uuid
    )
    if request_path.exists() or receipt_path.exists() or log_path.exists() or review_path.exists():
        raise FileExistsError(f"profile attempt already has artifacts; refusing retry: {node_id}")
    _verify_root(args)
    _load_inputs(args)
    deadline_left = int(contract.HARD_DEADLINE - contract.COLLECTION_RESERVE - time.time())
    if deadline_left < contract.MAX_PROFILE_SECONDS + 10:
        raise TimeoutError("not enough time remains for a complete reviewed 600-second profile")
    env = base_environment(gpu_uuid, args.source_root)
    env.update(
        {
            "WNN_PROJECT_ROOT": str(args.data_root),
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        }
    )
    owner = process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("campaign process identity is unavailable")
    with hold_gpu_slot(args.controller_root, gpu_uuid):
        _require_b300_headroom(gpu_uuid)
        deadline_left = int(contract.HARD_DEADLINE - contract.COLLECTION_RESERVE - time.time())
        if deadline_left < contract.MAX_PROFILE_SECONDS + 10:
            raise TimeoutError("not enough time remains after GPU admission for a full profile")
        _write_create_only(request_path, request)
        request_sha = contract.sha256_file(request_path)
        state.setdefault("profile_attempts", {})[node_id] = {
            "status": "launching",
            "request_path": str(request_path),
            "request_sha256": request_sha,
            "process_receipt": str(receipt_path),
            "log_path": str(log_path),
            "review_path": str(review_path),
            "gpu_uuid": gpu_uuid,
            "started_at_epoch": time.time(),
        }
        state["phase"] = "profiling"
        atomic_write_json(args.state_path, state)

        last_progress = 0.0

        def profile_progress() -> None:
            nonlocal last_progress
            now = time.monotonic()
            if receipt_path.is_file() and (last_progress == 0 or now - last_progress >= 5):
                process_receipt = contract.read_json(receipt_path)
                state["profile_attempts"][node_id]["current_process"] = process_receipt.get("child")
                state["profile_attempts"][node_id]["current_receipt"] = str(receipt_path)
                state["updated_at_epoch"] = time.time()
                atomic_write_json(args.state_path, state)
                last_progress = now

        code, timed_out, record = run_owned_command(
            request["argv"],
            cwd=args.source_root,
            env=env,
            log_path=log_path,
            receipt_path=receipt_path,
            owner={**owner, "controller_root": str(args.controller_root)},
            stage=0,
            trial_number=list(contract.TL3_IDS).index(node_id) + 1,
            mode="final_submit_engineering_r2",
            gpu_key=gpu_uuid,
            output_dir=Path(request["profile_output"]),
            timeout_seconds=contract.MAX_PROFILE_SECONDS,
            progress=profile_progress,
        )
    state["profile_attempts"][node_id].update(
        {"status": "completed" if code == 0 and not timed_out else "failed", "process": record}
    )
    atomic_write_json(args.state_path, state)
    if timed_out or code != 0:
        raise RuntimeError(f"profile failed for {node_id}; no automatic retry is allowed")
    grader_command = [
        str(args.python),
        str(args.grader),
        "--request",
        str(request_path),
        "--expected-request-sha256",
        request_sha,
        "--plan",
        str(args.plan),
        "--membership",
        str(args.membership),
        "--source-audit",
        str(args.source_audit),
        "--source-capsule",
        str(args.source_capsule),
        "--process",
        str(receipt_path),
        "--legacy-log",
        str(log_path),
        "--output-dir",
        str(review_path),
    ]
    review_env = base_environment("", args.source_root)
    review_env["WNN_PROJECT_ROOT"] = str(args.data_root)
    result = subprocess.run(
        grader_command,
        cwd=args.source_root,
        env=review_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    state["profile_attempts"][node_id]["grader"] = {
        "argv": grader_command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-8_000:],
        "stderr": result.stderr[-8_000:],
    }
    if result.returncode != 0:
        state["profile_attempts"][node_id]["status"] = "rejected"
        atomic_write_json(args.state_path, state)
        raise RuntimeError(f"independent profile grader rejected {node_id}")
    approval_path = review_path / "approved_profile_receipt.json"
    approval_sha = contract.sha256_file(approval_path)
    _validate_approval(
        node_id,
        approval_path,
        approval_sha,
        request_path,
        request_sha,
        contract.read_json(args.plan),
        contract.verify_sources(
            data_root=args.data_root,
            membership_path=args.membership,
            audit_path=args.source_audit,
            audit_sha256=contract.SOURCE_AUDIT_SHA256,
        ),
    )
    state["profile_attempts"][node_id].update(
        {
            "status": "approved",
            "approval_path": str(approval_path),
            "approval_sha256": approval_sha,
            "review_sha256": contract.sha256_file(review_path / "profile_review.json"),
        }
    )
    state["profiles_completed"] = sum(
        attempt.get("status") == "approved" for attempt in state["profile_attempts"].values()
    )
    atomic_write_json(args.state_path, state)


def _ready_to_profile(
    nodes: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    inner_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select only nodes whose actual fitting parents and required W18 cache are complete."""
    ready = []
    for node_id, node in nodes.items():
        if (
            node_id in profiles
            or inner_state.get("nodes", {}).get(node_id, {}).get("status") == "complete"
        ):
            continue
        if any(
            inner_state.get("nodes", {}).get(parent, {}).get("status") != "complete"
            for parent in node["parents"]
        ):
            continue
        if (
            node_id == "w18final_distilled_s5"
            and inner_state.get("cache", {}).get("status") != "complete"
        ):
            continue
        ready.append(node)
    return ready


def _director_command_matches(
    command: object,
    args: argparse.Namespace,
    profiles: dict[str, dict[str, Any]],
    created_epoch: object,
) -> bool:
    """Check that an adopted process is the exact frozen director wave."""
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
        return False
    script = str(args.director_script)
    if command.count(script) != 1:
        return False
    script_index = command.index(script)
    timeout_seconds = (
        int(command[3][:-1]) if len(command) > 3 and command[3][:-1].isdecimal() else 0
    )
    max_timeout = (
        int(contract.HARD_DEADLINE - float(created_epoch)) + 1
        if isinstance(created_epoch, (int, float)) and not isinstance(created_epoch, bool)
        else 0
    )
    if (
        script_index != 5
        or command[:2] != ["/usr/bin/timeout", "--signal=TERM"]
        or command[2] != "--kill-after=10s"
        or command[3][-1:] != "s"
        or not command[3][:-1].isdecimal()
        or timeout_seconds <= 0
        or timeout_seconds > max_timeout
        or max_timeout <= 0
        or command[4] != str(args.python)
    ):
        return False
    allowed_options = {
        "--action",
        "--runtime-root",
        "--source-root",
        "--data-root",
        "--plan",
        "--membership",
        "--source-audit",
        "--source-capsule",
        "--native-config",
        "--python",
        "--grader",
        "--root-admission",
        "--controller-root",
        "--max-lanes",
        "--profile-approval",
    }
    tail = command[script_index + 1 :]
    position = 0
    while position < len(tail):
        token = tail[position]
        option = token.partition("=")[0]
        if option not in allowed_options:
            return False
        if "=" not in token:
            position += 1
            if position >= len(tail) or tail[position].startswith("--"):
                return False
        position += 1
    required = (
        ("--action", "run"),
        ("--runtime-root", str(args.runtime_root)),
        ("--python", str(args.python)),
        ("--grader", str(args.grader)),
        ("--root-admission", str(args.root_admission)),
    )
    defaulted = (
        ("--source-root", str(args.source_root)),
        ("--data-root", str(args.data_root)),
        ("--plan", str(args.plan)),
        ("--membership", str(args.membership)),
        ("--source-audit", str(args.source_audit)),
        ("--source-capsule", str(args.source_capsule)),
        ("--native-config", str(args.native_config)),
        ("--controller-root", str(args.controller_root)),
        ("--max-lanes", str(contract.MAX_LANES)),
    )
    for option, expected in required:
        if contract.option_values(command, option) != [expected]:
            return False
    for option, expected in defaulted:
        values = contract.option_values(command, option)
        if values and values != [expected]:
            return False
    expected_args = {
        node_id: f"{item['sha256']}={item['gpu_uuid']}={item['path']}"
        for node_id, item in profiles.items()
    }
    supplied: dict[str, str] = {}
    for value in contract.option_values(command, "--profile-approval"):
        fields = value.split("=", 3)
        if len(fields) != 4 or fields[0] in supplied:
            return False
        supplied[fields[0]] = "=".join(fields[1:])
    defaults = {"source4_lstm192x1", "fin256_s3"}
    if any(expected_args.get(node_id) != value for node_id, value in supplied.items()):
        return False
    if (set(supplied) | defaults) != set(profiles):
        return False
    controller_values = contract.option_values(command, "--controller-root")
    return not controller_values or controller_values == [str(args.controller_root)]


def _owned_group_alive(pgid: int) -> bool:
    """Read procfs to see whether any process remains in one recorded owned group."""
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        close = raw.rfind(")")
        fields = raw[close + 2 :].split() if close >= 0 else []
        if len(fields) > 2 and fields[2].isdecimal() and int(fields[2]) == pgid:
            return True
    return False


def _require_complete(inner: dict[str, Any]) -> None:
    """Accept completion only with all 18 fixed fits, W18 cache, and no active workers."""
    nodes = inner.get("nodes", {})
    complete_ids = {name for name, item in nodes.items() if item.get("status") == "complete"}
    expected = set(contract.TL3_IDS) | {"source4_lstm192x1"}
    if (
        inner.get("phase") != "complete"
        or inner.get("scientific_fits_started") != contract.TOTAL_FITS
        or inner.get("scientific_fits_completed") != contract.TOTAL_FITS
        or complete_ids != expected
        or inner.get("cache", {}).get("status") != "complete"
        or inner.get("active_workers")
        or any(
            item.get("status") in {"failed", "reconciliation_required"} for item in nodes.values()
        )
    ):
        raise ValueError("controller state does not prove all 18 fits and cache completed")


def _require_identity(
    inner: dict[str, Any], plan: dict[str, Any], membership: Any, args: argparse.Namespace
) -> None:
    """Require the R1 identity, including its original root-admission hash."""
    if inner.get("identity") != _expected_controller_identity(plan, membership, args):
        raise ValueError("controller state identity differs from the frozen R1 director identity")


def _expected_controller_identity(
    plan: dict[str, Any], membership: Any, args: argparse.Namespace
) -> dict[str, object]:
    """Preserve the R1 state identity used by both R1 and R2 continuations."""
    return director._expected_identity(plan, membership, args) | {
        "admission_sha256": contract.sha256_file(args.root_admission)
    }


def _write_immutable_bytes(path: Path, value: bytes) -> None:
    """Create a byte-exact evidence copy, allowing only an identical prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != value:
            raise ValueError(f"immutable evidence path already differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _read_bound_json(path: Path, expected_sha256: str, schema: str) -> dict[str, Any]:
    """Read one non-symlink JSON proof only when its pinned bytes and schema match."""
    if path.is_symlink() or not path.is_file() or contract.sha256_file(path) != expected_sha256:
        raise ValueError(f"repair evidence bytes differ: {path}")
    value = contract.read_json(path, 16_000_000)
    if value.get("schema") != schema:
        raise ValueError(f"repair evidence schema differs: {path}")
    return value


def _owned_receipt(
    path: Path,
    expected_sha256: str,
    *,
    mode: str,
    stage: int,
    trial: int,
    gpu_uuid: str,
    output_dir: Path,
    command: list[str],
) -> dict[str, Any]:
    """Validate a finished owned child receipt against its named repair stage."""
    receipt = _read_bound_json(path, expected_sha256, "wnn.multistage-owned-process.v1")
    if (
        receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("mode") != mode
        or receipt.get("stage") != stage
        or receipt.get("trial_number") != trial
        or receipt.get("gpu_key") != gpu_uuid
        or receipt.get("output_dir") != str(output_dir.resolve())
        or receipt.get("command") != command
        or receipt.get("command_sha256") != contract.canonical_sha256(command)
    ):
        raise ValueError(f"owned repair stage receipt is incomplete or mismatched: {path}")
    child = receipt.get("child")
    if (
        not isinstance(child, dict)
        or type(child.get("pid")) is not int
        or type(child.get("pgid")) is not int
        or not isinstance(child.get("birth"), str)
        or len(command) < 5
        or not command[3].endswith("s")
        or receipt.get("timeout_seconds") != int(command[3][:-1]) + 10
    ):
        raise ValueError(f"owned repair stage omits its child identity: {path}")
    if process_identity(int(child["pid"])) == child or _owned_group_alive(int(child["pgid"])):
        raise RuntimeError(f"owned repair child group is still active: {path}")
    return receipt


def _timeout_command(command: object, expected: list[str], max_seconds: int) -> list[str]:
    """Check a generated GNU-timeout command with only its bounded duration variable."""
    if not isinstance(command, list) or len(command) != len(expected):
        raise ValueError("repair command length differs from its fixed entrypoint")
    duration = command[3] if len(command) > 3 else None
    if (
        not isinstance(duration, str)
        or re.fullmatch(r"[1-9][0-9]*s", duration) is None
        or int(duration[:-1]) > max_seconds
    ):
        raise ValueError("repair command timeout is outside its frozen stage cap")
    actual = list(command)
    actual[3] = expected[3]
    if actual != expected:
        raise ValueError("repair command differs from its fixed entrypoint")
    return list(command)


def _validate_native_repair(
    args: argparse.Namespace,
    node_record: dict[str, Any],
    membership: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one externally qualified native fit that replaces admission failure."""
    node_id, run = "source4_lstm192x1", "final-lstm192x1"
    proof_path = args.runtime_root / "receipts/native-admission-repair-r2.json"
    intent_path = args.runtime_root / "receipts/native-admission-repair-r2.intent.json"
    if node_record.get("status") != "failed" or "not an idle B300" not in str(
        node_record.get("error", "")
    ):
        raise ValueError("native repair does not match the recorded pre-child admission failure")
    if (args.controller_root / "receipts/fits/source4_lstm192x1.json").exists() or (
        args.controller_root / "logs/fits/source4_lstm192x1.log"
    ).exists():
        raise ValueError("the failed native admission unexpectedly has an original fit child")
    if proof_path.is_symlink() or not proof_path.is_file():
        raise FileNotFoundError(f"native repair proof is missing or linked: {proof_path}")
    proof = contract.read_json(proof_path, 16_000_000)
    if proof.get("schema") != "wnn.native-admission-repair.v1" or proof.get("status") != "complete":
        raise ValueError("native repair proof is not complete")
    intent = _read_bound_json(
        intent_path, str(proof.get("intent_sha256")), "wnn.native-admission-repair-intent.v1"
    )
    if (
        proof.get("intent_path") != str(intent_path)
        or proof.get("node_id") != node_id
        or proof.get("run") != run
        or intent.get("node_id") != node_id
        or intent.get("run") != run
        or intent.get("prior_node_record") != node_record
        or intent.get("prior_original_child_receipt_absent") is not True
        or intent.get("prior_original_log_absent") is not True
        or intent.get("prior_original_output_absent") is not True
        or intent.get("unique_scientific_fit_roles_cap") != contract.TOTAL_FITS
        or intent.get("foreign_signals") is not False
        or intent.get("source_commit") != contract.SOURCE_COMMIT
        or intent.get("plan_sha256") != contract.PLAN_SHA256
        or intent.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof.get("scientific_retraining") is not False
        or proof.get("independent_score_claim") != "forbidden"
        or proof.get("plan_sha256") != contract.PLAN_SHA256
        or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof.get("membership_sha256") != membership.manifest_sha256
        or proof.get("fit_members_sha256") != membership.fit_members_sha256
        or proof.get("fixed_completed_epochs") != 8
    ):
        raise ValueError("native repair ancestry, scope, or no-score claim differs")
    output = args.data_root / "artifacts/runs" / run
    command = intent.get("command")
    expected_command = list(node_record.get("argv", []))
    if not expected_command:
        raise ValueError("native failed node omits its original fixed argv")
    expected_command[3] = "1s"
    command = _timeout_command(command, expected_command, contract.MAX_FIT_SECONDS - 10)
    profile_path = (
        args.runtime_root / "profile-reviews/native-lstm-r2-final/approved_profile_receipt.json"
    )
    coexist_path = args.runtime_root / "receipts/native-coexist-qualified-r3.json"
    if (
        intent.get("profile_receipt") != str(profile_path)
        or intent.get("profile_receipt_sha256") != contract.NATIVE_PROFILE_SHA256
        or proof.get("full_fit_profile_sha256") != contract.NATIVE_PROFILE_SHA256
        or contract.sha256_file(profile_path) != contract.NATIVE_PROFILE_SHA256
        or intent.get("coexist_qualification_path") != str(coexist_path)
        or proof.get("coexist_qualification_sha256") != intent.get("coexist_qualification_sha256")
        or proof.get("gpu_uuid") != intent.get("gpu_uuid")
        or intent.get("gpu_uuid") != node_record.get("gpu_uuid")
        or intent.get("output") != str(output)
    ):
        raise ValueError("native repair profile, device, or output identity differs")
    coexist = _read_bound_json(
        coexist_path,
        str(intent.get("coexist_qualification_sha256")),
        "wnn.native-coexist-profile-qualified.v1",
    )
    approval_path = Path(str(coexist.get("profile_approval", "")))
    if (
        coexist.get("scientific_recipe_unchanged") is not True
        or coexist.get("gpu_uuid") != intent.get("gpu_uuid")
        or approval_path.is_symlink()
        or not approval_path.is_file()
        or contract.sha256_file(approval_path) != coexist.get("profile_approval_sha256")
        or approval_path
        != args.runtime_root
        / "profile-reviews/native-lstm-coexist-r3/approved_profile_receipt.json"
    ):
        raise ValueError("native coexist qualification is missing or mismatched")
    request_path = args.runtime_root / "inputs/native-lstm-coexist-r3.request.json"
    review_path = approval_path.parent / "profile_review.json"
    if (
        request_path.is_symlink()
        or review_path.is_symlink()
        or not request_path.is_file()
        or not review_path.is_file()
        or contract.sha256_file(request_path) != coexist.get("request_sha256")
    ):
        raise ValueError("native coexist profile request or review bytes differ")
    approval = contract.read_json(approval_path)
    review = contract.read_json(review_path)
    checks = review.get("checks")
    if (
        approval.get("approved") is not True
        or approval.get("request_sha256") != coexist.get("request_sha256")
        or approval.get("approval_review_sha256") != contract.sha256_file(review_path)
        or approval.get("receipt_schema") != "wnn.sequence-profile.v1"
        or approval.get("status") != "profile_complete"
        or approval.get("scientific_checkpoint_written") is not False
        or review.get("status") != "approved"
        or review.get("approved_for_full_fit") is not True
        or review.get("node_id") != node_id
        or review.get("plan_sha256") != contract.PLAN_SHA256
        or review.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or review.get("evidence_sha256", {}).get("request") != coexist.get("request_sha256")
        or review.get("evidence_sha256", {}).get("membership") != membership.manifest_sha256
        or review.get("evidence_sha256", {}).get("source_audit") != contract.SOURCE_AUDIT_SHA256
        or review.get("profile_mode") != "native_source4_lstm"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or review.get("failed_checks") != []
    ):
        raise ValueError("native coexist profile review is not independently approved")
    fit_receipt_path = args.controller_root / "receipts/fits/source4_lstm192x1-external-r2.json"
    fit_receipt = _owned_receipt(
        fit_receipt_path,
        str(proof.get("fit_process_receipt_sha256")),
        mode="final_submit_native_qualified_repair_r2",
        stage=1,
        trial=100,
        gpu_uuid=str(intent["gpu_uuid"]),
        output_dir=output,
        command=command,
    )
    if (
        proof.get("fit_process_receipt") != str(fit_receipt_path)
        or intent.get("process_receipt") != str(fit_receipt_path)
        or fit_receipt.get("timeout_seconds", 0) > contract.MAX_FIT_SECONDS
        or fit_receipt.get("cwd") != str(args.source_root.resolve())
    ):
        raise ValueError("native repair process receipt path or wall cap differs")
    terminal_path = output / "terminal_status.json"
    if (
        output.is_symlink()
        or not output.is_dir()
        or terminal_path.is_symlink()
        or not terminal_path.is_file()
        or proof.get("terminal_receipt") != str(terminal_path)
        or proof.get("terminal_receipt_sha256") != contract.sha256_file(terminal_path)
    ):
        raise ValueError("native terminal receipt bytes differ from repair proof")
    checkpoint = director.runtime.validate_fit_checkpoint(
        node={"node_id": node_id, "run": run}, output_dir=output, membership=membership
    )
    if (
        checkpoint.is_symlink()
        or proof.get("checkpoint_path") != str(checkpoint)
        or proof.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
    ):
        raise ValueError("native fixed-endpoint checkpoint differs from repair proof")
    return proof, fit_receipt


def _validate_orgbase_repair(
    args: argparse.Namespace,
    node_record: dict[str, Any],
    plan: dict[str, Any],
    membership: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the completed orgbase fit and its two source-ordered teacher exports."""
    node_id = "orgbase"
    plan_node = next(node for node in plan["nodes"] if node["id"] == node_id)
    run = str(plan_node["run"])
    proof_path = args.runtime_root / "receipts/orgbase-export-admission-repair-r2.json"
    intent_path = args.runtime_root / "receipts/orgbase-export-admission-repair-r2.intent.json"
    if node_record.get("status") != "failed" or "not an idle B300" not in str(
        node_record.get("error", "")
    ):
        raise ValueError("orgbase repair does not match the recorded export admission failure")
    if any(
        (args.controller_root / "receipts/exports" / f"orgbase-{split}.json").exists()
        for split in ("train", "valid")
    ):
        raise ValueError("orgbase has an original export child receipt; repair adoption is unsafe")
    if proof_path.is_symlink() or not proof_path.is_file():
        raise FileNotFoundError(f"orgbase repair proof is missing or linked: {proof_path}")
    proof = contract.read_json(proof_path, 16_000_000)
    if (
        proof.get("schema") != "wnn.orgbase-export-admission-repair.v1"
        or proof.get("status") != "complete"
    ):
        raise ValueError("orgbase export repair proof is not complete")
    intent = _read_bound_json(
        intent_path,
        str(proof.get("intent_sha256")),
        "wnn.orgbase-export-admission-repair-intent.v1",
    )
    fit_receipt_path = args.controller_root / "receipts/fits/orgbase.json"
    output = args.data_root / "artifacts/runs" / run
    if output.is_symlink() or not output.is_dir():
        raise ValueError("orgbase fixed-endpoint output directory is missing or linked")
    checkpoint = director.runtime.validate_fit_checkpoint(
        node={**plan_node, "node_id": node_id}, output_dir=output, membership=membership
    )
    if (
        proof.get("intent_path") != str(intent_path)
        or proof.get("node_id") != node_id
        or proof.get("run") != run
        or intent.get("node_id") != node_id
        or intent.get("run") != run
        or intent.get("prior_node_record") != node_record
        or intent.get("original_export_child_absent") is not True
        or intent.get("actual_fit_completed") is not True
        or intent.get("scientific_retraining") is not False
        or intent.get("foreign_signals") is not False
        or proof.get("scientific_retraining") is not False
        or proof.get("independent_score_claim") != "forbidden"
        or intent.get("fixed_endpoint") != 3
        or proof.get("fixed_endpoint") != 3
        or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or intent.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof.get("plan_sha256") != contract.PLAN_SHA256
        or intent.get("plan_sha256") != contract.PLAN_SHA256
        or proof.get("membership_sha256") != membership.manifest_sha256
        or intent.get("membership_sha256") != membership.manifest_sha256
        or proof.get("fit_members_sha256") != membership.fit_members_sha256
        or proof.get("fit_process_receipt") != str(fit_receipt_path)
        or intent.get("fit_process_receipt") != str(fit_receipt_path)
        or proof.get("fit_process_receipt_sha256") != intent.get("fit_process_receipt_sha256")
        or intent.get("checkpoint_path") != str(checkpoint)
        or intent.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
        or proof.get("checkpoint_path") != str(checkpoint)
        or proof.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
    ):
        raise ValueError("orgbase fit/export repair ancestry or no-score claim differs")
    fit_receipt = _owned_receipt(
        fit_receipt_path,
        str(proof.get("fit_process_receipt_sha256")),
        mode="final_submit_finite_chain",
        stage=1,
        trial=8,
        gpu_uuid=str(node_record.get("gpu_uuid")),
        output_dir=output,
        command=list(node_record.get("argv", [])),
    )
    if fit_receipt.get("cwd") != str(args.source_root.resolve()):
        raise ValueError("orgbase original fit receipt used a different source directory")
    exports: dict[str, Any] = {}
    raw_exports = proof.get("exports")
    if not isinstance(raw_exports, dict) or set(raw_exports) != {"train", "valid"}:
        raise ValueError("orgbase repair must contain exactly the train and valid exports")
    uuid = intent.get("gpu_uuid")
    for split in ("train", "valid"):
        item = raw_exports[split]
        expected_path = args.data_root / "cache" / f"pred_{run}_{split}.npy"
        receipt_path = (
            args.controller_root / "receipts/exports" / f"orgbase-{split}-external-r2.json"
        )
        command_template = [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=10s",
            "1s",
            str(args.python),
            str(args.source_root / "scripts/predict_run.py"),
            run,
            split,
            "--epoch",
            "3",
            "--final-fit-manifest",
            str(args.membership),
            "--final-fit-source-validation-receipt",
            str(args.source_audit),
            "--final-fit-source-validation-sha256",
            contract.SOURCE_AUDIT_SHA256,
            "--device",
            "cuda",
        ]
        receipt_path_actual = Path(str(item.get("receipt", "")))
        if (
            item.get("status") != "complete"
            or item.get("path") != str(expected_path)
            or item.get("prediction_role") != "fitting-only teacher features; no independent score"
            or item.get("independent_score_claim") != "forbidden"
            or str(receipt_path_actual) != str(receipt_path)
        ):
            raise ValueError(f"orgbase {split} export proof metadata differs")
        receipt_file = _read_bound_json(
            receipt_path, str(item.get("receipt_sha256")), "wnn.multistage-owned-process.v1"
        )
        command = _timeout_command(
            receipt_file.get("command"), command_template, contract.MAX_FIT_SECONDS - 10
        )
        export_receipt = _owned_receipt(
            receipt_path,
            str(item.get("receipt_sha256")),
            mode="final_submit_orgbase_qualified_export_repair_r2",
            stage=2,
            trial=101,
            gpu_uuid=str(uuid),
            output_dir=args.data_root / "cache",
            command=command,
        )
        if (
            export_receipt.get("timeout_seconds", 0) > contract.MAX_FIT_SECONDS
            or export_receipt.get("cwd") != str(args.source_root.resolve())
            or not expected_path.is_file()
            or expected_path.is_symlink()
            or expected_path.with_suffix(".json").is_symlink()
            or not expected_path.with_suffix(".json").is_file()
            or item.get("sha256") != contract.sha256_file(expected_path)
        ):
            raise ValueError(f"orgbase {split} export artifact bytes differ")
        validated = director.runtime.validate_export(args.data_root, run, split, membership)
        if validated != expected_path:
            raise ValueError(f"orgbase {split} source order or export path differs")
        exports[split] = {
            "status": "complete",
            "path": str(validated),
            "sha256": contract.sha256_file(validated),
            "receipt": str(receipt_path),
            "process": export_receipt,
            "prediction_role": "fitting-only teacher features; no independent score",
            "independent_score_claim": "forbidden",
        }
    return proof, fit_receipt, {"exports": exports, "checkpoint": str(checkpoint)}


def _validate_fin256_export_repair(
    args: argparse.Namespace,
    node_id: str,
    node_record: dict[str, Any],
    plan: dict[str, Any],
    membership: Any,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Adopt one completed fin256 fit and its independently receipt-bound exports."""
    if node_id not in {"fin256_s3", "fin256_s4"}:
        raise ValueError(f"unexpected export-only repair node: {node_id}")
    planned = next(item for item in plan["nodes"] if item["id"] == node_id)
    run = str(planned["run"])
    proof_path = args.runtime_root / "receipts" / f"{node_id}-export-admission-repair-r4.json"
    intent_path = (
        args.runtime_root / "receipts" / f"{node_id}-export-admission-repair-r4.intent.json"
    )
    proof = contract.read_json(proof_path, 16_000_000)
    intent = _read_bound_json(
        intent_path,
        str(proof.get("intent_sha256")),
        "wnn.final-submit-export-only-repair-intent.v1",
    )
    output = args.data_root / "artifacts/runs" / run
    expected_trial = contract.TL3_IDS.index(node_id) + 2
    reused = ["train", "valid"] if node_id == "fin256_s3" else ["train"]
    missing = [] if node_id == "fin256_s3" else ["valid"]
    fit_receipt_path = args.controller_root / "receipts/fits" / f"{node_id}.json"
    checkpoint = director.runtime.validate_fit_checkpoint(
        node={**planned, "node_id": node_id}, output_dir=output, membership=membership
    )
    profile_path = Path(str(node_record.get("profile_receipt", "")))
    error = str(node_record.get("error", ""))
    expected_failure = (
        "launched fit argv differs" in error
        if node_id == "fin256_s3"
        else "not an idle B300" in error
    )
    if (
        node_record.get("status") != "failed"
        or not expected_failure
        or output.is_symlink()
        or profile_path.is_symlink()
        or not profile_path.is_file()
        or node_record.get("profile_sha256") != contract.sha256_file(profile_path)
    ):
        raise ValueError(f"{node_id} does not match a completed-fit export-only failure")
    request_path = (
        args.runtime_root / "inputs/legacy-fin256-r2.request.json"
        if node_id == "fin256_s3"
        else args.runtime_root / "inputs" / f"{_profile_key(node_id)}.request.json"
    )
    profile = _validate_approval(
        node_id,
        profile_path,
        str(node_record["profile_sha256"]),
        request_path,
        contract.sha256_file(request_path),
        plan,
        membership,
    )
    if profile.get("gpu_uuid") != node_record.get("gpu_uuid"):
        raise ValueError(f"{node_id} fit GPU differs from its reviewed profile")
    fit_argv = director.execution.build_fit_argv(
        node={
            **planned,
            "node_id": node_id,
            "membership_path": str(args.membership),
            "audit_path": str(args.source_audit),
        },
        python=args.python,
        source_root=args.source_root,
        data_root=args.data_root,
        plan=plan,
        profile_path=profile_path,
    )
    fit_file = contract.read_json(fit_receipt_path, 2_000_000)
    fit_command = _timeout_command(
        fit_file.get("command"),
        ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", "1s", *fit_argv],
        contract.MAX_FIT_SECONDS - 10,
    )
    fit_receipt = _owned_receipt(
        fit_receipt_path,
        contract.sha256_file(fit_receipt_path),
        mode="final_submit_finite_chain",
        stage=1,
        trial=expected_trial,
        gpu_uuid=str(node_record.get("gpu_uuid")),
        output_dir=output,
        command=fit_command,
    )
    expected_reused_receipts = {
        split: str(args.controller_root / "receipts/exports" / f"{node_id}-{split}.json")
        for split in reused
    }
    expected_missing_receipts = {
        split: str(
            args.controller_root
            / "receipts/exports"
            / f"{node_id}-{split}-external-r4.json"
        )
        for split in missing
    }
    raw_exports = proof.get("exports")
    if (
        proof.get("schema") != "wnn.final-submit-export-only-repair.v1"
        or proof.get("status") != "complete"
        or proof.get("node_id") != node_id
        or proof.get("run") != run
        or proof.get("intent_path") != str(intent_path)
        or proof.get("plan_sha256") != contract.PLAN_SHA256
        or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof.get("membership_sha256") != membership.manifest_sha256
        or proof.get("fit_members_sha256") != membership.fit_members_sha256
        or proof.get("source_audit_sha256", proof.get("source_validation_sha256"))
        != contract.SOURCE_AUDIT_SHA256
        or proof.get("fixed_endpoint") != planned["endpoint"]
        or proof.get("fit_process_receipt") != str(fit_receipt_path)
        or proof.get("fit_process_receipt_sha256") != contract.sha256_file(fit_receipt_path)
        or proof.get("checkpoint_path") != str(checkpoint)
        or proof.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
        or proof.get("scientific_retraining") is not False
        or proof.get("independent_score_claim") != "forbidden"
        or intent.get("node_id") != node_id
        or intent.get("run") != run
        or intent.get("prior_node_record") != node_record
        or intent.get("schema") != "wnn.final-submit-export-only-repair-intent.v1"
        or intent.get("plan_sha256") != contract.PLAN_SHA256
        or intent.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or intent.get("membership_sha256") != membership.manifest_sha256
        or intent.get("fit_members_sha256") != membership.fit_members_sha256
        or intent.get("source_validation_sha256") != contract.SOURCE_AUDIT_SHA256
        or intent.get("fixed_endpoint") != planned["endpoint"]
        or intent.get("independent_score_claim") != "forbidden"
        or intent.get("completed_exports_reused") != reused
        or intent.get("missing_exports") != missing
        or intent.get("scientific_retraining") is not False
        or intent.get("foreign_signals") is not False
        or intent.get("fit_process_receipt") != str(fit_receipt_path)
        or intent.get("fit_process_receipt_sha256") != contract.sha256_file(fit_receipt_path)
        or intent.get("checkpoint_path") != str(checkpoint)
        or intent.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
        or not isinstance(raw_exports, dict)
        or set(raw_exports) != {"train", "valid"}
    ):
        raise ValueError(f"{node_id} export-only repair proof or intent differs")
    exports: dict[str, Any] = {}
    for split in ("train", "valid"):
        item = raw_exports[split]
        external = split in missing
        receipt_path = Path(
            expected_missing_receipts[split]
            if external
            else expected_reused_receipts[split]
        )
        expected_path = args.data_root / "cache" / f"pred_{run}_{split}.npy"
        exported = director.runtime.validate_export(args.data_root, run, split, membership)
        if (
            item.get("status") != "complete"
            or item.get("path") != str(expected_path)
            or item.get("sha256") != contract.sha256_file(expected_path)
            or item.get("receipt") != str(receipt_path)
            or item.get("receipt_sha256") != contract.sha256_file(receipt_path)
            or item.get("prediction_role")
            != "fitting-only teacher features; no independent score"
            or item.get("independent_score_claim") != "forbidden"
            or exported != expected_path
        ):
            raise ValueError(f"{node_id}/{split} export-only proof differs")
        receipt_file = contract.read_json(receipt_path, 2_000_000)
        export_gpu = (
            str(intent.get("gpu_uuid", "")) if external else str(node_record.get("gpu_uuid", ""))
        )
        mode = "final_submit_export_only_repair_r4" if external else "final_submit_finite_chain"
        trial = 103 if external else expected_trial
        command = [
            str(args.python),
            str(args.source_root / "scripts/predict_run.py"),
            run,
            split,
            "--epoch",
            str(planned["endpoint"]),
            "--final-fit-manifest",
            str(args.membership),
            "--final-fit-source-validation-receipt",
            str(args.source_audit),
            "--final-fit-source-validation-sha256",
            contract.SOURCE_AUDIT_SHA256,
            "--device",
            "cuda",
        ]
        if external:
            # Intent and process bind the stable UUID; this only records its current index.
            gpu_rows = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            ).stdout.splitlines()
            gpu_by_index = {
                fields[0].strip(): fields[1].strip()
                for line in gpu_rows
                if len(fields := line.split(",", 1)) == 2
            }
            if (
                gpu_by_index.get("2") != export_gpu
            ):
                raise ValueError(f"{node_id}/{split} repair GPU differs from the qualified GPU2")
        if item.get("process") != receipt_file:
            raise ValueError(f"{node_id}/{split} proof process differs from its receipt")
        wrapped_command = _timeout_command(
            receipt_file.get("command"),
            ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", "1s", *command],
            contract.MAX_FIT_SECONDS - 10,
        )
        process = _owned_receipt(
            receipt_path,
            contract.sha256_file(receipt_path),
            mode=mode,
            stage=2,
            trial=trial,
            gpu_uuid=export_gpu,
            output_dir=args.data_root / "cache",
            command=wrapped_command,
        )
        if process.get("cwd") != str(args.source_root.resolve()):
            raise ValueError(f"{node_id}/{split} export used another source directory")
        exports[split] = {
            "status": "complete",
            "path": str(exported),
            "sha256": contract.sha256_file(exported),
            "receipt": str(receipt_path),
            "process": process,
            "prediction_role": "fitting-only teacher features; no independent score",
            "independent_score_claim": "forbidden",
        }
    return proof, fit_receipt, checkpoint, {"exports": exports}


def _adopt_qualified_operational_repairs(
    args: argparse.Namespace,
    inner: dict[str, Any],
    plan: dict[str, Any],
    membership: Any,
) -> dict[str, Any]:
    """Resolve only the evidenced first-wave admission/export failures after workers exit."""
    expected_first = ROOT_PROFILE_NODES | {"fin256_s3", "source4_lstm192x1"}
    nodes = inner.get("nodes")
    if (
        inner.get("phase") not in {"failed", "awaiting_profile_or_dependency"}
        or inner.get("active_workers")
        or inner.get("active_lanes", 0) != 0
        or not isinstance(nodes, dict)
        or set(nodes) != expected_first
        or inner.get("scientific_fits_started") != 8
        or inner.get("cache", {}).get("status") != "not_started"
    ):
        raise ValueError("terminal state is outside the one qualified first-wave repair scope")
    failed = {node_id for node_id, record in nodes.items() if record.get("status") != "complete"}
    base_repairs = {"source4_lstm192x1", "orgbase"}
    fin256_repairs = {"fin256_s3", "fin256_s4"}
    if frozenset(failed) not in {
        frozenset(base_repairs),
        frozenset(base_repairs | fin256_repairs),
    }:
        raise ValueError("first-wave terminal state contains an unqualified failure")
    before_path = args.controller_root / "controller-state.json"
    if before_path.is_symlink() or not before_path.is_file():
        raise ValueError("controller state is missing or linked before repair adoption")
    before_bytes = before_path.read_bytes()
    if contract.read_json(before_path, 64_000_000) != inner:
        raise ValueError("controller state changed while qualified repairs were being checked")
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    snapshot = (
        args.controller_root
        / "receipts/repairs"
        / f"controller-state.before-repair-{before_sha}.json"
    )
    _write_immutable_bytes(snapshot, before_bytes)
    native_proof, native_receipt = _validate_native_repair(
        args, nodes["source4_lstm192x1"], membership
    )
    orgbase_proof, orgbase_fit_receipt, orgbase_data = _validate_orgbase_repair(
        args, nodes["orgbase"], plan, membership
    )
    fin256_data = {
        node_id: _validate_fin256_export_repair(args, node_id, nodes[node_id], plan, membership)
        for node_id in sorted(failed & fin256_repairs)
    }
    terminal_outcomes = json.loads(json.dumps(inner.get("terminal_outcomes", [])))
    repair_bindings = {
        "source4_lstm192x1": {
            "path": str(args.runtime_root / "receipts/native-admission-repair-r2.json"),
            "sha256": contract.sha256_file(
                args.runtime_root / "receipts/native-admission-repair-r2.json"
            ),
        },
        "orgbase": {
            "path": str(args.runtime_root / "receipts/orgbase-export-admission-repair-r2.json"),
            "sha256": contract.sha256_file(
                args.runtime_root / "receipts/orgbase-export-admission-repair-r2.json"
            ),
        },
    }
    for node_id, (_, _, _, _) in fin256_data.items():
        proof_path = args.runtime_root / "receipts" / f"{node_id}-export-admission-repair-r4.json"
        repair_bindings[node_id] = {
            "path": str(proof_path),
            "sha256": contract.sha256_file(proof_path),
        }
    native_node = nodes["source4_lstm192x1"]
    native_failure = json.loads(json.dumps(native_node))
    native_node.update(
        {
            "admission_failure": native_failure,
            "status": "complete",
            "node_id": "source4_lstm192x1",
            "run": "final-lstm192x1",
            "fit_process_receipt": native_proof["fit_process_receipt"],
            "fit_process": native_receipt,
            "fit_started_at_epoch": native_receipt.get("started_at_epoch"),
            "fit_finished_at_epoch": native_receipt.get("finished_at_epoch"),
            "checkpoint_path": native_proof["checkpoint_path"],
            "checkpoint_sha256": native_proof["checkpoint_sha256"],
            "exports": {},
            "operational_repair": repair_bindings["source4_lstm192x1"],
        }
    )
    actual_command = native_receipt["command"]
    native_node["argv"] = actual_command
    native_node["argv_sha256"] = native_receipt["command_sha256"]
    for key in ("error", "current_receipt", "current_process", "child_process", "current_stage"):
        native_node.pop(key, None)
    orgbase_node = nodes["orgbase"]
    orgbase_failure = json.loads(json.dumps(orgbase_node))
    orgbase_node.update(
        {
            "admission_failure": orgbase_failure,
            "status": "complete",
            "node_id": "orgbase",
            "run": str(next(node["run"] for node in plan["nodes"] if node["id"] == "orgbase")),
            "fit_process_receipt": str(args.controller_root / "receipts/fits/orgbase.json"),
            "fit_process": orgbase_fit_receipt,
            "fit_finished_at_epoch": orgbase_fit_receipt.get("finished_at_epoch"),
            "checkpoint_path": orgbase_proof["checkpoint_path"],
            "checkpoint_sha256": orgbase_proof["checkpoint_sha256"],
            "exports": orgbase_data["exports"],
            "operational_repair": repair_bindings["orgbase"],
        }
    )
    for key in ("error", "current_receipt", "current_process", "child_process", "current_stage"):
        orgbase_node.pop(key, None)
    for node_id, (_, fit_receipt, checkpoint, repair_data) in fin256_data.items():
        node_record = nodes[node_id]
        failure_record = json.loads(json.dumps(node_record))
        node_record.update(
            {
                "admission_failure": failure_record,
                "status": "complete",
                "fit_process_receipt": str(
                    args.controller_root / "receipts/fits" / f"{node_id}.json"
                ),
                "fit_process": fit_receipt,
                "fit_started_at_epoch": fit_receipt.get("started_at_epoch"),
                "fit_finished_at_epoch": fit_receipt.get("finished_at_epoch"),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": contract.sha256_file(checkpoint),
                "exports": repair_data["exports"],
                "operational_repair": repair_bindings[node_id],
                "argv": fit_receipt["command"],
                "argv_sha256": fit_receipt["command_sha256"],
            }
        )
        for key in ("error", "current_receipt", "current_process", "child_process", "current_stage"):
            node_record.pop(key, None)
    inner["scientific_fits_started"] = sum(
        bool(item.get("fit_attempted")) for item in nodes.values()
    )
    inner["scientific_fits_completed"] = sum(
        item.get("status") == "complete" for item in nodes.values()
    )
    inner["exports_completed"] = sum(
        sum(export.get("status") == "complete" for export in item.get("exports", {}).values())
        for item in nodes.values()
    )
    if (
        inner["scientific_fits_started"] != 8
        or inner["scientific_fits_completed"] != 8
        or inner["exports_completed"] < 2
        or inner.get("terminal_outcomes", []) != terminal_outcomes
    ):
        raise ValueError("repair adoption changed the fixed wave counts or terminal history")
    inner["phase"] = "awaiting_profile_or_dependency"
    inner["qualified_operational_repairs"] = inner.get("qualified_operational_repairs", []) + [
        {
            "schema": "wnn.final-submit-operational-repair-adoption.v1",
            "before_state_path": str(snapshot),
            "before_state_sha256": before_sha,
            "proofs": repair_bindings,
            "resolved_node_ids": ["source4_lstm192x1", "orgbase", *sorted(fin256_data)],
            "terminal_outcomes_preserved": True,
            "adopted_at_epoch": time.time(),
        }
    ]
    inner["updated_at_epoch"] = time.time()
    atomic_write_json(before_path, inner)
    return inner


def _wait_early_successor_panel(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, str] | None:
    """Wait for the root-owned four-fit worker to finish without touching R1 state."""
    profile_panel = args.runtime_root / "receipts/early-successor-profile-panel-r4.approved.json"
    if not profile_panel.is_file():
        return None
    launch_path = args.runtime_root / "receipts/fit-early-successors-r4-driver.launch.json"
    completion_path = args.runtime_root / "receipts/early-successor-full-fit-panel-r4.complete.json"
    while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
        if not launch_path.is_file() or launch_path.is_symlink():
            raise ValueError("early-successor approval exists without its one-shot worker receipt")
        launch = contract.read_json(launch_path, 4_000_000)
        child = launch.get("child") if isinstance(launch.get("child"), dict) else launch
        pid, pgid = child.get("pid"), child.get("pgid")
        birth = child.get("birth", child.get("start_ticks"))
        if type(pid) is not int or type(pgid) is not int or not isinstance(birth, (str, int)):
            raise ValueError("early-successor launch receipt omits PID/PGID/start identity")
        current = process_identity(pid)
        owner_alive = (
            current is not None
            and current.get("pgid") == pgid
            and str(current.get("birth", current.get("start_ticks"))) == str(birth)
        )
        alive = owner_alive or _owned_group_alive(pgid)
        if completion_path.is_file() and not completion_path.is_symlink() and not alive:
            return (
                contract.read_json(completion_path, 4_000_000),
                completion_path,
                launch_path,
                contract.sha256_file(launch_path),
            )
        if not alive:
            raise RuntimeError("early-successor worker exited without a complete proof panel")
        time.sleep(10)
    raise TimeoutError("collection reserve began while awaiting the four approved fits")


def _early_profile_entries(
    args: argparse.Namespace, plan: dict[str, Any], membership: Any
) -> tuple[dict[str, dict[str, Any]], str]:
    """Validate the exact independent profile panel and bind requests to recipes."""
    path = args.runtime_root / "receipts/early-successor-profile-panel-r4.approved.json"
    panel = contract.read_json(path, 1_000_000)
    entries = panel.get("approved", panel.get("profiles", panel.get("entries")))
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("early-successor profile panel must contain exactly four approvals")
    by_id = {str(item.get("node_id", "")): item for item in entries if isinstance(item, dict)}
    if set(by_id) != set(EARLY_SUCCESSOR_IDS) or len(by_id) != len(entries):
        raise ValueError("early-successor profile panel node set differs")
    result: dict[str, dict[str, Any]] = {}
    for node_id, item in by_id.items():
        approval_raw, request_raw = Path(item["path"]), Path(item["request_path"])
        approval_raw = (
            approval_raw if approval_raw.is_absolute() else args.runtime_root / approval_raw
        )
        request_raw = request_raw if request_raw.is_absolute() else args.runtime_root / request_raw
        approval_path, request_path = approval_raw.resolve(), request_raw.resolve()
        if (
            approval_raw.is_symlink()
            or request_raw.is_symlink()
            or approval_path != approval_raw.absolute()
            or request_path != request_raw.absolute()
        ):
            raise ValueError(f"early-successor profile evidence traverses a symlink: {node_id}")
        for evidence_path in (approval_path, request_path):
            try:
                evidence_path.relative_to(args.runtime_root)
            except ValueError as error:
                raise ValueError("early-successor profile evidence escapes runtime root") from error
        expected_sha, request_sha = str(item["sha256"]), str(item["request_sha256"])
        approval = _validate_approval(
            node_id, approval_path, expected_sha, request_path, request_sha, plan, membership
        )
        if approval.get("gpu_uuid") != item.get("gpu_uuid"):
            raise ValueError(f"early-successor profile GPU differs: {node_id}")
        result[node_id] = {
            "path": approval_path,
            "sha256": expected_sha,
            "gpu_uuid": str(item["gpu_uuid"]),
            "request_path": request_path,
            "request_sha256": request_sha,
        }
    if len({item["gpu_uuid"] for item in result.values()}) != 4:
        raise ValueError("early-successor profile panel reuses a GPU")
    return result, contract.sha256_file(path)


def _validate_early_stage_receipt(
    args: argparse.Namespace,
    path: Path,
    embedded: object,
    argv: list[str],
    *,
    stage: int,
    trial: int,
    gpu_uuid: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify one frozen fit/export command receipt against the actual owned process."""
    receipt = contract.read_json(path, 2_000_000)
    command, timeout = receipt.get("command"), receipt.get("timeout_seconds")
    if (
        receipt != embedded
        or receipt.get("schema") != "wnn.multistage-owned-process.v1"
        or receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("mode") != "final_submit_finite_chain"
        or receipt.get("stage") != stage
        or receipt.get("trial_number") != trial
        or receipt.get("gpu_key") != gpu_uuid
        or receipt.get("cwd") != str(args.source_root.resolve())
        or receipt.get("output_dir") != str(output_dir.resolve())
        or type(timeout) is not int
        or not 10 < timeout <= contract.MAX_FIT_SECONDS
        or not isinstance(command, list)
        or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
        or command[3:4] != [f"{timeout - 10}s"]
        or command[4:] != argv
        or receipt.get("command_sha256") != contract.canonical_sha256(command)
    ):
        raise ValueError(f"early-successor owned command receipt differs: {path.name}")
    return receipt


def _validate_early_successor_proof(
    args: argparse.Namespace,
    plan: dict[str, Any],
    membership: Any,
    inner: dict[str, Any],
    node: dict[str, Any],
    profile: dict[str, Any],
    intent: dict[str, Any],
    intent_sha: str,
    proof_path: Path,
    expected_proof_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completion proof, checkpoint, source-ordered exports, and ancestry."""
    node_id, run = node["node_id"], node["run"]
    if proof_path.is_symlink() or contract.sha256_file(proof_path) != expected_proof_sha:
        raise ValueError(f"early-successor completion proof bytes differ: {node_id}")
    proof = contract.read_json(proof_path, 8_000_000)
    trial = contract.TL3_IDS.index(node_id) + 2
    output = args.data_root / "artifacts/runs" / run
    parents = {}
    for parent_id in node["parents"]:
        parent = next(
            item
            for item in director._node_records(plan, contract.verify_plan(plan, args.plan), args)
            if item["node_id"] == parent_id
        )
        row = inner["nodes"].get(parent_id)
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ValueError(f"successor parent is not complete: {node_id} <- {parent_id}")
        checkpoint = director.runtime.validate_fit_checkpoint(
            node=parent,
            output_dir=args.data_root / "artifacts/runs" / parent["run"],
            membership=membership,
        )
        parents[parent_id] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": contract.sha256_file(checkpoint),
        }
        if (
            row.get("checkpoint_path") != parents[parent_id]["checkpoint_path"]
            or row.get("checkpoint_sha256") != parents[parent_id]["checkpoint_sha256"]
        ):
            raise ValueError(f"successor parent checkpoint changed: {parent_id}")
    frozen = next(item for item in intent["items"] if item.get("node_id") == node_id)
    if (
        proof.get("schema") != "wnn.final-early-successor-completion.v1"
        or proof.get("status") != "complete"
        or proof.get("node_id") != node_id
        or proof.get("run") != run
        or proof.get("plan_sha256") != contract.PLAN_SHA256
        or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof.get("membership_sha256") != membership.manifest_sha256
        or proof.get("fit_members_sha256") != membership.fit_members_sha256
        or proof.get("panel_path")
        != str(args.runtime_root / "inputs/early-successor-full-fit-panel-r4.json")
        or proof.get("panel_sha256") != intent_sha
        or proof.get("profile_path") != str(profile["path"])
        or proof.get("profile_sha256") != profile["sha256"]
        or proof.get("request_path") != str(profile["request_path"])
        or proof.get("request_sha256") != profile["request_sha256"]
        or proof.get("parents") != parents
        or proof.get("gpu_uuid") != profile["gpu_uuid"]
        or proof.get("trial") != trial
        or proof.get("fixed_endpoint") != node["endpoint"]
        or proof.get("independent_score_claim") != "forbidden"
        or frozen.get("run") != run
        or frozen.get("args") != node["args"]
        or frozen.get("profile_path") != str(profile["path"])
        or frozen.get("profile_sha256") != profile["sha256"]
        or frozen.get("gpu_uuid") != profile["gpu_uuid"]
        or frozen.get("parents") != parents
        or frozen.get("trial") != trial
    ):
        raise ValueError(f"early-successor proof is not bound to the frozen node: {node_id}")
    result = proof.get("result")
    if (
        not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("run") != run
    ):
        raise ValueError(f"early-successor result is incomplete: {node_id}")
    fit_path = args.controller_root / "receipts/fits" / f"{node_id}.json"
    fit_argv = director.execution.build_fit_argv(
        node=node,
        python=args.python,
        source_root=args.source_root,
        data_root=args.data_root,
        plan=plan,
        profile_path=Path(profile["path"]),
    )
    fit_process = _validate_early_stage_receipt(
        args,
        fit_path,
        result.get("fit_process"),
        fit_argv,
        stage=1,
        trial=trial,
        gpu_uuid=profile["gpu_uuid"],
        output_dir=output,
    )
    checkpoint = director.runtime.validate_fit_checkpoint(
        node=node, output_dir=output, membership=membership
    )
    if (
        result.get("node_id") != node_id
        or result.get("fit_process_receipt") != str(fit_path)
        or result.get("checkpoint_path") != str(checkpoint)
        or result.get("checkpoint_sha256") != contract.sha256_file(checkpoint)
        or checkpoint != output / f"model_ep{node['endpoint']}.pt"
    ):
        raise ValueError(f"early-successor fixed checkpoint differs: {node_id}")
    exports = result.get("exports")
    if not isinstance(exports, dict) or set(exports) != {"train", "valid"}:
        raise ValueError(f"early-successor exports are incomplete: {node_id}")
    for split in ("train", "valid"):
        export = exports[split]
        path = director.runtime.validate_export(args.data_root, run, split, membership)
        receipt_path = args.controller_root / "receipts/exports" / f"{node_id}-{split}.json"
        export_argv = [
            str(args.python),
            str(args.source_root / "scripts/predict_run.py"),
            run,
            split,
            "--epoch",
            str(node["endpoint"]),
            "--final-fit-manifest",
            str(args.membership),
            "--final-fit-source-validation-receipt",
            str(args.source_audit),
            "--final-fit-source-validation-sha256",
            contract.SOURCE_AUDIT_SHA256,
            "--device",
            "cuda",
        ]
        process = _validate_early_stage_receipt(
            args,
            receipt_path,
            export.get("process"),
            export_argv,
            stage=2,
            trial=trial,
            gpu_uuid=profile["gpu_uuid"],
            output_dir=args.data_root / "cache",
        )
        if (
            export.get("status") != "complete"
            or export.get("path") != str(path)
            or export.get("sha256") != contract.sha256_file(path)
            or export.get("receipt") != str(receipt_path)
            or export.get("prediction_role")
            != "fitting-only teacher features; no independent score"
            or export.get("independent_score_claim") != "forbidden"
            or process != contract.read_json(receipt_path, 2_000_000)
        ):
            raise ValueError(f"early-successor source export differs: {node_id}/{split}")
    closure = audit_final_fit_ancestor_closure(run, checkpoint.name, membership, args.data_root)
    required_runs = {parent["run"] for parent in parents.values()}
    if required_runs - {entry.get("run") for entry in closure} or any(
        entry.get("final_fit_manifest_sha256") != membership.manifest_sha256
        or entry.get("final_fit_members_sha256") != membership.fit_members_sha256
        or entry.get("source_validation_sha256") != membership.source_validation_sha256
        for entry in closure
    ):
        raise ValueError(f"early-successor ancestor closure differs: {node_id}")
    row = {
        "status": "complete",
        "fit_attempted": True,
        "node_id": node_id,
        "run": run,
        "started_at_epoch": proof.get("started_epoch"),
        "fit_started_at_epoch": result.get("fit_started_at_epoch"),
        "fit_finished_at_epoch": result.get("fit_finished_at_epoch"),
        "gpu_uuid": profile["gpu_uuid"],
        "profile_receipt": str(profile["path"]),
        "profile_sha256": profile["sha256"],
        "profile_request_sha256": profile["request_sha256"],
        "trial_number": trial,
        "argv": fit_process["command"],
        "argv_sha256": fit_process["command_sha256"],
        "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
        "plan_sha256": contract.PLAN_SHA256,
        "membership_sha256": membership.manifest_sha256,
        "fit_process_receipt": str(fit_path),
        "fit_process": fit_process,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": contract.sha256_file(checkpoint),
        "exports": exports,
        "ancestor_closure": closure,
        "independent_score_claim": "forbidden",
    }
    return proof, row


def _adopt_early_successors(
    args: argparse.Namespace, inner: dict[str, Any], plan: dict[str, Any], membership: Any
) -> dict[str, Any]:
    """Adopt the four root-run fits only after R1 is terminal and all proofs pass."""
    events = inner.get("qualified_early_successor_adoptions", [])
    if events:
        if len(events) != 1 or events[0].get("resolved_node_ids") != list(EARLY_SUCCESSOR_IDS):
            raise ValueError("early-successor adoption history is malformed")
        event = events[0]
        proofs = event.get("proofs")
        if not isinstance(proofs, dict) or set(proofs) != set(EARLY_SUCCESSOR_IDS):
            raise ValueError("early-successor adoption proof bindings are malformed")
        for node_id, binding in proofs.items():
            path = Path(str(binding.get("path", "")))
            row = inner.get("nodes", {}).get(node_id, {})
            if (
                path.is_symlink()
                or not path.is_file()
                or contract.sha256_file(path) != binding.get("sha256")
                or row.get("status") != "complete"
                or row.get("fit_attempted") is not True
            ):
                raise ValueError(f"previously adopted successor proof no longer matches: {node_id}")
        return inner
    completed = _wait_early_successor_panel(args)
    if completed is None:
        return inner
    completion, completion_path, launch_path, launch_sha = completed
    if (
        inner.get("phase") != "awaiting_profile_or_dependency"
        or inner.get("active_workers")
        or inner.get("active_lanes", 0) != 0
        or inner.get("scientific_fits_started") != 8
        or inner.get("scientific_fits_completed") != 8
    ):
        raise ValueError(
            "early successors can only be adopted after the repaired first wave is terminal"
        )
    before_path = args.controller_root / "controller-state.json"
    before_bytes = before_path.read_bytes()
    if contract.read_json(before_path, 64_000_000) != inner:
        raise ValueError("R1 controller state changed while early successor proofs were checked")
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    snapshot = (
        args.controller_root
        / "receipts/repairs"
        / f"controller-state.before-early-successors-{before_sha}.json"
    )
    _write_immutable_bytes(snapshot, before_bytes)
    expected_first = ROOT_PROFILE_NODES | {"fin256_s3", "source4_lstm192x1"}
    if set(inner.get("nodes", {})) != expected_first:
        raise ValueError("R1 state is not the exact repaired eight-node first wave")
    _require_identity(inner, plan, membership, args)
    panel_path = args.runtime_root / "inputs/early-successor-full-fit-panel-r4.json"
    panel = contract.read_json(panel_path, 4_000_000)
    panel_sha = contract.sha256_file(panel_path)
    profile_records, profile_panel_sha = _early_profile_entries(args, plan, membership)
    if (
        completion.get("schema") != "wnn.final-early-successor-panel-completion.v1"
        or completion.get("status") != "complete"
        or completion.get("plan_sha256") != contract.PLAN_SHA256
        or completion.get("panel_path") != str(panel_path)
        or completion.get("panel_sha256") != panel_sha
        or panel.get("schema") != "wnn.final-early-successor-fit-panel.v1"
        or panel.get("plan_sha256") != contract.PLAN_SHA256
        or panel.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or panel.get("membership_sha256") != membership.manifest_sha256
        or panel.get("profile_panel_sha256") != profile_panel_sha
        or panel.get("max_new_scientific_fits") != 4
        or panel.get("total_unique_fit_roles_cap") != contract.TOTAL_FITS
        or panel.get("write_shared_controller_state") is not False
        or panel.get("foreign_signals") is not False
    ):
        raise ValueError("early-successor completion panel differs from frozen admission")
    frozen = {
        item.get("node_id"): item for item in panel.get("items", []) if isinstance(item, dict)
    }
    completion_items = completion.get("items")
    if (
        set(frozen) != set(EARLY_SUCCESSOR_IDS)
        or not isinstance(completion_items, list)
        or len(completion_items) != 4
    ):
        raise ValueError("early-successor panel does not contain the exact four fits")
    proof_entries = {
        item.get("node_id"): item for item in completion_items if isinstance(item, dict)
    }
    if set(proof_entries) != set(EARLY_SUCCESSOR_IDS) or len(proof_entries) != 4:
        raise ValueError("early-successor completion proof list is incomplete")
    plan_nodes = contract.verify_plan(plan, args.plan)
    node_records = {
        node["node_id"]: node for node in director._node_records(plan, plan_nodes, args)
    }
    verified: dict[str, tuple[dict[str, Any], dict[str, Any], str, str]] = {}
    for node_id in EARLY_SUCCESSOR_IDS:
        entry = proof_entries[node_id]
        expected_path = args.runtime_root / "receipts" / f"early-successor-{node_id}.complete.json"
        proof_path = Path(str(entry.get("proof_path", "")))
        if proof_path != expected_path or entry.get("proof_sha256") != contract.sha256_file(
            proof_path
        ):
            raise ValueError(f"early-successor proof panel path/hash differs: {node_id}")
        proof, row = _validate_early_successor_proof(
            args,
            plan,
            membership,
            inner,
            node_records[node_id],
            profile_records[node_id],
            panel,
            panel_sha,
            proof_path,
            str(entry["proof_sha256"]),
        )
        if frozen[node_id].get("run") != node_records[node_id]["run"]:
            raise ValueError(f"early-successor frozen run differs: {node_id}")
        verified[node_id] = (proof, row, str(proof_path), str(entry["proof_sha256"]))
    terminal_history = json.loads(json.dumps(inner.get("terminal_outcomes", [])))
    for node_id, (_, row, _, _) in verified.items():
        if node_id in inner["nodes"]:
            raise ValueError(f"R1 already attempted early successor; refusing adoption: {node_id}")
        inner["nodes"][node_id] = row
    inner["scientific_fits_started"] = sum(
        bool(row.get("fit_attempted")) for row in inner["nodes"].values()
    )
    inner["scientific_fits_completed"] = sum(
        row.get("status") == "complete" for row in inner["nodes"].values()
    )
    inner["exports_completed"] = sum(
        sum(item.get("status") == "complete" for item in row.get("exports", {}).values())
        for row in inner["nodes"].values()
    )
    if (
        inner["scientific_fits_started"] != 12
        or inner["scientific_fits_completed"] != 12
        or inner.get("terminal_outcomes", []) != terminal_history
    ):
        raise ValueError("early-successor adoption changed fit counts or terminal history")
    proof_bindings = {
        node_id: {"path": path, "sha256": digest}
        for node_id, (_, _, path, digest) in verified.items()
    }
    inner["phase"] = "awaiting_profile_or_dependency"
    inner.setdefault("qualified_early_successor_adoptions", []).append(
        {
            "schema": "wnn.final-submit-early-successor-adoption.v1",
            "before_state_path": str(snapshot),
            "before_state_sha256": before_sha,
            "completion_panel_path": str(completion_path),
            "completion_panel_sha256": contract.sha256_file(completion_path),
            "worker_launch_receipt_path": str(launch_path),
            "worker_launch_receipt_sha256": launch_sha,
            "worker_exit_status": "unknown_after_detachment",
            "worker_exit_status_reason": "Popen wrapper did not observe or record child exit",
            "fit_panel_path": str(panel_path),
            "fit_panel_sha256": panel_sha,
            "profile_panel_path": str(
                args.runtime_root / "receipts/early-successor-profile-panel-r4.approved.json"
            ),
            "profile_panel_sha256": profile_panel_sha,
            "proofs": proof_bindings,
            "resolved_node_ids": list(EARLY_SUCCESSOR_IDS),
            "terminal_outcomes_preserved": True,
            "adopted_at_epoch": time.time(),
        }
    )
    inner["updated_at_epoch"] = time.time()
    atomic_write_json(before_path, inner)
    return inner


def _inner_worker_alive(inner: dict[str, Any]) -> bool:
    """Check only process identities recorded by the finite director."""
    for worker in inner.get("active_workers", {}).values():
        pid = worker.get("pid")
        pgid = worker.get("pgid")
        birth = worker.get("birth")
        if (
            type(pid) is int
            and type(pgid) is int
            and isinstance(birth, str)
            and process_identity(pid) == {"pid": pid, "pgid": pgid, "birth": birth}
        ):
            return True
    return False


def _adopt_director_wave(
    args: argparse.Namespace,
    state: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    membership: Any,
) -> dict[str, Any]:
    """Wait on an already-owned R1/R2 director wave without launching or signaling it."""
    runs = state.get("director_runs", [])
    adoptable_statuses = {"running", "adopted_running", "completed"}
    run_record = (
        runs[-1]
        if runs and runs[-1].get("status") in adoptable_statuses
        else None
    )
    receipt_path = (
        Path(run_record["receipt_path"])
        if run_record is not None and run_record.get("receipt_path")
        else args.adopt_director_receipt
    )
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"cannot adopt a running director without its receipt: {receipt_path}"
        )
    receipt = contract.read_json(receipt_path)
    if receipt.get("schema") == "wnn.multistage-owned-process.v1" and (
        receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
    ):
        raise RuntimeError("adopted director command did not complete successfully")
    command = receipt.get("command")
    created_epoch = receipt.get("created_epoch", receipt.get("started_at_epoch"))
    wave = int(run_record.get("wave", 1)) if run_record is not None else 1
    command_args = args
    if wave > 1:
        command_args = argparse.Namespace(**vars(args))
        command_args.root_admission = args.r2_root_admission
        command_args.director_script = args.r2_director_script
        _verify_root(command_args)
    first_wave_profile_ids = ROOT_PROFILE_NODES | {"fin256_s3", "source4_lstm192x1"}
    matched_profiles = profiles if wave > 1 else {
        node_id: profiles[node_id] for node_id in first_wave_profile_ids if node_id in profiles
    }
    if (
        (wave == 1 and set(matched_profiles) != first_wave_profile_ids)
        or not _director_command_matches(command, command_args, matched_profiles, created_epoch)
    ):
        raise ValueError("existing director command differs from the frozen plan/profile set")
    expected_admission_sha256 = contract.sha256_file(command_args.root_admission)
    if run_record is None and receipt.get("root_admission_sha256") != expected_admission_sha256:
        raise ValueError("existing director receipt is bound to another root admission")
    if (
        run_record is not None
        and run_record.get("root_admission_sha256") != expected_admission_sha256
    ):
        raise ValueError("campaign director record is bound to another root admission")
    identity = receipt.get("child")
    identity_source = identity if isinstance(identity, dict) else receipt
    identity = {
        "pid": identity_source.get("pid"),
        "pgid": identity_source.get("pgid"),
        "birth": identity_source.get("birth", identity_source.get("start_ticks")),
    }
    if any(type(identity.get(key)) is not int for key in ("pid", "pgid")) or not identity.get(
        "birth"
    ):
        raise ValueError("existing director receipt omits its PID/birth/PGID identity")
    expected_identity = _expected_controller_identity(plan, membership, args)
    if run_record is None:
        run_record = {
            "wave": wave,
            "status": "adopted_running",
            "receipt_path": str(receipt_path),
            "profile_ids": sorted(matched_profiles),
            "argv": command,
            "root_admission_sha256": expected_admission_sha256,
            "started_at_epoch": receipt.get("started_at_epoch", receipt.get("created_epoch")),
        }
        state.setdefault("director_runs", []).append(run_record)
    else:
        run_record["receipt_path"] = str(receipt_path)
        run_record["receipt"] = receipt
        run_record["argv"] = command
        run_record["root_admission_sha256"] = expected_admission_sha256
    inner_path = args.controller_root / "controller-state.json"
    while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
        inner = contract.read_json(inner_path, 64_000_000) if inner_path.is_file() else {}
        identity_record = inner.get("identity")
        if isinstance(identity_record, dict) and identity_record != expected_identity:
            raise ValueError("shared controller state identity differs from the frozen R1 identity")
        process_group_alive = (
            process_identity(int(identity["pid"])) == identity
            or _owned_group_alive(int(identity["pgid"]))
            or _inner_worker_alive(inner)
        )
        if process_group_alive:
            run_record["status"] = "adopted_running"
            run_record["current_process"] = identity
            run_record["latest_fit_counts"] = {
                "started": inner.get("scientific_fits_started", 0),
                "completed": inner.get("scientific_fits_completed", 0),
                "active_workers": len(inner.get("active_workers", {})),
            }
            state["updated_at_epoch"] = time.time()
            atomic_write_json(args.state_path, state)
            time.sleep(10)
            continue
        phase = inner.get("phase")
        if phase == "failed":
            if run_record.get("wave") != 1:
                raise RuntimeError("a later director wave failed; automatic repair is forbidden")
            inner = _adopt_qualified_operational_repairs(args, inner, plan, membership)
            phase = inner.get("phase")
        if phase in {"complete", "awaiting_profile_or_dependency"} and not inner.get(
            "active_workers"
        ):
            if not isinstance(identity_record, dict) or identity_record != expected_identity:
                raise ValueError("terminal shared state omits the exact frozen director identity")
            if phase == "complete":
                _require_complete(inner)
            elif run_record.get("wave") == 1:
                expected_first = ROOT_PROFILE_NODES | {"fin256_s3", "source4_lstm192x1"}
                complete_ids = {
                    name
                    for name, item in inner.get("nodes", {}).items()
                    if item.get("status") == "complete"
                }
                if (
                    inner.get("scientific_fits_started") != 8
                    or inner.get("scientific_fits_completed") != 8
                    or complete_ids != expected_first
                ):
                    raise ValueError(
                        "adopted first wave did not complete exactly the eight pinned roots"
                    )
            run_record.update({"status": "adopted_completed", "finished_at_epoch": time.time()})
            state["latest_fit_counts"] = {
                "started": inner.get("scientific_fits_started", 0),
                "completed": inner.get("scientific_fits_completed", 0),
                "exports_completed": inner.get("exports_completed", 0),
                "cache": inner.get("cache", {}).get("status"),
            }
            atomic_write_json(args.state_path, state)
            return inner
        if phase == "failed":
            raise RuntimeError("adopted director wave recorded a terminal failure")
        run_record["status"] = "reconciliation_required"
        state["updated_at_epoch"] = time.time()
        atomic_write_json(args.state_path, state)
        raise RuntimeError("adopted director exited with unresolved fit/cache ownership")
    raise TimeoutError("collection reserve began while adopting the R1 director wave")


def _director_pass(
    args: argparse.Namespace,
    plan: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the frozen R1 or R2 director with the currently approved profiles."""
    wave = len(state.get("director_runs", [])) + 1
    command_args = args
    if wave > 1:
        command_args = argparse.Namespace(**vars(args))
        command_args.root_admission = args.r2_root_admission
        command_args.director_script = args.r2_director_script
    _verify_root(command_args)
    _, loaded = _load_inputs(args)
    membership = loaded["membership"]
    if len(profiles) + HISTORICAL_PROFILE_COUNT > contract.MAX_PROFILES:
        raise ValueError("campaign profile count exceeds the 24-profile contract")
    key = f"director-wave-{wave:02d}"
    log_path = args.controller_root / "logs/campaign" / f"{key}.log"
    receipt_path = args.controller_root / "receipts/campaign" / f"{key}.json"
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "21590s",
        str(args.python),
        str(command_args.director_script),
        "--action",
        "run",
        "--runtime-root",
        str(args.runtime_root),
        "--source-root",
        str(args.source_root),
        "--data-root",
        str(args.data_root),
        "--plan",
        str(args.plan),
        "--membership",
        str(args.membership),
        "--source-audit",
        str(args.source_audit),
        "--source-capsule",
        str(args.source_capsule),
        "--native-config",
        str(args.native_config),
        "--python",
        str(args.python),
        "--grader",
        str(args.grader),
        "--root-admission",
        str(command_args.root_admission),
        "--controller-root",
        str(args.controller_root),
        "--max-lanes",
        str(contract.MAX_LANES),
    ]
    for node_id, item in profiles.items():
        command.extend(
            (
                "--profile-approval",
                f"{node_id}={item['sha256']}={item['gpu_uuid']}={item['path']}",
            )
        )
    remaining = int(contract.HARD_DEADLINE - contract.COLLECTION_RESERVE - time.time())
    timeout = remaining - 10
    if timeout <= 10:
        raise TimeoutError("collection reserve has begun; no new director pass may start")
    command[3] = f"{timeout - 10}s"
    state.setdefault("director_runs", []).append(
        {
            "wave": wave,
            "status": "running",
            "argv": command,
            "log_path": str(log_path),
            "receipt_path": str(receipt_path),
            "root_admission_sha256": contract.sha256_file(command_args.root_admission),
            "started_at_epoch": time.time(),
        }
    )
    atomic_write_json(args.state_path, state)
    env = base_environment("", args.source_root)
    env["WNN_PROJECT_ROOT"] = str(args.data_root)
    owner = process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("campaign process identity is unavailable")
    last_progress = 0.0

    def director_progress() -> None:
        nonlocal last_progress
        now = time.monotonic()
        if receipt_path.is_file() and (last_progress == 0 or now - last_progress >= 5):
            stage_receipt = contract.read_json(receipt_path)
            state["director_runs"][-1]["current_process"] = stage_receipt.get("child")
            state["updated_at_epoch"] = time.time()
            atomic_write_json(args.state_path, state)
            last_progress = now

    code, timed_out, receipt = run_owned_command(
        command,
        cwd=args.source_root,
        env=env,
        log_path=log_path,
        receipt_path=receipt_path,
        owner={**owner, "controller_root": str(args.controller_root)},
        stage=1,
        trial_number=wave,
        mode="final_submit_campaign_director",
        gpu_key="campaign-controller",
        output_dir=args.controller_root,
        timeout_seconds=timeout,
        progress=director_progress,
    )
    state["director_runs"][-1].update(
        {
            "status": "completed" if code == 0 and not timed_out else "failed",
            "receipt_path": str(receipt_path),
            "receipt": receipt,
            "finished_at_epoch": time.time(),
        }
    )
    atomic_write_json(args.state_path, state)
    if timed_out or code != 0:
        raise RuntimeError(f"frozen director pass failed; see {log_path}")
    child = receipt.get("child") if isinstance(receipt, dict) else None
    if (
        isinstance(child, dict)
        and type(child.get("pid")) is int
        and process_identity(int(child["pid"])) == child
    ) or (
        isinstance(child, dict)
        and type(child.get("pgid")) is int
        and _owned_group_alive(int(child["pgid"]))
    ):
        raise RuntimeError("director returned while an owned process group was still active")
    inner_path = args.controller_root / "controller-state.json"
    inner = contract.read_json(inner_path, 64_000_000)
    _require_identity(inner, plan, membership, args)
    failed_nodes = any(
        item.get("status") in {"failed", "reconciliation_required"}
        for item in inner.get("nodes", {}).values()
    )
    if inner.get("phase") == "failed" or failed_nodes:
        if (
            inner.get("phase") not in {"failed", "awaiting_profile_or_dependency"}
            or state["director_runs"][-1].get("wave") != 1
            or inner.get("active_workers")
        ):
            raise RuntimeError(
                "director recorded a terminal fit/cache failure; no retry is allowed"
            )
        inner = _adopt_qualified_operational_repairs(args, inner, plan, membership)
        _require_identity(inner, plan, membership, args)
    state["latest_fit_counts"] = {
        "started": inner.get("scientific_fits_started", 0),
        "completed": inner.get("scientific_fits_completed", 0),
        "exports_completed": inner.get("exports_completed", 0),
        "cache": inner.get("cache", {}).get("status"),
    }
    state["phase"] = inner.get("phase", "awaiting_profile_or_dependency")
    state["updated_at_epoch"] = time.time()
    atomic_write_json(args.state_path, state)
    return inner


def build_parser() -> argparse.ArgumentParser:
    """Create the finite campaign CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--source-capsule", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--root-admission", type=Path)
    parser.add_argument("--r2-root-admission", type=Path)
    parser.add_argument("--grader", type=Path)
    parser.add_argument("--director-script", type=Path)
    parser.add_argument("--r2-director-script", type=Path)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--profile-panel", type=Path)
    parser.add_argument("--controller-root", type=Path)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--adopt-director-receipt", type=Path)
    return parser


def main() -> None:
    """Validate the finite graph, then profile only ready nodes and advance the director."""
    args = build_parser().parse_args()
    _resolve_paths(args)
    _verify_root(args)
    plan, loaded = _load_inputs(args) if args.action == "preflight" else _load_pinned_inputs(args)
    nodes = loaded["nodes"]
    membership = loaded["membership"]
    profiles, panel_nodes = _load_profiles(args, plan, membership)
    missing_panel = sorted(ROOT_PROFILE_NODES - panel_nodes)
    if args.action == "preflight":
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "plan_sha256": contract.PLAN_SHA256,
                    "profile_approvals": sorted(profiles),
                    "profile_count_with_historical": len(profiles) + HISTORICAL_PROFILE_COUNT,
                    "missing_root_profile_panel_nodes": missing_panel,
                    "scientific_fits_started": False,
                    "cuda_touched": False,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return
    if missing_panel:
        raise ValueError(f"root profile panel is incomplete: {missing_panel}")
    controller_root = args.controller_root
    controller_root.mkdir(parents=True, exist_ok=True)
    state = (
        contract.read_json(args.state_path)
        if args.state_path.is_file()
        else {
            "schema": "wnn.final-submit-campaign-state.v1",
            "plan_sha256": contract.PLAN_SHA256,
            "phase": "running",
            "started_at_epoch": time.time(),
            "profile_attempts": {},
            "director_runs": [],
        }
    )
    if (
        state.get("schema") != "wnn.final-submit-campaign-state.v1"
        or state.get("plan_sha256") != contract.PLAN_SHA256
    ):
        raise ValueError("campaign state belongs to another plan or controller schema")
    state["profiles_approved"] = len(profiles)
    state["profile_count_with_historical"] = len(profiles) + HISTORICAL_PROFILE_COUNT
    with file_lock(controller_root / "campaign.lock", blocking=False):
        atomic_write_json(args.state_path, state)
        while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
            if len(profiles) + HISTORICAL_PROFILE_COUNT > contract.MAX_PROFILES:
                raise ValueError("campaign exceeded the 24-profile ceiling")
            inner_path = controller_root / "controller-state.json"
            inner = contract.read_json(inner_path, 64_000_000) if inner_path.is_file() else {}
            failed_nodes = any(
                item.get("status") in {"failed", "reconciliation_required"}
                for item in inner.get("nodes", {}).values()
            )
            if failed_nodes and inner.get("phase") in {
                "failed",
                "awaiting_profile_or_dependency",
            }:
                previous_director_runs = state.get("director_runs", [])
                if (
                    not previous_director_runs
                    or previous_director_runs[-1].get("wave") != 1
                    or previous_director_runs[-1].get("status")
                    not in {"completed", "adopted_completed"}
                ):
                    raise RuntimeError(
                        "qualified first-wave repair requires a terminal R1 director receipt"
                    )
                inner = _adopt_qualified_operational_repairs(args, inner, plan, membership)
                _require_identity(inner, plan, membership, args)
            if inner.get("phase") == "complete":
                inner = _adopt_early_successors(args, inner, plan, membership)
                _require_identity(inner, plan, membership, args)
                _require_complete(inner)
                state.update({"phase": "complete", "finished_at_epoch": time.time()})
                atomic_write_json(args.state_path, state)
                print(json.dumps(state, sort_keys=True, indent=2, allow_nan=False))
                return
            has_external_receipt = args.adopt_director_receipt.is_file()
            previous_director_runs = state.get("director_runs", [])
            resumable_first_wave = bool(previous_director_runs) and (
                previous_director_runs[-1].get("wave") == 1
                and previous_director_runs[-1].get("status")
                in {"running", "adopted_running", "completed"}
            )
            should_adopt = (
                inner.get("phase") == "running"
                or (
                    inner.get("phase") == "failed"
                    and has_external_receipt
                    and (not previous_director_runs or resumable_first_wave)
                )
                or (not inner and not state.get("director_runs") and has_external_receipt)
            )
            if should_adopt:
                inner = _adopt_director_wave(args, state, profiles, plan, membership)
                profiles, _ = _load_profiles(args, plan, membership)
            elif inner.get("phase") != "awaiting_profile_or_dependency":
                inner = _director_pass(args, plan, nodes, profiles, state)
            if inner.get("phase") in {
                "awaiting_profile_or_dependency",
                "complete",
            } and not inner.get("active_workers"):
                inner = _adopt_early_successors(args, inner, plan, membership)
            profiles, _ = _load_profiles(args, plan, membership)
            state["profiles_approved"] = len(profiles)
            state["profile_count_with_historical"] = len(profiles) + HISTORICAL_PROFILE_COUNT
            if (
                inner.get("scientific_fits_completed", 0) == contract.TOTAL_FITS
                and inner.get("cache", {}).get("status") == "complete"
            ):
                continue
            ready = _ready_to_profile(nodes, profiles, inner)
            if not ready:
                state["phase"] = "awaiting_external_profile_or_terminal_reconciliation"
                state["updated_at_epoch"] = time.time()
                atomic_write_json(args.state_path, state)
                raise RuntimeError(
                    "no ready profile or fit remains; inspect director and panel receipts"
                )
            used_gpus: set[str] = set()
            for node in ready:
                gpu_uuid = _select_idle_gpu(args, used_gpus)
                _run_profile(args, node, gpu_uuid, state)
                used_gpus.add(gpu_uuid)
                profiles, _ = _load_profiles(args, plan, membership)
            inner = _director_pass(args, plan, nodes, profiles, state)
            state["latest_fit_counts"] = {
                "started": inner.get("scientific_fits_started", 0),
                "completed": inner.get("scientific_fits_completed", 0),
                "exports_completed": inner.get("exports_completed", 0),
                "cache": inner.get("cache", {}).get("status"),
            }
            state["updated_at_epoch"] = time.time()
            atomic_write_json(args.state_path, state)
        state["phase"] = "collection_reserve"
        state["updated_at_epoch"] = time.time()
        atomic_write_json(args.state_path, state)
        print(json.dumps(state, sort_keys=True, indent=2, allow_nan=False))


def _select_idle_gpu(args: argparse.Namespace, used: set[str]) -> str:
    """Choose an unused B300 with 32 GiB free, without signaling other processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    )
    for candidate in (line.strip() for line in result.stdout.splitlines()):
        if not candidate or candidate in used:
            continue
        try:
            _require_b300_headroom(candidate)
        except (subprocess.SubprocessError, RuntimeError, ValueError):
            continue
        return candidate
    raise RuntimeError("no unused B300 with 32 GiB free is available for the next profile")


def _require_b300_headroom(gpu_uuid: str) -> None:
    """Require the UUID to identify a B300 with 32 GiB free at this observation."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    matches = [
        line.split(", ") for line in result.stdout.splitlines() if line.startswith(gpu_uuid + ",")
    ]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise RuntimeError(f"assigned GPU UUID is not uniquely visible: {gpu_uuid}")
    _, name, memory_free = matches[0]
    if "B300" not in name.upper() or int(memory_free) < 32_768:
        raise RuntimeError(
            f"assigned GPU lacks B300/32-GiB admission: {gpu_uuid}, {name}, {memory_free} MiB free"
        )


if __name__ == "__main__":
    main()
