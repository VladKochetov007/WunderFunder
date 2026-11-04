"""Run the fixed 18-fit final-submit DAG behind explicit root and profile gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import final_submit_chain_contract as contract
import final_submit_chain_execution as execution
import final_submit_chain_runtime as runtime

from wnn.multistage_search import atomic_write_json, file_lock
from wnn.multistage_search_runtime import process_identity


def _profile_inputs(
    args: argparse.Namespace, membership: Any, plan: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load the two pinned R2 profiles and any individually pinned reviewed profiles."""
    result: dict[str, dict[str, Any]] = {}
    review_root = args.runtime_root / "profile-reviews"
    defaults = {
        "source4_lstm192x1": (
            review_root / "native-lstm-r2-final/approved_profile_receipt.json",
            contract.NATIVE_PROFILE_SHA256,
            "native_source4_lstm",
            args.runtime_root / "inputs/native-lstm-r2.request.json",
        ),
        "fin256_s3": (
            review_root / "legacy-fin256-r2-final/approved_profile_receipt.json",
            contract.FIN256_PROFILE_SHA256,
            "legacy_tl3",
            args.runtime_root / "inputs/legacy-fin256-r2.request.json",
        ),
    }
    supplied: dict[str, tuple[str, str, str]] = {}
    for raw in args.profile_approval:
        fields = raw.split("=", 3)
        if len(fields) != 4 or fields[0] in supplied:
            raise ValueError("--profile-approval must be NODE=SHA256=GPU_UUID=APPROVAL_PATH")
        supplied[fields[0]] = (fields[1], fields[2], fields[3])
    allowed = set(contract.TL3_IDS) | {"source4_lstm192x1"}
    for node_id, (path, digest, mode, request_path) in defaults.items():
        if node_id in supplied:
            actual_path, actual_digest = Path(supplied[node_id][2]), supplied[node_id][0]
        elif not path.is_file():
            continue
        else:
            actual_path, actual_digest = path, digest
        approval = contract.validate_profile_approval(
            node_id=node_id,
            approval_path=actual_path,
            expected_sha256=actual_digest,
            plan_sha256=contract.PLAN_SHA256,
            membership=membership,
            mode=mode,
        )
        request = contract.read_json(request_path)
        request_hash = contract.sha256_file(request_path)
        if request.get("node_id") != node_id or (
            mode == "legacy_tl3" and approval.get("request_sha256") != request_hash
        ):
            raise ValueError(f"R2 profile approval request identity differs: {node_id}")
        evidence = contract.read_json(actual_path.parent / "profile_review.json").get(
            "evidence_sha256", {}
        )
        if evidence.get("request") != request_hash:
            raise ValueError(f"R2 profile review was not made from its frozen request: {node_id}")
        result[node_id] = {
            "path": actual_path,
            "sha256": actual_digest,
            "gpu_uuid": str(request["gpu_uuid"]),
            "mode": mode,
            "request_sha256": request_hash,
        }
        supplied.pop(node_id, None)
    for node_id, (digest, gpu_uuid, raw_path) in supplied.items():
        if node_id not in allowed:
            raise ValueError(f"profile approval names an unknown finite graph node: {node_id}")
        if node_id in result:
            raise ValueError(f"profile approval repeats a default node: {node_id}")
        mode = "native_source4_lstm" if node_id == "source4_lstm192x1" else "legacy_tl3"
        approval_path = Path(raw_path)
        approval = contract.validate_profile_approval(
            node_id=node_id,
            approval_path=approval_path,
            expected_sha256=digest,
            plan_sha256=contract.PLAN_SHA256,
            membership=membership,
            mode=mode,
        )
        result[node_id] = {
            "path": approval_path,
            "sha256": digest,
            "gpu_uuid": gpu_uuid,
            "mode": mode,
            "request_sha256": str(approval.get("request_sha256", "")),
        }
    if len(result) > contract.MAX_PROFILES:
        raise ValueError("profile approvals exceed the 24-profile ceiling")
    return result


def _verify_source_tree(source_root: Path, capsule_path: Path) -> None:
    """Compare every extracted file and link with the pinned capsule inventory."""
    expected_files: set[str] = set()
    expected_links: set[str] = set()
    seen: set[str] = set()
    with tarfile.open(capsule_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            name = relative.as_posix()
            if relative.is_absolute() or ".." in relative.parts or name in seen:
                raise ValueError(f"unsafe or duplicate source capsule member: {member.name}")
            seen.add(name)
            target = source_root.joinpath(*relative.parts)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"source capsule directory differs: {name}")
            elif member.isfile():
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"source capsule file is missing or unsafe: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"source capsule file cannot be read: {name}")
                digest = hashlib.sha256()
                with stream:
                    while block := stream.read(1 << 20):
                        digest.update(block)
                if digest.hexdigest() != contract.sha256_file(target):
                    raise ValueError(f"extracted source bytes differ from the capsule: {name}")
                expected_files.add(name)
            elif member.issym():
                if not target.is_symlink() or os.readlink(target) != member.linkname:
                    raise ValueError(f"source capsule link differs: {name}")
                expected_links.add(name)
            else:
                raise ValueError(f"unsupported source capsule member type: {name}")

    actual_files: set[str] = set()
    actual_links: set[str] = set()
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root).as_posix()
        parts = PurePosixPath(relative).parts
        if "__pycache__" in parts or path.suffix == ".pyc" or relative == ".cargo/config.toml":
            continue
        if path.is_symlink():
            actual_links.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        elif not path.is_dir():
            raise ValueError(f"unsupported extracted source entry: {relative}")
    if actual_files != expected_files or actual_links != expected_links:
        raise ValueError("extracted source inventory differs from the reviewed capsule")


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Any, dict[str, Any]]:
    """Verify the frozen plan, source capsule, data receipt, source tree, and profiles."""
    plan = contract.read_json(args.plan)
    nodes = contract.verify_plan(plan, args.plan)
    if contract.sha256_file(args.source_capsule) != contract.SOURCE_CAPSULE_SHA256:
        raise ValueError("source capsule bytes differ from the reviewed source archive")
    _verify_source_tree(args.source_root, args.source_capsule)
    audit_hash = contract.sha256_file(args.source_audit)
    membership = contract.verify_sources(
        data_root=args.data_root,
        membership_path=args.membership,
        audit_path=args.source_audit,
        audit_sha256=audit_hash,
    )
    if audit_hash != contract.SOURCE_AUDIT_SHA256:
        raise ValueError("source validation receipt differs from the frozen receiver audit")
    for node in nodes.values():
        argv = node["args"]
        if (
            Path(contract.option_values(argv, "--final-fit-manifest")[0]).resolve()
            != args.membership.resolve()
        ):
            raise ValueError(f"manifest path differs from the reviewed plan for {node['id']}")
        if (
            Path(contract.option_values(argv, "--final-fit-source-validation-receipt")[0]).resolve()
            != args.source_audit.resolve()
        ):
            raise ValueError(
                f"receiver receipt path differs from the reviewed plan for {node['id']}"
            )
    cache_values = contract.option_values(
        nodes["w18final_distilled_s5"]["args"], "--distill-cache-root"
    )
    if (
        len(cache_values) != 1
        or Path(cache_values[0]).resolve()
        != (args.data_root / "artifacts/runs/final-fit-w18-mixed-targets").resolve()
    ):
        raise ValueError("W18 cache output differs from the exact frozen plan cache path")
    native = plan["optional_source4_lstm"]
    config_path = Path(native["config_path"])
    if config_path.resolve() != args.native_config.resolve():
        raise ValueError("native config path differs from the amended frozen plan")
    if contract.sha256_file(config_path) != native["config_file_sha256"]:
        raise ValueError("native raw config bytes differ from the pinned R2 config")
    approvals = _profile_inputs(args, membership, plan)
    return plan, nodes, membership, approvals


def _gpu_map(args: argparse.Namespace, profiles: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Resolve approved GPU UUIDs to one-device CUDA visibility values."""
    result = {uuid: uuid for uuid in {item["gpu_uuid"] for item in profiles.values()}}
    for raw in args.gpu:
        if "=" not in raw:
            raise ValueError("--gpu must be GPU_UUID=CUDA_VISIBLE_DEVICES_VALUE")
        uuid, visible = raw.split("=", 1)
        if not uuid or not visible:
            raise ValueError("--gpu contains an empty identity")
        result[uuid] = visible
    for item in profiles.values():
        if item["gpu_uuid"] not in result:
            raise ValueError(f"no CUDA visibility mapping for profile GPU {item['gpu_uuid']}")
    return result


def _state_for(path: Path, identity: dict[str, object]) -> dict[str, Any]:
    """Load or create the single durable controller state, refusing identity drift."""
    if path.exists():
        state = contract.read_json(path, 64_000_000)
        if (
            state.get("schema") != "wnn.final-submit-chain-state.v1"
            or state.get("identity") != identity
        ):
            raise ValueError("controller state belongs to different plan/source/control bytes")
        return state
    return {
        "schema": "wnn.final-submit-chain-state.v1",
        "identity": identity,
        "phase": "prepared",
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "scientific_fits_started": 0,
        "scientific_fits_completed": 0,
        "profiles_approved": 0,
        "exports_completed": 0,
        "max_lanes": 0,
        "nodes": {},
        "active_workers": {},
        "cache": {"status": "not_started"},
        "terminal_outcomes": [],
    }


def _refresh_counts(state: dict[str, Any], profiles: dict[str, dict[str, Any]], lanes: int) -> None:
    """Update compact fit, profile, export, and active-worker counts for monitoring."""
    nodes = state["nodes"].values()
    state["scientific_fits_started"] = sum(bool(node.get("fit_attempted")) for node in nodes)
    state["scientific_fits_completed"] = sum(node.get("status") == "complete" for node in nodes)
    state["profiles_approved"] = len(profiles)
    state["exports_completed"] = sum(
        sum(item.get("status") == "complete" for item in node.get("exports", {}).values())
        for node in nodes
    )
    state["max_lanes"] = max(int(state.get("max_lanes", 0)), lanes)
    state["updated_at_epoch"] = time.time()


def _check_interrupted(state: dict[str, Any], state_path: Path) -> None:
    """Refuse duplicate launches after a controller interruption or unresolved child."""
    changed = False
    for node_id, record in state["nodes"].items():
        if record.get("status") != "running":
            continue
        receipt = Path(str(record.get("current_receipt", "")))
        child = contract.read_json(receipt).get("child") if receipt.is_file() else None
        if isinstance(child, dict) and process_identity(int(child.get("pid", -1))) == child:
            raise RuntimeError(f"{node_id} is still running under its recorded process identity")
        record["status"] = "reconciliation_required"
        record["terminal_reason"] = "controller stopped before recording a terminal worker result"
        changed = True
    cache = state.get("cache", {})
    if cache.get("status") == "running":
        receipt = Path(str(cache.get("current_receipt", "")))
        child = contract.read_json(receipt).get("child") if receipt.is_file() else None
        if isinstance(child, dict) and process_identity(int(child.get("pid", -1))) == child:
            raise RuntimeError(
                "the W18 cache builder is still running under its recorded process identity"
            )
        cache["status"] = "reconciliation_required"
        cache["terminal_reason"] = "controller stopped before recording a terminal cache result"
        changed = True
    if changed:
        state["phase"] = "reconciliation_required"
        state["updated_at_epoch"] = time.time()
        atomic_write_json(state_path, state)
        raise RuntimeError(
            "an owned worker ended without a terminal controller result; no retry was made"
        )
    if any(
        node.get("status") in {"failed", "reconciliation_required"}
        for node in state["nodes"].values()
    ):
        raise RuntimeError(
            "a prior fit/export failed or needs manual reconciliation; no retry was made"
        )


def _expected_identity(
    plan: dict[str, Any], membership: Any, args: argparse.Namespace
) -> dict[str, object]:
    """Keep the R1 state identity while separately admitting these R2 code bytes."""
    stable = {
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": contract.sha256_file(args.plan),
        "source_capsule_sha256": contract.sha256_file(args.source_capsule),
        "source_commit": contract.SOURCE_COMMIT,
        "source_validation_sha256": contract.SOURCE_AUDIT_SHA256,
        "membership_sha256": membership.manifest_sha256,
        "fit_members_sha256": membership.fit_members_sha256,
        "native_config_sha256": contract.sha256_file(args.native_config),
    }
    prior_state_path = args.controller_root / "controller-state.json"
    if prior_state_path.is_file():
        prior_state = contract.read_json(prior_state_path, 64_000_000)
        prior_identity = prior_state.get("identity")
        r1_root = args.runtime_root / "control-code-r1/scripts"
        r1_paths = (
            r1_root / "final_submit_chain.py",
            r1_root / "final_submit_chain_contract.py",
            r1_root / "final_submit_chain_runtime.py",
            r1_root / "final_submit_chain_execution.py",
        )
        r1_files = {
            f"scripts/{path.name}": contract.sha256_file(path) for path in r1_paths
        }
        r1_admission = contract.read_json(
            args.runtime_root / "ROOT_EXECUTION_ADMISSION_R1.json", 1_000_000
        )
        historical_files = {Path(name).name: digest for name, digest in r1_files.items()}
        r2_admission = contract.read_json(args.root_admission, 1_000_000)
        r2_admission_sha = contract.sha256_file(args.root_admission)
        execution_admissions = prior_state.get("execution_admissions", [])
        initial_ids = (
            set(contract.TL3_IDS[:7])
            | {"source4_lstm192x1", "ffull_s3", "ffull_s4", "frawW_s3ft", "frawW_s4ft"}
        )
        prior_nodes = prior_state.get("nodes", {})
        first_r2_entry = not execution_admissions
        if (
            not isinstance(prior_identity, dict)
            or {key: prior_identity.get(key) for key in stable} != stable
            or prior_identity.get("director_files") != historical_files
            or r1_admission.get("schema") != contract.ADMISSION_SCHEMA
            or r1_admission.get("approved") is not True
            or r1_admission.get("plan_sha256") != contract.PLAN_SHA256
            or r1_admission.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
            or r1_admission.get("director_files") != r1_files
            or r1_admission.get("director_sha256") != r1_files["scripts/final_submit_chain.py"]
            or r2_admission.get("approved") is not True
            or r2_admission.get("plan_sha256") != contract.PLAN_SHA256
            or r2_admission.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
            or (
                first_r2_entry
                and (
                    prior_state.get("phase") != "awaiting_profile_or_dependency"
                    or prior_state.get("scientific_fits_started") != 12
                    or prior_state.get("scientific_fits_completed") != 12
                    or prior_state.get("active_workers")
                    or prior_state.get("active_lanes", 0) != 0
                    or set(prior_nodes) != initial_ids
                    or any(row.get("status") != "complete" for row in prior_nodes.values())
                )
            )
            or (
                not first_r2_entry
                and (
                    len(execution_admissions) != 1
                    or execution_admissions[0].get("schema")
                    != "wnn.final-submit-r2-execution-admission.v1"
                    or execution_admissions[0].get("root_admission_sha256")
                    != r2_admission_sha
                    or execution_admissions[0].get("director_files")
                    != r2_admission.get("director_files")
                )
            )
        ):
            raise ValueError("R1 controller identity or admission differs before R2 continuation")
        return prior_identity
    raise ValueError("R2 continuation requires the existing terminal R1 controller state")


def _node_records(
    plan: dict[str, Any], nodes: dict[str, dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Materialize the fixed graph with explicit native node, paths, and dependency names."""
    result = [
        {
            **node,
            "node_id": node_id,
            "membership_path": str(args.membership),
            "audit_path": str(args.source_audit),
            "parents": list(node.get("parents", ())),
            "native": False,
        }
        for node_id, node in nodes.items()
    ]
    native = plan["optional_source4_lstm"]
    result.insert(
        0,
        {
            "node_id": "source4_lstm192x1",
            "run": native["run"],
            "endpoint": 7,
            "epochs": 8,
            "expected_updates": 7_840,
            "parents": [],
            "export_train": False,
            "membership_path": str(args.membership),
            "audit_path": str(args.source_audit),
            "native": True,
        },
    )
    return result


def _run(
    args: argparse.Namespace,
    plan: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    membership: Any,
    profiles: dict[str, dict[str, Any]],
    admission: dict[str, Any],
    gpu_map: dict[str, str],
) -> dict[str, Any]:
    """Advance only ready, independently profiled nodes; stop on any terminal error."""
    controller_root = args.controller_root
    controller_root.mkdir(parents=True, exist_ok=True)
    state_path = controller_root / "controller-state.json"
    identity = _expected_identity(plan, membership, args)
    state = _state_for(state_path, identity)
    _check_interrupted(state, state_path)
    if len(profiles) > admission["max_profiles"] or args.max_lanes > admission["max_lanes"]:
        raise ValueError("controller exceeds root-admitted profile or lane cap")
    if state["scientific_fits_started"] > contract.TOTAL_FITS:
        raise ValueError("controller state exceeds the 18-fit scientific budget")
    runtime.ensure_source_indexes(args.data_root, membership)
    all_nodes = _node_records(plan, nodes, args)
    state_lock = threading.RLock()

    r2_admission_record = {
        "schema": "wnn.final-submit-r2-execution-admission.v1",
        "root_admission_sha256": contract.sha256_file(args.root_admission),
        "director_files": admission.get("director_files"),
        "started_at_epoch": time.time(),
    }
    execution_admissions = state.setdefault("execution_admissions", [])
    if execution_admissions:
        if execution_admissions != [r2_admission_record] and not (
            len(execution_admissions) == 1
            and execution_admissions[0].get("schema") == r2_admission_record["schema"]
            and execution_admissions[0].get("root_admission_sha256")
            == r2_admission_record["root_admission_sha256"]
            and execution_admissions[0].get("director_files")
            == r2_admission_record["director_files"]
        ):
            raise ValueError("controller state already records another R2 execution admission")
    else:
        execution_admissions.append(r2_admission_record)
        atomic_write_json(state_path, state)

    def save() -> None:
        with state_lock:
            _refresh_counts(state, profiles, int(state.get("active_lanes", 0)))
            atomic_write_json(state_path, state)

    def progress(node_id: str, stage: str, receipt: dict[str, object]) -> None:
        with state_lock:
            record = state["nodes"][node_id]
            record["current_stage"] = stage
            if stage == "fit":
                record["current_receipt"] = str(
                    controller_root / "receipts/fits" / f"{node_id}.json"
                )
            elif stage.startswith("export:"):
                split = stage.partition(":")[2]
                record["current_receipt"] = str(
                    controller_root / "receipts/exports" / f"{node_id}-{split}.json"
                )
            command = receipt.get("command")
            if isinstance(command, list) and all(isinstance(token, str) for token in command):
                record["current_stage_argv"] = command
                record["current_stage_argv_sha256"] = receipt.get("command_sha256")
                record["effective_timeout_seconds"] = receipt.get("timeout_seconds")
                if stage == "fit":
                    record["argv"] = command
                    record["argv_sha256"] = receipt.get("command_sha256")
            child = receipt.get("child")
            if isinstance(child, dict):
                record["child_process"] = child
                state["active_workers"][node_id] = {
                    "node_id": node_id,
                    "stage": stage,
                    "pid": child.get("pid"),
                    "pgid": child.get("pgid"),
                    "birth": child.get("birth"),
                    "gpu_uuid": record.get("gpu_uuid"),
                    "receipt": record.get("current_receipt"),
                }
            save()

    def verify_before_fit(node: dict[str, Any], profile: dict[str, Any]) -> None:
        fresh_plan = contract.read_json(args.plan)
        contract.verify_plan(fresh_plan, args.plan)
        if contract.sha256_file(args.source_capsule) != contract.SOURCE_CAPSULE_SHA256:
            raise ValueError("source capsule changed before fit launch")
        _verify_source_tree(args.source_root, args.source_capsule)
        current_membership = contract.verify_sources(
            data_root=args.data_root,
            membership_path=args.membership,
            audit_path=args.source_audit,
            audit_sha256=contract.SOURCE_AUDIT_SHA256,
        )
        if current_membership != membership:
            raise ValueError("source membership changed before fit launch")
        contract.verify_root_admission(
            admission_path=args.root_admission,
            grader_path=args.grader,
            director_path=Path(__file__),
            helper_path=Path(__file__).with_name("final_submit_chain_contract.py"),
            runtime_helper_path=Path(__file__).with_name("final_submit_chain_runtime.py"),
            execution_helper_path=Path(__file__).with_name("final_submit_chain_execution.py"),
            plan_sha256=contract.PLAN_SHA256,
            source_capsule_sha256=contract.SOURCE_CAPSULE_SHA256,
            now_epoch=time.time(),
        )
        mode = "native_source4_lstm" if node["native"] else "legacy_tl3"
        contract.validate_profile_approval(
            node_id=node["node_id"],
            approval_path=Path(profile["path"]),
            expected_sha256=str(profile["sha256"]),
            plan_sha256=contract.PLAN_SHA256,
            membership=membership,
            mode=mode,
        )
        if (
            node["native"]
            and contract.sha256_file(args.native_config)
            != plan["optional_source4_lstm"]["config_file_sha256"]
        ):
            raise ValueError("native raw config changed before fit launch")

    while True:
        if time.time() >= contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
            state["phase"] = "collection_reserve"
            save()
            break
        if state["scientific_fits_started"] >= contract.TOTAL_FITS:
            break
        cache_record = state.get("cache", {})
        teachers = {name.removeprefix("tl3final_") for name, *_ in contract.W18_TEACHERS}
        exports_ready = all(
            state["nodes"].get(name, {}).get("status") == "complete"
            and set(state["nodes"][name].get("exports", {})) >= {"train", "valid"}
            for name in teachers
        )
        if exports_ready and cache_record.get("status") == "not_started":
            state["cache"] = {"status": "running", "started_at_epoch": time.time()}
            save()
            try:
                _verify_source_tree(args.source_root, args.source_capsule)
                cache_result = execution.build_w18_cache(
                    source_root=args.source_root,
                    data_root=args.data_root,
                    controller_root=controller_root,
                    python=args.python,
                    membership_path=args.membership,
                    audit_path=args.source_audit,
                    membership=membership,
                    output=Path(
                        contract.option_values(
                            nodes["w18final_distilled_s5"]["args"], "--distill-cache-root"
                        )[0]
                    ),
                    progress=lambda receipt: _cache_progress(
                        receipt, state, state_path, state_lock, profiles
                    ),
                )
                state["cache"] = cache_result
            except Exception as error:
                state["cache"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                state["phase"] = "failed"
                state["terminal_outcomes"].append(state["cache"])
                save()
                raise
            save()

        ready: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for node in all_nodes:
            node_id = node["node_id"]
            existing = state["nodes"].get(node_id, {})
            if existing.get("status") == "complete":
                continue
            if node_id not in profiles:
                continue
            if any(
                state["nodes"].get(parent, {}).get("status") != "complete"
                for parent in node["parents"]
            ):
                continue
            if (
                node_id == "w18final_distilled_s5"
                and state.get("cache", {}).get("status") != "complete"
            ):
                continue
            ready.append((node, profiles[node_id]))
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        used_gpus: set[str] = set()
        for item in ready:
            uuid = str(item[1]["gpu_uuid"])
            if uuid not in used_gpus:
                selected.append(item)
                used_gpus.add(uuid)
            if len(selected) >= args.max_lanes:
                break
        if not selected:
            break
        if state["scientific_fits_started"] + len(selected) > contract.TOTAL_FITS:
            raise ValueError("the finite controller would exceed 18 scientific fits")
        state["active_lanes"] = len(selected)
        for node, profile in selected:
            verify_before_fit(node, profile)
        futures = {}
        with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix="finalfit") as pool:
            for node, profile in selected:
                node_id = node["node_id"]
                gpu_uuid = str(profile["gpu_uuid"])
                visible = gpu_map[gpu_uuid]
                fit_argv = execution.build_fit_argv(
                    node=node,
                    python=args.python,
                    source_root=args.source_root,
                    data_root=args.data_root,
                    plan=plan,
                    profile_path=Path(profile["path"]),
                )
                command = [
                    "/usr/bin/timeout",
                    "--signal=TERM",
                    "--kill-after=10s",
                    "21590s",
                    *fit_argv,
                ]
                entry = {
                    "status": "running",
                    "fit_attempted": True,
                    "started_at_epoch": time.time(),
                    "node_id": node_id,
                    "run": node["run"],
                    "gpu_uuid": gpu_uuid,
                    "profile_receipt": str(profile["path"]),
                    "profile_sha256": profile["sha256"],
                    "argv": command,
                    "argv_sha256": contract.canonical_sha256(command),
                    "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
                    "plan_sha256": contract.PLAN_SHA256,
                    "membership_sha256": membership.manifest_sha256,
                    "raw_config_sha256": contract.sha256_file(args.native_config)
                    if node["native"]
                    else None,
                    "current_receipt": str(controller_root / "receipts/fits" / f"{node_id}.json"),
                    "exports": {},
                }
                state["nodes"][node_id] = entry
                state["active_workers"][node_id] = {
                    "node_id": node_id,
                    "stage": "starting",
                    "gpu_uuid": gpu_uuid,
                }
                state["phase"] = "running"
                save()
                trial = all_nodes.index(node) + 1
                futures[
                    pool.submit(
                        execution.execute_node,
                        node=node,
                        plan=plan,
                        membership=membership,
                        source_root=args.source_root,
                        data_root=args.data_root,
                        controller_root=controller_root,
                        python=args.python,
                        gpu_uuid=gpu_uuid,
                        gpu_visible=visible,
                        profile_path=Path(profile["path"]),
                        trial=trial,
                        progress=lambda stage, receipt, name=node_id: progress(
                            name, stage, receipt
                        ),
                    )
                ] = node
            failed = False
            for future in as_completed(futures):
                node = futures[future]
                node_id = node["node_id"]
                try:
                    result = future.result()
                    process_record = result.get("fit_process")
                    if not isinstance(process_record, dict) or process_record.get(
                        "command_sha256"
                    ) != state["nodes"][node_id].get("argv_sha256"):
                        raise ValueError("launched fit argv differs from its prelaunch digest")
                    state["nodes"][node_id].update(result)
                    state["nodes"][node_id]["status"] = "complete"
                except Exception as error:
                    state["nodes"][node_id]["status"] = "failed"
                    state["nodes"][node_id]["error"] = f"{type(error).__name__}: {error}"
                    state["terminal_outcomes"].append(
                        {
                            "node_id": node_id,
                            "status": "failed",
                            "error": state["nodes"][node_id]["error"],
                        }
                    )
                    failed = True
                state["active_workers"].pop(node_id, None)
                save()
        state["active_lanes"] = 0
        save()
        if failed:
            state["phase"] = "failed"
            save()
            break

    complete = (
        state["scientific_fits_completed"] == contract.TOTAL_FITS
        and state.get("cache", {}).get("status") == "complete"
    )
    state["phase"] = "complete" if complete else "awaiting_profile_or_dependency"
    state["finished_at_epoch"] = time.time() if complete else None
    save()
    return state


def _cache_progress(
    receipt: dict[str, object],
    state: dict[str, Any],
    state_path: Path,
    lock: Any,
    profiles: dict[str, dict[str, Any]],
) -> None:
    """Persist W18 cache worker ownership while it runs."""
    with lock:
        child = receipt.get("child")
        state["cache"]["current_process"] = child
        state["cache"]["current_receipt"] = str(state_path.parent / "receipts/cache/w18.json")
        state["active_workers"]["w18-cache"] = {
            "stage": "cache",
            "pid": child.get("pid") if isinstance(child, dict) else None,
            "pgid": child.get("pgid") if isinstance(child, dict) else None,
            "birth": child.get("birth") if isinstance(child, dict) else None,
        }
        state["updated_at_epoch"] = time.time()
        atomic_write_json(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    """Create the preflight and finite execution CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--source-capsule", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--root-admission", type=Path)
    parser.add_argument("--controller-root", type=Path)
    parser.add_argument("--profile-approval", action="append", default=[])
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--max-lanes", type=int, default=8)
    return parser


def main() -> None:
    """Validate fixed inputs or advance approved nodes without retrying terminal attempts."""
    args = build_parser().parse_args()
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
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.grader = args.grader.resolve()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError(f"the requested runtime interpreter is not executable: {args.python}")
    plan, nodes, membership, profiles = _load_inputs(args)
    gpu_map = _gpu_map(args, profiles)
    missing = sorted((set(contract.TL3_IDS) | {"source4_lstm192x1"}) - set(profiles))
    if args.action == "preflight":
        output = {
            "status": "preflight_passed",
            "plan_sha256": contract.PLAN_SHA256,
            "fit_count": contract.TOTAL_FITS,
            "tl3_fit_count": len(nodes),
            "configured_epochs_subtotal": 54,
            "total_configured_epochs": 62,
            "expected_updates_subtotal": 312_000,
            "total_expected_updates": 319_840,
            "membership_sha256": membership.manifest_sha256,
            "fit_members_sha256": membership.fit_members_sha256,
            "profiles_approved": len(profiles),
            "profiles_missing": missing,
            "ready_first_wave": [
                name for name in ("source4_lstm192x1", "fin256_s3") if name in profiles
            ],
            "independent_score_claim": "forbidden",
            "fit_started": False,
        }
        print(json.dumps(output, sort_keys=True, indent=2))
        return
    if args.root_admission is None:
        raise ValueError("--action run requires --root-admission")
    args.root_admission = args.root_admission.resolve()
    contract.verify_root_admission(
        admission_path=args.root_admission,
        grader_path=args.grader,
        director_path=Path(__file__),
        helper_path=Path(__file__).with_name("final_submit_chain_contract.py"),
        runtime_helper_path=Path(__file__).with_name("final_submit_chain_runtime.py"),
        execution_helper_path=Path(__file__).with_name("final_submit_chain_execution.py"),
        plan_sha256=contract.PLAN_SHA256,
        source_capsule_sha256=contract.SOURCE_CAPSULE_SHA256,
        now_epoch=time.time(),
    )
    if not 1 <= args.max_lanes <= contract.MAX_LANES:
        raise ValueError("--max-lanes must be between one and eight")
    with file_lock(args.controller_root / "director.lock", blocking=False):
        state = _run(
            args,
            plan,
            nodes,
            membership,
            profiles,
            contract.read_json(args.root_admission),
            gpu_map,
        )
    print(json.dumps(state, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
