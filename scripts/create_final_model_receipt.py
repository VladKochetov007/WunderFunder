"""Create the create-only receipt for a verified 18-fit final model set."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import final_submit_chain as director
import final_submit_chain_contract as contract
import final_submit_chain_runtime as runtime
import torch

from wnn.final_submit_manifest import TOTAL_SEQUENCES
from wnn.training.final_submit_bridge import final_fit_identity
from wnn.training.final_submit_bridge_cache import validate_final_distillation_cache
from wnn.training.final_submit_bridge_teachers import audit_final_fit_ancestor_closure


def _paths(args: argparse.Namespace) -> None:
    """Resolve the frozen campaign inputs and keep the Python symlink spelling."""
    args.runtime_root = args.runtime_root.resolve()
    defaults = {
        "source_root": args.runtime_root / "source",
        "data_root": args.runtime_root / "data",
        "plan": args.runtime_root / "inputs/final-plan-r2.json",
        "membership": args.runtime_root / "inputs/final-fit-membership-r1.json",
        "source_audit": args.runtime_root / "inputs/SOURCE_VALIDATION_R1.json",
        "source_capsule": args.runtime_root / "inputs/final-submit-source-39d83f739.tar.gz",
        "native_config": args.runtime_root / "inputs/LSTM_CONFIG_R2.json",
        "root_admission": args.runtime_root / "ROOT_EXECUTION_ADMISSION_R1.json",
        "grader": args.runtime_root / "control-grader-r2/scripts/review_final_submit_profile.py",
        "director_script": args.runtime_root / "control-code-r1/scripts/final_submit_chain.py",
        "controller_root": args.runtime_root / "control/final-submit-chain",
    }
    for name, default in defaults.items():
        setattr(args, name, (getattr(args, name) or default).resolve())
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    args.output = args.output.expanduser().absolute()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError(f"runtime interpreter is not executable: {args.python}")
    try:
        args.output.relative_to(args.runtime_root)
        args.output.parent.resolve().relative_to(args.runtime_root)
    except ValueError as error:
        raise ValueError("receipt output must resolve beneath the runtime root") from error
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"create-only final-model receipt exists: {args.output}")


def _packager(args: argparse.Namespace) -> Any:
    """Load the source-pinned packager contract without copying its validators."""
    source_scripts = str(args.source_root / "scripts")
    if source_scripts not in sys.path:
        sys.path.insert(0, source_scripts)
    return importlib.import_module("package_final_tl3")


def _verify_admission(args: argparse.Namespace) -> None:
    """Verify root admission bytes and pins, without applying its new-fit time gate."""
    admission = contract.read_json(args.root_admission, 1_000_000)
    files = {
        "scripts/final_submit_chain.py": contract.sha256_file(args.director_script),
        "scripts/final_submit_chain_contract.py": contract.sha256_file(
            args.director_script.with_name("final_submit_chain_contract.py")
        ),
        "scripts/final_submit_chain_runtime.py": contract.sha256_file(
            args.director_script.with_name("final_submit_chain_runtime.py")
        ),
        "scripts/final_submit_chain_execution.py": contract.sha256_file(
            args.director_script.with_name("final_submit_chain_execution.py")
        ),
    }
    expected = {
        "schema": contract.ADMISSION_SCHEMA,
        "approved": True,
        "plan_sha256": contract.PLAN_SHA256,
        "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
        "grader_sha256": contract.sha256_file(args.grader),
        "director_sha256": files["scripts/final_submit_chain.py"],
        "max_scientific_fits": contract.TOTAL_FITS,
        "max_profiles": contract.MAX_PROFILES,
        "max_fit_wall_seconds": contract.MAX_FIT_SECONDS,
        "max_profile_wall_seconds": contract.MAX_PROFILE_SECONDS,
        "max_lanes": contract.MAX_LANES,
        "hard_deadline_epoch": contract.HARD_DEADLINE,
        "collection_reserve_seconds": contract.COLLECTION_RESERVE,
        "foreign_process_signals": False,
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError("root admission differs from the frozen execution contract")
    if admission.get("director_files") != files:
        raise ValueError("root admission does not bind the frozen director file bytes")


def _verify_repair_history(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """If repairs were adopted, require their exact proofs and unchanged failure history."""
    nodes = state["nodes"]
    repaired = {
        name
        for name in ("source4_lstm192x1", "orgbase")
        if nodes[name].get("operational_repair") is not None
    }
    if not repaired:
        return
    if repaired != {"source4_lstm192x1", "orgbase"}:
        raise ValueError("only the paired native/orgbase admission repairs are allowed")
    events = state.get("qualified_operational_repairs")
    if not isinstance(events, list) or len(events) != 1:
        raise ValueError("controller state lacks one unique repair-adoption event")
    event = events[0]
    before_path = Path(str(event.get("before_state_path", "")))
    before_sha = event.get("before_state_sha256")
    expected_before = (
        args.controller_root
        / "receipts/repairs"
        / f"controller-state.before-repair-{before_sha}.json"
    )
    if (
        event.get("schema") != "wnn.final-submit-operational-repair-adoption.v1"
        or event.get("resolved_node_ids") != ["source4_lstm192x1", "orgbase"]
        or event.get("terminal_outcomes_preserved") is not True
        or before_path != expected_before
        or before_path.is_symlink()
        or not before_path.is_file()
        or contract.sha256_file(before_path) != before_sha
    ):
        raise ValueError("repair-adoption snapshot or ledger is invalid")
    before = contract.read_json(before_path, 64_000_000)
    expected_first = set(contract.TL3_IDS[:7]) | {"source4_lstm192x1"}
    prior_nodes = before.get("nodes")
    failures = (
        {name for name, row in prior_nodes.items() if row.get("status") != "complete"}
        if isinstance(prior_nodes, dict)
        else set()
    )
    if (
        before.get("phase") != "failed"
        or before.get("identity") != state.get("identity")
        or set(prior_nodes or {}) != expected_first
        or failures != {"source4_lstm192x1", "orgbase"}
        or state.get("terminal_outcomes") != before.get("terminal_outcomes")
        or any(
            "not an idle B300" not in str(prior_nodes[name].get("error", "")) for name in failures
        )
    ):
        raise ValueError("repair snapshot does not prove only the two admission failures")
    bindings = event.get("proofs")
    for node_id, filename, schema in (
        ("source4_lstm192x1", "native-admission-repair-r2.json", "wnn.native-admission-repair.v1"),
        (
            "orgbase",
            "orgbase-export-admission-repair-r2.json",
            "wnn.orgbase-export-admission-repair.v1",
        ),
    ):
        path = args.runtime_root / "receipts" / filename
        binding = bindings.get(node_id) if isinstance(bindings, dict) else None
        if (
            not isinstance(binding, dict)
            or binding != nodes[node_id].get("operational_repair")
            or binding.get("path") != str(path)
            or binding.get("sha256") != contract.sha256_file(path)
        ):
            raise ValueError(f"repair proof binding differs for {node_id}")
        proof = contract.read_json(path, 2_000_000)
        if (
            proof.get("schema") != schema
            or proof.get("status") != "complete"
            or proof.get("plan_sha256") != contract.PLAN_SHA256
            or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
            or proof.get("membership_sha256") != state["identity"]["membership_sha256"]
            or proof.get("fit_members_sha256") != state["identity"]["fit_members_sha256"]
            or proof.get("independent_score_claim") != "forbidden"
            or (node_id == "source4_lstm192x1" and proof.get("fixed_completed_epochs") != 8)
            or (node_id == "orgbase" and proof.get("fixed_endpoint") != 3)
        ):
            raise ValueError(f"repair proof has different ancestry or score semantics: {node_id}")


def _completed_graph(
    args: argparse.Namespace,
    plan: dict[str, Any],
    plan_nodes: dict[str, dict[str, Any]],
    membership: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require the exact terminal controller identity and all 18 completed fit records."""
    state = contract.read_json(args.controller_root / "controller-state.json", 64_000_000)
    identity = director._expected_identity(plan, membership, args) | {
        "admission_sha256": contract.sha256_file(args.root_admission)
    }
    expected = set(contract.TL3_IDS) | {"source4_lstm192x1"}
    rows = state.get("nodes")
    if (
        state.get("schema") != "wnn.final-submit-chain-state.v1"
        or state.get("identity") != identity
        or state.get("phase") != "complete"
        or state.get("scientific_fits_started") != contract.TOTAL_FITS
        or state.get("scientific_fits_completed") != contract.TOTAL_FITS
        or state.get("active_workers")
        or state.get("active_lanes", 0) != 0
        or not isinstance(rows, dict)
        or set(rows) != expected
        or any(
            row.get("status") != "complete" or not row.get("fit_attempted") for row in rows.values()
        )
    ):
        raise ValueError("controller state does not prove all 18 fixed fits completed")
    _verify_repair_history(args, state)
    nodes = director._node_records(plan, plan_nodes, args)
    if len(nodes) != contract.TOTAL_FITS or {node["node_id"] for node in nodes} != expected:
        raise ValueError("frozen plan does not materialize the exact 18-node graph")
    for node in nodes:
        row = rows[node["node_id"]]
        if row.get("run") != node["run"] or any(
            rows.get(parent, {}).get("status") != "complete" for parent in node["parents"]
        ):
            raise ValueError(f"node dependency closure is incomplete: {node['node_id']}")
    return state, nodes


def _verify_early_successor_adoption(
    args: argparse.Namespace,
    plan: dict[str, Any],
    plan_nodes: dict[str, dict[str, Any]],
    state: dict[str, Any],
    membership: Any,
) -> None:
    """Bind the four externally executed planned fits to the terminal controller state."""
    node_ids = ("ffull_s3", "ffull_s4", "frawW_s3ft", "frawW_s4ft")
    events = state.get("qualified_early_successor_adoptions")
    if not isinstance(events, list) or len(events) != 1:
        raise ValueError("final controller state lacks the single qualified early-fit adoption")
    event = events[0]
    proofs = event.get("proofs")
    if (
        event.get("schema") != "wnn.final-submit-early-successor-adoption.v1"
        or event.get("resolved_node_ids") != list(node_ids)
        or event.get("terminal_outcomes_preserved") is not True
        or event.get("worker_exit_status") != "unknown_after_detachment"
        or event.get("worker_exit_status_reason")
        != "Popen wrapper did not observe or record child exit"
        or not isinstance(proofs, dict)
        or set(proofs) != set(node_ids)
    ):
        raise ValueError("early-successor adoption event is malformed")
    before_path = Path(str(event.get("before_state_path", "")))
    before_sha = event.get("before_state_sha256")
    if (
        before_path
        != args.controller_root
        / "receipts/repairs"
        / f"controller-state.before-early-successors-{before_sha}.json"
        or before_path.is_symlink()
        or contract.sha256_file(before_path) != before_sha
    ):
        raise ValueError("early-successor adoption snapshot hash or path differs")
    before = contract.read_json(before_path, 64_000_000)
    prior_ids = set(before.get("nodes", {}))
    if (
        before.get("identity") != state.get("identity")
        or before.get("phase") != "awaiting_profile_or_dependency"
        or len(prior_ids) != 8
        or set(node_ids) & prior_ids
        or before.get("terminal_outcomes") != state.get("terminal_outcomes")
        or before.get("qualified_operational_repairs") != state.get("qualified_operational_repairs")
    ):
        raise ValueError("early-successor snapshot does not preserve the repaired first wave")
    panel_path = args.runtime_root / "inputs/early-successor-full-fit-panel-r4.json"
    completion_path = args.runtime_root / "receipts/early-successor-full-fit-panel-r4.complete.json"
    profile_path = args.runtime_root / "receipts/early-successor-profile-panel-r4.approved.json"
    launch_path = args.runtime_root / "receipts/fit-early-successors-r4-driver.launch.json"
    expected_paths = {
        "fit_panel_path": (panel_path, "fit_panel_sha256"),
        "completion_panel_path": (completion_path, "completion_panel_sha256"),
        "profile_panel_path": (profile_path, "profile_panel_sha256"),
        "worker_launch_receipt_path": (launch_path, "worker_launch_receipt_sha256"),
    }
    for path_field, (expected_path, sha_field) in expected_paths.items():
        actual_path = Path(str(event.get(path_field, "")))
        if (
            actual_path != expected_path
            or actual_path.is_symlink()
            or contract.sha256_file(actual_path) != event.get(sha_field)
        ):
            raise ValueError(f"early-successor adoption {path_field} binding differs")
    fit_panel = contract.read_json(panel_path, 4_000_000)
    completion = contract.read_json(completion_path, 4_000_000)
    profile_panel = contract.read_json(profile_path, 1_000_000)
    fit_panel_sha = contract.sha256_file(panel_path)
    if (
        fit_panel.get("schema") != "wnn.final-early-successor-fit-panel.v1"
        or fit_panel.get("plan_sha256") != contract.PLAN_SHA256
        or fit_panel.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or fit_panel.get("membership_sha256") != membership.manifest_sha256
        or fit_panel.get("profile_panel_sha256") != event.get("profile_panel_sha256")
        or fit_panel.get("max_new_scientific_fits") != 4
        or fit_panel.get("total_unique_fit_roles_cap") != contract.TOTAL_FITS
        or fit_panel.get("write_shared_controller_state") is not False
        or fit_panel.get("foreign_signals") is not False
        or completion.get("schema") != "wnn.final-early-successor-panel-completion.v1"
        or completion.get("status") != "complete"
        or completion.get("plan_sha256") != contract.PLAN_SHA256
        or completion.get("panel_path") != str(panel_path)
        or completion.get("panel_sha256") != fit_panel_sha
    ):
        raise ValueError("early-successor fit or completion panel identity differs")
    completion_items = completion.get("items")
    fit_items = fit_panel.get("items")
    if (
        not isinstance(completion_items, list)
        or len(completion_items) != 4
        or not isinstance(fit_items, list)
        or {item.get("node_id") for item in completion_items if isinstance(item, dict)}
        != set(node_ids)
        or {item.get("node_id") for item in fit_items if isinstance(item, dict)} != set(node_ids)
    ):
        raise ValueError("early-successor proof panel does not name the exact four planned fits")
    completion_by_id = {item["node_id"]: item for item in completion_items}
    fit_by_id = {item["node_id"]: item for item in fit_items}
    approved_profiles = profile_panel.get("approved")
    if not isinstance(approved_profiles, list) or len(approved_profiles) != 4:
        raise ValueError("early-successor profile panel is malformed")
    profile_by_id = {
        item.get("node_id"): item for item in approved_profiles if isinstance(item, dict)
    }
    if set(profile_by_id) != set(node_ids):
        raise ValueError("early-successor profile panel node set differs")
    for node_id in node_ids:
        binding = proofs[node_id]
        proof_path = Path(str(binding.get("path", "")))
        expected_path = args.runtime_root / "receipts" / f"early-successor-{node_id}.complete.json"
        row = state["nodes"].get(node_id)
        if (
            proof_path != expected_path
            or proof_path.is_symlink()
            or contract.sha256_file(proof_path) != binding.get("sha256")
            or completion_by_id[node_id].get("proof_path") != str(proof_path)
            or completion_by_id[node_id].get("proof_sha256") != binding.get("sha256")
            or not isinstance(row, dict)
            or row.get("status") != "complete"
            or row.get("fit_attempted") is not True
        ):
            raise ValueError(f"early-successor proof/state binding differs: {node_id}")
        proof = contract.read_json(proof_path, 8_000_000)
        result = proof.get("result")
        planned = plan_nodes.get(node_id)
        if not isinstance(planned, dict):
            raise ValueError(f"early-successor node is absent from frozen plan: {node_id}")
        profile = profile_by_id[node_id]
        approval_path = Path(str(profile.get("path", "")))
        request_path = Path(str(profile.get("request_path", "")))
        approval = contract.validate_profile_approval(
            node_id=node_id,
            approval_path=approval_path,
            expected_sha256=str(profile.get("sha256", "")),
            plan_sha256=contract.PLAN_SHA256,
            membership=membership,
            mode="legacy_tl3",
        )
        request_sha = contract.sha256_file(request_path)
        request = contract.read_json(request_path, 1_000_000)
        review = contract.read_json(approval_path.parent / "profile_review.json", 1_000_000)
        expected_parents = {}
        for parent_id in planned.get("parents", []):
            parent_row = state["nodes"].get(parent_id)
            if not isinstance(parent_row, dict) or parent_row.get("status") != "complete":
                raise ValueError(f"early-successor parent is incomplete: {node_id} <- {parent_id}")
            expected_parents[parent_id] = {
                "checkpoint_path": parent_row.get("checkpoint_path"),
                "checkpoint_sha256": parent_row.get("checkpoint_sha256"),
            }
        if (
            planned is None
            or proof.get("schema") != "wnn.final-early-successor-completion.v1"
            or proof.get("status") != "complete"
            or proof.get("node_id") != node_id
            or proof.get("run") != planned["run"]
            or proof.get("plan_sha256") != contract.PLAN_SHA256
            or proof.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
            or proof.get("membership_sha256") != membership.manifest_sha256
            or proof.get("fit_members_sha256") != membership.fit_members_sha256
            or proof.get("panel_path") != str(panel_path)
            or proof.get("panel_sha256") != fit_panel_sha
            or proof.get("trial") != contract.TL3_IDS.index(node_id) + 2
            or proof.get("fixed_endpoint") != planned["endpoint"]
            or proof.get("parents") != expected_parents
            or proof.get("independent_score_claim") != "forbidden"
            or proof.get("profile_path") != profile.get("path")
            or proof.get("profile_sha256") != profile.get("sha256")
            or proof.get("request_path") != profile.get("request_path")
            or proof.get("request_sha256") != profile.get("request_sha256")
            or request_sha != profile.get("request_sha256")
            or request.get("schema") != "wnn.final-submit-profile-request.v1"
            or request.get("node_id") != node_id
            or request.get("plan_sha256") != contract.PLAN_SHA256
            or request.get("mode") != "legacy_tl3"
            or request.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
            or request.get("final_fit_manifest_sha256") != membership.manifest_sha256
            or request.get("source_audit_sha256") != contract.SOURCE_AUDIT_SHA256
            or request.get("fit_members_sha256") != membership.fit_members_sha256
            or request.get("fit_member_count") != contract.FIT_MEMBERS
            or request.get("evaluation_member_count") != 0
            or request.get("no_eval_during_fit") is not True
            or request.get("independent_score_claim") != "forbidden"
            or request.get("gpu_uuid") != profile.get("gpu_uuid")
            or request.get("recipe_argv") != planned["args"]
            or request.get("recipe_sha256") != contract.canonical_sha256(planned["args"])
            or approval.get("request_sha256") != request_sha
            or review.get("evidence_sha256", {}).get("request") != request_sha
            or proof.get("gpu_uuid") != profile.get("gpu_uuid")
            or fit_by_id[node_id].get("run") != planned["run"]
            or fit_by_id[node_id].get("args") != planned["args"]
            or row.get("run") != proof.get("run")
            or row.get("gpu_uuid") != proof.get("gpu_uuid")
            or row.get("profile_receipt") != proof.get("profile_path")
            or row.get("profile_sha256") != proof.get("profile_sha256")
            or row.get("profile_request_sha256") != proof.get("request_sha256")
            or row.get("trial_number") != proof.get("trial")
            or not isinstance(result, dict)
            or result.get("status") != "complete"
            or result.get("checkpoint_path") != row.get("checkpoint_path")
            or result.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or result.get("fit_process_receipt") != row.get("fit_process_receipt")
            or result.get("fit_process") != row.get("fit_process")
            or result.get("exports") != row.get("exports")
        ):
            raise ValueError(f"early-successor proof differs from adopted node: {node_id}")


def _validate_fits(
    args: argparse.Namespace,
    plan: dict[str, Any],
    nodes: list[dict[str, Any]],
    state: dict[str, Any],
    membership: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Validate all fixed checkpoints, actual exports, and final-fit ancestor closures."""
    checkpoint_by_run: dict[str, dict[str, Any]] = {}
    closure_by_run: dict[str, list[dict[str, Any]]] = {}
    allowed_runs = {node["run"] for node in nodes}
    for index, node in enumerate(nodes):
        node_id, run = node["node_id"], node["run"]
        row = state["nodes"][node_id]
        root = args.data_root / "artifacts/runs" / run
        expected_path = root / (
            "model.pt" if node.get("native") else f"model_ep{node['endpoint']}.pt"
        )
        if expected_path.is_symlink() or not expected_path.is_file():
            raise ValueError(f"fixed checkpoint is missing or linked: {node_id}")
        checkpoint = runtime.validate_fit_checkpoint(
            node=node, output_dir=root, membership=membership
        )
        digest = contract.sha256_file(checkpoint)
        if row.get("checkpoint_path") != str(checkpoint) or row.get("checkpoint_sha256") != digest:
            raise ValueError(f"controller checkpoint hash differs: {node_id}")
        process_path = Path(str(row.get("fit_process_receipt", "")))
        repaired_native = node["node_id"] == "source4_lstm192x1" and bool(
            row.get("operational_repair")
        )
        expected_receipt = (
            args.controller_root / "receipts/fits/source4_lstm192x1-external-r2.json"
            if repaired_native
            else args.controller_root / "receipts/fits" / f"{node['node_id']}.json"
        )
        expected_trial = 100 if repaired_native else index + 1
        expected_mode = (
            "final_submit_native_qualified_repair_r2"
            if repaired_native
            else "final_submit_finite_chain"
        )
        process = contract.read_json(process_path, 2_000_000)
        if (
            process_path != expected_receipt
            or process != row.get("fit_process")
            or process.get("status") != "completed"
            or process.get("exit_code") != 0
            or process.get("timed_out") is not False
            or process.get("mode") != expected_mode
            or process.get("stage") != 1
            or process.get("trial_number") != expected_trial
            or process.get("gpu_key") != row.get("gpu_uuid")
            or process.get("output_dir") != str(root.resolve())
            or process.get("command_sha256") != contract.canonical_sha256(process.get("command"))
        ):
            raise ValueError(f"fit process receipt is not complete: {node_id}")
        if node.get("export_train") is True:
            exports = row.get("exports")
            if not isinstance(exports, dict) or set(exports) != {"train", "valid"}:
                raise ValueError(f"teacher lacks source-specific exports: {node_id}")
            for split in ("train", "valid"):
                export = exports[split]
                path = runtime.validate_export(args.data_root, run, split, membership)
                if (
                    export.get("status") != "complete"
                    or export.get("path") != str(path)
                    or export.get("sha256") != contract.sha256_file(path)
                    or export.get("independent_score_claim") != "forbidden"
                ):
                    raise ValueError(
                        f"teacher export differs from source membership: {node_id}/{split}"
                    )
        elif row.get("exports") != {}:
            raise ValueError(f"non-teacher node has unexpected exports: {node_id}")
        if not node.get("native"):
            closure = audit_final_fit_ancestor_closure(
                run, checkpoint.name, membership, args.data_root
            )
            if not closure or any(
                parent.get("run") not in allowed_runs
                or parent.get("final_fit_manifest_sha256") != membership.manifest_sha256
                or parent.get("final_fit_members_sha256") != membership.fit_members_sha256
                or parent.get("source_validation_sha256") != membership.source_validation_sha256
                for parent in closure
            ):
                raise ValueError(f"fit ancestor closure escapes the final-fit graph: {node_id}")
            closure_by_run[run] = closure
        checkpoint_by_run[run] = {"node": node, "path": checkpoint, "sha256": digest}
    return checkpoint_by_run, closure_by_run


def _validate_cache(
    args: argparse.Namespace,
    plan: dict[str, Any],
    state: dict[str, Any],
    membership: Any,
    checkpoints: dict[str, dict[str, Any]],
    closures: dict[str, list[dict[str, Any]]],
) -> None:
    """Validate W18 cache payloads and bind all six teachers to actual checkpoints."""
    node = next(item for item in plan["nodes"] if item["id"] == "w18final_distilled_s5")
    values = contract.option_values(node["args"], "--distill-cache-root")
    root = args.data_root / "artifacts/runs/final-fit-w18-mixed-targets"
    cache = state.get("cache")
    if (
        len(values) != 1
        or Path(values[0]).resolve() != root.resolve()
        or root.is_symlink()
        or not isinstance(cache, dict)
        or cache.get("status") != "complete"
        or Path(str(cache.get("path", ""))) != root
    ):
        raise ValueError("W18 cache is missing or differs from the frozen output path")
    verified = validate_final_distillation_cache(root, membership, "source", require_families=True)
    metadata = verified["metadata"]
    if (root / "metadata.json").is_symlink() or any(
        (root / str(entry.get("path", ""))).is_symlink()
        for entry in metadata.get("files", {}).values()
    ):
        raise ValueError("W18 cache metadata or payload includes a symlink")
    if cache.get("cache_sha256") != verified["metadata_sha256"]:
        raise ValueError("W18 cache metadata hash differs from the completed controller record")
    receipt_path = args.controller_root / "receipts/cache/w18.json"
    process = contract.read_json(receipt_path, 2_000_000)
    if (
        cache.get("receipt") != str(receipt_path)
        or cache.get("process") != process
        or process.get("status") != "completed"
        or process.get("exit_code") != 0
        or process.get("timed_out") is not False
        or process.get("mode") != "final_submit_finite_chain"
        or process.get("stage") != 3
        or process.get("trial_number") != 1
        or process.get("gpu_key") != "cpu-cache"
        or process.get("output_dir") != str(root.resolve())
    ):
        raise ValueError("W18 cache process receipt differs from the completed cache state")
    sources = metadata.get("sources")
    if not isinstance(sources, list) or len(sources) != len(contract.W18_TEACHERS):
        raise ValueError("W18 cache must contain exactly its six frozen teacher sources")
    for source, (run, family, coefficient, epoch) in zip(
        sources, contract.W18_TEACHERS, strict=True
    ):
        fitted = checkpoints.get(run)
        if (
            fitted is None
            or source.get("run") != run
            or source.get("family") != family
            or source.get("coefficient") != coefficient
            or source.get("epoch") != epoch
            or source.get("checkpoint_sha256") != fitted["sha256"]
            or fitted["node"]["endpoint"] != epoch
            or source.get("ancestor_closure") != closures.get(run)
        ):
            raise ValueError(f"W18 cache teacher does not match its completed fit: {run}")


def _receipt(
    package: Any, membership: Any, checkpoints: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build and validate the package contract's four serving-member identities."""
    members: dict[str, dict[str, str]] = {}
    for role, run in zip(package.ROLES, package.FINAL_RUNS, strict=True):
        fitted = checkpoints.get(run)
        if fitted is None:
            raise ValueError(f"serving checkpoint is absent: {role}")
        checkpoint = torch.load(fitted["path"], map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"serving checkpoint is malformed: {role}")
        config, _ = package._source_config(checkpoint, role)
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"serving checkpoint provenance is malformed: {role}")
        member: dict[str, str] = {
            "run": run,
            "checkpoint_sha256": fitted["sha256"],
            "configuration_sha256": package.canonical_sha256(config),
            "provenance_sha256": package.canonical_sha256(provenance),
        }
        if role == "source4_lstm":
            data = checkpoint.get("data")
            if not isinstance(data, dict):
                raise ValueError("Source4 data provenance is malformed")
            member["data_sha256"] = package.canonical_sha256(data)
        package._validate_final_checkpoint(role, checkpoint, member, membership)
        members[role] = member
    return {
        "schema": package.FINAL_RECEIPT_SCHEMA,
        "final_fit": final_fit_identity(membership),
        "fit_sequences": TOTAL_SEQUENCES,
        "independent_evaluation_members": [],
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "fixed_epochs": package.FINAL_FIXED_EPOCHS,
        "members": members,
    }


def _write(path: Path, value: dict[str, Any], package: Any) -> str:
    """Validate the exact receipt bytes and create the output without overwrite."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    package.validate_final_model_receipt(value, digest, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    package.load_final_model_receipt(path, digest)
    return digest


def build_parser() -> argparse.ArgumentParser:
    """Create the create-only receipt builder CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--source-capsule", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--root-admission", type=Path)
    parser.add_argument("--grader", type=Path)
    parser.add_argument("--director-script", type=Path)
    parser.add_argument("--controller-root", type=Path)
    return parser


def main() -> None:
    """Verify the complete fit graph and create one final-model receipt."""
    args = build_parser().parse_args()
    _paths(args)
    package = _packager(args)
    plan = contract.read_json(args.plan, 4_000_000)
    plan_nodes = contract.verify_plan(plan, args.plan)
    if contract.sha256_file(args.source_capsule) != contract.SOURCE_CAPSULE_SHA256:
        raise ValueError("source capsule differs from the pinned final-fit source")
    director._verify_source_tree(args.source_root, args.source_capsule)
    if (
        contract.sha256_file(args.native_config)
        != plan["optional_source4_lstm"]["config_file_sha256"]
    ):
        raise ValueError("native config differs from the final-fit plan")
    membership = contract.verify_sources(
        data_root=args.data_root,
        membership_path=args.membership,
        audit_path=args.source_audit,
        audit_sha256=contract.SOURCE_AUDIT_SHA256,
    )
    _verify_admission(args)
    state, nodes = _completed_graph(args, plan, plan_nodes, membership)
    _verify_early_successor_adoption(args, plan, plan_nodes, state, membership)
    checkpoints, closures = _validate_fits(args, plan, nodes, state, membership)
    _validate_cache(args, plan, state, membership, checkpoints, closures)
    receipt = _receipt(package, membership, checkpoints)
    digest = _write(args.output, receipt, package)
    print(
        json.dumps(
            {
                "status": "final_model_receipt_created",
                "schema": receipt["schema"],
                "path": str(args.output),
                "sha256": digest,
                "roles": sorted(receipt["members"]),
                "fit_sequences": receipt["fit_sequences"],
                "independent_evaluation_sequences": 0,
                "independent_score_claim": "forbidden",
                "scientific_fits_completed": contract.TOTAL_FITS,
                "training_or_cuda_started": False,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
