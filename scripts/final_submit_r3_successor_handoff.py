"""Run and adopt only the three fixed successor fits around the live R2 wave barrier."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any  # The pinned runtime reads dynamically typed JSON receipts.

SUCCESSORS = ("fttall_tlead_s5", "fttall_tlead_s6", "w18final_distilled_s5")
TLEADS = frozenset(SUCCESSORS[:2])
PROFILE_ROW_SCHEMA = "wnn.final-acceleration-profile-approval.v1"
PROFILE_GPU = {
    "fttall_tlead_s5": "GPU-76951d88-add3-0d6b-5bee-c0390915ca47",
    "fttall_tlead_s6": "GPU-e6d65cb1-aba4-14f5-d4e5-bf24ef1ee3d8",
    "w18final_distilled_s5": "GPU-1fc4d921-dc78-6711-c2a8-ec0589b7f106",
}


def _sha(path: Path) -> str:
    """Hash a regular file and reject symlinks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    """Create immutable sidecar evidence; an existing path means no retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _proc(pid: int) -> tuple[str, int, str] | None:
    """Read state, process group and start ticks for one Linux process."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"malformed proc stat for PID {pid}")
    return fields[0], int(fields[2]), fields[19]


def _cmdline(pid: int) -> list[str] | None:
    """Read one live Linux process command line without shell reconstruction."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [os.fsdecode(value) for value in raw.split(b"\0") if value]


def _children(pid: int) -> set[int]:
    """Read direct Linux child PIDs for an owned timeout wrapper."""
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return set()
    return {int(value) for value in raw.split()}


def _group_alive(pgid: int) -> bool:
    """Check whether any Linux process still belongs to a recorded stage group."""
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        if len(fields) > 2 and fields[2].isdecimal() and int(fields[2]) == pgid:
            return True
    return False


def _stopped(identity: dict[str, Any]) -> None:
    current = _proc(int(identity["pid"]))
    if (
        current is None
        or current[0] not in {"T", "t"}
        or current[1] != identity["pgid"]
        or current[2] != str(identity["birth"])
    ):
        raise RuntimeError("the exact outer campaign process is not stopped")


def _load(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load the exact R2 source capsule and return its pinned helpers."""
    source = args.root / "source"
    scripts = args.root / "control-code-r2/scripts"
    sys.path[:0] = [str(scripts), str(source / "src"), str(source)]
    import final_submit_chain as director
    import final_submit_chain_contract as contract
    import final_submit_chain_execution as execution
    import final_submit_chain_runtime as runtime

    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.training.final_submit_bridge_teachers import audit_final_fit_ancestor_closure

    return (
        director,
        contract,
        execution,
        runtime,
        atomic_write_json,
        file_lock,
        audit_final_fit_ancestor_closure,
    )


def _approval(
    args: argparse.Namespace,
    contract: Any,
    membership: Any,
    node: dict[str, Any],
    expected_gpu: str,
) -> dict[str, Any]:
    """Validate one root-approved R3 profile row, receipt, request and review."""
    node_id = str(node["node_id"])
    row_path = args.root / "receipts" / f"acceleration-profile-{node_id}-r1.approved.json"
    row = contract.read_json(row_path, 256_000)
    key = f"legacy-{node_id}-r3"
    approval_path = args.root / "profile-reviews" / f"{key}-review2/approved_profile_receipt.json"
    request_path = args.root / "inputs/r3" / f"{key}.request.json"
    review_path = approval_path.parent / "profile_review.json"
    expected = {
        "schema": PROFILE_ROW_SCHEMA,
        "node_id": node_id,
        "run": node["run"],
        "profile_path": str(approval_path),
        "request_path": str(request_path),
        "review_path": str(review_path),
        "gpu_uuid": expected_gpu,
        "plan_sha256": contract.PLAN_SHA256,
        "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
        "membership_sha256": membership.manifest_sha256,
        "fit_members_sha256": membership.fit_members_sha256,
        "source_validation_sha256": contract.SOURCE_AUDIT_SHA256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"R3 profile row differs for {node_id}")
    profile_sha, request_sha, review_sha = (
        row.get("profile_sha256"),
        row.get("request_sha256"),
        row.get("review_sha256"),
    )
    if (
        not all(isinstance(value, str) for value in (profile_sha, request_sha, review_sha))
        or _sha(approval_path) != profile_sha
        or _sha(request_path) != request_sha
        or _sha(review_path) != review_sha
    ):
        raise ValueError(f"R3 profile evidence hash differs for {node_id}")
    approval = contract.validate_profile_approval(
        node_id=node_id,
        approval_path=approval_path,
        expected_sha256=profile_sha,
        plan_sha256=contract.PLAN_SHA256,
        membership=membership,
        mode="legacy_tl3",
    )
    request, review = contract.read_json(request_path), contract.read_json(review_path)
    if (
        approval.get("request_sha256") != request_sha
        or review.get("evidence_sha256", {}).get("request") != request_sha
        or request.get("node_id") != node_id
        or request.get("recipe_argv") != node["args"]
        or request.get("recipe_sha256") != contract.canonical_sha256(node["args"])
        or request.get("plan_sha256") != contract.PLAN_SHA256
        or request.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or request.get("final_fit_manifest_sha256") != membership.manifest_sha256
        or request.get("source_audit_sha256") != contract.SOURCE_AUDIT_SHA256
        or request.get("fit_members_sha256") != membership.fit_members_sha256
        or request.get("fit_member_count") != contract.FIT_MEMBERS
        or request.get("evaluation_member_count") != 0
        or request.get("no_eval_during_fit") is not True
        or request.get("independent_score_claim") != "forbidden"
        or request.get("gpu_uuid") != row.get("gpu_uuid")
    ):
        raise ValueError(f"R3 profile request differs from the frozen recipe: {node_id}")
    return {
        "path": approval_path,
        "sha256": profile_sha,
        "request_path": request_path,
        "request_sha256": request_sha,
        "gpu_uuid": row["gpu_uuid"],
        "row_path": row_path,
        "row_sha256": _sha(row_path),
    }


def _expected_profile_paths(args: argparse.Namespace, node: dict[str, Any]) -> dict[str, Any]:
    """Pin the independent profile worker's paths/device without waiting for its row."""
    node_id = str(node["node_id"])
    key = f"legacy-{node_id}-r3"
    approval = args.root / "profile-reviews" / f"{key}-review2/approved_profile_receipt.json"
    return {
        "row_path": args.root / "receipts" / f"acceleration-profile-{node_id}-r1.approved.json",
        "profile_path": approval,
        "request_path": args.root / "inputs/r3" / f"{key}.request.json",
        "review_path": approval.parent / "profile_review.json",
        "gpu_uuid": PROFILE_GPU[node_id],
    }


def _wait_for_approval(
    args: argparse.Namespace,
    contract: Any,
    membership: Any,
    node: dict[str, Any],
    outer: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Wait only for this role's profile row, then validate its immutable evidence."""
    row_path = expected["row_path"]
    while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
        _stopped(outer)
        if row_path.exists():
            try:
                return _approval(args, contract, membership, node, str(expected["gpu_uuid"]))
            except json.JSONDecodeError:
                # The profile worker writes its final row directly; ignore only a partial JSON.
                time.sleep(1)
                continue
        time.sleep(2)
    raise TimeoutError(f"profile approval did not arrive before the fit reserve: {node['node_id']}")


def _owned_receipt(
    contract: Any,
    path: Path,
    process: dict[str, Any],
    argv: list[str],
    stage: int,
    trial: int,
    gpu: str,
    output: Path,
    cwd: Path,
) -> None:
    actual = contract.read_json(path, 2_000_000)
    timeout = actual.get("timeout_seconds")
    if (
        actual != process
        or actual.get("status") != "completed"
        or actual.get("exit_code") != 0
        or actual.get("timed_out") is not False
        or actual.get("mode") != "final_submit_finite_chain"
        or actual.get("stage") != stage
        or actual.get("trial_number") != trial
        or actual.get("gpu_key") != gpu
        or actual.get("output_dir") != str(output.resolve())
        or actual.get("cwd") != str(cwd.resolve())
        or type(timeout) is not int
        or actual.get("command", [])[3:4] != [f"{timeout - 10}s"]
        or actual["command"][:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
        or actual["command"][4:] != argv
        or actual.get("command_sha256") != contract.canonical_sha256(actual["command"])
    ):
        raise ValueError(f"owned process receipt differs: {path}")


def _validate_pause(
    args: argparse.Namespace,
    contract: Any,
    identity: dict[str, object],
    expected_existing: set[str],
    wave_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the root pause proof, outer campaign owner, and unchanged R2 launch."""
    admission = contract.read_json(args.pause_admission, 1_000_000)
    proof_path = Path(str(admission.get("pause_proof_path", "")))
    expected_proof = args.root / "receipts/acceleration-supervisor-pause-r1.json"
    if (
        admission.get("schema") != "wnn.acceleration-profile-root-admission.v1"
        or admission.get("approved") is not True
        or admission.get("plan_sha256") != contract.PLAN_SHA256
        or admission.get("source_capsule_sha256") != contract.SOURCE_CAPSULE_SHA256
        or proof_path != expected_proof
        or not proof_path.is_file()
        or proof_path.is_symlink()
        or _sha(proof_path) != admission.get("pause_proof_sha256")
    ):
        raise ValueError("root acceleration pause admission is missing or mismatched")
    proof = contract.read_json(proof_path, 2_000_000)
    outer = proof.get("outer_identity")
    resume_before = proof.get("required_resume_before_epoch")
    if (
        proof.get("schema") != "wnn.final-acceleration-supervisor-pause.v1"
        or proof.get("status") != "outer_python_paused"
        or proof.get("outer_identity") != admission.get("outer_identity")
        or proof.get("plan_sha256", contract.PLAN_SHA256) != contract.PLAN_SHA256
        or proof.get("source_capsule_sha256", contract.SOURCE_CAPSULE_SHA256)
        != contract.SOURCE_CAPSULE_SHA256
        or proof.get("expected_admitted_node_ids") != sorted(expected_existing)
        or proof.get("transferred_node_ids") != list(SUCCESSORS)
        or proof.get("foreign_process_signals") is not False
        or proof.get("fit_process_signals") is not False
        or proof.get("hard_deadline_epoch") != contract.HARD_DEADLINE
        or proof.get("latest_gpu_stage_end_epoch")
        != contract.HARD_DEADLINE - contract.COLLECTION_RESERVE
        or type(resume_before) not in {int, float}
        or not math.isfinite(resume_before)
        or resume_before != contract.HARD_DEADLINE - 1200
        or time.time() >= resume_before
        or not isinstance(outer, dict)
        or admission.get("max_transferred_scientific_fits") != 3
        or admission.get("max_global_scientific_fits") != contract.TOTAL_FITS
        or admission.get("max_new_profiles") != 3
        or admission.get("max_global_profiles") != contract.MAX_PROFILES
        or admission.get("hard_deadline_epoch") != contract.HARD_DEADLINE
    ):
        raise ValueError("root pause proof does not bind this fixed three-role transfer")
    _stopped(outer)

    launch_path = Path(str(proof.get("outer_launch_receipt_path", "")))
    expected_launch = args.root / "receipts/campaign-r8-driver.launch.json"
    launch = contract.read_json(launch_path, 2_000_000)
    if launch_path != expected_launch or _sha(launch_path) != proof.get(
        "outer_launch_receipt_sha256"
    ):
        raise ValueError("outer campaign launch receipt changed")
    command = launch.get("command")
    outer_script = Path(str(proof.get("outer_script_path", "")))
    if (
        outer_script != args.root / "control-campaign-r8/scripts/final_submit_campaign.py"
        or _sha(outer_script) != proof.get("outer_script_sha256")
        or proof.get("outer_script_sha256")
        != "27eea8a69ae228d473303e90a2266b0897fcf4d3088f3ae2de81a5a7e2541297"
        or not isinstance(command, list)
        or len(command) < 9
        or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
        or not str(command[3]).endswith("s")
        or command[4] != str(args.python)
        or command[5:7] != ["-u", str(outer_script)]
    ):
        raise ValueError("outer launch command is not the pinned R8 campaign")
    wrapper_pid = launch.get("pid")
    wrapper_pgid = launch.get("pgid")
    wrapper_birth = launch.get("start_ticks", launch.get("birth"))
    if (
        type(wrapper_pid) is not int
        or type(wrapper_pgid) is not int
        or not isinstance(wrapper_birth, str)
        or wrapper_pid != wrapper_pgid
        or outer.get("pgid") != wrapper_pgid
        or outer.get("pid") == wrapper_pid
        or _cmdline(wrapper_pid) != command
        or contract.canonical_sha256(_cmdline(int(outer["pid"])))
        != proof.get("outer_command_sha256")
        or _cmdline(int(outer["pid"])) != command[4:]
    ):
        raise ValueError("outer Python command line differs from its launch receipt")
    outer_argv = command[4:]
    required_options = {
        "--action": "run",
        "--runtime-root": str(args.root),
        "--source-root": str(args.source),
        "--data-root": str(args.data),
        "--plan": str(args.plan_path),
        "--membership": str(args.membership_path),
        "--source-audit": str(args.audit_path),
        "--source-capsule": str(args.capsule),
        "--native-config": str(args.native_config),
        "--python": str(args.python),
        "--root-admission": str(args.r1_admission),
        "--r2-root-admission": str(args.r2_admission),
        "--grader": str(args.grader),
        "--director-script": str(args.root / "control-code-r1/scripts/final_submit_chain.py"),
        "--r2-director-script": str(args.r2_script),
        "--controller-root": str(args.controller),
    }
    if any(
        contract.option_values(outer_argv, key) != [value]
        for key, value in required_options.items()
    ):
        raise ValueError("R8 campaign arguments do not pin the approved plan and control roots")
    deadline = launch.get("created_epoch", 0) + int(str(command[3])[:-1])
    if abs(deadline - contract.HARD_DEADLINE) >= 2 or deadline <= contract.HARD_DEADLINE - 1200:
        raise ValueError("outer campaign timeout does not preserve the frozen hard deadline")
    wrapper_proc = _proc(wrapper_pid)
    if (
        wrapper_proc is None
        or wrapper_proc[1:] != (wrapper_pgid, wrapper_birth)
        or wrapper_proc[0] in {"Z", "T", "t"}
    ):
        raise ValueError("outer GNU timeout wrapper identity is not live")

    before = proof.get("before_processes")
    after = proof.get("after_processes")
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
        raise ValueError("pause proof omits matching before/after process snapshots")
    required = {str(int(outer["pid"])), str(wrapper_pid)}
    r2 = proof.get("r2_identity")
    r2_python = proof.get("r2_python_identity")
    wave = contract.read_json(wave_path, 2_000_000)
    child = wave.get("child")
    if (
        not isinstance(r2, dict)
        or not isinstance(r2_python, dict)
        or not isinstance(child, dict)
        or r2 != {key: child.get(key) for key in ("pid", "pgid", "birth")}
        or proof.get("r2_wave_receipt_path") != str(wave_path)
        or proof.get("r2_wave_receipt_sha256") != _sha(wave_path)
        or proof.get("r2_command_sha256") != wave.get("command_sha256")
    ):
        raise ValueError("pause proof R2 process binding differs from its owned receipt")
    required.update((str(r2["pid"]), str(r2_python["pid"])))
    for row in after.values():
        if (
            isinstance(row, dict)
            and type(row.get("pid")) is int
            and row["pgid"] not in {outer["pgid"], r2["pgid"]}
        ):
            required.add(str(row["pid"]))
    if not required.issubset(before) or not required.issubset(after):
        raise ValueError("pause proof omits an owned outer/R2/fit process identity")
    for pid_key in before:
        old, new = before[pid_key], after[pid_key]
        if (
            not isinstance(old, dict)
            or not isinstance(new, dict)
            or any(
                old.get(key) != new.get(key) for key in ("pid", "pgid", "birth", "command_sha256")
            )
            or str(old.get("pid")) != pid_key
            or old.get("state") in {"T", "t", "Z"}
            or (new.get("state") in {"T", "t"}) != (pid_key == str(outer["pid"]))
        ):
            raise ValueError("pause proof shows a changed or incorrectly stopped process")
        live = _proc(int(pid_key))
        if live is not None and (
            live[1] != new.get("pgid")
            or live[2] != str(new.get("birth"))
            or contract.canonical_sha256(_cmdline(int(pid_key))) != new.get("command_sha256")
            or live[0] in {"Z"}
            or (pid_key == str(outer["pid"]) and live[0] not in {"T", "t"})
            or (pid_key != str(outer["pid"]) and live[0] in {"T", "t"})
        ):
            raise ValueError("live process state no longer matches the pause proof")
    r2_python_proc = _proc(int(r2_python.get("pid", -1))) if isinstance(r2_python, dict) else None
    if (
        r2["pgid"] == outer["pgid"]
        or r2_python["pgid"] != r2["pgid"]
        or r2_python_proc is None
        or r2_python_proc[0] in {"Z", "T", "t"}
        or r2_python_proc[1] != r2_python.get("pgid")
        or r2_python_proc[2] != str(r2_python.get("birth"))
        or _cmdline(int(r2["pid"])) != wave.get("command")
        or _cmdline(int(r2_python["pid"])) != wave.get("command", [])[4:]
        or _children(wrapper_pid) != {int(outer["pid"])}
        or _children(int(r2["pid"])) != {int(r2_python["pid"])}
    ):
        raise ValueError("R2 process group or command differs from the paused outer owner")

    saved_state = Path(str(proof.get("before_state_path", "")))
    if saved_state != args.root / "receipts/acceleration-controller-state-before-pause-r1.json":
        raise ValueError("pause proof refers to an unexpected before-state snapshot")
    saved = contract.read_json(saved_state, 64_000_000) if saved_state.is_file() else {}
    saved_nodes = saved.get("nodes", {})
    saved_complete = {
        key
        for key, row in saved_nodes.items()
        if isinstance(row, dict) and row.get("status") == "complete"
    }
    saved_running = {
        key
        for key, row in saved_nodes.items()
        if isinstance(row, dict) and row.get("status") == "running"
    }
    base_complete = expected_existing - {"fin256ft_s3", "fin256ft_s4", "fbc1k_s5"}
    if (
        not saved_state.is_file()
        or saved_state.is_symlink()
        or _sha(saved_state) != proof.get("before_state_sha256")
        or saved.get("identity") != identity
        or saved.get("phase") != "running"
        or saved.get("scientific_fits_started") != 15
        or saved.get("profiles_approved") != 15
        or not saved.get("qualified_early_successor_adoptions")
        or set(saved_nodes) != expected_existing
        or not base_complete.issubset(saved_complete)
        or not saved_running.issubset({"fin256ft_s3", "fin256ft_s4", "fbc1k_s5"})
        or saved_complete | saved_running != expected_existing
        or saved.get("scientific_fits_completed") != len(saved_complete)
        or set(saved.get("active_workers", {})) != saved_running
    ):
        raise ValueError("pause proof's before-state snapshot is not the same R1 lineage")
    return proof, wave


def _validate_r2_state(
    args: argparse.Namespace,
    contract: Any,
    execution: Any,
    membership: Any,
    nodes: dict[str, dict[str, Any]],
    state: dict[str, Any],
    identity: dict[str, object],
    wave: dict[str, Any],
    proof: dict[str, Any],
    outer: dict[str, Any],
) -> set[str]:
    """Require the exact 15 admitted roles and validate every live owned worker."""
    expected = (set(contract.TL3_IDS) | {"source4_lstm192x1"}) - set(SUCCESSORS)
    live_candidates = {"fin256ft_s3", "fin256ft_s4", "fbc1k_s5"}
    base_complete = expected - live_candidates
    rows = state.get("nodes")
    if (
        state.get("identity") != identity
        or not isinstance(rows, dict)
        or set(rows) != expected
        or state.get("scientific_fits_started") != 15
        or state.get("profiles_approved") != 15
        or state.get("phase") != "running"
        or not state.get("qualified_early_successor_adoptions")
        or any(row.get("fit_attempted") is not True for row in rows.values())
    ):
        raise ValueError("R2 controller state does not contain the exact 15 admitted roles")
    complete = {key for key, row in rows.items() if row.get("status") == "complete"}
    running = {key for key, row in rows.items() if row.get("status") == "running"}
    if (
        not base_complete.issubset(complete)
        or not running.issubset(live_candidates)
        or complete | running != expected
        or state.get("scientific_fits_completed") != len(complete)
        or state.get("active_lanes", 0) != len(running)
        or set(state.get("active_workers", {})) != running
        or any(row.get("status") in {"failed", "reconciliation_required"} for row in rows.values())
    ):
        raise ValueError("R2 state has an unowned, failed, or duplicate fit role")
    expected_profiles = {}
    for node_id in sorted(expected):
        row = rows[node_id]
        profile_path = Path(str(row.get("profile_receipt", "")))
        profile_sha = row.get("profile_sha256")
        gpu = row.get("gpu_uuid")
        if (
            not isinstance(profile_sha, str)
            or not isinstance(gpu, str)
            or _sha(profile_path) != profile_sha
        ):
            raise ValueError(f"R2 profile receipt is missing or changed: {node_id}")
        mode = "native_source4_lstm" if node_id == "source4_lstm192x1" else "legacy_tl3"
        contract.validate_profile_approval(
            node_id=node_id,
            approval_path=profile_path,
            expected_sha256=profile_sha,
            plan_sha256=contract.PLAN_SHA256,
            membership=membership,
            mode=mode,
        )
        expected_profiles[node_id] = f"{profile_sha}={gpu}={profile_path}"
    command = wave.get("command")
    timeout = wave.get("timeout_seconds")
    if (
        not isinstance(command, list)
        or type(timeout) is not int
        or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
        or command[3:4] != [f"{timeout - 10}s"]
    ):
        raise ValueError("R2 director command lacks the pinned owned timeout")
    fixed = [
        str(args.python),
        str(args.r2_script),
        "--action",
        "run",
        "--runtime-root",
        str(args.root),
        "--source-root",
        str(args.source),
        "--data-root",
        str(args.data),
        "--plan",
        str(args.plan_path),
        "--membership",
        str(args.membership_path),
        "--source-audit",
        str(args.audit_path),
        "--source-capsule",
        str(args.capsule),
        "--native-config",
        str(args.native_config),
        "--python",
        str(args.python),
        "--grader",
        str(args.grader),
        "--root-admission",
        str(args.r2_admission),
        "--controller-root",
        str(args.controller),
        "--max-lanes",
        str(contract.MAX_LANES),
    ]
    fixed_start = 4
    if command[fixed_start : fixed_start + len(fixed)] != fixed:
        raise ValueError("R2 command differs from the frozen director entrypoint")
    tail = command[fixed_start + len(fixed) :]
    actual_profiles = {}
    if len(tail) % 2:
        raise ValueError("R2 command has an incomplete profile approval pair")
    for index in range(0, len(tail), 2):
        if tail[index] != "--profile-approval":
            raise ValueError("R2 command includes an unapproved option")
        fields = tail[index + 1].split("=", 3)
        if len(fields) != 4 or fields[0] in actual_profiles:
            raise ValueError("R2 command repeats or corrupts a profile approval")
        actual_profiles[fields[0]] = "=".join(fields[1:])
    if actual_profiles != expected_profiles:
        raise ValueError("R2 command profile list is not the exact 15-node controller set")
    if (
        wave.get("status") != "running"
        or wave.get("mode") != "final_submit_campaign_director"
        or wave.get("stage") != 1
        or wave.get("trial_number") != 2
        or wave.get("gpu_key") != "campaign-controller"
        or wave.get("cwd") != str(args.source)
        or wave.get("output_dir") != str(args.controller)
        or wave.get("timed_out") is not False
        or wave.get("command_sha256") != contract.canonical_sha256(command)
        or wave.get("owner", {}).get("pid") != outer.get("pid")
        or wave.get("owner", {}).get("birth") != outer.get("birth")
        or wave.get("owner", {}).get("pgid") != outer.get("pgid")
        or wave.get("owner", {}).get("controller_root") != str(args.controller)
        or timeout + wave.get("started_at_epoch", 0)
        > contract.HARD_DEADLINE - contract.COLLECTION_RESERVE + 2
    ):
        raise ValueError("R2 owned process receipt does not match its live outer owner")
    child = wave.get("child")
    live = _proc(int(child["pid"])) if isinstance(child, dict) else None
    if (
        live is None
        or live[0] in {"Z", "T", "t"}
        or live[1] != child.get("pgid")
        or live[2] != str(child.get("birth"))
        or live[1] == outer.get("pgid")
        or _cmdline(int(child["pid"])) != command
    ):
        raise ValueError("R2 GNU timeout is not live under the recorded command and identity")
    active_groups: set[int] = set()
    for node_id in running:
        row = rows[node_id]
        active = state["active_workers"][node_id]
        child = {
            "pid": active.get("pid"),
            "pgid": active.get("pgid"),
            "birth": active.get("birth"),
        }
        current = _proc(int(child["pid"])) if type(child["pid"]) is int else None
        receipt_path = Path(str(active.get("receipt", "")))
        receipt = contract.read_json(receipt_path, 2_000_000)
        stage = row.get("current_stage")
        expected_stage_path = (
            args.controller / "receipts/fits" / f"{node_id}.json"
            if stage == "fit"
            else args.controller
            / "receipts/exports"
            / f"{node_id}-{str(stage).partition(':')[2]}.json"
        )
        receipt_state = receipt.get("status")
        if (
            (receipt_state == "running" and (current is None or current[0] in {"Z", "T", "t"}))
            or (receipt_state == "completed" and receipt.get("exit_code") != 0)
            or receipt_state not in {"running", "completed"}
            or (current is not None and current[1] != child["pgid"])
            or (current is not None and current[2] != child["birth"])
            or (receipt_state == "completed" and _group_alive(int(child["pgid"])))
            or child["pgid"] in {outer.get("pgid"), wave["child"].get("pgid")}
            or active.get("stage") != stage
            or active.get("gpu_uuid") != row.get("gpu_uuid")
            or receipt_path != expected_stage_path
            or row.get("current_receipt") != str(receipt_path)
            or row.get("child_process") != child
            or receipt.get("child") != child
            or receipt.get("owner", {}).get("pid") != proof.get("r2_python_identity", {}).get("pid")
            or receipt.get("owner", {}).get("birth")
            != proof.get("r2_python_identity", {}).get("birth")
            or receipt.get("owner", {}).get("controller_root") != str(args.controller)
            or receipt.get("command_sha256") != contract.canonical_sha256(receipt.get("command"))
            or (
                receipt_state == "running" and _cmdline(int(child["pid"])) != receipt.get("command")
            )
        ):
            raise ValueError(f"R2 worker is not live under its exact receipt: {node_id}")
        process_snapshot = proof.get("after_processes", {}).get(str(child["pid"]))
        if process_snapshot is not None and (
            process_snapshot.get("pgid") != child["pgid"]
            or process_snapshot.get("birth") != child["birth"]
            or process_snapshot.get("command_sha256") != receipt.get("command_sha256")
            or process_snapshot.get("state") in {"T", "t", "Z"}
        ):
            raise ValueError(f"R2 worker pause snapshot differs from its live receipt: {node_id}")
        trial = contract.TL3_IDS.index(node_id) + 2
        output = args.data / "artifacts/runs" / nodes[node_id]["run"]
        command = receipt.get("command", [])
        if stage == "fit":
            expected_argv = execution.build_fit_argv(
                node=nodes[node_id],
                python=args.python,
                source_root=args.source,
                data_root=args.data,
                plan=contract.read_json(args.plan_path),
                profile_path=Path(str(row["profile_receipt"])),
            )
            expected_stage = 1
        elif stage in {"export:train", "export:valid"}:
            split = stage.partition(":")[2]
            expected_argv = [
                str(args.python),
                str(args.source / "scripts/predict_run.py"),
                nodes[node_id]["run"],
                split,
                "--epoch",
                str(nodes[node_id]["endpoint"]),
                "--final-fit-manifest",
                str(args.membership_path),
                "--final-fit-source-validation-receipt",
                str(args.audit_path),
                "--final-fit-source-validation-sha256",
                contract.SOURCE_AUDIT_SHA256,
                "--device",
                "cuda",
            ]
            output = args.data / "cache"
            expected_stage = 2
        else:
            raise ValueError(f"R2 active worker has an unknown stage: {node_id}/{stage}")
        timeout = receipt.get("timeout_seconds")
        if (
            receipt.get("mode") != "final_submit_finite_chain"
            or receipt.get("stage") != expected_stage
            or receipt.get("trial_number") != trial
            or receipt.get("gpu_key") != row.get("gpu_uuid")
            or receipt.get("output_dir") != str(output.resolve())
            or receipt.get("cwd") != str(args.source)
            or type(timeout) is not int
            or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
            or command[3:4] != [f"{timeout - 10}s"]
            or command[4:] != expected_argv
        ):
            raise ValueError(f"R2 worker command differs from its frozen recipe: {node_id}")
        active_groups.add(int(child["pgid"]))
        if receipt_state == "running":
            children = _children(int(child["pid"]))
            if len(children) != 1:
                raise ValueError(f"R2 worker has an unexpected trainer child set: {node_id}")
            trainer_pid = next(iter(children))
            trainer = _proc(trainer_pid)
            if (
                trainer is None
                or trainer[0] in {"Z", "T", "t"}
                or trainer[1] != child["pgid"]
                or _cmdline(trainer_pid) != expected_argv
            ):
                raise ValueError(f"R2 trainer process differs from its frozen stage: {node_id}")
    if len(active_groups) != len(running):
        raise ValueError("R2 active fits share an owned process group")
    return expected


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("R3 requires the approved POSIX runtime")
    args.root = args.runtime_root.resolve()
    args.source = args.root / "source"
    args.data = args.root / "data"
    args.controller = args.root / "control/final-submit-chain"
    args.plan_path = args.root / "inputs/final-plan-r2.json"
    args.membership_path = args.root / "inputs/final-fit-membership-r1.json"
    args.audit_path = args.root / "inputs/SOURCE_VALIDATION_R1.json"
    args.capsule = args.root / "inputs/final-submit-source-39d83f739.tar.gz"
    args.native_config = args.root / "inputs/LSTM_CONFIG_R2.json"
    args.r2_script = args.root / "control-code-r2/scripts/final_submit_chain.py"
    args.r2_admission = args.root / "ROOT_EXECUTION_ADMISSION_R2.json"
    args.r1_admission = args.root / "ROOT_EXECUTION_ADMISSION_R1.json"
    args.grader = args.root / "control-grader-r2/scripts/review_final_submit_profile.py"
    args.pause_admission = args.root / "receipts/acceleration-profile-admission-r1.json"
    args.proofs = args.root / "receipts/final-submit-r3"
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError("requested R3 Python is not executable")

    director, contract, execution, runtime, atomic_write_json, file_lock, closure_fn = _load(args)
    plan = contract.read_json(args.plan_path, 4_000_000)
    planned = contract.verify_plan(plan, args.plan_path)
    if _sha(args.capsule) != contract.SOURCE_CAPSULE_SHA256:
        raise ValueError("R2 source capsule changed")
    director._verify_source_tree(args.source, args.capsule)
    if _sha(args.native_config) != plan["optional_source4_lstm"]["config_file_sha256"]:
        raise ValueError("native config changed")
    membership = contract.verify_sources(
        data_root=args.data,
        membership_path=args.membership_path,
        audit_path=args.audit_path,
        audit_sha256=contract.SOURCE_AUDIT_SHA256,
    )
    contract.verify_root_admission(
        admission_path=args.r2_admission,
        grader_path=args.grader,
        director_path=args.r2_script,
        helper_path=args.r2_script.with_name("final_submit_chain_contract.py"),
        runtime_helper_path=args.r2_script.with_name("final_submit_chain_runtime.py"),
        execution_helper_path=args.r2_script.with_name("final_submit_chain_execution.py"),
        plan_sha256=contract.PLAN_SHA256,
        source_capsule_sha256=contract.SOURCE_CAPSULE_SHA256,
        now_epoch=time.time(),
    )
    identity_args = argparse.Namespace(
        runtime_root=args.root,
        controller_root=args.controller,
        plan=args.plan_path,
        source_capsule=args.capsule,
        native_config=args.native_config,
        root_admission=args.r2_admission,
        membership=args.membership_path,
        source_audit=args.audit_path,
    )
    identity = director._expected_identity(plan, membership, identity_args)
    state_path = args.controller / "controller-state.json"
    initial_state = contract.read_json(state_path, 64_000_000)
    marker = initial_state.get("execution_admissions")
    if (
        initial_state.get("identity") != identity
        or not isinstance(marker, list)
        or len(marker) != 1
        or marker[0].get("root_admission_sha256") != _sha(args.r2_admission)
        or initial_state.get("scientific_fits_started") != 15
        or initial_state.get("profiles_approved") != 15
    ):
        raise ValueError("R3 does not own exactly the remaining three of the 18 planned roles")
    all_nodes = director._node_records(plan, planned, identity_args)
    nodes = {row["node_id"]: row for row in all_nodes}
    if set(SUCCESSORS) - set(nodes) or any(nodes[node]["export_train"] for node in SUCCESSORS):
        raise ValueError("R3 successor IDs or export contracts differ from the fixed plan")

    expected_existing = (set(contract.TL3_IDS) | {"source4_lstm192x1"}) - set(SUCCESSORS)
    wave_path = args.controller / "receipts/campaign/director-wave-02.json"
    pause_proof, wave = _validate_pause(args, contract, identity, expected_existing, wave_path)
    outer = pause_proof["outer_identity"]
    _validate_r2_state(
        args,
        contract,
        execution,
        membership,
        nodes,
        initial_state,
        identity,
        wave,
        pause_proof,
        outer,
    )

    expected_profiles = {
        node_id: _expected_profile_paths(args, nodes[node_id]) for node_id in SUCCESSORS
    }
    if len({row["gpu_uuid"] for row in expected_profiles.values()}) != len(SUCCESSORS):
        raise ValueError("R3 successor device assignments must be distinct")
    profiles: dict[str, dict[str, Any]] = {}
    command = wave["command"]
    child = wave["child"]
    for node_id in SUCCESSORS:
        node = nodes[node_id]
        if (
            node_id in initial_state["nodes"]
            or (args.data / "artifacts/runs" / node["run"]).exists()
            or (args.controller / "receipts/fits" / f"{node_id}.json").exists()
            or (args.controller / "logs/fits" / f"{node_id}.log").exists()
        ):
            raise FileExistsError(f"R3 role already has output or controller history: {node_id}")

    args.proofs.mkdir(parents=True, exist_ok=False)
    initial_bytes = state_path.read_bytes()
    initial_sha = hashlib.sha256(initial_bytes).hexdigest()
    _write(
        args.proofs / "intent.json",
        {
            "schema": "wnn.final-submit-r3-successor-intent.v1",
            "sidecar_path": str(Path(__file__).resolve()),
            "sidecar_sha256": _sha(Path(__file__).resolve()),
            "node_ids": list(SUCCESSORS),
            "plan_sha256": contract.PLAN_SHA256,
            "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
            "membership_sha256": membership.manifest_sha256,
            "fit_members_sha256": membership.fit_members_sha256,
            "source_validation_sha256": contract.SOURCE_AUDIT_SHA256,
            "outer_identity": outer,
            "outer_pause_admission_sha256": _sha(args.pause_admission),
            "r2_wave_receipt": str(wave_path),
            "r2_wave_command_sha256": contract.canonical_sha256(command),
            "r2_child": child,
            "initial_controller_state_sha256": initial_sha,
            "profile_rows": {
                key: {
                    "path": str(value["row_path"]),
                    "profile_path": str(value["profile_path"]),
                    "request_path": str(value["request_path"]),
                    "review_path": str(value["review_path"]),
                    "gpu_uuid": value["gpu_uuid"],
                }
                for key, value in expected_profiles.items()
            },
            "profile_admission_policy": "await_each_node_independently_no_global_barrier",
            "shared_controller_state_written_before_r2_terminal": False,
            "max_new_scientific_fits": 3,
            "total_fit_cap": contract.TOTAL_FITS,
            "profile_cap_after_adoption": 15 + len(SUCCESSORS),
            "independent_score_claim": "forbidden",
        },
    )

    def execute(node_id: str) -> dict[str, Any]:
        node = nodes[node_id]
        profile = _wait_for_approval(
            args,
            contract,
            membership,
            node,
            outer,
            expected_profiles[node_id],
        )
        profiles[node_id] = profile
        parents = ("fin256ft_s3", "fin256ft_s4") if node_id in TLEADS else tuple(node["parents"])
        while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
            _stopped(outer)
            live = contract.read_json(state_path, 64_000_000)
            if any(
                live.get("nodes", {}).get(parent, {}).get("status") != "complete"
                for parent in parents
            ):
                time.sleep(3)
                continue
            parent_checkpoints = {}
            for parent in parents:
                parent_node = nodes[parent]
                parent_row = live["nodes"][parent]
                checkpoint = runtime.validate_fit_checkpoint(
                    node=parent_node,
                    output_dir=args.data / "artifacts/runs" / parent_node["run"],
                    membership=membership,
                )
                if parent_row.get("checkpoint_path") != str(checkpoint) or parent_row.get(
                    "checkpoint_sha256"
                ) != _sha(checkpoint):
                    raise ValueError(f"R3 parent checkpoint differs: {parent}")
                parent_checkpoints[parent] = {"path": str(checkpoint), "sha256": _sha(checkpoint)}
            if node_id == "w18final_distilled_s5":
                cache = live.get("cache", {})
                cache_path = args.data / "artifacts/runs/final-fit-w18-mixed-targets"
                if cache.get("status") != "complete":
                    time.sleep(3)
                    continue
                metadata = execution.validate_final_distillation_cache(
                    cache_path, membership, "source", require_families=True
                )
                if cache.get("path") != str(cache_path) or cache.get(
                    "cache_sha256"
                ) != metadata.get("metadata_sha256"):
                    raise ValueError("W18 cache is not the validated frozen six-teacher cache")
            break
        else:
            raise TimeoutError(f"collection reserve began before {node_id} parents completed")

        output = args.data / "artifacts/runs" / node["run"]
        fit_receipt = args.controller / "receipts/fits" / f"{node_id}.json"
        fit_log = args.controller / "logs/fits" / f"{node_id}.log"
        if output.exists() or fit_receipt.exists() or fit_log.exists() or node_id in live["nodes"]:
            raise FileExistsError(f"R3 role already has an output or controller row: {node_id}")
        progress_path = args.proofs / f"{node_id}.progress.json"
        started = time.time()

        def progress(stage: str, receipt: dict[str, Any]) -> None:
            _stopped(outer)
            atomic_write_json(
                progress_path,
                {
                    "node_id": node_id,
                    "stage": stage,
                    "process": receipt,
                    "updated_epoch": time.time(),
                },
            )

        trial = contract.TL3_IDS.index(node_id) + 2
        result = execution.execute_node(
            node=node,
            plan=plan,
            membership=membership,
            source_root=args.source,
            data_root=args.data,
            controller_root=args.controller,
            python=args.python,
            gpu_uuid=profile["gpu_uuid"],
            gpu_visible=profile["gpu_uuid"],
            profile_path=profile["path"],
            trial=trial,
            progress=progress,
        )
        checkpoint = runtime.validate_fit_checkpoint(
            node=node, output_dir=output, membership=membership
        )
        process_path = Path(str(result["fit_process_receipt"]))
        process = contract.read_json(process_path, 2_000_000)
        argv = execution.build_fit_argv(
            node=node,
            python=args.python,
            source_root=args.source,
            data_root=args.data,
            plan=plan,
            profile_path=profile["path"],
        )
        _owned_receipt(
            contract,
            process_path,
            process,
            argv,
            1,
            trial,
            profile["gpu_uuid"],
            output,
            args.source,
        )
        closure = closure_fn(node["run"], checkpoint.name, membership, args.data)
        if {nodes[parent]["run"] for parent in parents} - {item.get("run") for item in closure}:
            raise ValueError(f"R3 ancestry closure is incomplete: {node_id}")
        proof = {
            "schema": "wnn.final-submit-r3-successor-completion.v1",
            "status": "complete",
            "node_id": node_id,
            "run": node["run"],
            "plan_sha256": contract.PLAN_SHA256,
            "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
            "membership_sha256": membership.manifest_sha256,
            "fit_members_sha256": membership.fit_members_sha256,
            "source_validation_sha256": contract.SOURCE_AUDIT_SHA256,
            "profile_path": str(profile["path"]),
            "profile_sha256": profile["sha256"],
            "request_path": str(profile["request_path"]),
            "request_sha256": profile["request_sha256"],
            "profile_row": str(profile["row_path"]),
            "profile_row_sha256": profile["row_sha256"],
            "parents": parent_checkpoints,
            "gpu_uuid": profile["gpu_uuid"],
            "trial": trial,
            "fixed_endpoint": node["endpoint"],
            "result": result,
            "ancestor_closure": closure,
            "started_epoch": started,
            "finished_epoch": time.time(),
            "independent_score_claim": "forbidden",
        }
        proof_path = args.proofs / f"{node_id}.complete.json"
        _write(proof_path, proof)
        return {"path": str(proof_path), "sha256": _sha(proof_path), "result": result}

    completed: dict[str, dict[str, Any]] = {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="finalr3") as pool:
        futures = {pool.submit(execute, node): node for node in SUCCESSORS}
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                completed[node] = future.result()
            except Exception as error:
                failures.append(f"{node}: {type(error).__name__}: {error}")
    if failures:
        _write(
            args.proofs / "failure.json",
            {
                "schema": "wnn.final-submit-r3-failure.v1",
                "failures": failures,
                "completed": completed,
            },
        )
        raise RuntimeError("R3 fit failure; no controller adoption or outer resume was made")

    lock_path = args.controller / "director.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("the R2 director lock is missing or linked")
    adopted = False
    while time.time() < contract.HARD_DEADLINE - contract.COLLECTION_RESERVE:
        _stopped(outer)
        try:
            lock = file_lock(lock_path, blocking=False)
            lock.__enter__()
        except BlockingIOError:
            time.sleep(2)
            continue
        try:
            before = state_path.read_bytes()
            before_sha = hashlib.sha256(before).hexdigest()
            snapshot = args.proofs / f"controller-before-{before_sha}.json"
            _write_bytes(snapshot, before)
            state = contract.read_json(state_path, 64_000_000)
            expected_existing = (set(contract.TL3_IDS) | {"source4_lstm192x1"}) - set(SUCCESSORS)
            if (
                state.get("identity") != identity
                or state.get("phase") != "awaiting_profile_or_dependency"
                or state.get("active_workers")
                or state.get("active_lanes", 0) != 0
                or set(state.get("nodes", {})) != expected_existing
                or state.get("scientific_fits_started") != 15
                or state.get("scientific_fits_completed") != 15
                or state.get("cache", {}).get("status") != "complete"
                or any(row.get("status") != "complete" for row in state["nodes"].values())
            ):
                raise ValueError("R2 is not terminal at the exact 15-fit/cache boundary")
            for node_id in SUCCESSORS:
                proof_info = completed[node_id]
                proof_path = Path(proof_info["path"])
                if _sha(proof_path) != proof_info["sha256"]:
                    raise ValueError(f"R3 completion proof changed: {node_id}")
                proof = contract.read_json(proof_path, 8_000_000)
                result, profile, node = proof.get("result"), profiles[node_id], nodes[node_id]
                checkpoint = runtime.validate_fit_checkpoint(
                    node=node,
                    output_dir=args.data / "artifacts/runs" / node["run"],
                    membership=membership,
                )
                if (
                    proof.get("schema") != "wnn.final-submit-r3-successor-completion.v1"
                    or proof.get("status") != "complete"
                    or proof.get("node_id") != node_id
                    or proof.get("run") != node["run"]
                    or proof.get("plan_sha256") != contract.PLAN_SHA256
                    or proof.get("membership_sha256") != membership.manifest_sha256
                    or proof.get("fit_members_sha256") != membership.fit_members_sha256
                    or proof.get("source_validation_sha256") != contract.SOURCE_AUDIT_SHA256
                    or proof.get("profile_path") != str(profile["path"])
                    or proof.get("profile_sha256") != profile["sha256"]
                    or proof.get("request_sha256") != profile["request_sha256"]
                    or proof.get("gpu_uuid") != profile["gpu_uuid"]
                    or not isinstance(result, dict)
                    or result.get("status") != "complete"
                    or result.get("checkpoint_path") != str(checkpoint)
                    or result.get("checkpoint_sha256") != _sha(checkpoint)
                ):
                    raise ValueError(f"R3 completion identity differs: {node_id}")
                process_path = args.controller / "receipts/fits" / f"{node_id}.json"
                process = contract.read_json(process_path, 2_000_000)
                argv = execution.build_fit_argv(
                    node=node,
                    python=args.python,
                    source_root=args.source,
                    data_root=args.data,
                    plan=plan,
                    profile_path=profile["path"],
                )
                _owned_receipt(
                    contract,
                    process_path,
                    process,
                    argv,
                    1,
                    contract.TL3_IDS.index(node_id) + 2,
                    profile["gpu_uuid"],
                    args.data / "artifacts/runs" / node["run"],
                    args.source,
                )
                if result.get("fit_process") != process or result.get("fit_process_receipt") != str(
                    process_path
                ):
                    raise ValueError(f"R3 process receipt differs: {node_id}")
                state["nodes"][node_id] = {
                    "status": "complete",
                    "fit_attempted": True,
                    "node_id": node_id,
                    "run": node["run"],
                    "started_at_epoch": proof["started_epoch"],
                    "fit_started_at_epoch": result["fit_started_at_epoch"],
                    "fit_finished_at_epoch": result["fit_finished_at_epoch"],
                    "gpu_uuid": profile["gpu_uuid"],
                    "profile_receipt": str(profile["path"]),
                    "profile_sha256": profile["sha256"],
                    "profile_request_sha256": profile["request_sha256"],
                    "trial_number": proof["trial"],
                    "argv": process["command"],
                    "argv_sha256": process["command_sha256"],
                    "source_capsule_sha256": contract.SOURCE_CAPSULE_SHA256,
                    "plan_sha256": contract.PLAN_SHA256,
                    "membership_sha256": membership.manifest_sha256,
                    "fit_process_receipt": str(process_path),
                    "fit_process": process,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": _sha(checkpoint),
                    "exports": result.get("exports", {}),
                    "ancestor_closure": proof["ancestor_closure"],
                    "independent_score_claim": "forbidden",
                    "r3_completion": {"path": str(proof_path), "sha256": proof_info["sha256"]},
                }
            state["scientific_fits_started"] = contract.TOTAL_FITS
            state["scientific_fits_completed"] = contract.TOTAL_FITS
            if state.get("profiles_approved") != 15 or 15 + len(SUCCESSORS) > contract.TOTAL_FITS:
                raise ValueError("R3 profile adoption exceeds the frozen 18-role budget")
            state["profiles_approved"] = 15 + len(SUCCESSORS)
            state["max_lanes"] = max(int(state.get("max_lanes", 0)), 3)
            state["active_workers"] = {}
            state["active_lanes"] = 0
            state["phase"] = "complete"
            state["finished_at_epoch"] = time.time()
            state["updated_at_epoch"] = time.time()
            adoption_path = args.proofs / "adoption.json"
            _write(
                adoption_path,
                {
                    "schema": "wnn.final-submit-r3-successor-adoption.v1",
                    "resolved_node_ids": list(SUCCESSORS),
                    "completion_proofs": completed,
                    "before_state_path": str(snapshot),
                    "before_state_sha256": before_sha,
                    "terminal_outcomes_preserved": True,
                    "state_adopted_under_director_lock": True,
                    "independent_score_claim": "forbidden",
                },
            )
            state.setdefault("qualified_r3_successor_adoptions", []).append(
                {
                    "path": str(adoption_path),
                    "sha256": _sha(adoption_path),
                    "schema": "wnn.final-submit-r3-successor-adoption.v1",
                }
            )
            atomic_write_json(state_path, state)
            adopted = True
        finally:
            lock.__exit__(None, None, None)
        break
    if not adopted:
        raise TimeoutError("R2 director lock did not release before the collection reserve")
    completion = args.proofs / "complete.json"
    _write(
        completion,
        {
            "schema": "wnn.final-submit-r3-handoff-completion.v1",
            "status": "complete",
            "node_ids": list(SUCCESSORS),
            "controller_state_sha256": _sha(state_path),
            "completion_proofs": completed,
            "total_fits": contract.TOTAL_FITS,
            "director_lock_released_before_resume": True,
            "independent_score_claim": "forbidden",
        },
    )
    _stopped(outer)
    os.kill(int(outer["pid"]), signal.SIGCONT)
    print(
        json.dumps(
            {"status": "complete", "proof": str(completion), "sha256": _sha(completion)},
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
