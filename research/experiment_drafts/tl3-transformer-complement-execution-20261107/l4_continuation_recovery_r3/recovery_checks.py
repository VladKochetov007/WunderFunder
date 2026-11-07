from __future__ import annotations

from pathlib import Path

from supervisor_checks import DEADLINE, digest, pinned, read

PARENTS = {
    101001: (
        "06a1bc3ce96a2033576dfc53085ebbfcff06663d390a9b5e3c39a0a7e978d93d",
        2139244,
        "2146947674",
    ),
    101002: (
        "909654a77629a17dd99c7a615d8d1997a115c7a7aad50f372ad3f22b61e5cc9a",
        2139245,
        "2146947675",
    ),
}


def non_zombie_group(pgid: int) -> bool:
    """Read-only admission rejects any still-running member of a historical owned group."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def validate_recovery(plan: dict, live: object, *, fresh_seed: int | None = None) -> None:
    """Bind the already-stopped original S1E8 states and preserve every failed S3 artifact."""
    if (
        plan.get("schema") != "wnn.gru-monitor-recovery.v3"
        or plan.get("approved") is not True
        or plan.get("reviewer") != "ROOT"
    ):
        raise ValueError("explicit ROOT repair plan required")
    if [(cell["seed"], cell["gpu_index"]) for cell in plan["cells"]] != [(101001, 0), (101002, 1)]:
        raise ValueError("recovery applies only to the two original GRU trajectories")
    pinned(plan["root_cleanup_proof"])
    required = {
        "recovery_supervisor.py",
        "recovery_checks.py",
        "canonical_resume_probe.py",
        "run_probe.py",
        "migrate_monitor_checkpoint_s4.py",
        "transition_support_s4.py",
        "archive_safe_state_s4.py",
        "supervisor.py",
        "supervisor_checks.py",
        "observed_worker.py",
    }
    if (
        len(plan["tool_pins"]) != len(required)
        or {Path(spec["path"]).name for spec in plan["tool_pins"]} != required
    ):
        raise ValueError("all R3, S4, inherited and observed-worker runtime files must be pinned")
    for spec in plan["historical_owned_receipts"]:
        value = pinned(spec)
        owner = value.get("child", value)
        if live(owner) or non_zombie_group(int(owner["pgid"])):
            raise ValueError("historical original/S3 process group is still live")
    for spec in plan["tool_pins"]:
        if digest(Path(spec["path"])) != spec["sha256"]:
            raise ValueError("reviewed recovery/migration/observation helper changed")
    runtime = Path(plan["runtime"])
    for cell in plan["cells"]:
        request = cell["request"]
        parent = pinned(cell["parent_request"])
        checkpoint_sha, pid, birth = PARENTS[cell["seed"]]
        if (
            parent["old_checkpoint_sha256"] != checkpoint_sha
            or parent["expected_next_epoch"] != 8
            or parent["expected_global_step"] != 51680
            or parent["stopped_trainer_owner"]["pid"] != pid
            or parent["stopped_trainer_owner"]["birth"] != birth
            or live(parent["stopped_trainer_owner"])
        ):
            raise ValueError("ROOT-proven original S1E8 frontier differs")
        fixed = (
            "old_output",
            "old_metadata",
            "old_checkpoint_sha256",
            "expected_next_epoch",
            "expected_global_step",
            "old_source_root",
            "old_source_manifest",
            "first_profile_receipt",
            "not_after_epoch",
            "stopped_trainer_owner",
        )
        if any(request[key] != parent[key] for key in fixed):
            raise ValueError("recovery changed the pinned original trajectory or clock")
        if (
            request["not_after_epoch"] != DEADLINE
            or float(pinned(request["first_profile_receipt"])["first_profile_epoch"]) + 82800
            != DEADLINE
            or Path(request["old_output"]) != runtime / f"runs/gru_s{cell['seed']}/fit"
            or Path(request["new_output"])
            != runtime / f"runs/gru_s{cell['seed']}_monitor_opt_r3/fit"
            or (
                (fresh_seed is None or fresh_seed == cell["seed"])
                and Path(request["new_output"]).exists()
            )
            or digest(Path(request["old_output"]) / "training_state.pt") != checkpoint_sha
        ):
            raise ValueError("original hash/deadline or fresh R3 sibling output differs")
        profile = pinned(request["approved_profile"])
        if (
            profile.get("approved") is not True
            or profile.get("reviewer") != "ROOT"
            or profile.get("updates") != 20
        ):
            raise ValueError("actual S4 native profile requires ROOT approval")


def probe_ready(path: Path, proof_sha: str, device: str) -> bool:
    """Require explicit CPU-only or real-CUDA zero-update canonical restoration evidence."""
    value = read(path)
    return (
        value.get("schema") == "wnn.gru-canonical-resume-probe.v3"
        and value.get("status") == "passed"
        and value.get("device") == device
        and value.get("transition_proof_sha256") == proof_sha
        and value.get("next_epoch") == 8
        and value.get("global_step") == 51680
        and value.get("optimizer_updates") == 0
        and value.get("forward_calls") == 0
        and value.get("cuda_rng_setter_stubbed") == (device == "cpu")
        and value.get("real_cuda_resume_exercised") == (device != "cpu")
    )
