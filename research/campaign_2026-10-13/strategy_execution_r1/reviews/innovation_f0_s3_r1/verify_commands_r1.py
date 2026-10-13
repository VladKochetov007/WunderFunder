#!/usr/bin/env python3
"""Stdlib-only exact pair/argv comparison for the frozen S09 R1 packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE = Path("/home/vlad/development/wunder/wnn-agent-worktrees/strategy-innovation-fit-20261013")
PACKET = SOURCE / "research/campaign_2026-10-13/strategy_execution_r1/innovation_f0_s3_r1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized(argv: list[str]) -> list[str]:
    result = list(argv)
    for flag, replacement in (
        ("--name", "RUN_NAME"),
        ("--resume-state", "RESUME_STATE"),
        ("--research-feature-control", "FEATURE_CONTROL"),
    ):
        if flag in result:
            result[result.index(flag) + 1] = replacement
    return result


def next_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main() -> None:
    names = ("MOMENTS_CONTROL", "INNOVATIONS")
    packets = {name: json.loads((PACKET / f"{name}_COMMANDS_R1.json").read_text()) for name in names}
    normalized_stage_vectors = {}
    mismatches = []
    stages = {}
    for name, packet in packets.items():
        if packet["fit_authorized"] or packet["export"]["launch_authorized"] or packet["grade"]["launch_authorized"]:
            mismatches.append(f"{name}: packet grants fit/export/grade authority")
        stages[name] = packet["stages"]
        normalized_stage_vectors[name] = []
        for stage in packet["stages"]:
            vector = stage["trainer_argv"]
            normalized_stage_vectors[name].append(normalized(vector))
            if stage["launch_authorized"]:
                mismatches.append(f"{name}/{stage['part']}: stage launch authority true")
            expected_resume = None if stage["part"] == "p1" else (
                f"{MAIN}/artifacts/runs/{stage['run_name'].rsplit('_p', 1)[0]}_p{int(stage['part'][-1]) - 1}/training_state.pt"
            )
            actual_resume = next_value(vector, "--resume-state")
            if actual_resume != expected_resume:
                mismatches.append(f"{name}/{stage['part']}: resume path is {actual_resume!r}, expected {expected_resume!r}")
            if actual_resume is not None and not Path(actual_resume).is_absolute():
                mismatches.append(f"{name}/{stage['part']}: resume path is not absolute")
            expected_stop = {"p1": "1", "p2": "3", "p3": "5"}[stage["part"]]
            if next_value(vector, "--stop-after-epoch") != expected_stop:
                mismatches.append(f"{name}/{stage['part']}: bad inclusive stop epoch")
            if next_value(vector, "--research-features") != "causal_innovations":
                mismatches.append(f"{name}/{stage['part']}: feature mode mismatch")
        if len({stage["run_name"] for stage in packet["stages"]}) != 3:
            mismatches.append(f"{name}: run-name collision")
        if len({stage["attempt_id"] for stage in packet["stages"]}) != 3:
            mismatches.append(f"{name}: attempt collision")
    if normalized_stage_vectors[names[0]] != normalized_stage_vectors[names[1]]:
        mismatches.append("normalized trainer argv vectors differ beyond the declared feature selector")
    for field in ("alpha", "epochs", "fixed_checkpoint_epoch", "onecycle_attempt_updates", "scale", "target_scale", "stored_teacher_amplitude", "training_precision", "export_precision", "representation"):
        if packets[names[0]]["fit_schedule"].get(field) != packets[names[1]]["fit_schedule"].get(field):
            mismatches.append(f"fit schedule differs: {field}")
    for name, packet in packets.items():
        profile = packet["device_profile"]
        vector = profile["worker_argv"]
        profile_check = {
            "shape": profile["shape"],
            "reserve_bytes": next_value(profile["controller_argv"], "--reserve-bytes"),
            "cpus": next_value(profile["controller_argv"], "--cpus"),
            "warmup": next_value(vector, "--profile-warmup"),
            "measured_steps": next_value(vector, "--profile-steps"),
            "profile_only": profile["profile_only"],
            "training_state_saved": profile["training_state_saved"],
            "memory_bound_bytes": profile["memory_bound_bytes"],
            "timeout_s": profile["max_wall_seconds"],
        }
        if profile_check["shape"] != {"batch": 16, "sequence_rows": 20000, "micro_rows": 2000, "n_in": 291}:
            mismatches.append(f"{name}: device profile does not target full 291-input fit shape")
        if (profile_check["reserve_bytes"] != "6000000000" or profile_check["cpus"] != "1"
                or profile_check["warmup"] != "5" or profile_check["measured_steps"] != "5"
                or not profile_check["profile_only"] or profile_check["training_state_saved"]):
            mismatches.append(f"{name}: profile bound/geometry is incorrect")
    cli = json.loads((PACKET / "CLI_PREFLIGHT_R1.json").read_text())
    if len(cli["checks"]) != 10 or any(item["status"] != "pass" for item in cli["checks"]):
        mismatches.append("actual parser receipt does not report ten passing exact-vector checks")

    # Packet checksum binding for this independent receipt.
    file_hashes = {
        filename: sha(PACKET / filename)
        for filename in (
            "RECIPE_R1.json", "MOMENTS_CONTROL_COMMANDS_R1.json", "INNOVATIONS_COMMANDS_R1.json",
            "PREFLIGHT_R1.json", "CLI_PREFLIGHT_R1.json",
        )
    }
    output = {
        "schema": "wnn.s09.independent-command-review.r1",
        "source_commit": "c6e774edc8d5e0c63715a83287cf36fd3969a779",
        "packet_sha256": file_hashes,
        "normalized_argv_pair_matches": normalized_stage_vectors[names[0]] == normalized_stage_vectors[names[1]],
        "stage_vectors": {
            name: [{
                "part": s["part"], "attempt_id": s["attempt_id"], "run_name": s["run_name"],
                "stop_after_epoch": next_value(s["trainer_argv"], "--stop-after-epoch"),
                "resume_state": next_value(s["trainer_argv"], "--resume-state"),
                "parent": next_value(s["trainer_argv"], "--init"),
                "split_manifest": next_value(s["trainer_argv"], "--split-manifest"),
                "feature_mode": next_value(s["trainer_argv"], "--research-features"),
                "feature_control": next_value(s["trainer_argv"], "--research-feature-control"),
                "target_cache_scale": next_value(s["trainer_argv"], "--distill-cache-scale"),
                "authorized": s["launch_authorized"],
            } for s in stages[name]] for name in names
        },
        "profiles": {name: {
            "shape": packet["device_profile"]["shape"],
            "reserve_bytes": next_value(packet["device_profile"]["controller_argv"], "--reserve-bytes"),
            "cpus": next_value(packet["device_profile"]["controller_argv"], "--cpus"),
            "warmup": next_value(packet["device_profile"]["worker_argv"], "--profile-warmup"),
            "measured_steps": next_value(packet["device_profile"]["worker_argv"], "--profile-steps"),
            "memory_bound_bytes": packet["device_profile"]["memory_bound_bytes"],
            "max_wall_seconds": packet["device_profile"]["max_wall_seconds"],
            "profile_only": packet["device_profile"]["profile_only"],
            "training_state_saved": packet["device_profile"]["training_state_saved"],
        } for name, packet in packets.items()},
        "actual_parser_receipt": {
            "checks": len(cli["checks"]),
            "all_pass": all(item["status"] == "pass" for item in cli["checks"]),
            "cuda_initialized": cli["cuda_initialized"],
            "post_parse_work_executed": cli["post_parse_work_executed"],
        },
        "schedule_scale_fields": {name: packet["fit_schedule"] for name, packet in packets.items()},
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
