from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transition_support import (
    SPLIT_SHA,
    copy_prefix,
    pinned_json,
    same_payload,
    trainer_files,
    validate_evaluations,
    validate_transition,
    verify_capsules,
    write_json_new,
)

from wnn.multistage_search_runtime import is_same_process
from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_monitor_events import MonitorEventWriter
from wnn.training.sequence_provenance import (
    file_sha256,
    recorder_for_run,
    validate_profile_receipt,
)

CPU_CONTEXT = DistributedContext(0, 0, 1, torch.device("cpu"), False)


def clone_checkpoint(
    original: Path,
    destination: Path,
    old_identity: dict[str, object],
    new_identity: dict[str, object],
    expected_sha: str,
) -> dict[str, object]:
    """Clone a pinned trusted safe state, changing only its admitted contract envelope."""
    if original.is_symlink() or file_sha256(original) != expected_sha:
        raise ValueError("parent training-state bytes differ from ROOT's stopped-boundary pin")
    old = load_training_state(original, old_identity, CPU_CONTEXT)
    if not 1 <= old["next_epoch"] < 12:
        raise ValueError("continuation needs a nonempty unfinished whole-epoch boundary")
    cloned = old | {"contract": new_identity}
    with destination.open("xb") as stream:
        torch.save(cloned, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checked = load_training_state(destination, new_identity, CPU_CONTEXT)
    if checked.keys() != old.keys() or any(
        not same_payload(old[key], checked[key]) for key in old if key != "contract"
    ):
        raise ValueError("checkpoint serialization changed model/optimizer/scheduler/RNG/offsets")
    if file_sha256(original) != expected_sha:
        raise ValueError("parent checkpoint changed during transition; original must be stopped")
    return checked


def prepare_output(
    old_output: Path,
    output: Path,
    config: dict[str, object],
    old_identity: dict[str, object],
    new_identity: dict[str, object],
    checkpoint_sha: str,
) -> dict[str, object]:
    """Create new canonical metadata and copied safe logs; never alter historical files."""
    old_root = old_output.resolve(strict=True)
    new_root = output.resolve()
    if new_root.is_relative_to(old_root) or old_root.is_relative_to(new_root):
        raise ValueError("new continuation must be a separate fresh sibling run")
    output.mkdir(parents=True, exist_ok=False)
    state = clone_checkpoint(
        old_output / "training_state.pt",
        output / "training_state.pt",
        old_identity,
        new_identity,
        checkpoint_sha,
    )
    offsets = {
        "rank0.jsonl": state["rank_log_offsets"][0],
        "metrics.jsonl": state["metrics_offset"],
        "monitor_events.jsonl": state["monitor_offsets"]["events"],
        "monitor_evaluations.jsonl": state["monitor_offsets"]["evaluations"],
    }
    logs = {
        name: copy_prefix(old_output / name, output / name, offset)
        for name, offset in offsets.items()
    }
    validate_evaluations(output / "monitor_evaluations.jsonl", state["next_epoch"])
    monitor = MonitorEventWriter(output / "monitor_events.jsonl", resume=True)
    monitor.restore_offsets(state["monitor_offsets"], state["global_step"])
    recorder_for_run(output, config, new_identity)
    return {
        "next_epoch": state["next_epoch"],
        "global_step": state["global_step"],
        "parent_checkpoint_sha256": checkpoint_sha,
        "continuation_checkpoint_sha256": file_sha256(output / "training_state.pt"),
        "run_metadata_sha256": file_sha256(output / "run_metadata.json"),
        "logs": logs,
        "unchanged_checkpoint_fields": sorted(state.keys() - {"contract"}),
    }


def migrate(request_path: Path, request_sha: str) -> dict[str, object]:
    """Require a pinned ROOT authorization and native exact-cell profile before writing."""
    if os.name != "posix":
        raise RuntimeError("production migration must run on the POSIX receiver")
    request = pinned_json({"path": str(request_path), "sha256": request_sha})
    if (
        request.get("schema") != "wnn.gru-monitor-transition-request.v1"
        or request.get("approved") is not True
        or request.get("reviewer") != "ROOT"
    ):
        raise ValueError("separate explicit ROOT transition approval is missing")
    clock = pinned_json(request["first_profile_receipt"])
    deadline = float(clock["first_profile_epoch"]) + 86400 - 3600
    if request.get("not_after_epoch") != deadline or not time.time() < deadline:
        raise ValueError("continuation exceeds the original L4 first-profile/reserve deadline")
    owner = request["stopped_trainer_owner"]
    if (
        not isinstance(owner, dict)
        or type(owner.get("pid")) is not int
        or not owner.get("birth")
        or is_same_process(owner)
    ):
        raise ValueError("original exact PID/birth trainer has not exited")
    old_files, new_files = verify_capsules(request)
    metadata = pinned_json(request["old_metadata"])
    config = load_config(Path(request["new_config"]["path"]))
    pinned_json(request["new_config"])
    profile = pinned_json(request["approved_profile"])
    if (
        profile.get("reviewer") != "ROOT"
        or profile.get("updates") != 20
        or profile.get("approved") is not True
    ):
        raise ValueError("exact-cell20-update S3 profile needs independent ROOT approval")
    new_identity = profile["identity"]
    validate_profile_receipt(Path(request["approved_profile"]["path"]), new_identity)
    original = Path(request["old_output"]) / "training_state.pt"
    if file_sha256(original) != request["old_checkpoint_sha256"]:
        raise ValueError("parent checkpoint differs from the approved boundary")
    old_identity = torch.load(original, map_location="cpu", weights_only=False)["contract"]
    validate_transition(metadata["config"], config, old_identity, new_identity)
    for identity, files in ((old_identity, old_files), (new_identity, new_files)):
        if identity["source_files"] != trainer_files(files):
            raise ValueError("native capsule differs from the profile/checkpoint source identity")
        data = identity["data"]
        if (
            data.get("split_sha256") != SPLIT_SHA
            or len(data.get("fit_members", [])) != 10325
            or len(data.get("evaluation_members", [])) != 1873
            or data.get("normalization", {}).get("scope") != "all_fit"
            or data.get("normalization", {}).get("sequence_count") != 10325
            or data.get("teacher") is not None
        ):
            raise ValueError("transition differs from official train-only/full-valid ancestry")
    old_output, output = Path(request["old_output"]), Path(request["new_output"])
    if (
        Path(request["old_metadata"]["path"]).resolve()
        != (old_output / "run_metadata.json").resolve()
    ):
        raise ValueError("pinned old metadata is not inside the original run")
    recorder_for_run(old_output, metadata["config"], old_identity)
    proof = prepare_output(
        old_output, output, config, old_identity, new_identity, request["old_checkpoint_sha256"]
    )
    if (
        proof["next_epoch"] != request["expected_next_epoch"]
        or proof["global_step"] != request["expected_global_step"]
    ):
        raise ValueError("copied checkpoint differs from the ROOT-approved whole-epoch frontier")
    proof.update(
        {
            "schema": "wnn.gru-monitor-transition-proof.v1",
            "request_sha256": request_sha,
            "request": request,
            "parent_identity": old_identity,
            "continuation_identity": new_identity,
            "parent_run": str(old_output),
            "continuation_run": str(output),
            "training_lineage": "epochs before next_epoch produced under S1; later updates under source-gru-monitor-archive-r1 (S3)",
            "retained_S1_artifacts": "original run and all epoch indexes/checkpoints byte-unchanged",
            "external_retention": "archive_safe_state.py must be ready before canonical resume",
            "training_tensor_optimizer_scheduler_rng_changes": [],
            "requested_final_epoch": 12,
            "scientific_fit_count_increment": 0,
            "created_at_epoch": time.time(),
        }
    )
    write_json_new(output / "transition_proof.json", proof)
    write_json_new(
        output / "transition_ready.json",
        {
            "schema": "wnn.gru-monitor-transition-ready.v1",
            "proof_sha256": file_sha256(output / "transition_proof.json"),
            "checkpoint_sha256": proof["continuation_checkpoint_sha256"],
            "request_sha256": request_sha,
            "not_after_epoch": deadline,
        },
    )
    return {
        "output": str(output),
        "proof_sha256": file_sha256(output / "transition_proof.json"),
        "next_epoch": proof["next_epoch"],
        "global_step": proof["global_step"],
    }


def main() -> None:
    """Prepare one approved continuation; never stop processes, launch fits or touch GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.request, args.request_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
