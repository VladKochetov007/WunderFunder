from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from transition_support import (
    pinned_json,
    trainer_files,
    validate_evaluations,
    validate_transition,
    verify_capsules,
    write_json_new,
)

from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_provenance import canonical_sha256, validate_profile_receipt


def inspect(template: dict) -> dict:
    """Validate the full old state/new approved identity before any trainer may be stopped."""
    old_files, new_files = verify_capsules(template)
    metadata = pinned_json(template["old_metadata"])
    pinned_json(template["new_config"])
    config = load_config(Path(template["new_config"]["path"]))
    profile = pinned_json(template["approved_profile"])
    if (
        profile.get("reviewer") != "ROOT"
        or profile.get("updates") != 20
        or profile.get("approved") is not True
    ):
        raise ValueError("exact S3 cell20-update ROOT-approved profile required before stop")
    identity = profile["identity"]
    validate_profile_receipt(Path(template["approved_profile"]["path"]), identity)
    path = Path(template["old_output"]) / "training_state.pt"
    with path.open("rb") as stream:
        loaded = torch.load(stream, map_location="cpu", weights_only=False)
    old = loaded["contract"]
    state = load_training_state(path, old, DistributedContext(0, 0, 1, torch.device("cpu"), False))
    validate_transition(metadata["config"], config, old, identity)
    if old["source_files"] != trainer_files(old_files) or identity["source_files"] != trainer_files(
        new_files
    ):
        raise ValueError("profile/checkpoint differs from exact source capsules")
    if not 1 <= state["next_epoch"] < 12 or state["global_step"] != state["next_epoch"] * 6460:
        raise ValueError("need an unfinished exact whole-epoch GRU boundary")
    # Read only the checkpointed full-valid prefix; unsafe later attempt bytes stay historical.
    data = (Path(template["old_output"]) / "monitor_evaluations.jsonl").read_bytes()
    prefix = data[: state["monitor_offsets"]["evaluations"]]
    with tempfile.NamedTemporaryFile(
        dir=Path(template["old_output"]).parents[2], suffix=".jsonl", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(prefix)
    try:
        validate_evaluations(temporary, state["next_epoch"])
    finally:
        temporary.unlink()
    return {
        "initial_next_epoch": state["next_epoch"],
        "contract_sha256": canonical_sha256(old),
        "metadata_sha256": template["old_metadata"]["sha256"],
        "new_identity_sha256": canonical_sha256(identity),
    }


def main() -> None:
    """Run read-only S3 CPU preflight and publish one immutable supervisor receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = pinned_json({"path": str(args.template), "sha256": args.template_sha256})
    write_json_new(args.output, inspect(template))


if __name__ == "__main__":
    main()
