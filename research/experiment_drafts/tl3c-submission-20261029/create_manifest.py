from __future__ import annotations

import argparse
import logging
from pathlib import Path

from candidate_contract import (
    MEMBERS,
    PR49_COMMIT,
    ancestor_names,
    endpoint,
    expected_source,
    split_contract,
    verify_ancestry,
    verify_candidate_args,
    verify_metadata,
)
from candidate_io import JsonObject, file_identity, object_value, write_json


def pin_run(runs_dir: Path, run: str) -> JsonObject:
    """Pin fresh endpoint and sidecar bytes without loading checkpoint tensors."""
    directory = runs_dir / run
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"run directory missing or symlinked: {directory}")
    return {
        "run": run,
        "checkpoint": file_identity(directory / f"model_ep{endpoint(run)}.pt"),
        "provenance": file_identity(directory / "provenance.json"),
        "training_source_sha256": expected_source(run),
    }


def main() -> None:
    """Freeze actual receiver artifacts after read-only split/source/lineage verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file; existing manifests are immutable")
    identity = file_identity(args.split)
    split = split_contract(identity)
    members = []
    for run in MEMBERS:
        record = pin_run(args.runs_dir, run)
        record["ancestors"] = [pin_run(args.runs_dir, name) for name in ancestor_names(run)]
        provenance = verify_metadata(record, split)
        verify_candidate_args(object_value(provenance["args"], "args"), run)
        verify_ancestry(record, split)
        members.append(record)
    write_json(
        args.output,
        {
            "schema": "tl3c-submission-inputs-v1",
            "training_reference_commit": PR49_COMMIT,
            "split_manifest": identity,
            "members": members,
            "teacher_prediction_consumption": "not proven by legacy provenance; receiver gate pending",
        },
    )
    logging.info("Frozen artifact identities: %s", args.output.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
