#!/usr/bin/env python3
"""Check a family-head-free core checkpoint against its source student tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--destination-run", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = root / "artifacts/runs" / args.source_run
    destination_dir = root / "artifacts/runs" / args.destination_run
    source_path = source_dir / f"model_ep{args.epoch}.pt"
    destination_path = destination_dir / f"model_ep{args.epoch}.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    destination = torch.load(destination_path, map_location="cpu", weights_only=False)
    source_state = source["model"]
    destination_state = destination["model"]
    if set(source_state) != set(destination_state):
        raise ValueError("core derivation changed tensor names")
    expected_rows = 23
    source_rows = int(source_state["head.2.weight"].shape[0])
    if source_rows != expected_rows + 8:
        raise ValueError(f"expected 31 source output rows, found {source_rows}")
    if destination_state["head.2.weight"].shape[0] != expected_rows:
        raise ValueError("derived core does not have the 23 declared primary head rows")
    if destination_state["head.2.bias"].shape[0] != expected_rows:
        raise ValueError("derived core bias does not have 23 declared primary head rows")

    changed = []
    for name, value in source_state.items():
        derived = destination_state[name]
        if name == "head.2.weight":
            if not torch.equal(derived, value[:expected_rows]):
                raise ValueError("primary output-weight rows changed during core derivation")
        elif name == "head.2.bias":
            if not torch.equal(derived, value[:expected_rows]):
                raise ValueError("primary output-bias rows changed during core derivation")
        elif not torch.equal(derived, value):
            changed.append(name)
    if changed:
        raise ValueError(f"non-family tensors changed during core derivation: {changed[:8]}")

    derivation = {
        "source_run": args.source_run,
        "source_epoch": args.epoch,
        "dropped_family_rows": 8,
    }
    if destination.get("core_derivation") != derivation:
        raise ValueError("checkpoint core_derivation metadata disagrees with source identity")
    for holder in (destination.get("args"), (destination.get("provenance") or {}).get("args")):
        if isinstance(holder, dict) and (
            holder.get("distill_family_head") is not False
            or holder.get("distill_family_weight") != 0.0
        ):
            raise ValueError("derived checkpoint still declares the family head active")

    source_provenance_path = source_dir / "provenance.json"
    destination_provenance_path = destination_dir / "provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text())
    destination_provenance = json.loads(destination_provenance_path.read_text())
    if destination_provenance.get("core_derivation") != derivation:
        raise ValueError("provenance core_derivation metadata disagrees")
    if (destination_provenance.get("fit_row_groups") != source_provenance.get("fit_row_groups")
            or destination_provenance.get("split_manifest_sha256") != source_provenance.get("split_manifest_sha256")):
        raise ValueError("core derivation changed fit membership or split binding")
    report = {
        "schema": "wnn.independent-core-derivation-audit.v1",
        "status": "passed",
        "source_run": args.source_run,
        "destination_run": args.destination_run,
        "epoch": args.epoch,
        "source_checkpoint_sha256": sha256(source_path),
        "destination_checkpoint_sha256": sha256(destination_path),
        "source_provenance_sha256": sha256(source_provenance_path),
        "destination_provenance_sha256": sha256(destination_provenance_path),
        "source_head_rows": source_rows,
        "core_head_rows": expected_rows,
        "dropped_family_rows": 8,
        "non_family_tensors_bitwise_equal": True,
        "primary_output_rows_bitwise_equal": True,
        "fit_membership_and_split_binding_unchanged": True,
        "limitations": [
            "This validates the core conversion only; it does not grade model quality or validate a served blend."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source": args.source_run, "core": args.destination_run}))


if __name__ == "__main__":
    main()
