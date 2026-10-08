#!/usr/bin/env python3
"""Independently verify a v2 core derivation against its actual producer bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    """Hash a file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON provenance sidecar."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_family_head_disabled(holder: Any, label: str) -> None:
    """Reject a derived core whose model arguments still enable family rows."""
    if isinstance(holder, dict) and (
        holder.get("distill_family_head") is not False
        or float(holder.get("distill_family_weight", -1.0)) != 0.0
    ):
        raise ValueError(f"{label} still declares family-head training active")


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
    checkpoint_name = f"model_ep{args.epoch}.pt"
    source_path = source_dir / checkpoint_name
    destination_path = destination_dir / checkpoint_name
    source_provenance_path = source_dir / "provenance.json"
    destination_provenance_path = destination_dir / "provenance.json"

    source_hash = sha256_file(source_path)
    destination_hash = sha256_file(destination_path)
    source_provenance_hash = sha256_file(source_provenance_path)
    destination_provenance_hash = sha256_file(destination_provenance_path)
    source_provenance = load_json(source_provenance_path)
    destination_provenance = load_json(destination_provenance_path)

    if source_provenance.get("checkpoint_sha256", {}).get(checkpoint_name) != source_hash:
        raise ValueError("source sidecar does not bind the actual 31-row checkpoint bytes")
    if destination_provenance.get("checkpoint_sha256", {}).get(checkpoint_name) != destination_hash:
        raise ValueError("destination sidecar does not bind the actual 23-row core bytes")

    derivation = {
        "source_run": args.source_run,
        "source_epoch": args.epoch,
        "source_checkpoint_sha256": source_hash,
        "source_provenance_sha256": source_provenance_hash,
        "dropped_family_rows": 8,
    }
    if destination_provenance.get("core_derivation") != derivation:
        raise ValueError("destination sidecar derivation metadata disagrees with actual source")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    destination = torch.load(destination_path, map_location="cpu", weights_only=False)
    source_state = source.get("model")
    destination_state = destination.get("model")
    if not isinstance(source_state, dict) or not isinstance(destination_state, dict):
        raise ValueError("source or destination checkpoint has no model state dictionary")
    if set(source_state) != set(destination_state):
        raise ValueError("core derivation changed the model tensor-name set")

    source_weight = source_state.get("head.2.weight")
    source_bias = source_state.get("head.2.bias")
    core_weight = destination_state.get("head.2.weight")
    core_bias = destination_state.get("head.2.bias")
    if source_weight is None or source_bias is None or core_weight is None or core_bias is None:
        raise ValueError("checkpoint is missing primary output-head tensors")
    if tuple(source_weight.shape) != (31, 256) or tuple(source_bias.shape) != (31,):
        raise ValueError("source is not the expected 31-row student checkpoint")
    if tuple(core_weight.shape) != (23, 256) or tuple(core_bias.shape) != (23,):
        raise ValueError("destination is not the expected 23-row family-free core")

    for name, value in source_state.items():
        derived = destination_state[name]
        if name == "head.2.weight":
            if not torch.equal(derived, value[:23]):
                raise ValueError("retained primary output weights differ from source rows 0:23")
        elif name == "head.2.bias":
            if not torch.equal(derived, value[:23]):
                raise ValueError("retained primary output bias differs from source rows 0:23")
        elif not torch.equal(derived, value):
            raise ValueError(f"non-family tensor changed in core derivation: {name}")

    checkpoint_derivation = destination.get("core_derivation")
    if checkpoint_derivation != derivation:
        raise ValueError("checkpoint embedded derivation metadata disagrees with source bytes")
    require_family_head_disabled(destination.get("args"), "checkpoint args")
    require_family_head_disabled(
        (destination.get("provenance") or {}).get("args"), "embedded provenance args"
    )
    require_family_head_disabled(destination_provenance.get("args"), "sidecar provenance args")
    if destination_provenance.get("fit_row_groups") != source_provenance.get(
        "fit_row_groups"
    ) or destination_provenance.get("split_manifest_sha256") != source_provenance.get(
        "split_manifest_sha256"
    ):
        raise ValueError("core conversion changed fit membership or split binding")

    report = {
        "schema": "wnn.independent-core-identity-v2-audit.v1",
        "status": "passed",
        "source_run": args.source_run,
        "destination_run": args.destination_run,
        "epoch": args.epoch,
        "source_checkpoint_sha256": source_hash,
        "destination_checkpoint_sha256": destination_hash,
        "source_provenance_sha256": source_provenance_hash,
        "destination_provenance_sha256": destination_provenance_hash,
        "source_checkpoint_map_sha256": source_provenance["checkpoint_sha256"][checkpoint_name],
        "destination_checkpoint_map_sha256": destination_provenance["checkpoint_sha256"][
            checkpoint_name
        ],
        "source_head_rows": 31,
        "core_head_rows": 23,
        "dropped_family_rows": 8,
        "non_family_tensors_bitwise_equal": True,
        "primary_output_rows_bitwise_equal": True,
        "fit_membership_and_split_binding_unchanged": True,
        "core_map_binding": "actual_destination_core_checkpoint_sha256",
        "limitations": [
            "This validates checkpoint derivation identity and fit-split preservation only; it makes no quality claim."
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "source_checkpoint_sha256": source_hash,
                "core_checkpoint_sha256": destination_hash,
            }
        )
    )


if __name__ == "__main__":
    main()
