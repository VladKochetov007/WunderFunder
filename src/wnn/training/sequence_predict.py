from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_models import build_sequence_model

# Checkpoint payloads contain tensors and JSON-compatible provenance, validated on load.


def load_endpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load only a completed fixed-final endpoint with explicit FP32 ancestry."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("unsupported endpoint; native/legacy weights require an explicit adapter")
    config = payload["config"]
    if payload["completed_epochs"] != config["epochs"] or payload["epoch"] != config["epochs"] - 1:
        raise ValueError("cache/score requires the fixed-final completed checkpoint")
    for name, value in payload["model"].items():
        if value.is_floating_point() and (
            value.dtype != torch.float32 or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid strict-FP32 checkpoint tensor: {name}")
    model = build_sequence_model(config["model"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    x: torch.Tensor,
    chunk_rows: int,
    device: torch.device,
) -> np.ndarray:
    """Reset once per sequence, forward every warm-up row, and carry bounded causal state."""
    state = None
    outputs = []
    for start in range(0, len(x), chunk_rows):
        prediction, hints, state = model(
            x[start : start + chunk_rows].unsqueeze(0).to(device), state
        )
        outputs.append(torch.cat((prediction, hints), -1).squeeze(0).cpu().numpy())
    values = np.concatenate(outputs).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite endpoint output")
    return values


def _producer(checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    data = payload["data"]
    return {
        "checkpoint_sha256": file_sha256(checkpoint),
        "split_sha256": data["split_sha256"],
        "fit_members": data["fit_members"],
        "teacher": data["teacher"],
        "config": payload["config"],
        "provenance": payload["provenance"],
    }


def export_predictions(args: argparse.Namespace) -> None:
    """Export disjoint rank shards, publishing a complete manifest only after all ranks finish."""
    # Import the shared runtime here to keep all precision policy in one implementation.
    from wnn.training.sequence_distributed import (
        collective_output_setup,
        initialize_distributed,
    )

    runtime = initialize_distributed()
    device, rank, world = runtime.device, runtime.rank, runtime.world_size
    split = load_campaign_split(args.split_manifest, args.data_root)
    model, payload = load_endpoint(args.checkpoint, device)
    if payload["data"]["split_sha256"] != split.sha256:
        raise ValueError("endpoint does not exclude this evaluation fold")
    fitting = [("train", g) for g in split.train_groups] + [
        ("valid", g) for g in split.valid_groups
    ]
    if payload["data"]["fit_members"] != [list(member) for member in fitting]:
        raise ValueError("endpoint fitting membership is not this source-qualified split")
    members = (
        fitting if args.membership == "fit" else [("valid", g) for g in split.evaluation_groups]
    )
    collective_output_setup(args.output, False, runtime)
    if world > 1:
        dist.barrier()
    dataset = SequenceRecords(
        args.data_root,
        members,
        0,
        input_encoding=payload["config"]["model"].get("input_encoding", "raw112"),
        training_mask="need_prediction" if args.membership == "fit" else "is_scored",
    )
    entries = []
    for index in range(rank, len(dataset), world):
        source, group = members[index]
        values = predict_sequence(
            model, dataset[index]["x"], payload["config"]["chunk_rows"], device
        )
        if args.membership == "evaluation":
            values = values[:, :2].copy()
        name = f"{source}-{group}.npy"
        target = args.output / name
        temporary = args.output / f"{name}.partial"
        with temporary.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        temporary.replace(target)
        entries.append(
            {"split": source, "row_group": group, "file": name, "sha256": file_sha256(target)}
        )
    part = args.output / f"rank-{rank}.json"
    part.write_text(json.dumps(entries, sort_keys=True))
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_entries = [
            entry
            for worker in range(world)
            for entry in json.loads((args.output / f"rank-{worker}.json").read_text())
        ]
        entry_map = {(entry["split"], entry["row_group"]): entry for entry in all_entries}
        if len(all_entries) != len(members) or set(entry_map) != set(members):
            raise ValueError("missing or duplicate export shards")
        manifest = {
            "schema": "sequence-teacher-cache-v1"
            if args.membership == "fit"
            else "sequence-predictions-v1",
            "complete": True,
            "input_schema": "raw112-v1",
            "membership": args.membership,
            "split_sha256": split.sha256,
            "hint_dim": payload["config"]["model"]["hint_dim"] if args.membership == "fit" else 0,
            "sequence_length": SEQUENCE_LENGTH,
            "producer": _producer(args.checkpoint, payload),
            "world_size": world,
            "sequences": [entry_map[member] for member in members],
        }
        temporary = args.output / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "manifest.json")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    """Export complete fitting-only KD targets or held-out predictions on allocated GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership", choices=("fit", "evaluation"), required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) < 1:
        parser.error("invalid distributed world size")
    export_predictions(args)
