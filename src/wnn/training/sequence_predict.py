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
from wnn.training.sequence_blend import compose_prediction
from wnn.training.sequence_blend_prediction import blend_producer, prediction_reference
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_epoch_prediction import (
    EPOCH_KIND,
    epoch_admission,
    validate_epoch_payload,
)
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import forward_context

# Checkpoint payloads contain tensors and JSON-compatible provenance, validated on load.


def load_endpoint(
    checkpoint: Path,
    device: torch.device,
    admission: dict[str, Any] | None = None,
    *,
    allow_blend_head: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a fixed-final endpoint or an explicitly admitted retained epoch."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if admission is not None:
        validate_epoch_payload(payload, admission)
    elif payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("unsupported endpoint; native/legacy weights require an explicit adapter")
    config = payload["config"]
    if config.get("blend_objective_alpha", 0) > 0 and not allow_blend_head:
        raise ValueError("composite-trained head requires explicit reference-bound export")
    if admission is None and (
        payload["completed_epochs"] != config["epochs"] or payload["epoch"] != config["epochs"] - 1
    ):
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
    precision_mode = str(getattr(model, "precision_mode", "fp32"))
    for start in range(0, len(x), chunk_rows):
        with forward_context(precision_mode, device.type):
            prediction, hints, state = model(
                x[start : start + chunk_rows].unsqueeze(0).to(device), state
            )
        outputs.append(torch.cat((prediction.float(), hints.float()), -1).squeeze(0).cpu().numpy())
    values = np.concatenate(outputs).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite endpoint output")
    return values


def _producer(checkpoint: Path, payload: dict[str, Any]) -> dict[str, Any]:
    data = payload["data"]
    producer = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "split_sha256": data["split_sha256"],
        "fit_members": data["fit_members"],
        "teacher": data["teacher"],
        "config": payload["config"],
        "provenance": payload["provenance"],
    }
    if "reference" in data:
        producer["reference"] = data["reference"]
    return producer


def export_predictions(args: argparse.Namespace) -> None:
    """Export disjoint rank shards, publishing a complete manifest only after all ranks finish."""
    # Import the shared runtime here to keep all precision policy in one implementation.
    from wnn.training.sequence_distributed import (
        collective_output_setup,
        coordinated_call,
        initialize_distributed,
    )

    runtime = initialize_distributed()
    device, rank, world = runtime.device, runtime.rank, runtime.world_size
    split = load_campaign_split(args.split_manifest, args.data_root)
    admission = None
    index_path, index_hash = (
        getattr(args, "epoch_index", None),
        getattr(args, "epoch_index_sha256", None),
    )
    if bool(index_path) != bool(index_hash):
        raise ValueError("epoch evaluation requires both --epoch-index and --epoch-index-sha256")
    if index_path is not None:
        raw_split = json.loads(args.split_manifest.read_text())
        if (
            args.membership != "evaluation"
            or raw_split.get("schema") != "wnn.sequence-inner-split.v1"
        ):
            raise ValueError("retained epochs are admitted only for inner evaluation")
        admission = epoch_admission(args.checkpoint, index_path, index_hash)
    reference_path = getattr(args, "reference_cache", None)
    if reference_path is not None and admission is not None:
        raise ValueError("composite exports require the fixed-final endpoint, not epoch selection")
    model, payload = load_endpoint(
        args.checkpoint, device, admission, allow_blend_head=reference_path is not None
    )
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
    reference, alpha = prediction_reference(
        reference_path, getattr(args, "blend_alpha", None), payload, split, members, args.membership
    )
    collective_output_setup(args.output, False, runtime)
    if world > 1:
        dist.barrier()
    dataset = SequenceRecords(args.data_root, members, 0)
    entries = []
    for index in range(rank, len(dataset), world):
        source, group = members[index]
        values = predict_sequence(
            model, dataset[index]["x"], payload["config"]["chunk_rows"], device
        )
        if reference is not None:
            baseline, _ = reference.read(source, group, 0)
            values = compose_prediction(torch.from_numpy(values[:, :2]), baseline, alpha).numpy()
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
    if reference is not None:
        coordinated_call(
            lambda: reference.verify_unchanged() if rank == 0 else None,
            "reference cache stability after export",
            runtime,
        )
    if rank == 0:
        all_entries = [
            entry
            for worker in range(world)
            for entry in json.loads((args.output / f"rank-{worker}.json").read_text())
        ]
        entry_map = {(entry["split"], entry["row_group"]): entry for entry in all_entries}
        if len(all_entries) != len(members) or set(entry_map) != set(members):
            raise ValueError("missing or duplicate export shards")
        producer = _producer(args.checkpoint, payload)
        if reference is not None:
            producer = blend_producer(producer, reference.identity, alpha)
        if admission is not None:
            producer.update({"kind": EPOCH_KIND, "epoch_admission": admission})
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
            "producer": producer,
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
    parser.add_argument("--epoch-index", type=Path)
    parser.add_argument("--epoch-index-sha256")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--blend-alpha", type=float)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) < 1:
        parser.error("invalid distributed world size")
    export_predictions(args)
