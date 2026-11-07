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
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    SEQUENCE_INNER_SPLIT_SCHEMA,
    file_sha256,
    load_campaign_split,
)
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_epoch_prediction import (
    EPOCH_KIND,
    epoch_admission,
    validate_epoch_payload,
)
from wnn.training.sequence_fit_utils import collate_sequences
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import forward_context
from wnn.training.sequence_swa import SWA_SCHEMA, validate_swa_payload

# Checkpoint payloads contain tensors and JSON-compatible provenance, validated on load.


def load_endpoint(
    checkpoint: Path, device: torch.device, admission: dict[str, Any] | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a fixed-final endpoint or an explicitly admitted retained epoch."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_schema") == SWA_SCHEMA:
        if admission is not None:
            raise ValueError("SWA readouts cannot be admitted as a single retained epoch")
        validate_swa_payload(payload)
    elif admission is not None:
        validate_epoch_payload(payload, admission)
    elif payload.get("checkpoint_schema") != "wnn.sequence-fit.v1":
        raise ValueError("unsupported endpoint; native/legacy weights require an explicit adapter")
    config = payload["config"]
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
    if x.ndim != 2:
        raise ValueError("single-sequence prediction requires a [rows,features] tensor")
    return predict_sequence_batch(model, x.unsqueeze(0), chunk_rows, device)[0]


@torch.no_grad()
def predict_sequence_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    chunk_rows: int,
    device: torch.device,
) -> np.ndarray:
    """Predict independent complete sequences together with one causal state per sequence."""
    if x.ndim != 3 or x.shape[0] < 1 or chunk_rows <= 0:
        raise ValueError("batched prediction requires [batch,rows,features] and positive chunks")
    state = None
    outputs = []
    precision_mode = str(getattr(model, "precision_mode", "fp32"))
    for start in range(0, x.shape[1], chunk_rows):
        with forward_context(precision_mode, device.type):
            prediction, hints, state = model(x[:, start : start + chunk_rows].to(device), state)
        outputs.append(torch.cat((prediction.float(), hints.float()), -1))
    values = torch.cat(outputs, dim=1).cpu().numpy().astype(np.float32, copy=False)
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
    report_only_primary_valid = bool(getattr(args, "report_only_primary_valid", False))
    admission = None
    index_path, index_hash = (
        getattr(args, "epoch_index", None),
        getattr(args, "epoch_index_sha256", None),
    )
    if bool(index_path) != bool(index_hash):
        raise ValueError("epoch evaluation requires both --epoch-index and --epoch-index-sha256")
    if index_path is not None:
        if args.membership != "evaluation":
            raise ValueError("retained epochs are inference-only and require evaluation membership")
        if split.schema in (SEQUENCE_INNER_SPLIT_SCHEMA, MULTISTAGE_AUXILIARY_SPLIT_SCHEMA):
            if report_only_primary_valid:
                raise ValueError("report-only primary mode requires the chronological split")
        elif split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
            if not report_only_primary_valid:
                raise ValueError(
                    "chronological retained epochs require --report-only-primary-valid"
                )
        else:
            raise ValueError(
                "retained epochs are supported only for inner or chronological reports"
            )
        admission = epoch_admission(args.checkpoint, index_path, index_hash)
    elif report_only_primary_valid:
        raise ValueError("--report-only-primary-valid requires a retained epoch checkpoint")
    model, payload = load_endpoint(args.checkpoint, device, admission)
    if payload["data"]["split_sha256"] != split.sha256:
        raise ValueError("endpoint does not exclude this evaluation fold")
    fitting = [("train", g) for g in split.train_groups]
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        fitting += [("valid", g) for g in split.valid_groups]
    if payload["data"]["fit_members"] != [list(member) for member in fitting]:
        raise ValueError("endpoint fitting membership is not this source-qualified split")
    batch_size = getattr(args, "batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("prediction batch size must be a positive integer")
    members = (
        fitting
        if args.membership == "fit"
        else [(split.evaluation_source, g) for g in split.evaluation_groups]
    )
    collective_output_setup(args.output, False, runtime)
    if world > 1:
        dist.barrier()
    input_feature_mode = payload["config"]["model"].get("input_feature_mode", "raw112")
    dataset = SequenceRecords(
        args.data_root,
        members,
        0,
        training_mask=split.evaluation_mask if args.membership == "evaluation" else "is_scored",
        input_feature_mode=input_feature_mode,
    )
    entries = []
    rank_indices = list(range(rank, len(dataset), world))
    for batch_start in range(0, len(rank_indices), batch_size):
        batch_indices = rank_indices[batch_start : batch_start + batch_size]
        batch = collate_sequences(dataset, batch_indices)
        batch_values = predict_sequence_batch(
            model, batch["x"], payload["config"]["chunk_rows"], device
        )
        for batch_offset, index in enumerate(batch_indices):
            source, group = members[index]
            values = batch_values[batch_offset]
            if args.membership == "evaluation":
                values = values[:, :2].copy()
            name = f"{source}-{group}.npy"
            target = args.output / name
            temporary = args.output / f"{name}.partial"
            with temporary.open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(target)
            entries.append(
                {
                    "split": source,
                    "row_group": group,
                    "file": name,
                    "sha256": file_sha256(target),
                }
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
        producer = _producer(args.checkpoint, payload)
        if admission is not None:
            producer.update({"kind": EPOCH_KIND, "epoch_admission": admission})
            if report_only_primary_valid:
                producer["primary_valid_report_only"] = True
        manifest = {
            "schema": "sequence-teacher-cache-v1"
            if args.membership == "fit"
            else "sequence-predictions-v1",
            "complete": True,
            "input_schema": payload["data"]["input_schema"],
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
    parser.add_argument("--report-only-primary-valid", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sequences per forward batch; each sequence retains an independent reset state",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) < 1:
        parser.error("invalid distributed world size")
    export_predictions(args)
