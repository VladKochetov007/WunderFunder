"""Bounded fit-only gradient decomposition for paired GRU256x2 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from gradient_decomposition_core import (
    BATCH,
    CHUNK,
    GPU_LIMIT,
    STARTS,
    measure,
)
from gru_checkpoint_io import load_gru_models

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.features import N_EVENT_FEATURES
from wnn.outcomes import sequence_class, structure_labels
from wnn.resources import configure_strict_fp32, configure_torch
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs
from wnn.training.prediction_inputs import configure_prediction_inputs
from wnn.training.targets import next_change_labels


def digest(data: bytes) -> str:
    """SHA-256 for a small in-memory identity."""
    return hashlib.sha256(data).hexdigest()


def group_digest(groups: Sequence[int], *, sorted_membership: bool = False) -> str:
    """Hash row-group IDs in signed little-endian int64 form."""
    values = sorted(map(int, groups)) if sorted_membership else list(map(int, groups))
    return digest(np.asarray(values, dtype="<i8").tobytes())


def state_digest(model: torch.nn.Module) -> str:
    """Hash all fitted parameters and normalization buffers."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        h.update(np.asarray(tensor.shape, dtype="<i8").tobytes() + tensor.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    """Run the fixed sample on one CUDA device and emit a content-bound JSON result."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--panel", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--deadline-seconds", type=float, default=300)
    args = cli.parse_args()
    if not 0 < args.deadline_seconds <= 300:
        cli.error("--deadline-seconds must be in (0, 300]")
    if args.output.exists():
        raise FileExistsError(args.output)
    panel_path = args.panel.resolve(strict=True)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    project, source = Path(panel["paths"]["project_root"]), Path(panel["paths"]["source_root"])
    split_path = Path(panel["paths"]["split_manifest"])
    split = load_campaign_split(split_path, project)
    if len(split.train_groups) != 10325 or len(split.evaluation_groups) != 1873:
        raise ValueError("official fit/validation sequence counts differ")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires the admitted single CUDA device")
    torch.cuda.set_device(device)
    os.environ["WNN_THREADS"] = "8"
    cap = min(GPU_LIMIT, int(os.environ.get("WNN_GPU_MEMORY_BYTES", str(GPU_LIMIT))))
    os.environ["WNN_GPU_MEMORY_BYTES"] = str(cap)
    configure_torch()
    configure_strict_fp32()
    torch.set_num_threads(8)
    control_cells = sorted(
        [
            cell
            for cell in panel["cells"]
            if cell.get("configuration_id") == "gru256x2_control"
            and cell.get("role") == "control"
            and cell.get("seed") in (1001, 1002)
        ],
        key=lambda cell: cell["seed"],
    )
    models, fit_order = load_gru_models(
        panel,
        split,
        device,
        [cell["cell_id"] for cell in control_cells],
        (2, 3),
        control_recipe=True,
        training_mode=True,
    )
    for record in models:
        record["weight_sha256_before"] = state_digest(record["model"])
    selected = [g for g in fit_order if g not in CORRUPT_TRAIN_ROW_GROUPS][: BATCH * 8]
    if len(selected) != BATCH * 8:
        raise ValueError("fit order lacks eight complete non-corrupt B32 blocks")
    env = TrainingEnvironment(root=project, source_root=source)
    configure_prediction_inputs(env, models[0]["args"])
    if env.inputs.tiedlead:
        raise ValueError("this native211 control cannot use tied-lead inputs")
    data = SequenceFile(project / "datasets" / "train.parquet")
    results, deadline = [], time.monotonic() + args.deadline_seconds
    for batch_i, start in enumerate(STARTS):
        groups = selected[batch_i * BATCH : (batch_i + 1) * BATCH]
        seqs = [data.read(group, targets=True, mask=False) for group in groups]
        if any(s.targets is None for s in seqs):
            raise ValueError("fit source is missing target columns")
        raw = np.stack([s.features for s in seqs])
        target_np = np.stack([s.targets for s in seqs]).astype(np.float32, copy=False)
        x = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
        inputs = build_inputs(x, environment=env)
        del x, raw, target_np
        if inputs.shape != (BATCH, SEQUENCE_LENGTH, 112 + N_EVENT_FEATURES):
            raise ValueError(f"causal input shape differs: {tuple(inputs.shape)}")
        valid = (torch.arange(SEQUENCE_LENGTH, device=device)[None] >= WARMUP).expand(
            BATCH, -1
        )
        spread, branch, cls = structure_labels(y, sequence_class(y))
        nx_rows, nx_up, nx_ok = next_change_labels(y, valid)
        labels = (valid, spread, branch, cls, nx_rows, nx_up, nx_ok)
        for record in models:
            measured = measure(record, inputs, y, labels, start, deadline)
            results.append(
                {
                    "seed": record["seed"],
                    "epoch": record["epoch"],
                    "batch_index": batch_i,
                    "row_groups": groups,
                    "row_groups_ordered_sha256": group_digest(groups),
                    "window": {"start": start, "stop": start + CHUNK, "context_rows": start},
                    **measured,
                }
            )
        del inputs, y, labels, spread, branch, cls, nx_rows, nx_up, nx_ok
    for record in models:
        after = state_digest(record["model"])
        if after != record["weight_sha256_before"] or any(
            p.grad is not None for p in record["model"].parameters()
        ):
            raise RuntimeError("weights/buffers changed or gradients were left attached")
        record["weight_sha256_after"] = after
    summary = []
    for seed in (1001, 1002):
        for epoch in (2, 3):
            rows = [r for r in results if r["seed"] == seed and r["epoch"] == epoch]
            shared = [r["gradient"]["groups"]["shared"] for r in rows]
            ratios = [x["aux_to_main_norm_ratio"] for x in shared]
            cosines = [x["cosine"]["main_vs_aux_sum"] for x in shared]
            summary.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "batches": len(rows),
                    "shared_aux_to_main_norm_ratio_mean": (
                        float(np.mean(ratios)) if all(x is not None for x in ratios) else None
                    ),
                    "shared_main_vs_aux_sum_cosine_mean": (
                        float(np.mean(cosines)) if all(x is not None for x in cosines) else None
                    ),
                    "shared_component_norm_means": {
                        name: float(np.mean([x["norm"][name] for x in shared]))
                        for name in (
                            "main_wmse",
                            "weighted_struct",
                            "weighted_next",
                            "combined",
                        )
                    },
                }
            )
    output = {
        "schema": "wnn.gru-control-gradient-decomposition.v1",
        "status": "complete",
        "mode": "model.train(); dropout=0; autograd only; no optimizer/update; no BatchNorm; strict FP32; TF32/autocast disabled",
        "purpose": "fit-only gradient diagnostic; no validation or is_scored mask loaded",
        "objective": {
            "main": "chunk-normalized |clip(y,-2,2)| weighted MSE",
            "structural": "0.3 * canonical spread/branch/class CE",
            "next_change": "0.3 * canonical next-change log-distance MSE + sign BCE",
            "warmup_rows": WARMUP,
            "auxiliary_labels_use_full_fit_sequence_targets": True,
            "is_scored_used_as_loss_mask": False,
            "training_clip_norm": 1.0,
        },
        "limits": {
            "max_wall_seconds": args.deadline_seconds,
            "gpu_allocator_cap_bytes": cap,
            "cpu_threads": 8,
        },
        "bindings": {
            "panel_sha256": file_sha256(panel_path),
            "diagnostic_script_sha256": file_sha256(Path(__file__).resolve()),
            "diagnostic_core_sha256": file_sha256(
                Path(__file__).with_name("gradient_decomposition_core.py")
            ),
            "source_binding_sha256": panel["identity"]["source_binding_sha256"],
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_manifest_sha256": split.sha256,
            "split_file_sha256": file_sha256(split_path),
            "fit_order_sha256_seed1001": group_digest(fit_order),
            "fit_membership_sorted_sha256": group_digest(fit_order, sorted_membership=True),
            "selected_groups_ordered_sha256": group_digest(selected),
            "selected_groups_sorted_sha256": group_digest(selected, sorted_membership=True),
            "excluded_corrupt_fit_groups": sorted(set(fit_order) & CORRUPT_TRAIN_ROW_GROUPS),
            "window_plan_sha256": digest(
                json.dumps(
                    [{"batch": i, "start": s, "stop": s + CHUNK} for i, s in enumerate(STARTS)],
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ),
            "checkpoints": [
                {
                    k: r[k]
                    for k in (
                        "seed",
                        "epoch",
                        "run_name",
                        "checkpoint",
                        "checkpoint_sha256",
                        "weight_sha256_before",
                        "weight_sha256_after",
                    )
                }
                for r in models
            ],
        },
        "batch_count": len(results),
        "checkpoint_summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
