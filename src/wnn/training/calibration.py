from __future__ import annotations

import argparse
import json
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch
from torch.utils.data import DataLoader

from wnn.run_identity import sample_order_identity
from wnn.training.checkpoints import resume_contract
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs
from wnn.training.reporting import emit
from wnn.training.targets import main_row_weights
from wnn.training_sampling import EpochOrderSampler
from wnn.training_views import origin_windows


@torch.no_grad()
def calibrate_distill_reference(
    args: argparse.Namespace,
    train_loader: DataLoader,
    prop_model: torch.nn.Module | None,
    provenance: dict[str, Any],
    device: torch.device,
    total: Any,
    *,
    environment: TrainingEnvironment | None = None,
) -> None:
    """Measure MAIN denominator on actual fit batches; do not fit or inspect eval rows."""
    environment = environment if environment is not None else TrainingEnvironment()
    from wnn.training_calibration import RecordedOrder, summarize_masses

    update_rows = args.update_rows or args.chunk
    masses, orders = ([], [])
    order = RecordedOrder(train_loader.sampler)
    original_sampler = train_loader.batch_sampler.sampler
    train_loader.batch_sampler.sampler = order
    for epoch in range(args.calibration_epochs):
        if isinstance(train_loader.sampler, EpochOrderSampler):
            train_loader.sampler.set_epoch(epoch)
        orders.append(sample_order_identity(train_loader))
        for batch, (x, target, scored, *rest) in enumerate(train_loader):
            x, target, scored = (x.to(device), target.to(device), scored.to(device))
            y = target[..., :2]
            valid = (
                torch.arange(x.shape[1], device=device)[None].expand(x.shape[:2])
                >= environment.warmup
            )
            cached = rest[0].to(device) if rest else None
            weights = main_row_weights(
                build_inputs(x, environment=environment),
                y,
                valid,
                scored,
                prop_model,
                cached,
                args,
                x,
                environment=environment,
            )
            mass = y.float().clamp(-2, 2).abs() * weights[..., None]
            windows = origin_windows(
                x.shape[1],
                update_rows,
                args.chunk,
                args.origin_view,
                args.origin_seed,
                epoch,
                batch,
            )
            masses.extend(float(mass[:, window].sum()) for window in windows)
        orders[-1].update(order.identity(args.batch))
    train_loader.batch_sampler.sampler = original_sampler
    report = summarize_masses(masses)
    report.update(
        {
            "format": "wnn.distill-reference-mass",
            "version": 1,
            "estimator": "mean over actual sampled fit batches and update windows",
            "calibration_epochs": args.calibration_epochs,
            "sample_order": orders,
            "contract": resume_contract(
                args, provenance, train_loader, total, environment=environment
            ),
        }
    )
    path = Path(args.calibrate_distill_reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    emit(
        json.dumps(
            {
                "reference_mass": report["reference_mass"],
                "calibration": str(path),
                "updates": report["updates"],
            }
        ),
        flush=True,
    )
