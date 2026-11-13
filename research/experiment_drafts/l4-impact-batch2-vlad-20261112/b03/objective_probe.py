"""Receiver-only numerical oracle for the B03 target, clock, normalization and gradients."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path

import torch

from wnn.data import I0
from wnn.training.sequence_distributed import DistributedContext, configure_strict_fp32
from wnn.training.sequence_objectives import (
    global_window_denominators,
    objective_numerators,
    scaled_objective,
    since_event_row_weights,
)
from wnn.training.sequence_provenance import file_sha256, source_identity

PACKET = Path(__file__).resolve().parent
B03_MODE = "standalone_b03_convex_kd_wmse_v1"
ATOL = 2e-6


def causal_weights(device: torch.device) -> torch.Tensor:
    """Verify that the full-sequence clock resets at a joint price event and crosses row 1000."""
    inputs = torch.zeros(1, 1110, 112, device=device)
    inputs[:, 7:, I0.bid_px.start] = 1
    inputs[:, 17:, I0.bid_px.start] = 2
    inputs[:, 17:, I0.ask_px.start] = 1
    row = torch.arange(inputs.shape[1], device=device)
    age = torch.where(row >= 17, row - 17, row).float()
    expected = (1 / (1 + age / 32)).clamp_min(0.25).unsqueeze(0)
    weights = since_event_row_weights(inputs, 32.0, 0.25)
    torch.testing.assert_close(weights, expected, rtol=0, atol=1e-7)
    future = inputs.clone()
    future[:, 1002:, I0.bid_px.start] = 4
    future[:, 1002:, I0.ask_px.start] = 4
    changed = since_event_row_weights(future, 32.0, 0.25)
    torch.testing.assert_close(weights[:, :1002], changed[:, :1002], rtol=0, atol=0)
    if changed[0, 1002].item() != 1 or weights[0, 1000].item() != 0.25:
        raise ValueError("future perturbation or TBPTT age continuity failed")
    return weights[:, [0, 7, 17, 18, 999, 1000]]


def cell_probe(
    age: torch.Tensor, tau: int, alpha: float, context: DistributedContext
) -> dict[str, float | int]:
    """Check scalar loss, prediction gradient, detached teacher and alpha-zero invariance."""
    device = context.device
    y = torch.tensor(
        [[4.0, -1.0], [-3, 0.5], [1.25, 0.0], [-2, 1.25], [-0.375, 0.5], [0.75, -0.875]],
        device=device,
    ).unsqueeze(0)
    t = torch.tensor(
        [[-3.5, 0.25], [2, -1.5], [-0.75, 3.5], [1.5, 4], [0.25, -0.75], [-1.25, 1.5]],
        device=device,
    ).unsqueeze(0)
    mask = torch.tensor([[True, False, True, True, True, True]], device=device)
    t[~mask] = float("nan")
    y[~mask] = float("nan")
    teacher = t.requires_grad_()
    z = torch.linspace(-0.75, 0.75, y.numel(), device=device).reshape_as(y).requires_grad_()
    hints = z.detach()[..., :0]
    denom = global_window_denominators(y, mask, hints, [(0, 3), (3, 6)], context)
    label = torch.where(mask[..., None], y, 0.0).clamp(-2, 2)
    teach = torch.where(mask[..., None], teacher.detach(), 0.0).clamp(-2, 2)
    weights = label.abs() * (age[..., None] if tau else 1.0)
    mass = label.abs().sum()
    target = (1.0 - alpha) * label + alpha * teach
    oracle = (weights * (z.detach() - target).square()).sum() / mass
    derivative = 2 * weights * (z.detach() - target) / mass

    def objective(cached: torch.Tensor) -> torch.Tensor:
        total = z.new_zeros(())
        for start, stop in ((0, 3), (3, 6)):
            sl = slice(start, stop)
            sums = objective_numerators(
                z[:, sl],
                hints[:, sl],
                y[:, sl],
                cached[:, sl],
                hints[:, sl],
                mask[:, sl],
                objective_mode=B03_MODE,
                context=context,
                target_alpha=alpha,
                label_row_weights=age[:, sl] if tau else None,
            )
            total = total + scaled_objective(sums, denom, 1, alpha, 0.0, microchunks=2)
        return total

    actual = objective(teacher)
    actual.backward()
    torch.testing.assert_close(denom[0], mass, rtol=0, atol=ATOL)
    torch.testing.assert_close(actual.detach(), oracle, rtol=0, atol=ATOL)
    torch.testing.assert_close(z.grad, derivative, rtol=0, atol=ATOL)
    if teacher.grad is not None:
        raise ValueError("B03 objective propagated into teacher cache")
    delta = float((objective(-teacher).detach() - actual.detach()).abs())
    if (alpha == 0 and delta != 0) or (alpha > 0 and delta < 1e-4):
        raise ValueError("teacher perturbation did not distinguish the alpha controls")
    return {
        "tau": tau,
        "target_alpha": alpha,
        "loss": actual.item(),
        "original_label_mass": mass.item(),
        "teacher_flip_delta": delta,
        "max_abs_gradient_error": (z.grad - derivative).abs().max().item(),
    }


def main() -> None:
    """Run four deterministic checks on one allocated L4 and retain the source-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or torch.cuda.device_count() != 1:
        raise ValueError("B03 oracle needs an allocated Linux host with exactly one visible L4")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    if "L4" not in name:
        raise ValueError("B03 numerical oracle requires the L4 receiver")
    root = Path(os.environ["WF_RUNS"]).resolve(strict=True)
    native = root / "runtime"
    module_source = Path(source_identity.__code__.co_filename).resolve()
    if not module_source.is_relative_to(native / "src"):
        raise ValueError("unexpected Python runtime imports")
    sha = source_identity(native, native / "scripts/train_sequence.py")["sha256"]
    if sha != json.loads((PACKET / "runtime_manifest.json").read_bytes())["source_sha256"]:
        raise ValueError("B03 probe source differs from the frozen overlay")
    qualification = file_sha256(root / "qualification.json")
    configure_strict_fp32()
    context = DistributedContext(0, 0, 1, device, False)
    age = causal_weights(device)
    cells = [cell_probe(age, tau, alpha, context) for tau in (0, 32) for alpha in (0.0, 0.5)]
    receipt = {
        "schema": "wnn.b03.objective-probe.v1",
        "status": "passed",
        "source_sha256": sha,
        "qualification_sha256": qualification,
        "device": name,
        "torch": str(torch.__version__),
        "clock_checked_across_row": 1000,
        "atol": ATOL,
        "cells": cells,
    }
    with tempfile.NamedTemporaryFile(dir=args.output.parent, mode="w", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.link(stream.name, args.output)


if __name__ == "__main__":
    main()
