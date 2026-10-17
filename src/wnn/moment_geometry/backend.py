"""Adapt prepared causal arrays to the canonical training components, without a new update engine."""

from __future__ import annotations

import math

import torch

from wnn.run_identity import EpochUpdateCounter
from wnn.training.cli import build_parser
from wnn.training.environment import TrainingEnvironment
from wnn.training.forward_backward import backward_standard, forward_standard
from wnn.training.optimization import build_optimization
from wnn.training.profiling import StageTimer
from wnn.training.state import TrainingRun
from wnn.training.updates import optimizer_step


def prepare_run(
    width: int,
    total_updates: int,
    epochs: int,
    device: torch.device,
    seed: int,
    hidden: int = 32,
    layers: int = 1,
) -> TrainingRun:
    """Bind the exact pilot recipe to shared model/optimizer/scheduler/update implementations."""
    args = build_parser().parse_args(
        [
            "--name",
            "moment-geometry-pilot",
            "--hidden",
            "32",
            "--layers",
            "1",
            "--loss",
            "wmse",
            "--precision",
            "fp32",
            "--sched",
            "sgdr",
            "--sync-free",
            "--grad-stats",
        ]
    )
    args.seed = seed
    args.hidden, args.layers = hidden, layers
    args.lr, args.wd, args.beta2 = 0.001, 0.01, 0.999
    args.epochs, args.batch, args.chunk = epochs, 4, 512
    args.warmup_frac, args.restarts, args.wsd_final = 0.0, 0, 0.01
    args.swa_frac, args.eval_ema, args.ema_teacher, args.td_alpha = 0.0, False, 0.0, 0.0
    args.clip, args.grad_noise, args.clip_per_module = 1.0, 0.0, ""
    environment = TrainingEnvironment()
    run = TrainingRun(environment)
    run.options.args, run.options.device = args, device
    run.network.model = environment.make_model(vars(args), width, 2).to(device)
    run.network.total = total_updates
    run.data.provenance = {}
    if not build_optimization(run):
        raise RuntimeError("unexpected preflight-only branch")
    run.progress.timer = StageTimer(False)
    run.progress.epoch_counter = EpochUpdateCounter(device)
    run.progress.activation_profile_updates = []
    return run


def step_prepared(
    run: TrainingRun, inputs: torch.Tensor, targets: torch.Tensor, start: int, stop: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Use canonical forward/backward/update paths on one prepared chronological chunk."""
    run.batch.inputs, run.batch.sl = inputs, slice(start, stop)
    forward_standard(run)
    valid = torch.ones((len(inputs), stop - start), dtype=torch.bool, device=inputs.device)
    if start < min(stop, 99):
        valid[:, : min(stop, 99) - start] = False
    loss = run.optimization.objective(run.batch.out, targets[:, start:stop], valid)
    run.batch.loss = loss
    lr = float(run.optimization.opt.param_groups[0]["lr"])
    backward_standard(run)
    optimizer_step(run)
    norm = torch.linalg.vector_norm(run.optimization.grad_stats.rows[-1])
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite pilot gradient norm")
    return loss.detach(), norm.detach(), lr


def expected_learning_rate(step: int, total: int) -> float:
    """Expose the frozen endpoint-inclusive cosine equation for diagnostic checks."""
    fraction = 1.0 if total <= 1 else step / (total - 1)
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * fraction))
