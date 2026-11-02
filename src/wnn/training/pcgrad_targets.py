"""Target-wise PCGrad on the shared trunk of the plain supervised GRU."""

from __future__ import annotations

import os

import torch

from wnn.training.state import TrainingRun


def validate_pcgrad(run: TrainingRun) -> None:
    """Reject unsupported training modes before any model fit starts."""
    args = run.options.args
    if not args.target_pcgrad:
        return
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or args.precision != "fp32"
        or args.arch != "gru"
        or args.cell != "gru"
        or args.enc != "linear"
        or args.numeric_map != "none"
        or bool(args.distill)
        or bool(args.distill_cache_root)
        or args.loss != "wmse"
        or args.loss_norm != "chunk"
        or args.td_alpha > 0
        or args.filtered_residual_weight != 0
        or args.qat != "none"
        or run.options.controlled_loss
    ):
        run.options.ap.error(
            "--target-pcgrad requires one GPU, fp32, plain GRU, chunk WMSE, "
            "no TD/QAT/filtered loss and the standard update clock"
        )


def pcgrad_backward(run: TrainingRun) -> None:
    """Project task gradients on the shared encoder and GRU, retaining auxiliary gradients."""
    if run.batch.teach is not None or run.network.initializer_teacher is not None:
        raise ValueError("target PCGrad currently supports label-only training, not KD")
    if run.optimization.scaler.is_enabled():
        raise ValueError("target PCGrad requires a disabled gradient scaler")
    model = run.network.model
    params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and (name.startswith("inp.") or name.startswith("gru."))
    ]
    if not params:
        raise ValueError("target PCGrad found no trainable plain GRU trunk parameters")
    sl = run.batch.sl
    target = run.batch.y[:, sl].float().clamp(-2, 2)
    weight = target.abs() * run.batch.main_valid[:, sl][..., None]
    errors = weight * (run.batch.out[..., :2].float() - target).square()
    denominator = weight.sum().clamp_min(1e-8)
    task0 = errors[..., 0].sum() / denominator
    task1 = errors[..., 1].sum() / denominator
    grad0 = torch.autograd.grad(task0, params, retain_graph=True, allow_unused=True)
    grad1 = torch.autograd.grad(task1, params, retain_graph=True, allow_unused=True)
    zero = run.batch.loss.new_zeros(())
    dot = sum(
        [
            (a * b).sum()
            for a, b in zip(grad0, grad1, strict=True)
            if a is not None and b is not None
        ],
        zero,
    )
    norm0 = sum([(a * a).sum() for a in grad0 if a is not None], zero)
    norm1 = sum([(b * b).sum() for b in grad1 if b is not None], zero)
    # The symmetric projection uses original gradients, not already-modified ones.
    coefficient0 = torch.where(dot < 0, dot / norm1.clamp_min(1e-20), zero)
    coefficient1 = torch.where(dot < 0, dot / norm0.clamp_min(1e-20), zero)
    run.batch.loss.backward()
    for parameter, a, b in zip(params, grad0, grad1, strict=True):
        if parameter.grad is None:
            continue
        if b is not None:
            parameter.grad.add_(b * -coefficient0)
        if a is not None:
            parameter.grad.add_(a * -coefficient1)
