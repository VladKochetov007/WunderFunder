from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch
from torch import nn

LOG_VARIANCE_MIN = -4.0
LOG_VARIANCE_MAX = 4.0
LOG_VARIANCE_OUTPUTS = 2


def configuration_error(args: argparse.Namespace, controlled_loss: bool) -> str | None:
    """Return why the opt-in mean/logvariance recipe is incompatible with this run."""
    enabled = bool(args.mean_variance_head)
    objective = args.mean_variance_objective
    if objective == "nll" and not enabled:
        return "--mean-variance-objective nll requires --mean-variance-head 1"
    if not enabled:
        return None
    if args.arch != "gru" or args.cell != "gru" or args.enc != "linear":
        return "--mean-variance-head requires the plain GRU with a linear input encoder"
    if args.precision != "fp32" or args.qat != "none" or args.recurrent_rank:
        return "--mean-variance-head requires dense strict FP32 without QAT"
    if args.loss != "wmse" or args.loss_norm != "chunk":
        return "--mean-variance-head supports only chunk-normalized --loss wmse"
    if controlled_loss or args.td_alpha or args.ema_teacher or args.eval_ema or args.swa_frac:
        return "--mean-variance-head does not support controlled, TD, EMA, or SWA objectives"
    if objective == "nll":
        has_teacher = bool(
            args.distill or args.distill_cache_root or args.final_distill_initializer
        )
        if not has_teacher:
            return "the mean/logvariance NLL requires the fixed MainT3 blended teacher target"
        if (
            args.distill_alpha != 0.9
            or args.distill_alpha_far is not None
            or args.distill_boot_h != 0
            or args.distill_reanchor != 0.0
            or args.distill_cache_scale != 1.0
        ):
            return "the mean/logvariance NLL requires the fixed 0.9 teacher / 0.1 label target"
    return None


def weighted_diagonal_gaussian_nll(
    mean: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted Gaussian NLL in the canonical two-target clipped-weight scale."""
    if (
        mean.shape != target.shape
        or mean.shape != log_variance.shape
        or mean.shape != weights.shape
    ):
        raise ValueError("mean, target, logvariance and weights must have identical shapes")
    mean = mean.float()
    target = target.float()
    log_variance = log_variance.float().clamp(LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)
    weights = weights.float()
    residual_squared = (mean - target).square()
    terms = torch.exp(-log_variance) * residual_squared + log_variance
    return (weights * terms).sum() / weights.sum().clamp_min(1e-8)


def append_log_variance_output_head(model: nn.Module) -> nn.Module:
    """Append two zero-initialized output rows without changing existing model outputs."""
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Linear):
        raise ValueError("mean/logvariance output requires a sequential linear readout head")
    output = head[-1]
    if output.bias is None:
        raise ValueError("mean/logvariance output requires a biased readout layer")
    old_width = output.out_features
    device = output.weight.device
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(
            output.in_features,
            old_width + LOG_VARIANCE_OUTPUTS,
            bias=True,
            device="cpu",
            dtype=output.weight.dtype,
        )
    with torch.no_grad():
        expanded.weight[:old_width].copy_(output.weight.detach().to("cpu"))
        expanded.bias[:old_width].copy_(output.bias.detach().to("cpu"))
        expanded.weight[old_width:].zero_()
        expanded.bias[old_width:].zero_()
    head[-1] = expanded.to(device=device)
    return model


def extend_mean_variance_state(
    parent_state: Mapping[str, torch.Tensor],
    initialized_state: Mapping[str, torch.Tensor],
    *,
    family_outputs: int = 0,
) -> dict[str, torch.Tensor]:
    """Copy parent tensors and preserve initialized family plus zero logvariance rows."""
    if parent_state.keys() != initialized_state.keys():
        raise ValueError("parent and student checkpoint tensor names differ")
    expected_extra = family_outputs + LOG_VARIANCE_OUTPUTS
    if family_outputs not in (0, 8):
        raise ValueError("only the existing eight-output family extension is supported")
    merged = {name: value.clone() for name, value in initialized_state.items()}
    for name, parent_value in parent_state.items():
        student_value = initialized_state[name]
        if name in ("head.2.weight", "head.2.bias"):
            if (
                parent_value.ndim != student_value.ndim
                or parent_value.shape[0] < 2
                or student_value.shape[0] != parent_value.shape[0] + expected_extra
                or parent_value.shape[1:] != student_value.shape[1:]
            ):
                raise ValueError(
                    "student output head must append family rows and two logvariance rows"
                )
            merged[name][: parent_value.shape[0]].copy_(parent_value)
        elif parent_value.shape != student_value.shape:
            raise ValueError(f"warm-start tensor shape differs outside output head: {name}")
        else:
            merged[name].copy_(parent_value)
    return merged
