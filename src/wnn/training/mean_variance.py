from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch
from torch import nn

LOG_VARIANCE_OUTPUTS = 2


def configuration_error(args: argparse.Namespace, controlled_loss: bool) -> str | None:
    """Return why the optional WMSE checkpoint-layout head is incompatible."""
    if not bool(args.mean_variance_head):
        return None
    if args.arch != "gru" or args.cell != "gru" or args.enc != "linear":
        return "--mean-variance-head requires the plain GRU with a linear input encoder"
    if args.precision != "fp32" or args.qat != "none" or args.recurrent_rank:
        return "--mean-variance-head requires dense strict FP32 without QAT"
    if args.loss != "wmse" or args.loss_norm != "chunk":
        return "--mean-variance-head supports only chunk-normalized --loss wmse"
    if getattr(args, "dynamics_aux_head", False):
        return "--mean-variance-head cannot be combined with the dynamics auxiliary head"
    if controlled_loss or args.td_alpha or args.ema_teacher or args.eval_ema or args.swa_frac:
        return "--mean-variance-head does not support controlled, TD, EMA, or SWA objectives"
    return None


def append_log_variance_output_head(model: nn.Module) -> nn.Module:
    """Append two dormant zero rows without changing existing model outputs or RNG."""
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
) -> dict[str, torch.Tensor]:
    """Copy a matching parent state while retaining two initialized trailing zero rows."""
    if parent_state.keys() != initialized_state.keys():
        raise ValueError("parent and student checkpoint tensor names differ")
    merged = {name: value.clone() for name, value in initialized_state.items()}
    for name, parent_value in parent_state.items():
        student_value = initialized_state[name]
        if name in ("head.2.weight", "head.2.bias"):
            if (
                parent_value.ndim != student_value.ndim
                or parent_value.shape[0] < 2
                or student_value.shape[0] != parent_value.shape[0] + LOG_VARIANCE_OUTPUTS
                or parent_value.shape[1:] != student_value.shape[1:]
            ):
                raise ValueError("student output head must append exactly two trailing rows")
            merged[name][: parent_value.shape[0]].copy_(parent_value)
        elif parent_value.shape != student_value.shape:
            raise ValueError(f"warm-start tensor shape differs outside output head: {name}")
        else:
            merged[name].copy_(parent_value)
    return merged
