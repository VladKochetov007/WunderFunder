from __future__ import annotations

import argparse
import copy
import hashlib

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch

from wnn.training.environment import TrainingEnvironment
from wnn.training.forward import detach_state


def final_initializer_distillation_error(args: argparse.Namespace) -> str | None:
    """Return why the opt-in initializer-target arm is outside its supported recipe."""
    if not args.final_distill_initializer:
        return None
    if not args.final_model_adaptation or not args.init or (not args.inherit_init_preprocessing):
        return "--final-distill-initializer requires final-model adaptation from an inherited initializer"
    if (
        args.precision != "fp32"
        or args.arch != "gru"
        or args.cell != "gru"
        or (args.enc != "linear")
        or (args.recurrent_rank != 0)
    ):
        return (
            "--final-distill-initializer requires a dense FP32 plain GRU with linear input encoding"
        )
    if not args.distill_cache_root or args.distill:
        return (
            "--final-distill-initializer requires a validated --distill-cache-root, not --distill"
        )
    if args.loss != "wmse" or args.loss_norm != "chunk":
        return "--final-distill-initializer requires cached wMSE with chunk normalization"
    if (
        args.distill_alpha != 0.9
        or args.distill_alpha_far is not None
        or args.distill_boot_h != 0
        or (args.distill_reanchor != 0.0)
        or (args.distill_cache_scale != 1.0)
        or (args.distill_family_weight != 0.0)
    ):
        return "--final-distill-initializer fixes alpha=0.9 and disables cache scaling, bootstrap, reanchor, and family targets"
    if (
        args.update_rows != 0
        or args.detach_rows != 0
        or args.distill_denominator != "chunk"
        or args.detach_boundary_jitter
        or (args.origin_view != "off")
        or (args.credit_mode != "deterministic")
    ):
        return "--final-distill-initializer supports only default chunk updates and deterministic causal credit"
    if (
        args.ema_teacher != 0.0
        or args.ema_metric_weight
        or args.ema_reference_rows != 0
        or args.eval_ema
        or (args.swa_frac != 0.0)
        or (args.td_alpha != 0.0)
    ):
        return "--final-distill-initializer does not support EMA, SWA, or TD targets"
    if args.resume_state or args.save_training_state:
        return "--final-distill-initializer does not support training-state continuation"
    return None


def final_initializer_state_sha256(model: torch.nn.Module) -> str:
    """Fingerprint the complete ordered tensor state of the frozen initializer teacher."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def final_initializer_distillation_target(
    labels: torch.Tensor, predictions: torch.Tensor
) -> torch.Tensor:
    """Apply the fixed clipped-label/initializer 0.1/0.9 target rule."""
    return 0.1 * labels.float().clamp(-2, 2) + 0.9 * predictions.float().clamp(-2, 2)


def make_final_initializer_teacher(model: torch.nn.Module) -> torch.nn.Module:
    """Clone the loaded, pre-QAT FP32 initializer as a fixed inference teacher."""
    if not getattr(model, "_inference_fp32", False):
        raise ValueError("initializer teacher must be created from an FP32 inference model")
    if getattr(model, "_qat_preset", "none") != "none":
        raise ValueError("initializer teacher must be cloned before QAT configuration")
    if any(value.dtype != torch.float32 for value in model.state_dict().values()):
        raise ValueError("initializer teacher requires an all-FP32 model state")
    teacher = copy.deepcopy(model)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def replace_final_initializer_chunk(
    teacher: torch.nn.Module,
    inputs: torch.Tensor,
    rows: slice,
    state: Any,
    target_batch: torch.Tensor,
    *,
    environment: TrainingEnvironment | None = None,
) -> Any:
    """Overwrite one device-batch target slice with causal frozen-teacher predictions."""
    environment = environment if environment is not None else TrainingEnvironment()
    with torch.no_grad(), torch.autocast(inputs.device.type, enabled=False):
        output, next_state = environment.forward_chunk(teacher, inputs, rows, state)
    prediction = output[..., :2].float().detach()
    if prediction.shape != target_batch[:, rows].shape:
        raise ValueError("initializer output does not match the cached main-target batch shape")
    target_batch[:, rows] = prediction
    return detach_state(next_state)
