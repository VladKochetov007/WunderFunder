from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from attrs import define
from torch import nn

from wnn.optim import MuonAdamW
from wnn.resources import configure_strict_fp32
from wnn.training.cli import build_parser
from wnn.training.muon_precision import muon_precision_error
from wnn.training.reporting import emit
from wnn.training.runner import run_training

if TYPE_CHECKING:
    from wnn.training.environment import TrainingEnvironment


@define
class ProfileCapture:
    """Hold references to the model and optimizer built by the canonical trainer."""

    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    setup_cuda_peak_allocated_bytes: int | None = None
    setup_cuda_peak_reserved_bytes: int | None = None


def profile_argument_error(args: argparse.Namespace) -> str | None:
    """Require the bounded, fit-only full-shape teacher profile recipe."""
    if args.device != "cuda":
        return "teacher profile requires --device cuda"
    if (args.batch, args.chunk, args.update_rows) != (32, 1000, 0):
        return "teacher profile requires batch 32, chunk 1000 and one update per chunk"
    if (args.profile_warmup, args.profile_steps, args.max_batches) != (2, 4, 6):
        return (
            "teacher profile requires two warmup batches, four measured batches and six max batches"
        )
    if not args.grad_stats:
        return (
            "teacher profile requires --grad-stats for canonical FP32 gradient-finiteness evidence"
        )
    if not args.no_eval_during_fit:
        return "teacher profile requires --no-eval-during-fit"
    if args.fast or args.prop_cache:
        return "teacher profile must not require an unbound propensity cache"
    if not args.propensity:
        return "teacher profile requires the source-bound propensity checkpoint"
    if args.seed != 3 or args.propensity != "fg19_propensity_clean_f0_seed0_e4":
        return "teacher profile is the seed-3 run with the named clean F0 propensity checkpoint"
    if (
        args.arch != "gru"
        or args.cell != "gru"
        or args.enc != "linear"
        or (args.hidden, args.layers, args.head_layers) != (384, 2, 1)
    ):
        return "teacher profile requires the H384x2 linear-encoder GRU with head-layers 1"
    if (args.epochs, args.checkpoint_policy, args.checkpoint_epoch) != (3, "fixed", 2):
        return "teacher profile requires three epochs and fixed zero-based checkpoint epoch 2"
    if args.loss != "wmse" or args.loss_norm != "chunk":
        return "teacher profile requires chunk-normalized WMSE"
    if (args.lr, args.wd, args.beta2, args.sched, args.warmup_frac, args.clip) != (
        0.001,
        0.01,
        0.999,
        "onecycle",
        0.05,
        1.0,
    ):
        return "teacher profile optimizer schedule differs from the bound candidate"
    if (args.prop_power, args.prop_eps, args.valid_true_mask) != (2.0, 0.03, 0.03) or (
        args.aux_struct,
        args.aux_next,
        args.since_weight,
        args.since_floor,
    ) != (0.3, 0.3, 32.0, 0.25):
        return "teacher profile propensity or auxiliary weighting differs from the bound candidate"
    if (
        args.aux_tier != 0
        or args.aux_mask != 0
        or args.aux_sign != 0
        or args.aux_quote != 0
        or args.aux
    ):
        return "teacher profile does not admit additional auxiliary objectives"
    if args.mean_variance_head != 0 or args.representation_mode is not None:
        return "teacher profile requires the legacy 23-output, 211-input model"
    if (
        args.optim != "muon"
        or args.muon_scope != "all"
        or args.muon_ortho_dtype != "fp32"
        or not args.strict_muon_fp32
        or args.precision != "fp32"
    ):
        return "teacher profile requires strict FP32 Muon scope all"
    precision_error = muon_precision_error(args)
    if precision_error is not None:
        return precision_error
    if args.resume_state or args.init:
        return "teacher profile must start from fresh model initialization"
    if args.update_rows != 0 or args.detach_rows != 0 or args.origin_view != "off":
        return "teacher profile requires one canonical 1000-row update window at a time"
    if args.activation_backend != "none" or args.parity_dump:
        return "teacher profile does not enable a compiled backend or parity dump"
    return None


def configure_profile_runtime() -> None:
    """Set the declared single-CPU strict-FP32 runtime before canonical setup."""
    if torch.cuda.is_initialized():
        raise RuntimeError("profile runtime must be configured before CUDA initialization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    configure_strict_fp32()
    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError("teacher profile requires FP32 default dtype")


def install_capture_hooks(environment: TrainingEnvironment, capture: ProfileCapture) -> None:
    """Capture canonical model/optimizer references without altering their construction."""
    make_model = environment.make_model
    build_optimizer = environment.build_optimizer

    def capture_model(args_dict: dict[str, object], n_in: int, n_out: int) -> nn.Module:
        """Retain the canonical model without changing its construction."""
        if capture.model is not None:
            raise RuntimeError("canonical trainer constructed more than one primary model")
        model = make_model(args_dict, n_in, n_out)
        capture.model = model
        return model

    def capture_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
        """Retain canonical optimizer state and the setup memory peak."""
        if capture.optimizer is not None:
            raise RuntimeError("canonical trainer constructed more than one primary optimizer")
        optimizer = build_optimizer(model, args)
        capture.optimizer = optimizer
        torch.cuda.synchronize(args.device)
        capture.setup_cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(args.device))
        capture.setup_cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(args.device))
        return optimizer

    environment.make_model = capture_model
    environment.build_optimizer = capture_optimizer


def _finite_tensors(tensors: Sequence[torch.Tensor]) -> bool:
    """Check captured tensors one at a time to avoid an additional full-size stack."""
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _read_cuda_peak(device: str) -> tuple[int, int]:
    """Snapshot peak CUDA memory before post-profile validation allocates tensors."""
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def profile_state_summary(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    """Return finite/dtype and optimizer-policy summaries for captured post-profile state."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    model_tensors = parameters + buffers
    if not parameters or not model_tensors:
        raise RuntimeError("canonical profile returned an empty model")
    if any(tensor.dtype != torch.float32 for tensor in model_tensors):
        raise RuntimeError("profile model parameters/buffers are not all FP32")
    if not _finite_tensors(model_tensors):
        raise RuntimeError("profile model contains nonfinite parameters or buffers")

    muon_groups = [group for group in optimizer.param_groups if group.get("muon") is True]
    if not isinstance(optimizer, MuonAdamW) or not muon_groups:
        raise RuntimeError("strict teacher profile did not construct MuonAdamW groups")
    if any(group.get("muon_fp32_ortho") is not True for group in muon_groups):
        raise RuntimeError("a Muon parameter group does not require FP32 orthogonalization")

    state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    floating_state = [tensor for tensor in state_tensors if tensor.is_floating_point()]
    if not floating_state:
        raise RuntimeError("profile produced no floating-point optimizer state")
    if any(tensor.dtype != torch.float32 for tensor in floating_state):
        raise RuntimeError("optimizer floating-point state is not all FP32")
    if not _finite_tensors(floating_state):
        raise RuntimeError("optimizer state contains nonfinite values")

    return {
        "model_parameter_tensors": len(parameters),
        "model_buffer_tensors": len(buffers),
        "model_parameter_count": sum(parameter.numel() for parameter in parameters),
        "model_all_fp32_finite": True,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_group_count": len(optimizer.param_groups),
        "muon_group_count": len(muon_groups),
        "all_muon_groups_fp32_ortho": True,
        "optimizer_state_tensor_count": len(state_tensors),
        "optimizer_float_state_tensor_count": len(floating_state),
        "optimizer_float_state_dtype_counts": dict(
            Counter(str(tensor.dtype) for tensor in floating_state)
        ),
        "optimizer_float_state_all_fp32_finite": True,
    }


def run_teacher_profile(argv: list[str] | None = None) -> None:
    """Run a bounded canonical fit profile and append memory/state evidence."""
    from wnn.training.environment import TrainingEnvironment

    argv = list(argv or [])
    parser = build_parser()
    args = parser.parse_args(argv)
    error = profile_argument_error(args)
    if error is not None:
        parser.error(error)
    configure_profile_runtime()
    environment = TrainingEnvironment(argv=argv)
    capture = ProfileCapture()
    install_capture_hooks(environment, capture)
    with torch.autocast(device_type="cuda", enabled=False):
        run_training(argv, environment=environment)
    torch.cuda.synchronize(args.device)
    fit_peak_allocated_bytes, fit_peak_reserved_bytes = _read_cuda_peak(args.device)
    if capture.model is None or capture.optimizer is None:
        raise RuntimeError("canonical profile returned without captured model and optimizer")
    setup_peak_allocated_bytes = capture.setup_cuda_peak_allocated_bytes
    setup_peak_reserved_bytes = capture.setup_cuda_peak_reserved_bytes
    if setup_peak_allocated_bytes is None or setup_peak_reserved_bytes is None:
        raise RuntimeError("canonical profile returned without pre-fit CUDA peak evidence")
    summary = profile_state_summary(capture.model, capture.optimizer)
    tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)
    autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    matmul_precision = torch.get_float32_matmul_precision()
    if tf32_matmul or tf32_cudnn or autocast_enabled or matmul_precision != "highest":
        raise RuntimeError("strict FP32 runtime flags changed during the canonical profile")
    summary.update(
        {
            "device": args.device,
            "setup_cuda_peak_allocated_bytes": setup_peak_allocated_bytes,
            "setup_cuda_peak_reserved_bytes": setup_peak_reserved_bytes,
            "fit_cuda_peak_allocated_bytes": fit_peak_allocated_bytes,
            "fit_cuda_peak_reserved_bytes": fit_peak_reserved_bytes,
            "cuda_peak_allocated_bytes": max(setup_peak_allocated_bytes, fit_peak_allocated_bytes),
            "cuda_peak_reserved_bytes": max(setup_peak_reserved_bytes, fit_peak_reserved_bytes),
            "float32_matmul_precision": matmul_precision,
            "cuda_matmul_tf32": tf32_matmul,
            "cudnn_tf32": tf32_cudnn,
            "autocast_enabled_after_profile": autocast_enabled,
            "expected_optimizer_attempts": 120,
            "expected_measured_optimizer_updates": 80,
            "scope": "canonical fit-only profile state; no evaluation or checkpoint save",
        }
    )
    emit("TEACHER_PROFILE_STATE", json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    """Run the teacher profile using the canonical command-line parser."""
    import sys

    run_teacher_profile(sys.argv[1:])
