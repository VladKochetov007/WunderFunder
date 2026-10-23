from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.i1_history import I1_HISTORY_MODE, I1_INPUT_WIDTH
from wnn.run_identity import tensor_identity
from wnn.training.environment import TrainingEnvironment
from wnn.training.normalization import fit_normalization

PARENT_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES


def validate_i1_warm_normalization(
    args: Namespace, checkpoint: Mapping[str, object] | None
) -> None:
    """Fail closed unless appended i1 normalization has the supported parent and child."""
    mode = args.i1_warm_normalization
    if mode == "inherit":
        return
    if mode != "fit-appended":
        raise ValueError(f"unsupported i1 warm normalization mode: {mode}")
    if not args.init:
        raise ValueError("fit-appended i1 normalization requires an initializer checkpoint")
    if not args.inherit_init_preprocessing:
        raise ValueError("fit-appended i1 normalization requires inherited preprocessing")
    if checkpoint is None:
        raise ValueError("fit-appended i1 normalization requires a loaded initializer checkpoint")
    if args.resume_state:
        raise ValueError("fit-appended i1 normalization cannot resume training state")
    if args.research_features != I1_HISTORY_MODE:
        raise ValueError("fit-appended i1 normalization requires i1_history child inputs")
    if args.drop_i1:
        raise ValueError("fit-appended i1 normalization requires --drop-i1 to be disabled")
    if args.research_feature_control not in ("none", "all"):
        raise ValueError("fit-appended i1 normalization supports only none or all controls")
    if args.research_feature_adapter_only:
        raise ValueError("fit-appended i1 normalization does not support feature-adapter-only mode")

    parent_args = checkpoint.get("args")
    parent_state = checkpoint.get("model")
    if not isinstance(parent_args, Mapping) or not isinstance(parent_state, Mapping):
        raise ValueError("initializer lacks standard model arguments or state")
    _validate_standard_parent_args(parent_args)
    parent_width = _stored_parent_width(parent_state)
    if parent_width != PARENT_INPUT_WIDTH:
        raise ValueError(
            "fit-appended i1 normalization requires a standard 211-slot parent; "
            f"initializer stores {parent_width} input slots"
        )


def fit_i1_appended_normalization(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    environment: TrainingEnvironment,
) -> dict[str, object]:
    """Fit canonical training statistics only for the 156 appended history slots."""
    if model.mean.shape != (I1_INPUT_WIDTH,) or model.std.shape != (I1_INPUT_WIDTH,):
        raise ValueError("appended i1 normalization requires the fixed 367-slot model")
    parent_mean = model.mean[:PARENT_INPUT_WIDTH].detach().clone()
    parent_std = model.std[:PARENT_INPUT_WIDTH].detach().clone()
    parent_mean_identity = tensor_identity(parent_mean)
    parent_std_identity = tensor_identity(parent_std)
    try:
        fit_normalization(model, loader, device, batches=2, environment=environment)
    finally:
        model.mean[:PARENT_INPUT_WIDTH].copy_(parent_mean)
        model.std[:PARENT_INPUT_WIDTH].copy_(parent_std)
    if tensor_identity(model.mean[:PARENT_INPUT_WIDTH]) != parent_mean_identity:
        raise RuntimeError("fitting appended i1 slots changed inherited parent means")
    if tensor_identity(model.std[:PARENT_INPUT_WIDTH]) != parent_std_identity:
        raise RuntimeError("fitting appended i1 slots changed inherited parent scales")
    return {
        "mode": "fit-appended",
        "estimator": "canonical fit_normalization over first 2 fitting-loader batches; "
        "flatten sequence rows and retain rows [::7]",
        "parent_width": PARENT_INPUT_WIDTH,
        "child_width": I1_INPUT_WIDTH,
        "appended_width": I1_INPUT_WIDTH - PARENT_INPUT_WIDTH,
        "parent_mean_identity": parent_mean_identity,
        "parent_std_identity": parent_std_identity,
        "appended_mean_identity": tensor_identity(model.mean[PARENT_INPUT_WIDTH:]),
        "appended_std_identity": tensor_identity(model.std[PARENT_INPUT_WIDTH:]),
        "fit_membership": "existing training loader only; inherited fit-row membership",
    }


def _validate_standard_parent_args(parent: Mapping[str, object]) -> None:
    """Require the parent feature layout represented by the 211-slot prefix."""
    required_values = {
        "research_features": "none",
        "research_feature_control": "none",
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "privileged_mode": "none",
        "numeric_map": "none",
        "sentinel_indicators": "none",
        "sentinel_values": "unchanged",
    }
    for name, expected in required_values.items():
        actual = parent.get(name, expected)
        if actual != expected:
            raise ValueError(f"fit-appended i1 normalization requires parent {name}={expected!r}")
    disabled_flags = ("online_norm", "raw_only", "heuristics", "drop_i1")
    if any(bool(parent.get(name, False)) for name in disabled_flags):
        raise ValueError("fit-appended i1 normalization requires a standard unmodified parent view")
    if parent.get("prop_input", ""):
        raise ValueError("fit-appended i1 normalization does not support propensity inputs")
    if int(parent.get("sentinel_base_width", 0) or 0):
        raise ValueError("fit-appended i1 normalization does not support sentinel inputs")


def _stored_parent_width(state: Mapping[str, object]) -> int:
    """Derive and cross-check parent input width from normalization and projection state."""
    mean = state.get("mean")
    std = state.get("std")
    input_weight = state.get("inp.weight")
    if not all(isinstance(value, torch.Tensor) for value in (mean, std, input_weight)):
        raise ValueError("initializer lacks normalization buffers or input projection weight")
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert isinstance(input_weight, torch.Tensor)
    if mean.ndim != 1 or std.shape != mean.shape or input_weight.ndim != 2:
        raise ValueError("initializer normalization or input projection has an unsupported shape")
    width = int(mean.shape[0])
    if input_weight.shape[1] != width:
        raise ValueError("initializer projection width disagrees with stored normalization")
    return width
