from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from attrs import frozen
from torch import nn

from wnn.training.environment import TrainingEnvironment
from wnn.training.models import make_model


@frozen
class ModelLoad:
    """Frozen model reconstruction and its exact persisted training arguments."""

    model: nn.Module
    environment: TrainingEnvironment
    args: dict[str, object]
    n_in: int
    state_keys: list[str]


def _require_plain_r01_args(args: dict[str, object], n_in: int) -> None:
    """Reject checkpoint metadata that would change the fixed F0 input or model."""
    expected: dict[str, object] = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 512,
        "layers": 1,
        "dropout": 0,
        "precision": "fp32",
        "loss": "wmse",
        "head_layers": 1,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "online_norm": False,
        "raw_only": False,
        "heuristics": False,
        "research_features": "none",
        "research_feature_control": "none",
        "privileged_mode": "none",
        "sentinel_indicators": "none",
    }
    for key, value in expected.items():
        observed = args.get(key, value)
        if observed != value:
            raise ValueError(f"checkpoint argument {key}={observed!r} differs from fixed R01")
    if n_in != 211:
        raise ValueError(f"checkpoint input width must be 211, got {n_in}")
    if any(
        args.get(key) not in (None, 512)
        for key in ("encoder_width", "recurrent_width", "readout_width")
    ):
        raise ValueError("checkpoint contains noncanonical independent GRU widths")
    if any(
        bool(args.get(key))
        for key in ("propensity", "prop_cache", "prop_input", "quote_cache_root", "distill")
    ):
        raise ValueError("checkpoint contains an unsupported cached or auxiliary input")
    if args.get("aux_next_k", "") not in (None, ""):
        raise ValueError("checkpoint requests auxiliary next-step output heads")
    # The observed CLI default is inert unless an auxiliary objective is active.
    auxiliary_objectives_disabled = all(
        args.get(key, 0) == 0
        for key in (
            "aux_struct",
            "aux_next",
            "aux_tier",
            "aux_mask",
            "aux_sign",
            "aux_quote",
        )
    )
    for key, value in args.items():
        if key.startswith("aux_") and isinstance(value, (int, float)) and value != 0:
            if key == "aux_weight" and value == 0.5 and auxiliary_objectives_disabled:
                continue
            raise ValueError(f"checkpoint contains unsupported nonzero {key}")


def _load_restricted_checkpoint(checkpoint_path: Path) -> object:
    """Load on CPU while allowlisting only PyTorch's version metadata wrapper."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def parameter_inventory(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], dict[str, list[int]], list[dict[str, object]]]:
    """Classify each trainable tensor, excluding final target rows from shared blocks."""
    blocks: dict[str, list[int]] = {"trunk_inp_gru": [], "shared_head_hidden": []}
    parameters: list[tuple[str, nn.Parameter]] = []
    inventory: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"unexpected frozen parameter: {name}")
        if name.startswith(("inp.", "gru.")):
            block = "trunk_inp_gru"
        elif name.startswith("head.0."):
            block = "shared_head_hidden"
        elif name.startswith("head.2."):
            block = "target_output_excluded"
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
        index = len(parameters)
        parameters.append((name, parameter))
        if block != "target_output_excluded":
            blocks[block].append(index)
        inventory.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "block": block,
            }
        )
    return parameters, blocks, inventory


def load_frozen_model(checkpoint_path: Path, source_root: Path, data_root: Path) -> ModelLoad:
    """Load the strict R01 model on CPU from a restricted weights-only checkpoint."""
    checkpoint = _load_restricted_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise ValueError("restricted checkpoint lacks plain args/model mappings")
    args = checkpoint["args"]
    n_in = int(checkpoint.get("n_in", -1))
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("restricted checkpoint lacks an inference state dictionary")
    _require_plain_r01_args(args, n_in)
    expected_shapes = {
        "mean": (211,),
        "std": (211,),
        "inp.weight": (512, 211),
        "inp.bias": (512,),
        "gru.weight_ih_l0": (1536, 512),
        "gru.weight_hh_l0": (1536, 512),
        "gru.bias_ih_l0": (1536,),
        "gru.bias_hh_l0": (1536,),
        "head.0.weight": (512, 1024),
        "head.0.bias": (512,),
        "head.2.weight": (2, 512),
        "head.2.bias": (2,),
    }
    if set(state) != set(expected_shapes):
        raise ValueError("checkpoint state keys differ from the fixed plain GRU")
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape or state[name].dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} has unexpected shape or dtype")
        if not torch.isfinite(state[name]).all():
            raise ValueError(f"checkpoint tensor {name} is nonfinite")
    environment = TrainingEnvironment(source_root=source_root, root=data_root, argv=[])
    model = make_model(args, n_in, 2).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.eval()
    if (
        not torch.isfinite(model.mean).all()
        or not torch.isfinite(model.std).all()
        or torch.any(model.std <= 0)
    ):
        raise ValueError("checkpoint normalization buffers must be finite with positive scales")
    _, _, inventory = parameter_inventory(model)
    counts = {name: 0 for name in ("trunk_inp_gru", "shared_head_hidden", "target_output_excluded")}
    for item in inventory:
        counts[str(item["block"])] += int(item["numel"])
    if counts != {
        "trunk_inp_gru": 1_684_480,
        "shared_head_hidden": 524_800,
        "target_output_excluded": 1_026,
    }:
        raise ValueError("checkpoint parameter inventory differs from R01 terminal architecture")
    return ModelLoad(model, environment, args, n_in, sorted(state))


def state_fingerprint(model: nn.Module) -> str:
    """Hash all model parameter and buffer names, metadata and current bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
