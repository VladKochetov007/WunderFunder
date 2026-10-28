from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from wnn.galam import GALAMState, galam_config_from_mapping
from wnn.galam.variant_config import galam_variant_from_mapping, validate_variant_config
from wnn.galam.variants import build_galam_variant


def galam_model_settings(raw: Mapping[str, object]) -> dict[str, object]:
    """Resolve explicit GALAM settings without silently applying Transformer defaults.

    Every architecture parameter must be present in the nested ``galam`` record.
    Common sequence-runtime aliases are derived and checked when supplied; their
    values are not independent architecture choices.
    """
    nested = raw.get("galam")
    if not isinstance(nested, Mapping):
        raise ValueError("GALAM requires an explicit model.galam configuration")
    variant = galam_variant_from_mapping(nested)
    config_values = {name: value for name, value in nested.items() if name != "variant"}
    config = galam_config_from_mapping(config_values)
    validate_variant_config(config, variant)
    if (config.input_features, config.output_features) != (112, 2):
        raise ValueError("sequence training requires the explicit 112-to-2 GALAM adaptation")
    if config.timestamp_source != "ordinal":
        raise ValueError("the sequence loader supplies ordinal steps, not physical timestamps")
    expected: dict[str, object] = {
        "architecture": "galam",
        "hidden": config.width,
        "layers": config.lstm_layers
        if variant == "schematic_lstm_local_only"
        else config.global_blocks,
        "heads": config.attention_heads,
        "ff_multiplier": config.feed_forward_width / config.width,
        "context_rows": config.attention_history,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "hint_projection": "learned",
        "attention_backend": "math",
        "compile_blocks": False,
    }
    if set(raw) - (set(expected) | {"galam", "precision_mode"}):
        raise ValueError("unknown GALAM runtime model field")
    for name, value in expected.items():
        if name in raw and (type(raw[name]) is bool) != (type(value) is bool):
            raise ValueError(f"GALAM runtime alias {name!r} has an incompatible type")
        if name in raw and raw[name] != value:
            raise ValueError(f"GALAM runtime alias {name!r} disagrees with model.galam")
    if raw.get("precision_mode", "fp32") != "fp32":
        raise ValueError("this unprofiled GALAM reconstruction currently requires fp32")
    return expected | {"galam": dict(nested), "precision_mode": "fp32"}


def validate_galam_batch_plan(
    config: Mapping[str, object], fit_sequences: int, world_size: int
) -> None:
    """Reject singleton training batches before any GALAM fitting or profiling.

    The canonical loop pads rank membership to an equal sequence count but keeps
    the final partial batch. Per-time BatchNorm needs at least two batch members;
    changing that normalization or silently dropping data would change the recipe.
    """
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("architecture") != "galam":
        return
    batch_size = config.get("batch_size")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("GALAM per-time BatchNorm requires batch_size >= 2 per rank")
    if type(fit_sequences) is not int or type(world_size) is not int:
        raise TypeError("sequence and rank counts must be integers")
    if min(fit_sequences, world_size) < 1:
        raise ValueError("positive fitting sequence count and world size required")
    if fit_sequences % world_size:
        raise ValueError(
            "GALAM BatchNorm cannot consume zero-weight DDP padding rows; "
            "use a world size dividing the source-qualified fitting membership"
        )
    per_rank = math.ceil(fit_sequences / world_size)
    if per_rank < 2 or per_rank % batch_size == 1:
        raise ValueError(
            "GALAM batch plan produces a singleton BatchNorm batch; "
            "choose a batch size compatible with actual chronological membership"
        )


class GALAMSequenceAdapter(nn.Module):
    """Expose the selected GALAM template or labelled adaptation to sequence training.

    This adapter adds train-only external normalization and the runtime's empty
    representation-hint output. Neither is a claim about the original recipe.
    """

    def __init__(self, settings: Mapping[str, object]) -> None:
        super().__init__()
        resolved = galam_model_settings(settings)
        nested = resolved["galam"]
        if not isinstance(nested, Mapping):
            raise TypeError("validated GALAM settings must retain a mapping")
        variant = galam_variant_from_mapping(nested)
        config_values = {name: value for name, value in nested.items() if name != "variant"}
        self.variant = variant
        self.galam = build_galam_variant(galam_config_from_mapping(config_values), variant)
        self.register_buffer("mean", torch.zeros(112, dtype=torch.float32))
        self.register_buffer("std", torch.ones(112, dtype=torch.float32))
        self.precision_mode = "fp32"

    def forward(
        self, values: Tensor, state: GALAMState | None = None
    ) -> tuple[Tensor, Tensor, GALAMState]:
        """Return causal predictions, no hint targets, and explicit resumable state."""
        if values.ndim != 3 or values.shape[-1] != 112 or values.shape[0] < 1:
            raise ValueError("GALAM sequence input must have shape [batch, rows, 112]")
        if values.dtype != torch.float32:
            raise TypeError("GALAM sequence inputs must be fp32")
        if torch.is_autocast_enabled(values.device.type):
            raise ValueError("GALAM reduced-precision behavior has not been profiled")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("GALAM sequence parameters must be fp32")
        if any(
            buffer.is_floating_point() and buffer.dtype != torch.float32
            for buffer in self.buffers()
        ):
            raise TypeError("GALAM floating state buffers must be fp32")
        if state is None:
            state = self.galam.reset_state(values.shape[0], values.device, values.dtype)
        prediction, next_state = self.galam.forward_chunk((values - self.mean) / self.std, state)
        return prediction, values.new_empty((*values.shape[:2], 0)), next_state
