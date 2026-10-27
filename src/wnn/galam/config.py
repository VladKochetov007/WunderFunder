from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

KNOWN_GRAPH_DETAILS = frozenset(
    {
        "separate_global_and_local_memories",
        "local_batch_norm_then_lstm",
        "temporal_attention_in_both_memories",
        "learned_relative_timestamp_bias",
        "query_key_rms_normalization",
        "schematic_global_conv_mixer",
        "schematic_conv_gelu_batchnorm_order",
        "global_residual_post_layer_norm",
        "gated_fusion",
        "fusion_gate_linear_silu_linear_sigmoid",
        "post_gate_channel_product",
        "decoder_post_rms_norm_and_gelu_ffn",
    }
)

HISTORICAL_SHAPE_DETAIL = "historical_32_input_and_output_widths"

UNRESOLVED_DETAILS = frozenset(
    {
        "width_and_attention_heads",
        "lstm_and_attention_block_counts",
        "attention_history_and_convolution_kernel",
        "relative_time_scale_and_source",
        "fusion_channel_order",
        "fusion_bypass_source_equation",
        "hidden_cell_projection_and_ffn_widths",
        "convolution_hidden_width",
        "batch_norm_axis_and_running_stat_convention",
        "author_geglu_text_vs_schematic_conv_mixer",
        "connectome_output_adaptation",
    }
)


@dataclass(frozen=True)
class GALAMConfig:
    """All shape and source-uncertain GALAM choices for one declared use."""

    input_features: int
    output_features: int
    width: int
    attention_heads: int
    lstm_layers: int
    global_blocks: int
    local_attention_blocks: int
    attention_history: int
    conv_kernel_size: int
    conv_hidden_width: int
    relative_bias_width: int
    timestamp_source: Literal["ordinal", "provided"]
    timestamp_scale: float
    feed_forward_width: int
    fusion_mode: Literal["diagram_channel_product"]
    fusion_channel_order: tuple[
        Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
        Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
        Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
        Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
    ]
    faithful_details: frozenset[str]
    assumed_details: frozenset[str]
    role: Literal["historical_32_to_32", "connectome_112_to_2", "custom"]

    def __post_init__(self) -> None:
        """Reject invalid dimensions and incomplete fidelity declarations."""
        integer_fields = (
            self.input_features,
            self.output_features,
            self.width,
            self.attention_heads,
            self.lstm_layers,
            self.global_blocks,
            self.local_attention_blocks,
            self.attention_history,
            self.conv_kernel_size,
            self.conv_hidden_width,
            self.relative_bias_width,
            self.feed_forward_width,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields
        ):
            raise ValueError("GALAM dimensions, depths, and bounded histories must be positive")
        if self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if (
            isinstance(self.timestamp_scale, bool)
            or not isinstance(self.timestamp_scale, (int, float))
            or not math.isfinite(self.timestamp_scale)
            or self.timestamp_scale <= 0
        ):
            raise ValueError("timestamp_scale must be finite and positive")
        if self.timestamp_source not in ("ordinal", "provided"):
            raise ValueError("timestamp_source must be ordinal or provided")
        if self.role not in ("historical_32_to_32", "connectome_112_to_2", "custom"):
            raise ValueError("role must be historical_32_to_32, connectome_112_to_2, or custom")
        if self.role == "historical_32_to_32" and (self.input_features, self.output_features) != (
            32,
            32,
        ):
            raise ValueError("historical_32_to_32 role requires 32 inputs and 32 outputs")
        if self.role == "connectome_112_to_2" and (self.input_features, self.output_features) != (
            112,
            2,
        ):
            raise ValueError("connectome_112_to_2 role requires 112 inputs and 2 outputs")
        if not isinstance(self.faithful_details, frozenset) or not isinstance(
            self.assumed_details, frozenset
        ):
            raise ValueError("fidelity detail sets must be immutable frozensets")
        if any(
            not isinstance(detail, str) for detail in self.faithful_details | self.assumed_details
        ):
            raise ValueError("fidelity details must be strings")
        if self.fusion_mode != "diagram_channel_product":
            raise ValueError("fusion_mode must be diagram_channel_product")
        allowed_channels = {
            "global_memory",
            "local_memory",
            "hidden_projection",
            "cell_projection",
        }
        if (
            not isinstance(self.fusion_channel_order, tuple)
            or len(self.fusion_channel_order) != 4
            or any(item not in allowed_channels for item in self.fusion_channel_order)
            or set(self.fusion_channel_order) != allowed_channels
        ):
            raise ValueError("fusion_channel_order must order each product source exactly once")
        if self.faithful_details & self.assumed_details:
            raise ValueError("a detail cannot be both source-supported and assumed")
        expected = KNOWN_GRAPH_DETAILS | UNRESOLVED_DETAILS
        if self.role == "historical_32_to_32":
            expected = expected - {"connectome_output_adaptation"} | {HISTORICAL_SHAPE_DETAIL}
        elif self.role == "connectome_112_to_2":
            expected = expected - {HISTORICAL_SHAPE_DETAIL}
        else:
            expected = expected - {"connectome_output_adaptation", HISTORICAL_SHAPE_DETAIL}
        required_faithful = KNOWN_GRAPH_DETAILS
        required_assumed = UNRESOLVED_DETAILS
        if self.role == "historical_32_to_32":
            required_faithful = required_faithful | {HISTORICAL_SHAPE_DETAIL}
            required_assumed = required_assumed - {"connectome_output_adaptation"}
        elif self.role == "connectome_112_to_2":
            required_faithful = required_faithful - {HISTORICAL_SHAPE_DETAIL}
        else:
            required_assumed = required_assumed - {"connectome_output_adaptation"}
        if not required_faithful <= self.faithful_details:
            raise ValueError("source-supported details cannot be marked assumed")
        if not required_assumed <= self.assumed_details:
            raise ValueError("unresolved details must remain marked assumed")
        if self.faithful_details | self.assumed_details != expected:
            raise ValueError(
                "faithful_details and assumed_details must classify every declared detail"
            )


def historical_32_config() -> GALAMConfig:
    """Return a clearly parameterized 32-feature historical architecture template."""
    return _template_config(32, 32, "historical_32_to_32")


def connectome_112_to_2_config() -> GALAMConfig:
    """Return the separate 112-input, 2-output Connectome adaptation template."""
    return _template_config(112, 2, "connectome_112_to_2")


def _template_config(
    input_features: int,
    output_features: int,
    role: Literal["historical_32_to_32", "connectome_112_to_2"],
) -> GALAMConfig:
    """Build example dimensions while marking each unresolved choice as an assumption."""
    faithful = KNOWN_GRAPH_DETAILS
    assumed = UNRESOLVED_DETAILS
    if role == "historical_32_to_32":
        faithful = faithful | {HISTORICAL_SHAPE_DETAIL}
        assumed = assumed - {"connectome_output_adaptation"}
    faithful = frozenset(faithful)
    assumed = frozenset(assumed)
    return GALAMConfig(
        input_features=input_features,
        output_features=output_features,
        width=64,
        attention_heads=4,
        lstm_layers=1,
        global_blocks=1,
        local_attention_blocks=1,
        attention_history=64,
        conv_kernel_size=5,
        conv_hidden_width=128,
        relative_bias_width=32,
        timestamp_source="ordinal",
        timestamp_scale=1.0,
        feed_forward_width=128,
        fusion_mode="diagram_channel_product",
        fusion_channel_order=(
            "global_memory",
            "local_memory",
            "hidden_projection",
            "cell_projection",
        ),
        faithful_details=faithful,
        assumed_details=assumed,
        role=role,
    )


def galam_config_from_mapping(values: Mapping[str, object]) -> GALAMConfig:
    """Parse a complete JSON-style mapping without filling missing assumptions."""
    required = {
        "input_features",
        "output_features",
        "width",
        "attention_heads",
        "lstm_layers",
        "global_blocks",
        "local_attention_blocks",
        "attention_history",
        "conv_kernel_size",
        "conv_hidden_width",
        "relative_bias_width",
        "timestamp_source",
        "timestamp_scale",
        "feed_forward_width",
        "fusion_mode",
        "fusion_channel_order",
        "faithful_details",
        "assumed_details",
        "role",
    }
    missing = required - values.keys()
    extra = values.keys() - required
    if missing or extra:
        raise ValueError(f"config keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    scalar_ints = required - {
        "timestamp_scale",
        "timestamp_source",
        "fusion_mode",
        "fusion_channel_order",
        "faithful_details",
        "assumed_details",
        "role",
    }
    for name in scalar_ints:
        _mapping_int(values, name)
    scale = values["timestamp_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("timestamp_scale must be numeric")
    timestamp_source = values["timestamp_source"]
    if timestamp_source not in ("ordinal", "provided"):
        raise ValueError("timestamp_source must be ordinal or provided")
    role = values["role"]
    if role not in ("historical_32_to_32", "connectome_112_to_2", "custom"):
        raise ValueError("role must be a recognized GALAM config role")
    fusion_mode = values["fusion_mode"]
    if fusion_mode != "diagram_channel_product":
        raise ValueError("fusion_mode must be diagram_channel_product")
    order = values["fusion_channel_order"]
    allowed_channels = {
        "global_memory",
        "local_memory",
        "hidden_projection",
        "cell_projection",
    }
    if not isinstance(order, (list, tuple)) or len(order) != 4:
        raise ValueError("fusion_channel_order must contain four product sources")
    if any(item not in allowed_channels for item in order) or set(order) != allowed_channels:
        raise ValueError("fusion_channel_order must order each product source exactly once")
    return GALAMConfig(
        input_features=_mapping_int(values, "input_features"),
        output_features=_mapping_int(values, "output_features"),
        width=_mapping_int(values, "width"),
        attention_heads=_mapping_int(values, "attention_heads"),
        lstm_layers=_mapping_int(values, "lstm_layers"),
        global_blocks=_mapping_int(values, "global_blocks"),
        local_attention_blocks=_mapping_int(values, "local_attention_blocks"),
        attention_history=_mapping_int(values, "attention_history"),
        conv_kernel_size=_mapping_int(values, "conv_kernel_size"),
        conv_hidden_width=_mapping_int(values, "conv_hidden_width"),
        relative_bias_width=_mapping_int(values, "relative_bias_width"),
        timestamp_source=cast(Literal["ordinal", "provided"], timestamp_source),
        timestamp_scale=float(scale),
        feed_forward_width=_mapping_int(values, "feed_forward_width"),
        fusion_mode=cast(Literal["diagram_channel_product"], fusion_mode),
        fusion_channel_order=cast(
            tuple[
                Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
                Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
                Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
                Literal["global_memory", "local_memory", "hidden_projection", "cell_projection"],
            ],
            tuple(order),
        ),
        faithful_details=_string_set(values["faithful_details"], "faithful_details"),
        assumed_details=_string_set(values["assumed_details"], "assumed_details"),
        role=cast(Literal["historical_32_to_32", "connectome_112_to_2", "custom"], role),
    )


def _mapping_int(values: Mapping[str, object], name: str) -> int:
    """Read one validated integer config field from an untyped mapping."""
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _string_set(value: object, name: str) -> frozenset[str]:
    """Convert JSON string arrays or equivalent collections to an immutable set."""
    if not isinstance(value, (list, tuple, set, frozenset)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must be a collection of strings")
    return frozenset(value)
