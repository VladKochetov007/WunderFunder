from __future__ import annotations

from collections.abc import Mapping

import torch

from wnn.i1_history import I1_HISTORY_MODE, I1_INPUT_WIDTH, i1_history_input_mask_indices
from wnn.nn import RNNModel


def configure_i1_input_mask(
    model: torch.nn.Module, args: Mapping[str, object], input_width: int
) -> None:
    """Attach the checkpoint-configured i1 mask after model normalization.

    The model factory and explicit environment hook both call this boundary so
    standard and injected constructors restore the same feature behavior.
    """
    validate_i1_input_configuration(args, input_width)
    mode = str(args.get("research_features", "none"))
    if mode != I1_HISTORY_MODE:
        return
    if not isinstance(model, RNNModel):
        raise ValueError("i1_history input masks require the normalized recurrent model")
    indices = i1_history_input_mask_indices(
        str(args.get("research_feature_control", "none")), bool(args.get("drop_i1", False))
    )
    model._i1_input_mask_indices = indices


def validate_i1_input_configuration(args: Mapping[str, object], input_width: int) -> None:
    """Reject i1 mask configurations whose width or feature semantics are unsupported."""
    mode = str(args.get("research_features", "none"))
    drop_i1 = bool(args.get("drop_i1", False))
    if mode != I1_HISTORY_MODE:
        if drop_i1:
            raise ValueError("--drop-i1 requires --research-features i1_history")
        return
    control = str(args.get("research_feature_control", "none"))
    if control not in ("none", "all"):
        raise ValueError("i1_history supports only --research-feature-control none or all")
    if input_width != I1_INPUT_WIDTH:
        raise ValueError(
            f"i1_history requires the fixed {I1_INPUT_WIDTH}-column input, got {input_width}"
        )
    unsupported = [
        name
        for name, active in (
            ("architecture", args.get("arch", "gru") not in ("gru", "gated")),
            ("cell", args.get("cell", "gru") != "gru"),
            ("encoder", args.get("enc", "linear") != "linear"),
            ("online normalization", bool(args.get("online_norm", False))),
            ("heuristics", bool(args.get("heuristics", False))),
            ("raw-only inputs", bool(args.get("raw_only", False))),
            ("propensity inputs", bool(args.get("prop_input", ""))),
            ("privileged inputs", args.get("privileged_mode", "none") != "none"),
            ("numeric maps", args.get("numeric_map", "none") != "none"),
            (
                "sentinel preprocessing",
                bool(args.get("sentinel_base_width", 0))
                or args.get("sentinel_indicators", "none") != "none"
                or args.get("sentinel_values", "unchanged") != "unchanged",
            ),
        )
        if active
    ]
    if unsupported:
        raise ValueError(f"i1_history input masks do not support: {', '.join(unsupported)}")
