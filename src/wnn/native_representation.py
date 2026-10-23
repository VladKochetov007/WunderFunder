"""Input-layout validation for native submission exports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wnn.i1_history import (
    I1_HISTORY_MODE,
    I1_INPUT_WIDTH,
    i1_history_input_mask_indices,
    validate_i1_history_prior,
)

N_RAW = 112
N_EVENT = 99
N_BASE = N_RAW + N_EVENT
N_REPRESENTATION = 256
N_NATIVE_EMA_INPUT = N_BASE + N_REPRESENTATION
REPRESENTATION_SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"


def native_input_metadata(
    args: Mapping[str, object],
    state: Mapping[str, np.ndarray],
    checkpoint_n_in: object,
    history_prior: object = None,
    checkpoint_provenance: object = None,
) -> dict[str, object]:
    """Validate a checkpoint's native input layout and describe its stream contract.

    Args:
        args: Checkpoint training arguments.
        state: Float32 checkpoint state after loading.
        checkpoint_n_in: Checkpoint's declared model input width, if present.
        history_prior: Checkpoint's fitting-only i1-history prior, if present.
        checkpoint_provenance: Checkpoint's stored training provenance, if present.

    Returns:
        JSON-safe metadata for the build receipt and packaged member list.

    Raises:
        ValueError: If the input width, mode, or appended normalizers are unsupported.
    """
    mean = state.get("mean")
    std = state.get("std")
    if mean is None or std is None or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("native export requires one-dimensional mean and std buffers")
    width = int(mean.shape[0])
    if std.shape != mean.shape:
        raise ValueError("native export mean and std buffers must have the same width")
    if isinstance(checkpoint_n_in, bool):
        raise ValueError("checkpoint n_in must be an integer input width")
    if checkpoint_n_in is not None and (
        not isinstance(checkpoint_n_in, int) or checkpoint_n_in != width
    ):
        raise ValueError("checkpoint n_in does not match its normalization buffer width")
    if state.get("inp.weight") is None or state["inp.weight"].ndim != 2:
        raise ValueError("native export requires a matrix input projection")
    if state["inp.weight"].shape[1] != width:
        raise ValueError("input projection width does not match normalization buffers")

    research_mode = args.get("research_features", "none")
    if research_mode == I1_HISTORY_MODE:
        if args.get("representation_mode") not in (None, "none"):
            raise ValueError("i1_history cannot be combined with a native representation mode")
        if width != I1_INPUT_WIDTH or checkpoint_n_in != I1_INPUT_WIDTH:
            raise ValueError(
                f"i1_history requires checkpoint n_in={I1_INPUT_WIDTH} and matching buffers"
            )
        if not isinstance(history_prior, Mapping):
            raise ValueError("i1_history checkpoint lacks its fitting-only prior")
        prior = validate_i1_history_prior(history_prior)
        expected_prior = {
            "alpha_definition": "1 - 2**(-1/half_life_rows)",
            "mean_initialization": "first_observation",
            "energy_initialization": "zero",
            "energy_delta": "current_i1_row_minus_previous_i1_row",
            "first_row_history": "zero",
        }
        if any(prior.get(key) != value for key, value in expected_prior.items()):
            raise ValueError("i1_history prior does not match the supported streaming semantics")
        if (
            args.get("arch", "gru") != "gru"
            or args.get("cell", "gru") != "gru"
            or args.get("enc", "linear") != "linear"
            or bool(args.get("online_norm", False))
            or bool(args.get("heuristics", False))
            or bool(args.get("raw_only", False))
            or bool(args.get("prop_input", ""))
            or args.get("privileged_mode", "none") != "none"
            or args.get("numeric_map", "none") != "none"
            or bool(args.get("sentinel_base_width", 0))
            or args.get("sentinel_indicators", "none") != "none"
            or args.get("sentinel_values", "unchanged") != "unchanged"
        ):
            raise ValueError("i1_history checkpoint has unsupported model input transforms")
        control = args.get("research_feature_control", "none")
        drop_i1 = args.get("drop_i1", False)
        if not isinstance(drop_i1, bool):
            raise ValueError("i1_history drop_i1 must be a boolean checkpoint argument")
        mask_indices = i1_history_input_mask_indices(str(control), drop_i1)
        if not isinstance(checkpoint_provenance, Mapping):
            raise ValueError("i1_history checkpoint lacks stored mask provenance")
        feature_control = checkpoint_provenance.get("causal_feature_control")
        recorded_mask = (
            feature_control.get("zeroed_model_input_indices")
            if isinstance(feature_control, Mapping)
            else None
        )
        if (
            not isinstance(recorded_mask, (list, tuple))
            or any(type(index) is not int for index in recorded_mask)
            or tuple(recorded_mask) != mask_indices
            or feature_control.get("kind") != control
            or feature_control.get("drop_i1") is not drop_i1
            or feature_control.get("mask_point") != "after stored model normalization"
        ):
            raise ValueError("i1_history mask provenance disagrees with checkpoint arguments")
        return {
            "schema": "wnn.native_input_layout.v1",
            "representation_mode": I1_HISTORY_MODE,
            "input_width": width,
            "layout": [
                "raw112",
                "event99",
                "i1_residual_h32_52",
                "i1_residual_h512_52",
                "i1_log_energy_h512_52",
            ],
            "history_prior": prior,
            "post_normalization_mask_indices": list(recorded_mask),
            "mask_point": "after stored model normalization",
            "fast_activations": False,
        }
    if research_mode not in (None, "none"):
        raise ValueError(f"native export does not support research feature mode {research_mode!r}")

    raw_mode = args.get("representation_mode")
    if raw_mode in (None, "none"):
        mode = "none"
    elif isinstance(raw_mode, str):
        mode = raw_mode
    else:
        raise ValueError("checkpoint representation_mode must be a string or none")

    if mode in ("sparse", "ema_sparse"):
        raise ValueError(f"native export does not support representation mode {mode!r}")
    if mode == "none":
        if width not in (N_BASE, N_BASE + N_RAW):
            raise ValueError(f"native model width {width} has no supported representation metadata")
        online_norm = args.get("online_norm", False)
        if not isinstance(online_norm, bool) or online_norm != (width == N_BASE + N_RAW):
            raise ValueError("checkpoint online_norm flag does not match its native input width")
        layout = ["raw112", "event99"]
        if width == N_BASE + N_RAW:
            layout.append("online_norm112")
        return {
            "schema": "wnn.native_input_layout.v1",
            "representation_mode": mode,
            "input_width": width,
            "layout": layout,
        }

    if mode not in ("base", "ema"):
        raise ValueError(f"unknown checkpoint representation mode {mode!r}")
    if width != N_NATIVE_EMA_INPUT:
        raise ValueError(
            f"representation mode {mode!r} requires {N_NATIVE_EMA_INPUT} inputs, got {width}"
        )
    if checkpoint_n_in is None:
        raise ValueError("467-input representation checkpoint must declare n_in=467")
    if not np.array_equal(mean[N_BASE:], np.zeros(N_REPRESENTATION, dtype=mean.dtype)):
        raise ValueError("467-input checkpoint appended means must all be zero")
    if not np.array_equal(std[N_BASE:], np.ones(N_REPRESENTATION, dtype=std.dtype)):
        raise ValueError("467-input checkpoint appended scales must all be one")

    suffix = ["zero128", "zero112", "zero16"]
    if mode == "ema":
        suffix[1] = "causal_signed_raw_delta_ema112_hl32"
    return {
        "schema": "wnn.native_input_layout.v1",
        "representation_mode": mode,
        "input_width": width,
        "layout": ["raw112", "event99", *suffix],
        "producer_contract": {
            "git_commit": REPRESENTATION_SOURCE_COMMIT,
            "path": "src/wnn/training/input_representation.py",
        },
    }
