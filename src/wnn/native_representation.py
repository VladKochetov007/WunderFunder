"""Input-layout validation for native submission exports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

N_RAW = 112
N_EVENT = 99
N_BASE = N_RAW + N_EVENT
N_REPRESENTATION = 256
N_NATIVE_EMA_INPUT = N_BASE + N_REPRESENTATION
REPRESENTATION_SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"


def native_input_metadata(
    args: Mapping[str, object], state: Mapping[str, np.ndarray], checkpoint_n_in: object
) -> dict[str, object]:
    """Validate a checkpoint's native input layout and describe its stream contract.

    Args:
        args: Checkpoint training arguments.
        state: Float32 checkpoint state after loading.
        checkpoint_n_in: Checkpoint's declared model input width, if present.

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
