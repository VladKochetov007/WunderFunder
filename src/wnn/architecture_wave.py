"""Trainer adapter for the scratch Transformer/Mamba research wave.

Imports of the optional CUDA Mamba package happen only when that model is selected.
These backends export research predictions, not competition CPU submission packages.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

ARCHITECTURES = ("causal-transformer", "mamba2")
MODEL_SOURCES = {
    "causal-transformer": "src/wnn/causal_transformer.py",
    "mamba2": "src/wnn/mamba2_regressor.py",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shape options stored verbatim in each research checkpoint."""
    group = parser.add_argument_group("streaming architecture wave")
    group.add_argument("--attention-heads", type=int, default=8)
    group.add_argument("--attention-window", type=int, default=256)
    group.add_argument("--attention-ffn-mult", type=int, default=4)
    group.add_argument("--mamba-state-size", type=int, default=64)
    group.add_argument("--mamba-head-dim", type=int, default=64)
    group.add_argument("--mamba-expand", type=int, default=2)
    group.add_argument("--mamba-conv-width", type=int, default=4)
    group.add_argument("--mamba-chunk-size", type=int, default=256)


# Saved argparse/checkpoint dictionaries contain heterogeneous values, hence Any.
def rejection(options: dict[str, Any]) -> str | None:
    """Fail early for GRU-only modifiers rather than silently ignoring them."""
    arch = options.get("arch")
    if arch not in ARCHITECTURES:
        return None
    defaults = {
        "numeric_map": "none",
        "qat": "none",
        "recurrent_rank": 0,
        "privileged_mode": "none",
        "init": "",
        "rnn_init": "default",
        "cell": "gru",
        "enc": "linear",
        "dropout": 0.0,
        "optim_switch": "",
        "credit_mode": "deterministic",
        "readout_activation": "gelu",
        "sentinel_indicators": "none",
        "sentinel_values": "unchanged",
        "sentinel_base_width": 0,
        "init_qmc": False,
        "activation_backend": "none",
        "zoneout": 0.0,
        "freeze": "",
        "head_layers": 1,
        "gate_ce": 0.0,
    }
    rejected = [
        "--" + name.replace("_", "-")
        for name, default in defaults.items()
        if options.get(name, default) != default
    ]
    if options.get("optim") == "muon":
        rejected.append("Muon")
    if any(
        options.get(name) is not None
        for name in (
            "encoder_width",
            "recurrent_width",
            "readout_width",
        )
    ):
        rejected.append("independent GRU widths")
    hidden, layers = options.get("hidden", 256), options.get("layers", 4)
    if hidden <= 0 or layers <= 0:
        rejected.append("non-positive hidden/layers")
    if arch == "causal-transformer":
        heads = options.get("attention_heads", 8)
        if heads <= 0 or hidden % heads or (hidden // heads) % 2:
            rejected.append("attention heads must divide hidden into even RoPE dimensions")
        if options.get("attention_window", 256) <= 0 or options.get("attention_ffn_mult", 4) <= 0:
            rejected.append("non-positive attention window/FFN multiplier")
    else:
        names = (
            "mamba_state_size",
            "mamba_head_dim",
            "mamba_expand",
            "mamba_conv_width",
            "mamba_chunk_size",
        )
        values = [
            options.get(name, default)
            for name, default in zip(
                names,
                (64, 64, 2, 4, 256),
                strict=True,
            )
        ]
        if any(value <= 0 for value in values):
            rejected.append("non-positive Mamba shape")
        elif hidden * values[2] % values[1]:
            rejected.append("Mamba expanded width must be divisible by head dimension")
        if values[4] > 0 and values[4] & (values[4] - 1):
            rejected.append("Mamba scan chunk size must be a power of two")
        if options.get("device", "cuda") != "cuda":
            rejected.append("Mamba-2 requires a Linux CUDA host")
    return f"{arch} scratch backend rejects " + ", ".join(rejected) if rejected else None


def make_sequence_model(options: dict[str, Any], n_in: int, n_out: int) -> nn.Module:
    """Construct the selected streaming model, preserving the shared head ABI."""
    error = rejection(options)
    if error is not None:
        raise ValueError(error)
    common = dict(n_in=n_in, hidden=options["hidden"], layers=options["layers"], n_out=n_out)
    if options["arch"] == "causal-transformer":
        from wnn.causal_transformer import CausalTransformer

        return CausalTransformer(
            **common,
            heads=options.get("attention_heads", 8),
            window=options.get("attention_window", 256),
            ffn_mult=options.get("attention_ffn_mult", 4),
        )
    if options["arch"] == "mamba2":
        from wnn.mamba2_regressor import Mamba2Regressor

        return Mamba2Regressor(
            **common,
            state_size=options.get("mamba_state_size", 64),
            head_dim=options.get("mamba_head_dim", 64),
            expand=options.get("mamba_expand", 2),
            conv_width=options.get("mamba_conv_width", 4),
            chunk_size=options.get("mamba_chunk_size", 256),
        )
    raise ValueError(f"unknown streaming architecture: {options['arch']}")
