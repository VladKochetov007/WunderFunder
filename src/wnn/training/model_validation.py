from __future__ import annotations

import argparse

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any


def apply_block_memory_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    """Use four block-memory layers unless the command explicitly sets --layers."""
    if args.arch != "block-memory":
        return
    if any(token == "--layers" or token.startswith("--layers=") for token in argv):
        return
    args.layers = 4


def block_memory_rejection(args: argparse.Namespace) -> str | None:
    """Reject modifiers that this raw-scratch teacher would ignore or crash on.

    Numeric maps, QAT, factorization, sentinel preprocessing, fixed readout
    mixtures and QMC initialization are also rejected later for non-GRU models.
    Privileged inputs, parent ``--init``, GRU ``--rnn-init``, Muon and random
    credit scaling are not, so this arch names them here.

    Returns:
        A parser error message, or None when the recipe can be constructed.
    """
    if args.arch != "block-memory":
        return None
    rejected: list[str] = []
    if args.numeric_map != "none":
        rejected.append("--numeric-map")
    if args.qat != "none":
        rejected.append("--qat")
    if args.recurrent_rank:
        rejected.append("--recurrent-rank")
    if args.privileged_mode != "none":
        rejected.append("--privileged-mode")
    if args.init:
        rejected.append("--init")
    if args.rnn_init != "default":
        rejected.append("--rnn-init")
    if args.cell != "gru":
        rejected.append("--cell")
    if args.enc != "linear":
        rejected.append("--enc")
    if args.dropout != 0:
        rejected.append("--dropout")
    if args.optim == "muon" or args.optim_switch:
        rejected.append("Muon or --optim-switch")
    if args.credit_mode != "deterministic":
        rejected.append("--credit-mode")
    if args.readout_activation != "gelu":
        rejected.append("--readout-activation")
    if (
        args.sentinel_indicators != "none"
        or args.sentinel_values != "unchanged"
        or args.sentinel_base_width
    ):
        rejected.append("sentinel preprocessing")
    if args.init_qmc:
        rejected.append("--init-qmc")
    if args.block_heads <= 0 or args.block_rows <= 0 or args.hidden <= 0 or (args.layers <= 0):
        rejected.append("non-positive hidden, layers, heads or block rows")
    elif args.hidden % args.block_heads:
        rejected.append("--hidden not divisible by --block-heads")
    if not rejected:
        return None
    return "block-memory raw scratch rejects " + ", ".join(rejected)


def independent_width_configuration_error(
    args_dict: dict[str, Any], n_in: int | None = None
) -> str | None:
    """Validate the independent-width shape against the native 211-feature GRU path."""
    requested = (
        any(
            args_dict.get(name) is not None
            for name in ("encoder_width", "recurrent_width", "readout_width")
        )
        or args_dict.get("head_layers", 1) != 1
    )
    if not requested:
        return None
    incompatible = []
    if args_dict.get("student_frontier_contract"):
        incompatible.append("equal-width --student-frontier-contract")
    if args_dict.get("arch", "gru") != "gru":
        incompatible.append("--arch")
    if args_dict.get("cell", "gru") != "gru":
        incompatible.append("--cell")
    if args_dict.get("enc", "linear") != "linear":
        incompatible.append("--enc")
    if args_dict.get("numeric_map", "none") != "none":
        incompatible.append("--numeric-map")
    if args_dict.get("readout_activation", "gelu") != "gelu":
        incompatible.append("--readout-activation")
    if args_dict.get("loss") == "ce_joint":
        incompatible.append("--loss ce_joint output decoding")
    if args_dict.get("recurrent_rank", 0):
        incompatible.append("--recurrent-rank")
    if n_in is not None and n_in != 211:
        incompatible.append("non-native input width")
    if args_dict.get("research_features", "none") != "none":
        incompatible.append("--research-features")
    if args_dict.get("privileged_mode", "none") != "none":
        incompatible.append("--privileged-mode")
    if args_dict.get("raw_only", False):
        incompatible.append("--raw-only")
    if args_dict.get("heuristics", False):
        incompatible.append("--heuristics")
    if args_dict.get("online_norm", False):
        incompatible.append("--online-norm")
    if args_dict.get("prop_input", ""):
        incompatible.append("--prop-input")
    if (
        args_dict.get("sentinel_base_width", 0)
        or args_dict.get("sentinel_indicators", "none") != "none"
        or args_dict.get("sentinel_values", "unchanged") != "unchanged"
    ):
        incompatible.append("sentinel preprocessing")
    if incompatible:
        reasons = ", ".join(incompatible)
        return f"independent widths/head depth require the native-compatible linear-input GELU GRU; {reasons} is unsupported"
    return None
