from __future__ import annotations

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch

from wnn.block_memory_attention import BlockMemoryAttention
from wnn.event_content import EventContentModel
from wnn.nn import DualClockModel, GatedRNNModel, RNNModel, TCNModel, TwoClockModel
from wnn.sentinel import SentinelInputProjection
from wnn.specialist import PrefixAdapterGRUModel, SpecialistGRUModel
from wnn.structured_state import StructuredStateModel
from wnn.tabm_gru import TabMGRUModel
from wnn.training.i1_inputs import configure_i1_input_mask
from wnn.training.model_validation import independent_width_configuration_error


def make_model(args_dict: dict[str, Any], n_in: int, n_out: int) -> torch.nn.Module:
    """Build a model and restore checkpoint-declared semantic input masks."""
    model = _make_model(args_dict, n_in, n_out)
    configure_i1_input_mask(model, args_dict, n_in)
    return model


def _make_model(args_dict: dict[str, Any], n_in: int, n_out: int) -> torch.nn.Module:
    """Construct the requested architecture with the historical constructor order."""
    width_error = independent_width_configuration_error(args_dict, n_in)
    if width_error is not None:
        raise ValueError(width_error)
    numeric_map = args_dict.get("numeric_map", "none")
    if numeric_map != "none" and (
        args_dict.get("arch", "gru") != "gru"
        or args_dict.get("cell", "gru") != "gru"
        or args_dict.get("enc", "linear") != "linear"
    ):
        raise ValueError("numeric maps require the plain GRU with linear input encoding")
    if args_dict.get("readout_activation", "gelu") != "gelu" and (
        args_dict.get("arch", "gru") != "gru"
        or args_dict.get("cell", "gru") != "gru"
        or args_dict.get("enc", "linear") != "linear"
    ):
        raise ValueError("fixed readout mixture requires a plain GRU and linear encoder")
    if args_dict.get("arch") == "specialist":
        if (
            args_dict.get("cell", "gru") != "gru"
            or args_dict.get("enc", "linear") != "linear"
            or args_dict["layers"] != 1
            or (args_dict["dropout"] != 0)
        ):
            raise ValueError(
                "specialist requires one-layer GRUs, dropout0, and linear input encoding"
            )
        return SpecialistGRUModel(
            n_in,
            args_dict["hidden"],
            args_dict.get("expert_hidden", 32),
            args_dict.get("num_experts", 4),
            args_dict["layers"],
            n_out=n_out,
            route=args_dict.get("specialist_route", "learned"),
            prefix_rows=args_dict.get("prefix_rows", 99),
            hash_seed=args_dict.get("hash_seed", 1729),
        )
    if args_dict.get("arch") == "prefix-adapter":
        if args_dict.get("cell", "gru") != "gru" or args_dict.get("enc", "linear") != "linear":
            raise ValueError("prefix-adapter currently requires --cell gru and --enc linear")
        return PrefixAdapterGRUModel(
            n_in,
            args_dict["hidden"],
            args_dict["layers"],
            args_dict["dropout"],
            n_out=n_out,
            rank=args_dict.get("adapter_rank", 8),
            mode=args_dict.get("adapter_mode", "conditional"),
            prefix_rows=args_dict.get("prefix_rows", 99),
        )
    if args_dict.get("arch") == "structured":
        if (
            args_dict["hidden"] != 96
            or args_dict["layers"] != 2
            or args_dict["dropout"] != 0
            or (args_dict.get("enc", "linear") != "linear")
        ):
            raise ValueError("structured card B requires hidden96, layers2, dropout0, enc linear")
        model = StructuredStateModel(n_in, n_out, args_dict.get("decay_init", "multi"))
        for prefix in filter(None, args_dict.get("freeze", "").split(",")):
            if not any((name.startswith(prefix) for name, _ in model.named_parameters())):
                raise ValueError(
                    f"unknown structured freeze prefix {prefix!r}; backbone is inp,blocks"
                )
        return model
    if args_dict.get("arch") == "event-content":
        if (
            args_dict["hidden"] != 96
            or args_dict["layers"] != 1
            or args_dict["dropout"] != 0
            or (args_dict.get("enc", "linear") != "linear")
        ):
            raise ValueError("D1 requires hidden96,layers1,dropout0,enc linear")
        return EventContentModel(n_in, n_out, args_dict.get("memory_arm", "endpoint-content"))
    if args_dict.get("arch") == "tcn":
        if args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear":
            raise ValueError("tcn requires dropout0, enc linear")
        model = TCNModel(
            n_in,
            args_dict["hidden"],
            args_dict["layers"],
            n_out=n_out,
            mix=bool(args_dict.get("tcn_mix", 0)),
        )
        for prefix in filter(None, args_dict.get("freeze", "").split(",")):
            if not any((name.startswith(prefix) for name, _ in model.named_parameters())):
                raise ValueError(f"unknown tcn freeze prefix {prefix!r}; trunk is inp,tcn")
        return model
    if args_dict.get("arch") == "twoclock":
        if args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear":
            raise ValueError("twoclock requires dropout0, enc linear")
        model = TwoClockModel(
            n_in, args_dict["hidden"], args_dict["layers"], args_dict["event_hidden"], n_out=n_out
        )
        frozen = tuple(filter(None, args_dict.get("freeze", "").split(",")))
        loose = [
            name
            for name, _ in model.named_parameters()
            if frozen and (not name.startswith(frozen)) and (not name.startswith("head."))
        ]
        if loose:
            raise ValueError(
                f"twoclock --freeze must cover the whole trunk (inp,gru); trainable: {loose}"
            )
        return model
    if args_dict.get("arch") == "gated":
        return GatedRNNModel(
            n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out
        )
    if args_dict.get("arch", "gru") == "dual":
        return DualClockModel(
            n_in,
            args_dict["hidden"],
            args_dict["layers"],
            args_dict["event_hidden"],
            args_dict["dropout"],
            n_out=n_out,
        )
    if args_dict.get("arch") == "block-memory":
        if (
            args_dict.get("cell", "gru") != "gru"
            or args_dict.get("enc", "linear") != "linear"
            or args_dict.get("dropout", 0) != 0
        ):
            raise ValueError("block-memory needs gru cell, linear encoder, dropout 0")
        heads = args_dict.get("block_heads", 8)
        block_rows = args_dict.get("block_rows", 256)
        hidden = args_dict["hidden"]
        if heads <= 0 or block_rows <= 0 or hidden <= 0 or hidden % heads:
            raise ValueError("block-memory hidden must be divisible by positive heads")
        model = BlockMemoryAttention(
            n_in, hidden, args_dict["layers"], heads, block_rows, n_out=n_out
        )
        for prefix in filter(None, args_dict.get("freeze", "").split(",")):
            if not any((name.startswith(prefix) for name, _ in model.named_parameters())):
                raise ValueError(
                    f"unknown block-memory freeze prefix {prefix!r}; trunk is project,layers"
                )
        return model
    if args_dict.get("ensemble_k", 1) > 1:
        return _make_tabm(args_dict, n_in, n_out)
    model = RNNModel(
        n_in,
        args_dict["hidden"],
        args_dict["layers"],
        args_dict["dropout"],
        n_out=n_out,
        cell=args_dict.get("cell", "gru"),
        enc=args_dict.get("enc", "linear"),
        tok_dim=args_dict.get("tok_dim", 32),
        att_layers=args_dict.get("att_layers", 1),
        conv_k=args_dict.get("conv_k", 16),
        readout_activation=args_dict.get("readout_activation", "gelu"),
        numeric_map=numeric_map,
        ple_intervals=args_dict.get("ple_intervals", 8),
        encoder_width=args_dict.get("encoder_width"),
        recurrent_width=args_dict.get("recurrent_width"),
        readout_width=args_dict.get("readout_width"),
        head_layers=args_dict.get("head_layers", 1),
        dynamics_aux_head=bool(args_dict.get("dynamics_aux_head", False)),
    )
    if args_dict.get("sentinel_indicators", "none") != "none":
        model.inp = SentinelInputProjection(model.inp, args_dict["sentinel_base_width"])
    return model


def _make_tabm(args_dict: dict[str, Any], n_in: int, n_out: int) -> torch.nn.Module:
    """TabM-mini GRU ensemble; only the plain linear-encoder GRU layout is materializable."""
    unsupported = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "numeric_map": "none",
        "readout_activation": "gelu",
        "head_layers": 1,
        "dropout": 0,
        "dynamics_aux_head": False,
        "sentinel_indicators": "none",
        "encoder_width": None,
        "recurrent_width": None,
        "readout_width": None,
    }
    for key, expected in unsupported.items():
        if args_dict.get(key, expected) != expected:
            raise ValueError(f"--ensemble-k requires {key}={expected!r}")
    return TabMGRUModel(
        n_in, args_dict["hidden"], args_dict["layers"], n_out, args_dict["ensemble_k"]
    )
