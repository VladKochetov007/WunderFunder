"""Grow a dense GRU's widths while preserving its initial inference function."""

from __future__ import annotations

from collections.abc import Mapping

import torch

State = Mapping[str, torch.Tensor]
_BASE = {
    "mean",
    "std",
    "inp.weight",
    "inp.bias",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
}


def _layout(state: State) -> tuple[int, int, int, int]:
    """Validate the legacy linear-encoder, one-hidden-readout GRU layout."""
    if not _BASE.issubset(state) or "gru.weight_hh_l0" not in state:
        raise ValueError("expected a dense linear-encoder GRU checkpoint")
    encoder, n_in = state["inp.weight"].shape
    recurrent = state["gru.weight_hh_l0"].shape[1]
    readout = state["head.0.weight"].shape[0]
    n_out = state["head.2.weight"].shape[0]
    expected = {
        "mean": (n_in,),
        "std": (n_in,),
        "inp.weight": (encoder, n_in),
        "inp.bias": (encoder,),
        "head.0.weight": (readout, recurrent + encoder),
        "head.0.bias": (readout,),
        "head.2.weight": (n_out, readout),
        "head.2.bias": (n_out,),
    }
    layers = 0
    while f"gru.weight_ih_l{layers}" in state:
        inputs = encoder if layers == 0 else recurrent
        expected.update(
            {
                f"gru.weight_ih_l{layers}": (3 * recurrent, inputs),
                f"gru.weight_hh_l{layers}": (3 * recurrent, recurrent),
                f"gru.bias_ih_l{layers}": (3 * recurrent,),
                f"gru.bias_hh_l{layers}": (3 * recurrent,),
            }
        )
        layers += 1
    if not layers or state.keys() != expected.keys():
        raise ValueError("only unidirectional dense GRUs with one hidden head are supported")
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"incompatible {name} shape: expected {shape}")
        if state[name].dtype not in (torch.float32, torch.float64):
            raise ValueError("growth requires dense FP32 or FP64 state")
    return encoder, recurrent, readout, layers


def grow_gru_state(parent: State, initialized_child: State) -> dict[str, torch.Tensor]:
    """Embed parent units and retain random new units behind zero connectors.

    Both models must use the same GRU equations, linear/GELU encoder, readout
    activation, input preprocessing and output semantics. Widths may increase;
    layer count, input and output widths must match. Copy all output rows,
    including auxiliary heads. No caller state is mutated. The function is
    preserved in exact arithmetic; changed matrix shapes can alter FP rounding.
    """
    ep, hp, mp, lp = _layout(parent)
    ec, hc, mc, lc = _layout(initialized_child)
    if lp != lc or min(ec - ep, hc - hp, mc - mp) < 0:
        raise ValueError("growth cannot shrink widths or change recurrent depth")
    if (ep, hp, mp) == (ec, hc, mc):
        raise ValueError("at least one width must grow")
    if (
        parent["mean"].shape != initialized_child["mean"].shape
        or parent["head.2.bias"].shape != initialized_child["head.2.bias"].shape
    ):
        raise ValueError("input and complete output widths must match")
    if any(parent[key].dtype != initialized_child[key].dtype for key in parent):
        raise ValueError("growth cannot change parameter precision")
    child = {name: value.detach().clone() for name, value in initialized_child.items()}
    with torch.no_grad():
        for name in ("mean", "std"):
            child[name].copy_(parent[name])
        child["inp.weight"][:ep].copy_(parent["inp.weight"])
        child["inp.bias"][:ep].copy_(parent["inp.bias"])
        for layer in range(lp):
            old_inputs = ep if layer == 0 else hp
            for gate in range(3):
                old_rows = slice(gate * hp, (gate + 1) * hp)
                rows = slice(gate * hc, gate * hc + hp)
                for kind, columns in (("ih", old_inputs), ("hh", hp)):
                    name = f"gru.weight_{kind}_l{layer}"
                    child[name][rows].zero_()
                    child[name][rows, :columns].copy_(parent[name][old_rows])
                    bias = f"gru.bias_{kind}_l{layer}"
                    child[bias][rows].copy_(parent[bias][old_rows])
        # Readout columns concatenate recurrent output THEN the encoder skip.
        child["head.0.weight"][:mp].zero_()
        child["head.0.weight"][:mp, :hp].copy_(parent["head.0.weight"][:, :hp])
        child["head.0.weight"][:mp, hc : hc + ep].copy_(parent["head.0.weight"][:, hp:])
        child["head.0.bias"][:mp].copy_(parent["head.0.bias"])
        child["head.2.weight"].zero_()
        child["head.2.weight"][:, :mp].copy_(parent["head.2.weight"])
        child["head.2.bias"].copy_(parent["head.2.bias"])
    return child


def grow_gru_hidden(hidden: torch.Tensor, width: int) -> torch.Tensor:
    """Embed an old [layer, batch, unit] state with zero initial new units."""
    if hidden.ndim != 3 or width < hidden.shape[2]:
        raise ValueError("hidden state must have three axes and cannot shrink")
    result = hidden.new_zeros((*hidden.shape[:2], width))
    result[..., : hidden.shape[2]] = hidden
    return result
