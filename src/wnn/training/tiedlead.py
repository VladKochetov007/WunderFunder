from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from wnn.data import I1, N_FEATURES
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES
from wnn.nn import RNNModel

TIEDLEAD_BASE_WIDTH = 211
TIEDLEAD_LEAD_WIDTH = 52
TIEDLEAD_INPUT_WIDTH = TIEDLEAD_BASE_WIDTH + TIEDLEAD_LEAD_WIDTH
TIEDLEAD_HIDDEN = 256
TIED_ENCODER_HIDDEN = 128
TIED_ENCODER_WIDTH = 64
_I0 = slice(0, 52)
_I1 = slice(52, 104)
_EVENT_I0PX = FEATURE_NAMES.index("ev_i0px")


def tiedlead_configuration_error(args: Mapping[str, object]) -> str | None:
    """Reject options that would silently change the fixed tiedlead profile."""
    fixed = {
        "arch": "tiedlead",
        "cell": "gru",
        "enc": "linear",
        "hidden": TIEDLEAD_HIDDEN,
        "layers": 2,
        "dropout": 0.0,
        "precision": "fp32",
        "numeric_map": "none",
        "readout_activation": "gelu",
    }
    invalid = [
        f"--{name.replace('_', '-')}={args.get(name)!r}"
        for name, value in fixed.items()
        if args.get(name, value) != value
    ]
    disabled = {
        "raw_only": "--raw-only",
        "heuristics": "--heuristics",
        "online_norm": "--online-norm",
        "init_qmc": "--init-qmc",
        "recurrent_rank": "--recurrent-rank",
        "sentinel_base_width": "sentinel preprocessing",
    }
    invalid.extend(option for name, option in disabled.items() if args.get(name, 0))
    for name, option, default in (
        ("research_features", "--research-features", "none"),
        ("privileged_mode", "--privileged-mode", "none"),
        ("prop_input", "--prop-input", ""),
        ("init", "--init", ""),
        ("rnn_init", "--rnn-init", "default"),
        ("qat", "--qat", "none"),
        ("activation_backend", "--activation-backend", "none"),
        ("sentinel_indicators", "sentinel preprocessing", "none"),
    ):
        if args.get(name, default) != default:
            invalid.append(option)
    if args.get("sentinel_values", "unchanged") != "unchanged":
        invalid.append("sentinel preprocessing")
    if args.get("distill_family_weight", 0.0):
        invalid.append("--distill-family-weight (family-head integration is separate)")
    if any(
        args.get(name) is not None for name in ("encoder_width", "recurrent_width", "readout_width")
    ):
        invalid.append("independent encoder/recurrent/readout widths")
    if args.get("head_layers", 1) != 1:
        invalid.append("--head-layers")
    return None if not invalid else "tiedlead profile rejects " + ", ".join(dict.fromkeys(invalid))


def append_causal_i1_lead(raw: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Return each row's i1 block at the most recent i0-price event.

    The first row is an initial event. The operation is sequence-local and
    causal; callers build the complete sequence before chunked model forward.
    """
    if raw.ndim != 3 or raw.shape[-1] != N_FEATURES:
        raise ValueError(f"tiedlead requires (batch,time,{N_FEATURES}) raw inputs")
    if event.ndim != 3 or event.shape[:2] != raw.shape[:2] or event.shape[-1] != N_EVENT_FEATURES:
        raise ValueError("tiedlead event features do not match raw sequence shape")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("tiedlead requires a nonempty batch of nonempty sequences")
    if (I1.bid_px.start, I1.trade_vol.stop) != (_I1.start, _I1.stop):
        raise RuntimeError("canonical i1 feature block no longer matches the tiedlead contract")
    event_flag = event[..., _EVENT_I0PX] > 0.5
    event_flag = event_flag.clone()
    event_flag[:, 0] = True
    rows = torch.arange(raw.shape[1], device=raw.device).expand(raw.shape[0], -1)
    last_event = torch.cummax(torch.where(event_flag, rows, torch.zeros_like(rows)), dim=1).values
    i1 = raw[..., _I1]
    return torch.gather(i1, 1, last_event[..., None].expand(-1, -1, TIEDLEAD_LEAD_WIDTH))


def build_tiedlead_inputs(raw: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Append the causal lead after the canonical 112 raw and 99 event channels."""
    base = torch.cat((raw, event), dim=-1)
    if base.shape[-1] != TIEDLEAD_BASE_WIDTH:
        raise ValueError(f"tiedlead base input order must have width {TIEDLEAD_BASE_WIDTH}")
    result = torch.cat((base, append_causal_i1_lead(raw, event).to(base.dtype)), dim=-1)
    if result.shape[-1] != TIEDLEAD_INPUT_WIDTH:
        raise RuntimeError("tiedlead input assembly produced an unexpected width")
    return result


class TiedLeadRNNModel(RNNModel):
    """Canonical GRU with one shared raw-scale instrument/lead encoder."""

    def __init__(self, n_in: int, n_out: int) -> None:
        """Build the fixed 263-input, 256-hidden, two-layer GRU profile."""
        if n_in != TIEDLEAD_INPUT_WIDTH or n_out < 2:
            raise ValueError("tiedlead requires 263 inputs and at least two outputs")
        super().__init__(n_in, TIEDLEAD_HIDDEN, 2, 0.0, n_out=n_out, cell="gru", enc="linear")
        self.tied = nn.Sequential(
            nn.Linear(104, TIED_ENCODER_HIDDEN),
            nn.GELU(),
            nn.Linear(TIED_ENCODER_HIDDEN, TIED_ENCODER_WIDTH),
            nn.GELU(),
        )
        self.tied_out = nn.Linear(2 * TIED_ENCODER_WIDTH, TIEDLEAD_HIDDEN)
        self.lead_out = nn.Linear(TIED_ENCODER_WIDTH, TIEDLEAD_HIDDEN)
        nn.init.zeros_(self.tied_out.weight)
        nn.init.zeros_(self.tied_out.bias)
        nn.init.zeros_(self.lead_out.weight)
        nn.init.zeros_(self.lead_out.bias)

    def encode(self, xn: torch.Tensor) -> torch.Tensor:
        """Add tied branch residuals before GELU; deltas reset at every call."""
        if xn.ndim != 3 or xn.shape[-1] != TIEDLEAD_INPUT_WIDTH or xn.shape[1] == 0:
            raise ValueError("tiedlead encoder expects a nonempty (batch,time,263) tensor")
        raw = xn * self.std + self.mean
        delta = torch.cat(
            (
                torch.zeros_like(raw[:, :1, :104]),
                raw[:, 1:, :104] - raw[:, :-1, :104],
            ),
            dim=1,
        )
        i0 = torch.cat((raw[..., _I0], delta[..., _I0]), dim=-1)
        i1 = torch.cat((raw[..., _I1], delta[..., _I1]), dim=-1)
        lead = raw[..., TIEDLEAD_BASE_WIDTH:TIEDLEAD_INPUT_WIDTH]
        lead_input = torch.cat((raw[..., _I1], raw[..., _I1] - lead), dim=-1)
        projected = self.inp(xn)
        projected = projected + self.tied_out(torch.cat((self.tied(i0), self.tied(i1)), dim=-1))
        projected = projected + self.lead_out(self.tied(lead_input))
        return F.gelu(projected)
