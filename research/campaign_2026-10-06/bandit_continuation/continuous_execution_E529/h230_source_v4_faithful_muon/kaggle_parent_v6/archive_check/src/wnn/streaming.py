"""One-row-per-call graph: event features and GRU with explicit state tensors.

Exported to ONNX so the submission runs a single ``session.run`` per row, like
the organizer baseline. State: previous row, step counter, last-event steps for
(i0 price, i0 trade, i1 price), held i0 price deltas, GRU hidden, first-row flag.
Must equal ``wnn.features.event_features`` + ``wnn.nn.RNNModel`` on a sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from wnn.data import I0, I1

_CAP = 10_000.0


def initial_state(model):
    layers = model.gru.num_layers
    hidden = model.gru.hidden_size
    return [
        torch.zeros(1, 112),  # prev row
        torch.zeros(1),  # step
        torch.zeros(1, 3),  # last event step: i0 px, i0 trade, i1 px
        torch.zeros(1, 22),  # held i0 price deltas at last price event
        torch.zeros(layers, 1, hidden),  # GRU hidden
        torch.ones(1),  # first row flag
    ]


class StreamingGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, row, prev, step, last, held, h, first):
        prev = torch.where(first[:, None] > 0.5, row, prev)
        d = row - prev
        ne = (prev != row).to(row.dtype)
        f_px = ne[:, I0.bid_px.start] * ne[:, I0.ask_px.start]
        f_bv = ne[:, I0.bid_vol].amax(-1)
        f_av = ne[:, I0.ask_vol].amax(-1)
        f_tp = ne[:, I0.trade_px].amax(-1)
        f_tv = ne[:, I0.trade_vol].amax(-1)
        f_i1p = torch.clamp(ne[:, I1.bid_px.start] + ne[:, I1.ask_px.start], max=1.0)
        f_i1v = ne[:, I1.bid_vol.start : I1.ask_vol.stop].amax(-1)
        f_i1t = ne[:, I1.trade_vol].amax(-1)
        flags = torch.stack([f_px, f_bv, f_av, f_tp, f_tv, f_i1p, f_i1v, f_i1t], -1)
        events = torch.stack([f_px, f_tv, f_i1p], -1)
        last = torch.where(events > 0.5, step[:, None], last)
        since = torch.log1p(torch.clamp(step[:, None] - last, max=_CAP))
        px = slice(I0.bid_px.start, I0.ask_px.stop)
        held = torch.where(f_px[:, None] > 0.5, d[:, px], held)
        feats = torch.cat(
            [d[:, px], d[:, I0.bid_vol.start : I0.ask_vol.stop], d[:, I1.bid_px.start : I1.ask_px.stop], flags, since, held],
            -1,
        )
        x = torch.cat([row, feats], -1)[:, None, :]
        y, h = self.model(x, h)
        return y[:, 0, :2], row, step + 1, last, held, h, torch.zeros_like(first)
