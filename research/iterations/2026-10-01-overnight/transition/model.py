"""Isolated head extension; unchanged backbone and original checkpoint parameter names."""
from __future__ import annotations
import torch
from torch import nn
from wnn.nn import RNNModel
from features import WIDTH


class FeatureHeadModel(RNNModel):
    def __init__(self, n_in, hidden=128, layers=2, dropout=0.0, n_out=23):
        super().__init__(n_in, hidden, layers, dropout, n_out)
        # Paired baseline RNG trajectory must not change from constructing an extra layer.
        with torch.random.fork_rng(devices=[]):
            self.head_extra = nn.Linear(WIDTH, hidden, bias=False)
            nn.init.zeros_(self.head_extra.weight)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incoming = dict(state_dict)
        if "head_extra.weight" not in incoming:
            incoming["head_extra.weight"] = torch.zeros_like(self.head_extra.weight)
        return super().load_state_dict(incoming, strict=strict, assign=assign)

    def forward(self, x, h=None):
        width = self.mean.numel()
        if x.shape[-1] != width + WIDTH:
            raise ValueError(f"expected {width}+{WIDTH} columns, got {x.shape[-1]}")
        z = self.encode((x[..., :width]-self.mean)/self.std)
        o, h = self.gru(z, h)
        activation = self.head[0](torch.cat([o, z], dim=-1))
        activation = activation + self.head_extra(x[..., width:])
        return self.head[2](self.head[1](activation)), h
