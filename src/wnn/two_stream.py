"""Two-stream recurrent model (research, --arch twostream): separate GRUs for instrument i0 and i1.

Each stream gets its own instrument's raw columns, the shared a0..a7 clocks and its own causal event features;
the head sees both GRU outputs and both projections. Motivation: in the single GRU the i1 dynamics compete with
the dominant i0 inputs for the same state; a dedicated i1 memory can track i1's own event process (lead-lag at
event level) without being overwritten. Input layout: [raw 112 | event 99] (wnn.features.FEATURE_NAMES).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

RAW, EV = 112, 112  # event block starts at column 112 of the model input


def _stream_columns():
    a = list(range(104, 112))
    i0 = list(range(0, 52)) + a + [EV + j for j in range(0, 44)] + [EV + 66 + j for j in range(5)] \
        + [EV + 74, EV + 75] + [EV + 77 + j for j in range(22)]
    i1 = list(range(52, 104)) + a + [EV + 44 + j for j in range(22)] + [EV + 71 + j for j in range(3)] + [EV + 76]
    return i0, i1


class TwoStreamModel(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128, layers: int = 2, n_out: int = 2, hidden1: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        i0, i1 = _stream_columns()
        assert max(i0 + i1) < n_in
        self.register_buffer("idx0", torch.tensor(i0))
        self.register_buffer("idx1", torch.tensor(i1))
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        h1 = hidden1 or hidden
        self.inp0, self.inp1 = nn.Linear(len(i0), hidden), nn.Linear(len(i1), h1)
        self.gru0 = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.gru1 = nn.GRU(h1, h1, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden + 2 * h1, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x, h=None):
        xn = (x - self.mean) / self.std
        z0 = F.gelu(self.inp0(xn.index_select(-1, self.idx0)))
        z1 = F.gelu(self.inp1(xn.index_select(-1, self.idx1)))
        h0, h1 = (None, None) if h is None else h
        o0, h0 = self.gru0(z0, h0)
        o1, h1 = self.gru1(z1, h1)
        return self.head(torch.cat([o0, o1, z0, z1], dim=-1)), (h0, h1)
