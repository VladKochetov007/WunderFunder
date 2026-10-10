"""Causal scoring-propensity network P(is_scored | rows <= t)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PropensityNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, h=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.gru(z, h)
        return self.out(torch.cat([o, z], -1)).squeeze(-1), h


@torch.no_grad()
def propensity(model, inputs, chunk: int = 4000):
    """Probabilities (B, T) for already-built inputs, streaming over chunks."""
    h, outs = None, []
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast(inputs.device.type, dtype=torch.float16, enabled=inputs.is_cuda):
            logit, h = model(inputs[:, s : s + chunk], h)
        outs.append(torch.sigmoid(logit.float()))
    return torch.cat(outs, 1)


def load_propensity(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PropensityNet(ck["n_in"], ck["hidden"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()
