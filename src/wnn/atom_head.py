"""RNNModel with a class-3 distributional ("atom") head and a metric-optimal decode (research, --atom-head).

Class-3 targets take few joint values (t0, t1) (64 atoms cover 99.9% of the metric weight, cache/class3_atoms.npy).
The atom head predicts a categorical distribution pi over the atoms (trained with |y|-weighted cross-entropy on
class-3 rows). Its decode is the per-target optimum of the |y|-weighted squared error:
    d_j = sum_k pi_k g_kj |g_kj| / sum_k pi_k |g_kj|
The output mixes the regression head and the decode: y_hat = reg + lambda * p3 * (d - reg), with p3 the model's own
class-3 probability (aux class logits) and lambda = sigmoid(learned scalar). The atom logits are appended at the
end of the output vector for the loss; prediction uses the first two columns only.
"""

from __future__ import annotations

import torch
from torch import nn

from wnn.nn import RNNModel


class AtomRNNModel(RNNModel):
    def __init__(self, n_in, hidden, layers, dropout, n_out, class_slice, atoms, **kw):
        super().__init__(n_in, hidden, layers, dropout, n_out=n_out, **kw)
        self.class_slice = class_slice
        self.register_buffer("atoms", torch.as_tensor(atoms, dtype=torch.float32))  # (K, 2)
        k = self.atoms.shape[0]
        self.atom_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.mix_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x, h=None):
        z = self.encode((x - self.mean) / self.std)
        o, h = self.gru(z, h)
        feats = torch.cat([o, z], dim=-1)
        out = self.head(feats).float()
        logits = self.atom_head(feats).float()
        pi = torch.softmax(logits, dim=-1)
        g = self.atoms  # (K, 2)
        ga = g.abs()
        dec = (pi @ (g * ga)) / (pi @ ga).clamp_min(1e-6)  # (B, T, 2)
        a, b = self.class_slice
        p3 = torch.softmax(out[..., a:b], dim=-1)[..., 3:4]
        lam = torch.sigmoid(self.mix_logit)
        reg = out[..., :2]
        mixed = reg + lam * p3 * (dec - reg)
        return torch.cat([mixed, out[..., 2:], logits], dim=-1), h
