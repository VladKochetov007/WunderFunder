import torch
from torch import nn
from wnn.nn import RNNModel


class CompressModel(RNNModel):
    def __init__(self, n_in, hidden, layers, dropout, n_out, head_hidden=0):
        super().__init__(n_in, hidden, layers, dropout, n_out)
        self.family_head = nn.Linear(hidden, 8)
        width = head_hidden or hidden
        if width != hidden:
            # Match the ordinary h256 model's CPU RNG consumption exactly.
            # These replacement tensors are subsequently loaded from the
            # function-preserving initializer, not freshly fitted from scratch.
            with torch.random.fork_rng(devices=[]):
                self.head = nn.Sequential(nn.Linear(2*hidden, width), nn.GELU(), nn.Linear(width, n_out))
                self.family_head = nn.Linear(width, 8)

    def forward(self, x, h=None):
        z = self.encode((x-self.mean)/self.std)
        o, h = self.gru(z, h)
        f = self.head[1](self.head[0](torch.cat([o, z], -1)))
        self.family_output = self.family_head(f)
        return self.head[2](f), h
