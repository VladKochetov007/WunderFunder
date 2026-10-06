import torch
from torch import nn
from wnn.nn import RNNModel


class CompressModel(RNNModel):
    """Causal class-conditioned residual using the existing readout features."""

    def __init__(self, n_in, hidden, layers, dropout, n_out, class_mode="none"):
        super().__init__(n_in, hidden, layers, dropout, n_out)
        assert class_mode in ("none", "uniform", "soft")
        self.family_head = nn.Linear(hidden, 8)
        self.class_mode = class_mode
        if class_mode != "none":
            assert n_out == 23, "Class logits must occupy columns 17:21"
            self.class_res = nn.Linear(hidden, 8)
            nn.init.zeros_(self.class_res.weight)
            nn.init.zeros_(self.class_res.bias)

    def forward(self, x, h=None):
        z = self.encode((x - self.mean) / self.std)
        o, h = self.gru(z, h)
        f = self.head[1](self.head[0](torch.cat([o, z], -1)))
        self.family_output = self.family_head(f)
        out = self.head[2](f)
        if self.class_mode != "none":
            prob = torch.softmax(out[..., 17:21].detach().float(), -1)
            if self.class_mode == "uniform":
                prob = torch.full_like(prob, 0.25)
            # Equal row-wise residual Jacobian norm in both arms. Otherwise a
            # confident gate has four times the uniform head's effective rate.
            prob = (prob / prob.square().sum(-1, keepdim=True).sqrt()).to(out.dtype)
            residual = self.class_res(f).reshape(*f.shape[:-1], 4, 2)
            delta = (residual * prob.unsqueeze(-1)).sum(-2)
            out = torch.cat((out[..., :2] + delta, out[..., 2:]), -1)
        return out, h
