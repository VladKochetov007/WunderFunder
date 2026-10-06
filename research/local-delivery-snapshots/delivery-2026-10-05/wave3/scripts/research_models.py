"""Wave3 causal, frozen-backbone adapters. No target-derived deployment gates."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from wnn.nn import RNNModel
from wnn.outcomes import N_SP_AUX


class AdapterModel(RNNModel):
    def __init__(self, n_in, hidden, layers, dropout, n_out, mode):
        super().__init__(n_in, hidden, layers, dropout, n_out)
        self.mode = mode
        self.adapter = nn.Sequential(nn.Linear(2 * hidden, 16), nn.GELU(), nn.Linear(16, 4))
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        self.last_penalty = None
        self.last_moments = None

    def forward(self, x, h=None):
        z = self.encode((x-self.mean)/self.std)
        o, h = self.gru(z, h)
        feat = torch.cat([o, z], -1)
        base = self.head(feat)
        raw = self.adapter(feat.detach()).float()
        if self.mode in ('quiet', 'shared'):
            delta = .5 * (raw[..., :2] + raw[..., 2:])
            gate = 1.
            if self.mode == 'quiet':
                # Class probabilities are predicted from causal features, never labels.
                start = 2 + N_SP_AUX + 3
                gate = base[..., start:start+4].float().softmax(-1)[..., 3:4].detach()
            delta = gate * delta
            pred = base[..., :2].float() + delta
            self.last_penalty = .05 * delta.square().mean()
        else:
            den = .05 + F.softplus(raw[..., 2:] + math.log(math.expm1(.35)))
            num = .4 * base[..., :2].float().detach() + raw[..., :2]
            self.last_moments = (num, den)
            if self.mode == 'ratio':
                pred = .75 * base[..., :2].float() + .25 * num / den
            else:
                pred = base[..., :2].float() + .5 * (raw[..., :2] + raw[..., 2:])
            self.last_penalty = .01 * (pred-base[..., :2].float()).square().mean()
        return torch.cat([pred, base[..., 2:].float()], -1), h


def adapter_loss(model, y, scored, valid):
    loss = model.last_penalty
    if model.mode == 'ratio':
        yy = y.float().clamp(-2, 2)
        weight = yy.abs() * scored[..., None].float()
        num, den = model.last_moments
        per = (num-weight*yy).square() + (den-weight).square()
        v = valid[..., None].float()
        loss = loss + (per*v).sum() / (2*v.sum()).clamp_min(1.)
    return loss
