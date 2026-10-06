"""Isolated positive-state minGRU prototype; no trainer/native integration.

Operator: https://arxiv.org/html/2410.01201v3 (section3 and appendixB).
The blocked associative log scan below is independently implemented, not copied
from the paper's cumulative-log-sum example. All scan arithmetic/carry is FP32.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def positive_g(x):
    return torch.where(x >= 0, x + .5, torch.sigmoid(x))


def log_g(x):
    # Keep both torch.where branches defined; g is nondifferentiable at exactly0.
    return torch.where(x >= 0, torch.log(torch.clamp_min(x, 0) + .5), -F.softplus(-x))


def blocked_scan(gate_logits, candidate_logits, initial=None, block_size=64):
    """[B,T,H] -> states, using associative affine composition in log space.

    Carry is the actual nonnegative state, never g(state). Zero initial state is
    handled linearly, so there is no log(0) and its derivative remains defined.
    Block carry is NOT detached; truncation is a future trainer responsibility.
    """
    if gate_logits.shape != candidate_logits.shape or gate_logits.ndim != 3:
        raise ValueError('matching [batch,time,hidden] logits required')
    if block_size <= 0 or gate_logits.shape[1] == 0:
        raise ValueError('positive block size and nonempty sequence required')
    with torch.autocast(device_type=gate_logits.device.type, enabled=False):
        k, v = gate_logits.float(), candidate_logits.float()
        carry = torch.zeros_like(k[:, 0]) if initial is None else initial.float()
        if carry.shape != k[:, 0].shape:
            raise ValueError('initial state must be [batch,hidden]')
        if not torch.isfinite(carry).all() or (carry < 0).any():
            raise ValueError('state must be finite and nonnegative')
        results = []
        for begin in range(0, k.shape[1], block_size):
            kb, vb = k[:, begin:begin+block_size], v[:, begin:begin+block_size]
            la = -F.softplus(kb)
            lb = -F.softplus(-kb) + log_g(vb)
            # (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
            # Prefix doubling is out-of-place to preserve autograd versions.
            distance = 1
            while distance < la.shape[1]:
                next_b = torch.logaddexp(lb[:, distance:], la[:, distance:] + lb[:, :-distance])
                next_a = la[:, distance:] + la[:, :-distance]
                la = torch.cat((la[:, :distance], next_a), dim=1)
                lb = torch.cat((lb[:, :distance], next_b), dim=1)
                distance *= 2
            value = torch.exp(lb) + torch.exp(la) * carry[:, None]
            results.append(value)
            carry = value[:, -1]
        return torch.cat(results, dim=1), carry


def streaming_states(gate_logits, candidate_logits, initial=None):
    """Literal recurrence oracle, also supports float64 for independent checks."""
    carry = torch.zeros_like(gate_logits[:, 0]) if initial is None else initial
    states = []
    for k, v in zip(gate_logits.unbind(1), candidate_logits.unbind(1)):
        z = torch.sigmoid(k)
        carry = (1 - z) * carry + z * positive_g(v)
        states.append(carry)
    return torch.stack(states, 1), carry


class MinGRUCell(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Linear(hidden, hidden)
        self.candidate = nn.Linear(hidden, hidden)

    def forward(self, x, initial=None, block_size=64):
        return blocked_scan(self.gate(x), self.candidate(x), initial, block_size)

    def step(self, x, initial=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            k, v = self.gate(x.float()), self.candidate(x.float())
            state = torch.zeros_like(k) if initial is None else initial.float()
            z = torch.sigmoid(k)
            return (1 - z) * state + z * positive_g(v)


class MinimalRNN(nn.Module):
    """Same normalization, input linear+GELU and [state,input] head as RNNModel.

    Two H256 minGRUs, no additional normalization/residual/convolution modules.
    CPU prototype supports FP32 parameters only; AMP/native are not qualified.
    """
    def __init__(self, n_in=211, hidden=256, layers=2, n_out=2):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.register_buffer('mean', torch.zeros(n_in))
        self.register_buffer('std', torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.cells = nn.ModuleList([MinGRUCell(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x, h=None, block_size=64):
        z = F.gelu(self.inp((x-self.mean)/self.std))
        value, states = z, []
        for i, cell in enumerate(self.cells):
            value, last = cell(value, None if h is None else h[i], block_size)
            states.append(last)
        return self.head(torch.cat((value, z), -1)), torch.stack(states)

    def step(self, x, h=None):
        z = F.gelu(self.inp((x-self.mean)/self.std))
        value, states = z, []
        for i, cell in enumerate(self.cells):
            value = cell.step(value, None if h is None else h[i])
            states.append(value)
        return self.head(torch.cat((value, z), -1)), torch.stack(states)
