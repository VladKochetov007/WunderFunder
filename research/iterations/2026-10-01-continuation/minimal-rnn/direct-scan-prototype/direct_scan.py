"""Isolated direct positive affine scan; same parameters and recurrence as reference.

Deliberately follows FP32 step coefficients a=1-sigmoid(k), b=sigmoid(k)*g(v).
No alternate coefficients, log fallback, clipping or tolerance-based rescue.
"""
import torch
from torch.nn import functional as F
from torch_reference import MinimalRNN, positive_g


def direct_scan(gate_logits, candidate_logits, initial=None, block_size=64):
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
        result = []
        for begin in range(0, k.shape[1], block_size):
            z = torch.sigmoid(k[:, begin:begin+block_size])
            a = 1-z
            b = z*positive_g(v[:, begin:begin+block_size])
            distance = 1
            while distance < a.shape[1]:
                next_b = b[:, distance:] + a[:, distance:]*b[:, :-distance]
                next_a = a[:, distance:]*a[:, :-distance]
                a = torch.cat((a[:, :distance], next_a), dim=1)
                b = torch.cat((b[:, :distance], next_b), dim=1)
                distance *= 2
            value = b+a*carry[:, None]
            result.append(value)
            carry = value[:, -1]
        return torch.cat(result, dim=1), carry


class DirectMinimalRNN(MinimalRNN):
    """Only forward scan differs; inherited step and state_dict remain unchanged."""
    def forward(self, x, h=None, block_size=64):
        z = F.gelu(self.inp((x-self.mean)/self.std))
        value, states = z, []
        for i, cell in enumerate(self.cells):
            value, last = direct_scan(cell.gate(value), cell.candidate(value), None if h is None else h[i], block_size)
            states.append(last)
        return self.head(torch.cat((value,z),-1)), torch.stack(states)
