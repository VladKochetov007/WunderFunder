"""Learned interval integration -> event GRU; no data or checkpoint I/O.

Current event closes the interval, after contributing its fast-row projection.
The instantaneous control has the same modules and carries the same integrator.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from wnn.nn import RNNModel

EVENT_INDEX = 178  # full-prefix211 input, raw112 + event flag66; before normalization


def interval_means(values, events, incoming_sum, incoming_count):
    """Vectorized inclusive interval sums. Carries are post-event-reset values.

    Float16/bfloat16 projections accumulate in FP32; FP32/FP64 stay as supplied.
    No state mutation or internal detach. Prefix subtraction is numerically
    compared with the literal streaming recurrence in generated checks.
    """
    b, t, d = values.shape
    assert events.shape == (b, t) and events.dtype == torch.bool
    assert incoming_sum.shape == (b, d) and incoming_count.shape == (b,)
    if t == 0:
        return values.to(incoming_sum.dtype), incoming_sum, incoming_count
    v = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    assert incoming_sum.dtype == v.dtype and incoming_count.dtype == torch.long
    rows = torch.arange(t, device=v.device).expand(b, t)
    no_event = torch.full_like(rows, -1)
    latest = torch.cummax(torch.where(events, rows, no_event), dim=1).values
    previous = torch.cat([no_event[:, :1], latest[:, :-1]], dim=1)
    prefix = torch.cat([torch.zeros_like(v[:, :1]), torch.cumsum(v, dim=1)], dim=1)
    origin = torch.gather(prefix, 1, (previous+1)[..., None].expand(b, t, d))
    sums = prefix[:, 1:]-origin + torch.where((previous < 0)[..., None], incoming_sum[:, None], 0.)
    counts = rows-previous + torch.where(previous < 0, incoming_count[:, None], 0)
    means = sums/counts[..., None].to(v.dtype)
    final_sum = torch.where(events[:, -1, None], torch.zeros_like(sums[:, -1]), sums[:, -1])
    final_count = torch.where(events[:, -1], torch.zeros_like(counts[:, -1]), counts[:, -1])
    return means, final_sum, final_count


class IntervalMemoryModel(RNNModel):
    """Fixed211/fast256x1/project32/slow64, readout23 by default.

    State: (row_GRU_h, event_GRU_h[B,64], interval_sum[B,32], count[B]).
    All are tensors, compatible with train_aux.detach_state's tuple recursion.
    """
    def __init__(self, arm: str, n_out: int = 23):
        if arm not in ('accumulated', 'instantaneous'):
            raise ValueError(arm)
        super().__init__(211, 256, 1, 0., n_out=n_out)
        self.arm = arm
        self.interval_proj = nn.Linear(256, 32)
        self.event_gru = nn.GRU(33, 64, batch_first=True)
        # Preserve the original GEMM shape/arithmetic and add a zero residual.
        # Widening head.0 itself can change baseline rounding even with zero pads.
        self.extra_weight = nn.Parameter(torch.zeros(256, 32+64))

    def load_plain_baseline(self, state):
        """Explicit strict transplantation of every ordinary GRU tensor.

        No partial load: source keys/shapes/dtypes must exactly equal the full
        ordinary model subset; new modules retain their paired initialization.
        """
        own = self.state_dict()
        ordinary = {k for k in own if not k.startswith(('interval_proj.', 'event_gru.')) and k != 'extra_weight'}
        if set(state) != ordinary:
            raise ValueError(('plain checkpoint keys', sorted(set(state)^ordinary)))
        merged = dict(own)
        for k in ordinary:
            if state[k].shape != own[k].shape or state[k].dtype != own[k].dtype:
                raise ValueError(('plain checkpoint shape/dtype', k))
            if not bool(torch.isfinite(state[k]).all()):
                raise ValueError(('nonfinite plain state', k))
            merged[k] = state[k]
        if not bool((state['std'] > 0).all()):
            raise ValueError('nonpositive normalization scale')
        merged['extra_weight'] = torch.zeros_like(own['extra_weight'])
        self.load_state_dict(merged, strict=True)

    def freeze_trunk(self):
        """H2 trains original head + extra_weight; all state generators frozen."""
        for module in (self.inp, self.gru, self.interval_proj, self.event_gru):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, x, state=None):
        if x.shape[-1] != 211:
            raise ValueError('Requires the unchanged full-prefix211 input')
        return self.forward_events(x, x[..., EVENT_INDEX] > .5, state)

    def forward_events(self, x, events, state=None):
        """Explicit event interface for generated tests; production uses forward."""
        b, t, _ = x.shape
        if t == 0:
            if state is None:
                raise ValueError('Empty first chunk has no established carry')
            return x.new_empty((b, 0, self.head[2].out_features)), state
        if state is None:
            # Row0 has no preceding observation; never close an interval there.
            events = torch.cat([torch.zeros_like(events[:, :1]), events[:, 1:]], dim=1)
        z = self.encode((x-self.mean)/self.std)
        if state is None:
            row_state = None
            slow = z.new_zeros((b, 64))
            dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
            total = torch.zeros((b, 32), device=z.device, dtype=dtype)
            count = torch.zeros(b, device=z.device, dtype=torch.long)
        else:
            row_state, slow, total, count = state
        h, row_state = self.gru(z, row_state)
        v = torch.tanh(self.interval_proj(h))
        mean, final_total, final_count = interval_means(v, events, total, count)
        u = mean.to(z.dtype) if self.arm == 'accumulated' else v

        # Same inclusive count as the integrator; the current event closes it.
        rows = torch.arange(t, device=x.device).expand(b, t)
        negative = torch.full_like(rows, -1)
        last = torch.cummax(torch.where(events, rows, negative), dim=1).values
        previous = torch.cat([negative[:, :1], last[:, :-1]], dim=1)
        lengths = rows-previous + torch.where(previous < 0, count[:, None], 0)
        slow_input = torch.cat([u, torch.log1p(lengths.to(z.dtype))[..., None]], dim=-1)
        counts = events.sum(1)
        n_upto = torch.cumsum(events.long(), dim=1)
        held = slow[:, None, :].expand(b, t, 64)
        max_count = int(counts.max())  # one host sync per chunk, like old dual clock
        if max_count:
            order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)[:, :max_count]
            packed = torch.gather(slow_input, 1, order[..., None].expand(b, max_count, 33))
            event_out, _ = self.event_gru(packed, slow[None].contiguous())
            event_out = event_out.to(z.dtype)
            index = (n_upto-1).clamp_min(0)
            selected = torch.gather(event_out, 1, index[..., None].expand(b, t, 64))
            held = torch.where((n_upto > 0)[..., None], selected, held)
            tail = event_out[torch.arange(b, device=x.device), (counts-1).clamp_min(0)]
            slow = torch.where((counts > 0)[:, None], tail, slow)
        base = self.head[0](torch.cat([h, z], dim=-1))
        addition = F.linear(torch.cat([u, held], dim=-1), self.extra_weight)
        out = self.head[2](self.head[1](base+addition))
        return out, (row_state, slow, final_total, final_count)
