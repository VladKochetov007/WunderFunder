"""Frozen-trunk CTXMEM head with exact original parameter names and small CPU memory."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from wnn.nn import RNNModel
from memory import RANK, WIDTH, event_histories, fixed_projection, selected_features


class ContextMemoryModel(RNNModel):
    def __init__(self, n_in=211, hidden=128, layers=2, dropout=0.0, n_out=23, arm='memory'):
        if (n_in, hidden, layers, dropout) != (211, 128, 2, 0.0) or arm not in ('memory', 'current'):
            raise ValueError('CTXMEM01 requires fixed211/128/2/0 and declared arm')
        super().__init__(n_in, hidden, layers, dropout, n_out)
        self.arm = arm
        self.inp.requires_grad_(False)
        self.gru.requires_grad_(False)
        self.register_buffer('ctx_projection', torch.from_numpy(fixed_projection()))
        # Preserve the baseline initialization and subsequent training-shuffle RNG.
        with torch.random.fork_rng(devices=[]):
            self.head_extra = nn.Linear(WIDTH, hidden, bias=False)
            nn.init.zeros_(self.head_extra.weight)
        torch.backends.cuda.matmul.allow_tf32 = False

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incoming = dict(state_dict)
        if 'head_extra.weight' not in incoming:
            incoming['head_extra.weight'] = torch.zeros_like(self.head_extra.weight)
        if 'ctx_projection' not in incoming:
            incoming['ctx_projection'] = self.ctx_projection.detach().clone()
        if not torch.equal(incoming['ctx_projection'].cpu(), torch.from_numpy(fixed_projection())):
            raise ValueError('Checkpoint changed the fixed projection')
        return super().load_state_dict(incoming, strict=strict, assign=assign)

    def forward(self, x, state=None):
        if x.ndim != 3 or x.shape[-1] != 211:
            raise ValueError('Expected BxTx211 built causal inputs')
        if x.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Fixed projection requires TF32 disabled')
        if state is None:
            recurrent, memory = None, None
        else:
            recurrent = state[0]
            memory = tuple(t.detach().cpu().numpy() for t in state[1:])
        if x.shape[1] == 0:
            return x.new_empty((x.shape[0], 0, self.head[2].out_features)), state
        # Frozen backbone has no gradient path through q or the CPU histories.
        # Original head remains differentiable, as does the extra linear map.
        with torch.no_grad():
            z = self.encode((x - self.mean) / self.std)
            o, recurrent = self.gru(z, recurrent)
            with torch.autocast(device_type=x.device.type, enabled=False):
                q = torch.nn.functional.linear(o.float(), self.ctx_projection.float())
            q_cpu = q.detach().cpu().numpy()
            events = (x[..., 178] > .5).detach().cpu().numpy()
            histories, next_memory = event_histories(q_cpu, events, memory)
            extra_cpu = selected_features(q_cpu, histories, self.arm)
            extra = torch.from_numpy(np.ascontiguousarray(extra_cpu)).to(device=x.device)
        activation = self.head[0](torch.cat((o, z), dim=-1))
        activation = activation + self.head_extra(extra)
        out = self.head[2](self.head[1](activation))
        carried = (recurrent, *(torch.from_numpy(value) for value in next_memory))
        return out, carried
