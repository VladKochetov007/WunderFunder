"""Causal segment-recurrent Transformer (Transformer-XL style) for 20k-row streams (research, --arch tfm).

Rows are split into blocks of S rows. At every layer, a block attends causally to itself and to the previous
block's inputs to that layer (the memory), so the receptive field per layer is <= 2S rows and grows with depth;
the last block of a chunk is carried to the next chunk as state (detached between chunks), so the model is causal
and chunk-invariant. Positions use ALiBi (per-head linear distance penalty), so no absolute position leaks the
sequence step. Pre-LN blocks: attention + GELU MLP. The head sees the final hidden state and the input projection.

Intended as a high-capacity TEACHER (several M parameters); per-row streaming cost is ~20-30x a hidden-128 GRU.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def alibi_slopes(heads: int) -> torch.Tensor:
    start = 2 ** (-8.0 / heads)
    return torch.tensor([start ** (i + 1) for i in range(heads)])


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.heads, self.dropout = heads, dropout



class XLTransformer(nn.Module):
    def __init__(self, n_in: int, d: int = 256, layers: int = 6, heads: int = 8, ff: int = 1024, block: int = 250,
                 n_out: int = 2, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.d, self.s, self.nl = d, block, layers
        self.inp = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_out))
        # ALiBi + causal mask over [memory S | block S]: query i (block position) at absolute S + i
        s = block
        qpos = torch.arange(s)[:, None] + s
        kpos = torch.arange(2 * s)[None, :]
        dist = (qpos - kpos).float()
        bias = -alibi_slopes(heads)[:, None, None] * dist[None]
        bias = bias.masked_fill((kpos > qpos)[None], float("-inf"))
        self.register_buffer("bias", bias)

    def init_state(self, b, device, dtype):
        return tuple(torch.zeros(b, self.s, self.d, device=device, dtype=dtype) for _ in range(self.nl)), \
            torch.zeros(b, dtype=torch.bool, device=device)

    def forward(self, x, state=None):
        b, t, _ = x.shape
        s = self.s
        if t % s:
            raise ValueError(f"chunk length {t} must be a multiple of the block size {s}")
        z = F.gelu(self.inp((x - self.mean) / self.std))
        nb = t // s
        if state is None:
            state = self.init_state(b, x.device, z.dtype)
        mems, _ = state
        # at the start of a sequence the memory is all zeros (no information); training and streaming inference
        # both start that way, so one shared (H, S, 2S) mask suffices and the per-sample mask is not materialised
        full_bias = self.bias
        h = z.reshape(b, nb, s, self.d)
        new_mems = []
        for layer, blk in enumerate(self.blocks):
            prev = torch.cat([mems[layer].to(h.dtype)[:, None], h[:, :-1]], dim=1)  # previous block's layer input
            new_mems.append(h[:, -1].detach())
            h = self._block(blk, h.reshape(b * nb, s, self.d), prev.reshape(b * nb, s, self.d), full_bias)
            h = h.reshape(b, nb, s, self.d)
        y = self.ln(h.reshape(b, t, self.d))
        out = self.head(torch.cat([y, z], dim=-1))
        return out, (tuple(new_mems), torch.ones(b, dtype=torch.bool, device=x.device))

    @staticmethod
    def _block(blk, x, mem, bias):
        # per-sample bias: SDPA broadcasts (N, H, S, 2S)
        n, s, d = x.shape
        hh = blk.heads
        kv_in = blk.ln1(torch.cat([mem, x], dim=1))
        qkv = blk.qkv(kv_in)
        q = qkv[:, s:, :d].reshape(n, s, hh, d // hh).transpose(1, 2)
        k = qkv[..., d:2 * d].reshape(n, 2 * s, hh, d // hh).transpose(1, 2)
        v = qkv[..., 2 * d:].reshape(n, 2 * s, hh, d // hh).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                             dropout_p=blk.dropout if blk.training else 0.0)
        x = x + blk.proj(att.transpose(1, 2).reshape(n, s, d))
        return x + blk.mlp(blk.ln2(x))
