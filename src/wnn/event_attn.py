"""GRU + causal attention over the last K i0 price-event tokens (research model, --arch attn).

Why: the GRU compresses history into a fixed state (effective memory ~1000 rows). Class-3 outcomes are
decided by the next book events, and the recent sequence of book states is exactly what a fixed state
blurs. Here every i0 price event writes a token (row projection + log gap since the previous event) into a
ring of the last K tokens; every row queries that ring with its GRU output (multi-head dot-product
attention) and the head sees [GRU output | row projection | attention context].

Causality: a row attends only to tokens of events at rows <= t (its own row included when it is an event).
Chunked training carries (GRU state, last K tokens, token validity) across chunks, like the dual-clock model.
Inference cost per row: one query projection (H x D), K dot products and K value mixes (~2*K*D), plus one
token projection per event.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EventAttnModel(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128, layers: int = 2, n_out: int = 2, k: int = 32,
                 d: int = 64, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.k, self.d, self.heads = k, d, heads
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.tok = nn.Linear(hidden + 1, 2 * d)  # -> key, value of an event token
        self.null_kv = nn.Parameter(torch.zeros(2 * d))  # always-present null token (empty ring early on)
        self.q = nn.Linear(hidden, d)
        self.out_ctx = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(2 * hidden + d, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def init_state(self, b: int, device, dtype):
        ring = torch.zeros(b, self.k, 2 * self.d, device=device, dtype=dtype)
        valid = torch.zeros(b, self.k, dtype=torch.bool, device=device)
        gap = torch.zeros(b, device=device, dtype=torch.long)  # rows since the last event at chunk start
        return None, ring, valid, gap

    def forward(self, x, events, state=None):
        """x (B, T, n_in) raw model inputs, events (B, T) bool; state from init_state / a previous chunk."""
        b, t, _ = x.shape
        dev = x.device
        z = F.gelu(self.inp((x - self.mean) / self.std))
        if state is None:
            state = self.init_state(b, dev, z.dtype)
        h, ring, valid, gap_in = state
        o, h = self.gru(z, h)

        # gap (rows since previous event) at each row, carried across chunks
        rows = torch.arange(t, device=dev).expand(b, t)
        neg = torch.full_like(rows, -1)
        last = torch.cummax(torch.where(events, rows, neg), dim=1).values
        prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)
        gap = torch.where(prev >= 0, rows - prev, rows + 1 + gap_in[:, None])
        kv_all = self.tok(torch.cat([z, torch.log1p(gap.to(z.dtype))[..., None]], dim=-1))  # (B, T, 2D)

        # compact this chunk's event tokens after the carried ring: tokens (B, K + T, 2D)
        order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)
        ev_kv = torch.gather(kv_all, 1, order[..., None].expand(b, t, 2 * self.d))
        n_ev = events.sum(1)
        ev_valid = torch.arange(t, device=dev)[None] < n_ev[:, None]
        tokens = torch.cat([ring.to(ev_kv.dtype), ev_kv], dim=1)
        tvalid = torch.cat([valid, ev_valid], dim=1)
        # row r sees the K tokens ending at its latest event: positions c_r .. c_r + K - 1, c_r = #events <= r
        c = torch.cumsum(events.long(), dim=1)
        idx = c[..., None] + torch.arange(self.k, device=dev)  # (B, T, K) into tokens
        win = tokens.gather(1, idx.reshape(b, -1, 1).expand(b, t * self.k, 2 * self.d)).reshape(b, t, self.k, 2 * self.d)
        wvalid = tvalid.gather(1, idx.reshape(b, -1)).reshape(b, t, self.k)
        null = self.null_kv.to(win.dtype).expand(b, t, 1, 2 * self.d)
        win = torch.cat([null, win], dim=2)
        wvalid = torch.cat([torch.ones_like(wvalid[..., :1]), wvalid], dim=2)
        key, val = win.split(self.d, dim=-1)
        hd = self.d // self.heads
        q = self.q(o).reshape(b, t, self.heads, hd)
        key = key.reshape(b, t, self.k + 1, self.heads, hd)
        val = val.reshape(b, t, self.k + 1, self.heads, hd)
        score = torch.einsum("btnh,btknh->btnk", q.float(), key.float()) / math.sqrt(hd)
        score = score.masked_fill(~wvalid[:, :, None, :], float("-inf"))
        att = torch.softmax(score, dim=-1)
        ctx = torch.einsum("btnk,btknh->btnh", att, val.float()).reshape(b, t, self.d).to(o.dtype)
        ctx = self.out_ctx(ctx)
        out = self.head(torch.cat([o, z, ctx], dim=-1))

        # new carried ring: the last K tokens after this chunk
        end = n_ev + self.k  # exclusive end index into tokens
        ridx = (end[:, None] - self.k + torch.arange(self.k, device=dev)[None])
        new_ring = tokens.gather(1, ridx[..., None].expand(b, self.k, 2 * self.d)).detach()
        new_valid = tvalid.gather(1, ridx)
        last_row = last[:, -1]
        gap_out = torch.where(last_row >= 0, t - 1 - last_row, gap_in + t)
        return out, (h, new_ring, new_valid, gap_out)
