"""v8 research architectures: clock-bank minGRU, diagonal state-space bank and dual-tower late fusion.

All three are causal, keep O(1) state per row, start from zero state at every sequence start and return
``(output, state)`` from ``forward(x, state=None)`` so the trainer's TBPTT loop (``forward_chunk`` ->
``detach_state``) treats them exactly like the GRU path. The input ``x`` is the raw model input of
``train_rnn.build_inputs``: columns 0-111 are the raw market columns, 112-210 the event features and any
further columns are appended research inputs. Instrument 0 occupies columns 0-51 and instrument 1 columns
52-103, each as 11 bid px, 11 ask px, 11 bid vol, 11 ask vol, 4 trade px, 4 trade vol; 104-111 are extras.

How they differ from the existing recurrent archs in ``wnn.nn``:

* ``twoclock`` / ``dual``: one dense GRU that rewrites its whole state on every row plus ONE small GRU that
  steps on a6 / i0-price events. ``clockbank`` has no dense every-row recurrence at all: the state is
  partitioned into banks and each bank is frozen (exact copy) unless its own column group changed on that row.
  ``dualtower`` has no every-row recurrence either: a book tower and a trade tower each step only on their own
  events and meet only in the readout (late fusion), with the gap since each event as an explicit input.
* ``dssm`` is a linear time-invariant diagonal recurrence (no tanh, no gate in the state path); the
  nonlinearity sits between layers only. Nothing in the repository resembles it (``structured`` uses
  gated multi-decay blocks, ``tcn`` is a convolution).

Serving: none of these are supported by the Rust engine (see the README). They are research/teacher models.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

N_RAW = 112
N_RAW_MARKET = 104  # columns 0-103 are the two instruments; 104-111 are extras
_EV = N_RAW  # offset of the 99 event features inside the model input
SCAN_BLOCK = 32


def _inst(offset: int) -> dict[str, tuple[int, int]]:
    return {
        "px": (offset, offset + 22), "vol": (offset + 22, offset + 44), "trade": (offset + 44, offset + 52),
        "bid_px": (offset, offset + 11), "ask_px": (offset + 11, offset + 22),
    }


def _cols(*ranges: tuple[int, int]) -> list[int]:
    return [c for a, b in ranges for c in range(a, b)]


I0, I1 = _inst(0), _inst(52)

# Clock groups. A bank ticks on rows where any column of its group differs from the previous row. The always-on
# base bank (appended automatically) ticks on every row.
CLOCK_LAYOUTS: dict[str, list[tuple[str, list[int]]]] = {
    "semantic": [
        ("i0_px", _cols(I0["px"])), ("i0_vol", _cols(I0["vol"])), ("i0_trade", _cols(I0["trade"])),
        ("i1_px", _cols(I1["px"])), ("i1_vol", _cols(I1["vol"])), ("i1_trade", _cols(I1["trade"])),
        ("extras", _cols((104, 112))),
    ],
    # proposal hint: a fast-clock block (columns 0-21, 44-47, 52-69) and the remainder in three slower groups
    "hint": [
        ("fast", _cols((0, 22), (44, 48), (52, 70))),
        ("i0_vol_trade_vol", _cols((22, 44), (48, 52))),
        ("i1_rest", _cols((70, 104))),
        ("extras", _cols((104, 112))),
    ],
}


def changed_columns(raw: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column "differs from the previous row" flags (B, T, C) and the last raw row (B, 1, C) to carry.

    Row 0 of a sequence compares with itself (no change), matching ``wnn.features``; chunked calls pass the
    last row of the previous chunk so they equal the full-sequence call.
    """
    raw = raw.float()
    prev = raw[:, :1] if prev is None else prev.to(raw.dtype)
    shifted = torch.cat([prev, raw[:, :-1]], dim=1)
    return raw != shifted, raw[:, -1:].clone()


def affine_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor, block: int = SCAN_BLOCK) -> torch.Tensor:
    """Exact h_t = a_t * h_{t-1} + b_t for a (B, T, C), b (B, T, C), h0 (B, C), computed in float32.

    Two-level scan: an elementwise loop over ``block`` rows inside every block in parallel, a loop over blocks
    that carries the incoming state, then the carried state is added back through the within-block cumulative
    product (all factors a in [0, 1], so nothing overflows). About T/block + block sequential steps.
    """
    a, b, h0 = a.float(), b.float(), h0.float()
    bsz, t, c = a.shape
    pad = (-t) % block
    if pad:
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, pad))
    nb = (t + pad) // block
    a, b = a.reshape(bsz, nb, block, c), b.reshape(bsz, nb, block, c)
    h, local = torch.zeros(bsz, nb, c, dtype=a.dtype, device=a.device), []
    for i in range(block):
        h = a[:, :, i] * h + b[:, :, i]
        local.append(h)
    local = torch.stack(local, dim=2)
    decay = torch.cumprod(a, dim=2)
    carry, incoming = h0, []
    for k in range(nb):
        incoming.append(carry)
        carry = decay[:, k, -1] * carry + local[:, k, -1]
    out = local + decay * torch.stack(incoming, dim=1)[:, :, None]
    return out.reshape(bsz, nb * block, c)[:, :t]


def _head(width_in: int, width: int, n_out: int) -> nn.Sequential:
    # head.0 / head.2 keep the Linear roles of RNNModel, so aux columns, family head and predict_run work
    return nn.Sequential(nn.Linear(width_in, width), nn.GELU(), nn.Linear(width, n_out))


class _V8Base(nn.Module):
    def _norm_buffers(self, n_in: int) -> None:
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def muon_matrix_blocks(self) -> dict[str, int]:  # parameter name -> orthogonalisation blocks (see wnn.optim)
        return {}

    def check_freeze_prefixes(self, prefixes) -> None:
        for prefix in prefixes:
            if not any(n.startswith(prefix) for n, _ in self.named_parameters()):
                raise ValueError(f"unknown freeze prefix {prefix!r} for {type(self).__name__}")


# ---------------------------------------------------------------------------------------------------------
# (A) clock-bank minGRU
# ---------------------------------------------------------------------------------------------------------

class ClockBankLayer(nn.Module):
    """minGRU whose channels are partitioned into banks; a bank holds its state unless its clock ticked.

    z = sigmoid(Wz x + bz [+ own]), c = tanh(Wc x [+ own]); m = tick of the channel's bank;
    h_t = (1 - m z) h_{t-1} + m z c. The candidate and gate do not read h, so the recurrence is an exact
    affine scan; the nonlinearity between rows lives only in the stack of layers.
    """

    def __init__(self, width: int, bank_dims: list[int], own_cols: list[list[int]] | None):
        super().__init__()
        self.gz, self.gc = nn.Linear(width, width), nn.Linear(width, width, bias=False)
        self.bank_dims = bank_dims
        self.register_buffer("chan_bank", torch.repeat_interleave(torch.arange(len(bank_dims)),
                                                                  torch.tensor(bank_dims)), persistent=False)
        # first layer only: each bank also reads its own raw column group directly (structural specialisation)
        self.own_cols = own_cols
        if own_cols is not None:
            self.own = nn.ModuleList(nn.Linear(len(c), 2 * d, bias=False) if c else nn.Identity()
                                     for c, d in zip(own_cols, bank_dims))
        with torch.no_grad():  # per-bank memory timescales: bank k keeps ~T_k = 2 + 3k ticks (z0 = 1 / T_k)
            lo = 0
            for k, d in enumerate(bank_dims):
                self.gz.bias[lo:lo + d] = -math.log(1.0 + 3.0 * k)
                lo += d

    def forward(self, x, tick, h0, xn_cols=None):
        pre_z, pre_c = self.gz(x), self.gc(x)
        if self.own_cols is not None and xn_cols is not None:
            oz, oc = [], []
            for proj, cols, d in zip(self.own, self.own_cols, self.bank_dims):
                if cols:
                    z_part, c_part = proj(xn_cols[..., cols]).chunk(2, dim=-1)
                else:
                    z_part = c_part = x.new_zeros(*x.shape[:-1], d)
                oz.append(z_part)
                oc.append(c_part)
            pre_z, pre_c = pre_z + torch.cat(oz, -1), pre_c + torch.cat(oc, -1)
        m = tick[..., self.chan_bank].float()  # (B, T, width): 1 where the channel's bank ticks
        gate = m * torch.sigmoid(pre_z.float())
        h = affine_scan(1.0 - gate, gate * torch.tanh(pre_c.float()), h0)
        return h, h[:, -1]


class ClockBankModel(_V8Base):
    """Encoder -> ``layers`` clock-bank minGRU layers -> head on [top state, encoder skip].

    ``hidden`` is split evenly over the clock groups of ``layout`` plus one always-on base bank (the remainder
    goes to the base bank). State: per-layer bank states plus the last raw row used for the change flags.
    """

    def __init__(self, n_in: int, hidden: int = 384, layers: int = 2, n_out: int = 2, layout: str = "semantic"):
        super().__init__()
        if layout not in CLOCK_LAYOUTS:
            raise ValueError(f"unknown clock-bank layout {layout!r}")
        if n_in < N_RAW:
            raise ValueError("clockbank needs the 112 raw columns at the start of the input")
        groups = CLOCK_LAYOUTS[layout]
        nb = len(groups) + 1
        if hidden < 4 * nb:
            raise ValueError(f"--hidden {hidden} too small for {nb} clock banks")
        dims = [hidden // nb] * (nb - 1)
        dims.append(hidden - sum(dims))
        self._norm_buffers(n_in)
        self.layout, self.hidden, self.n_layers = layout, hidden, layers
        self.group_cols = [c for _, c in groups]
        self.inp = nn.Linear(n_in, hidden)
        own = self.group_cols + [[]]
        self.layers = nn.ModuleList(ClockBankLayer(hidden, dims, own if i == 0 else None) for i in range(layers))
        self.head = _head(2 * hidden, hidden, n_out)

    def tick_flags(self, raw, prev):
        ne, last = changed_columns(raw, prev)
        flags = [ne[..., c].any(-1) for c in self.group_cols]
        flags.append(torch.ones_like(flags[0]))  # base bank: every row
        return torch.stack(flags, dim=-1), last

    def forward(self, x, state=None):
        xn = (x - self.mean) / self.std
        z = F.gelu(self.inp(xn))
        tick, last = self.tick_flags(x[..., :N_RAW], None if state is None else state[-1])
        cols = xn[..., :N_RAW]
        out, new = z, []
        for i, layer in enumerate(self.layers):
            h0 = z.new_zeros(z.shape[0], self.hidden) if state is None else state[i]
            seq, h_last = layer(out, tick, h0, cols)
            out = seq.to(z.dtype)
            new.append(h_last)
        return self.head(torch.cat([out, z], dim=-1)), (*new, last)

    def muon_matrix_blocks(self):
        return {n: 1 for n, p in self.named_parameters()
                if p.ndim == 2 and n.startswith("layers.") and (".gz.weight" in n or ".gc.weight" in n)}


# ---------------------------------------------------------------------------------------------------------
# (B) diagonal state-space bank
# ---------------------------------------------------------------------------------------------------------

class DSSMLayer(nn.Module):
    """h_t = a (.) h_{t-1} + g (.) (B x_t);  y = gelu(C h_t) + x_t  with per-channel decay a and gain g.

    The state path is linear and time invariant (no tanh, no gate, no dependence of a on the data); decays
    start log-uniformly spread over timescales 2..``t_max`` rows. The gain starts at sqrt(1 - a^2) (clipped)
    so slow channels do not blow up on a constant input.
    """

    def __init__(self, width: int, t_max: float = 512.0):
        super().__init__()
        self.inp = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width)
        timescale = torch.exp(torch.linspace(math.log(2.0), math.log(t_max), width))
        a = 1.0 - 1.0 / timescale
        self.logit_a = nn.Parameter(torch.log(a / (1.0 - a)))
        gain = torch.sqrt(1.0 - a * a).clamp_min(0.05)
        self.raw_g = nn.Parameter(torch.log(torch.expm1(gain)))

    def forward(self, x, h0):
        a = torch.sigmoid(self.logit_a.float())
        u = F.softplus(self.raw_g.float()) * self.inp(x).float()
        h = affine_scan(a.expand_as(u), u, h0)
        return F.gelu(self.out(h.to(x.dtype))) + x, h[:, -1]


class DSSMModel(_V8Base):
    """Encoder -> ``layers`` diagonal LTI state-space layers -> head on [top output, encoder skip]."""

    def __init__(self, n_in: int, hidden: int = 384, layers: int = 2, n_out: int = 2, t_max: float = 512.0):
        super().__init__()
        self._norm_buffers(n_in)
        self.hidden = hidden
        self.inp = nn.Linear(n_in, hidden)
        self.layers = nn.ModuleList(DSSMLayer(hidden, t_max) for _ in range(layers))
        self.head = _head(2 * hidden, hidden, n_out)

    def forward(self, x, state=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        out, new = z, []
        for i, layer in enumerate(self.layers):
            h0 = z.new_zeros(z.shape[0], self.hidden) if state is None else state[i]
            out, h_last = layer(out, h0)
            new.append(h_last)
        return self.head(torch.cat([out, z], dim=-1)), tuple(new)

    def muon_matrix_blocks(self):
        return {n: 1 for n, p in self.named_parameters()
                if p.ndim == 2 and n.startswith("layers.") and (".inp.weight" in n or ".out.weight" in n)}


# ---------------------------------------------------------------------------------------------------------
# (C) dual-tower book / trade late fusion with event-gated updates
# ---------------------------------------------------------------------------------------------------------

# columns of the 211-wide input each tower reads: raw columns plus the matching event features
# (event features start at 112: d_i0px 0-21, d_i0vol 22-43, d_i1px 44-65, flags 66-73, since 74-76, last_d_i0px 77-98;
# flags 66+3, 66+4, 66+7 and since 75 are trade events, the rest of the event features are book events)
BOOK_RAW = _cols(I0["px"], I0["vol"], I1["px"], I1["vol"])
TRADE_RAW = _cols(I0["trade"], I1["trade"])
BOOK_IN = (BOOK_RAW + [_EV + k for k in range(0, 66)] + [_EV + 66 + k for k in (0, 1, 2, 5, 6)]
           + [_EV + 74, _EV + 76] + [_EV + k for k in range(77, 99)])
TRADE_IN = TRADE_RAW + _cols((104, 112)) + [_EV + 66 + 3, _EV + 66 + 4, _EV + 66 + 7, _EV + 75]


def held_event_gru(grus: nn.ModuleList, u, events, state, gap_in):
    """Event-gated stack of single-layer GRUs over the compacted event rows.

    u: (B, T, E) per-row embedding; events: (B, T) bool; state: list of (B, H) per layer; gap_in: (B,) rows
    since the last event at the end of the previous chunk (0 at a sequence start = a virtual event at row -1).
    Returns the top-layer output held from the latest event at or before each row, the rows since that event
    (0 on an event row), the new per-layer state and the new gap. Between events nothing is updated.
    """
    b, t, _ = u.shape
    dev = u.device
    rows = torch.arange(t, device=dev).expand(b, t)
    neg = torch.full_like(rows, -1)
    last = torch.cummax(torch.where(events, rows, neg), dim=1).values  # latest event row at or before each row
    prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)  # latest event strictly before each row
    gap_prev = torch.where(prev >= 0, rows - prev, rows + 1 + gap_in[:, None])  # gap to the previous event
    gap_now = torch.where(last >= 0, rows - last, rows + 1 + gap_in[:, None])
    gap_now = torch.where(events, torch.zeros_like(gap_now), gap_now)
    counts = events.sum(1)
    n_upto = torch.cumsum(events.long(), 1)
    held = state[-1][:, None, :].expand(b, t, state[-1].shape[-1])
    new_state = list(state)
    maxc = int(counts.max())
    if maxc > 0:
        order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)[:, :maxc]  # event rows, in order
        x = torch.cat([torch.gather(u, 1, order[..., None].expand(b, maxc, u.shape[-1])),
                       torch.log1p(torch.gather(gap_prev, 1, order).to(u.dtype))[..., None]], dim=-1)
        idx = (n_upto - 1).clamp_min(0)
        has_event = counts > 0
        last_idx = (counts - 1).clamp_min(0)
        for i, gru in enumerate(grus):
            y, _ = gru(x.to(state[i].dtype), state[i][None].contiguous())
            h_last = y[torch.arange(b, device=dev), last_idx]
            new_state[i] = torch.where(has_event[:, None], h_last, state[i])
            x = y
        held = torch.where((n_upto > 0)[..., None],
                           torch.gather(y, 1, idx[..., None].expand(b, t, y.shape[-1])).to(u.dtype), held.to(u.dtype))
    last_row = last[:, -1]
    gap_out = torch.where(last_row >= 0, t - 1 - last_row, gap_in + t)
    return held, gap_now, new_state, gap_out


class DualTowerModel(_V8Base):
    """Book tower and trade tower, each an event-gated GRU stack, fused only in the readout.

    The book tower steps on rows where any book column (price or volume of either instrument) changed, the
    trade tower on rows where any trade px/vol column changed. Each reads only its own columns (plus matching
    event features) and the log gap since its previous event; between events its state and output are held.
    The head reads [book, trade, per-row skip, log1p(gaps)]. There is no every-row recurrence.
    ``hidden`` = book tower width, ``trade_hidden`` = trade tower width, ``layers`` GRU layers per tower.
    """

    def __init__(self, n_in: int, hidden: int = 224, layers: int = 2, trade_hidden: int = 160, n_out: int = 2,
                 skip: int = 128):
        super().__init__()
        if n_in < _EV + 99:
            raise ValueError("dualtower needs the 211-wide input (112 raw + 99 event features)")
        self._norm_buffers(n_in)
        self.hidden, self.trade_hidden, self.n_layers = hidden, trade_hidden, layers
        self.register_buffer("book_in", torch.tensor(BOOK_IN), persistent=False)
        self.register_buffer("trade_in", torch.tensor(TRADE_IN), persistent=False)
        self.book_cols, self.trade_cols = BOOK_RAW, TRADE_RAW
        self.book_proj, self.trade_proj = nn.Linear(len(BOOK_IN), hidden), nn.Linear(len(TRADE_IN), trade_hidden)
        # layer 0 reads [embedding, log gap since the previous event]; deeper layers read the layer below
        self.book_gru = nn.ModuleList(nn.GRU(hidden + (i == 0), hidden, batch_first=True) for i in range(layers))
        self.trade_gru = nn.ModuleList(nn.GRU(trade_hidden + (i == 0), trade_hidden, batch_first=True)
                                       for i in range(layers))
        self.skip = nn.Linear(n_in, skip)
        self.head = _head(hidden + trade_hidden + skip + 2, hidden, n_out)

    def forward(self, x, state=None):
        b = x.shape[0]
        xn = (x - self.mean) / self.std
        ne, last = changed_columns(x[..., :N_RAW], None if state is None else state[-1])
        ev_book, ev_trade = ne[..., self.book_cols].any(-1), ne[..., self.trade_cols].any(-1)
        ub = F.gelu(self.book_proj(xn[..., self.book_in]))
        ut = F.gelu(self.trade_proj(xn[..., self.trade_in]))
        if state is None:
            sb = [ub.new_zeros(b, self.hidden, dtype=torch.float32) for _ in range(self.n_layers)]
            st = [ub.new_zeros(b, self.trade_hidden, dtype=torch.float32) for _ in range(self.n_layers)]
            gb = gt = torch.zeros(b, device=x.device, dtype=torch.long)
        else:
            n = self.n_layers
            sb, st, (gb, gt) = list(state[:n]), list(state[n:2 * n]), state[2 * n:2 * n + 2]
        held_b, gap_b, sb, gb = held_event_gru(self.book_gru, ub, ev_book, sb, gb)
        held_t, gap_t, st, gt = held_event_gru(self.trade_gru, ut, ev_trade, st, gt)
        skip = F.gelu(self.skip(xn))
        gaps = torch.log1p(torch.stack([gap_b, gap_t], dim=-1).to(skip.dtype))
        out = self.head(torch.cat([held_b.to(skip.dtype), held_t.to(skip.dtype), skip, gaps], dim=-1))
        return out, (*sb, *st, gb, gt, last)

    def muon_matrix_blocks(self):
        return {n: 3 for n, p in self.named_parameters()
                if p.ndim == 2 and (n.startswith("book_gru.") or n.startswith("trade_gru.")) and ".weight_" in n}
