"""Recurrent models and metric-aligned losses (research side, torch)."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from wnn.event_content import ENDPOINT
from wnn.numeric_ple import NumericFeatureMap
from wnn.tiedlead import (
    TIEDLEAD_BASE_INPUT_WIDTH,
    TIEDLEAD_BOOK_WIDTH,
    TIEDLEAD_INPUT_WIDTH,
    TIEDLEAD_INSTRUMENT_WIDTH,
    TIED_ENCODER_HIDDEN,
    TIED_ENCODER_WIDTH,
    tied_book_deltas,
)

METRIC_CLIP = 2.0


class FixedReadoutMixture(nn.Module):
    """Parameter-free E488 mixture, restricted to the readout activation."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Mix equal GELU and LeakyReLU contributions without modifying inputs."""
        return 0.5 * F.gelu(values) + 0.5 * F.leaky_relu(values, negative_slope=0.01)


class RNNReadout(nn.Sequential):
    """One- or two-hidden-layer readout with the legacy parameter names intact."""

    def __init__(self, input_width: int, width: int, n_out: int, layers: int,
                 activation: nn.Module) -> None:
        first = nn.Linear(input_width, width)
        extra = nn.Linear(width, width) if layers == 2 else None
        output = nn.Linear(width, n_out)
        super().__init__(first, activation, output)
        self.extra = extra

    def _legacy_layers(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self._modules["0"], self._modules["1"], self._modules["2"]

    def __getitem__(self, index: int | slice) -> nn.Module:
        layers = self._legacy_layers()
        if isinstance(index, slice):
            return nn.Sequential(*layers[index])
        return layers[index]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        normalized = index + 3 if index < 0 else index
        if normalized not in (0, 1, 2):
            raise IndexError("readout index out of range")
        self._modules[str(normalized)] = module

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._legacy_layers())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first, activation, output = self._legacy_layers()
        values = activation(first(values))
        if self.extra is not None:
            values = activation(self.extra(values))
        return output(values)


class RNNModel(nn.Module):
    """Input projection -> stacked GRU -> head fed by GRU output and the projection skip."""

    def __init__(
        self, n_in: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0, n_out: int = 2,
        cell: str = "gru", enc: str = "linear", tok_dim: int = 32, att_layers: int = 1, conv_k: int = 16,
        readout_activation: str = "gelu", numeric_map: str = "none", ple_intervals: int = 8,
        encoder_width: int | None = None, recurrent_width: int | None = None,
        readout_width: int | None = None, head_layers: int = 1,
    ):
        super().__init__()
        if readout_activation not in ("gelu", "fixed_mix"):
            raise ValueError("readout_activation must be gelu or fixed_mix")
        if numeric_map not in ("none", *NumericFeatureMap.MODES):
            raise ValueError(f"unknown numeric map mode: {numeric_map}")
        encoder_width = hidden if encoder_width is None else encoder_width
        recurrent_width = hidden if recurrent_width is None else recurrent_width
        readout_width = hidden if readout_width is None else readout_width
        if min(encoder_width, recurrent_width, readout_width) <= 0:
            raise ValueError("encoder, recurrent and readout widths must be positive")
        if head_layers not in (1, 2):
            raise ValueError("head_layers must be 1 or 2")
        self.encoder_width = encoder_width
        self.recurrent_width = recurrent_width
        self.readout_width = readout_width
        self.head_layers = head_layers
        # Keep the historical attribute as the recurrent-state width.
        self.hidden = recurrent_width
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.numeric_feature_map = (
            None if numeric_map == "none" else NumericFeatureMap(n_in, numeric_map, ple_intervals)
        )
        input_width = n_in if self.numeric_feature_map is None else self.numeric_feature_map.output_width
        self.inp = nn.Linear(input_width, encoder_width)
        # Optional per-row encoders, added as residuals to the linear projection (research screening only;
        # the native engine supports enc="linear").
        self.enc = enc
        if enc == "mlp":
            self.enc_norm = nn.LayerNorm(encoder_width)
            self.enc_up = nn.Linear(encoder_width, 4 * encoder_width)
            self.enc_down = nn.Linear(2 * encoder_width, encoder_width)
        elif enc == "tiedlead":
            if n_in != TIEDLEAD_INPUT_WIDTH:
                raise ValueError(
                    f"tiedlead requires exactly {TIEDLEAD_INPUT_WIDTH} input channels"
                )
            if cell != "gru" or dropout != 0.0 or numeric_map != "none":
                raise ValueError("tiedlead requires a no-dropout GRU and no numeric map")
            self.tied = nn.Sequential(
                nn.Linear(TIEDLEAD_BOOK_WIDTH, TIED_ENCODER_HIDDEN),
                nn.GELU(),
                nn.Linear(TIED_ENCODER_HIDDEN, TIED_ENCODER_WIDTH),
                nn.GELU(),
            )
            self.tied_out = nn.Linear(2 * TIED_ENCODER_WIDTH, encoder_width)
            self.lead_out = nn.Linear(TIED_ENCODER_WIDTH, encoder_width)
            nn.init.zeros_(self.tied_out.weight)
            nn.init.zeros_(self.tied_out.bias)
            nn.init.zeros_(self.lead_out.weight)
            nn.init.zeros_(self.lead_out.bias)
        elif enc == "tokens":
            self.groups = token_groups(n_in)
            self.tok_in = nn.ModuleList(nn.Linear(b - a, tok_dim) for a, b in self.groups)
            self.cls = nn.Parameter(torch.zeros(1, 1, tok_dim))
            layer = nn.TransformerEncoderLayer(tok_dim, 4, 2 * tok_dim, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.att = nn.TransformerEncoder(layer, att_layers, enable_nested_tensor=False)
            self.tok_out = nn.Linear(2 * tok_dim, encoder_width)
        elif enc == "conv":
            self.conv_k = conv_k
            self.conv_dw = nn.Conv1d(encoder_width, encoder_width, conv_k, groups=encoder_width)
            self.conv_pw = nn.Linear(encoder_width, encoder_width)
        elif enc not in ("linear", "tiedlead"):
            raise ValueError(enc)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        # Attribute name kept as ``gru`` so existing checkpoints and exporters stay valid.
        self.gru = rnn(
            encoder_width,
            recurrent_width,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = RNNReadout(
            recurrent_width + encoder_width,
            readout_width,
            n_out,
            head_layers,
            nn.GELU() if readout_activation == "gelu" else FixedReadoutMixture(),
        )

    def encode(self, xn: torch.Tensor) -> torch.Tensor:
        if self.enc == "tiedlead":
            encoded, _ = self._encode_tiedlead(xn, None)
            return encoded
        if self.numeric_feature_map is not None:
            xn = self.numeric_feature_map(xn)
        u = self.inp(xn)
        if self.enc == "tokens":
            b, t, _ = xn.shape
            flat = xn.reshape(b * t, -1)
            toks = torch.stack([lin(flat[:, a:c]) for lin, (a, c) in zip(self.tok_in, self.groups)], dim=1)
            toks = self.att(torch.cat([self.cls.expand(b * t, -1, -1), toks], dim=1))
            pooled = torch.cat([toks[:, 0], toks[:, 1:].mean(1)], dim=-1)
            u = u + self.tok_out(pooled).reshape(b, t, -1)
        z = F.gelu(u)
        if self.enc == "mlp":
            a, g = self.enc_up(self.enc_norm(z)).chunk(2, dim=-1)
            z = z + self.enc_down(F.gelu(a) * torch.sigmoid(g))
        elif self.enc == "conv":
            c = F.pad(z.transpose(1, 2), (self.conv_k - 1, 0))  # causal; zero left context at chunk starts
            z = z + F.gelu(self.conv_pw(self.conv_dw(c).transpose(1, 2)))
        return z

    def _encode_tiedlead(
        self, xn: torch.Tensor, previous_book: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add raw-scale shared book and lead residuals while carrying chunk deltas."""
        if xn.ndim != 3 or xn.shape[1] == 0 or xn.shape[-1] != TIEDLEAD_INPUT_WIDTH:
            raise ValueError(
                f"tiedlead expects a nonempty (batch,time,{TIEDLEAD_INPUT_WIDTH}) tensor"
            )
        raw = xn * self.std + self.mean
        books = raw[..., :TIEDLEAD_BOOK_WIDTH]
        deltas, next_book = tied_book_deltas(books, previous_book)
        i0 = torch.cat(
            (
                raw[..., :TIEDLEAD_INSTRUMENT_WIDTH],
                deltas[..., :TIEDLEAD_INSTRUMENT_WIDTH],
            ),
            dim=-1,
        )
        i1 = torch.cat(
            (
                raw[..., TIEDLEAD_INSTRUMENT_WIDTH:TIEDLEAD_BOOK_WIDTH],
                deltas[..., TIEDLEAD_INSTRUMENT_WIDTH:TIEDLEAD_BOOK_WIDTH],
            ),
            dim=-1,
        )
        lead = raw[..., TIEDLEAD_BASE_INPUT_WIDTH:TIEDLEAD_INPUT_WIDTH]
        raw_i1 = raw[..., TIEDLEAD_INSTRUMENT_WIDTH:TIEDLEAD_BOOK_WIDTH]
        lead_input = torch.cat((raw_i1, raw_i1 - lead), dim=-1)
        projected = self.inp(xn)
        projected = projected + self.tied_out(torch.cat((self.tied(i0), self.tied(i1)), dim=-1))
        projected = projected + self.lead_out(self.tied(lead_input))
        return F.gelu(projected), next_book

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Apply the recurrent model and return any architecture-specific chunk state."""
        normalized = (x - self.mean) / self.std
        if self.enc == "tiedlead":
            if h is None:
                recurrent_state, previous_book = None, None
            elif isinstance(h, tuple) and len(h) == 2:
                recurrent_state, previous_book = h
            else:
                raise TypeError("tiedlead state must be (GRU hidden, previous raw book) or None")
            z, next_book = self._encode_tiedlead(normalized, previous_book)
            recurrent, next_recurrent_state = self.gru(z, recurrent_state)
            next_state = (next_recurrent_state, next_book)
        else:
            z = self.encode(normalized)
            recurrent, next_state = self.gru(z, h)
        return self.head(torch.cat([recurrent, z], dim=-1)), next_state


class GatedRNNModel(RNNModel):
    """Factor the target pair into a shared level and gated movement size.

    The sign head emits a probability, while the size head remains positive.
    Their product is the conditional movement used by both target outputs.
    The sign logit is returned after the ordinary auxiliary outputs for BCE.
    """

    def __init__(self, n_in: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.0, n_out: int = 2) -> None:
        super().__init__(n_in, hidden, layers, dropout, n_out)
        self.sign_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.gate_mode = "soft"

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode((x - self.mean) / self.std)
        recurrent, h = self.gru(z, h)
        features = torch.cat([recurrent, z], dim=-1)
        raw = self.head(features)
        sign_logit = self.sign_head(features)
        soft_sign = torch.tanh(sign_logit.float() / 2)
        if self.gate_mode == "hard":
            gate = torch.where(sign_logit >= 0, 1.0, -1.0)
        elif self.gate_mode == "confident":
            gate = torch.where(soft_sign.abs() >= 0.5, soft_sign.sign(), soft_sign)
        else:
            gate = soft_sign
        movement = gate * F.softplus(raw[..., 1:2].float())
        pair = torch.cat([raw[..., :1].float() + movement,
                          raw[..., :1].float() - movement], dim=-1)
        return torch.cat([pair, raw[..., 2:].float(), sign_logit.float()], dim=-1), h


class CausalConvBlock(nn.Module):
    """Pre-norm residual causal conv, kernel 2, dilation d: x(t) + W [a(t - d); a(t)] + b, a = GELU(LayerNorm(x)).

    ``tail`` holds a() of the d rows before the chunk (zeros at a sequence start); the returned tail is the state
    for the next chunk, so chunked calls equal one full-sequence call. ``mix`` adds GELU + 1x1 conv after the conv.
    """

    def __init__(self, channels: int, dilation: int, mix: bool = False):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Linear(2 * channels, channels)  # input columns [a(t - d), a(t)]
        self.mix = nn.Sequential(nn.GELU(), nn.Linear(channels, channels)) if mix else None

    def forward(self, x, tail=None):
        a = F.gelu(self.norm(x))
        b, t, c = a.shape
        if tail is None:
            tail = a.new_zeros(b, self.dilation, c)
        ext = torch.cat([tail.to(a.dtype), a], dim=1)  # ext[:, j] = a(j - d) in chunk rows
        y = self.conv(torch.cat([ext[:, :t], a], dim=-1))
        if self.mix is not None:
            y = self.mix(y)
        return x + y, ext[:, t:].clone()  # clone: a view would keep the whole extended chunk alive


class TCN(nn.Module):
    """Causal dilated blocks with dilations 1, 2, ..., 2^(L-1) (receptive field 2^L rows), then a LayerNorm."""

    def __init__(self, channels: int, layers: int, mix: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(CausalConvBlock(channels, 2**i, mix) for i in range(layers))
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, h=None):
        tails = []
        for i, block in enumerate(self.blocks):
            x, tail = block(x, None if h is None else h[i])
            tails.append(tail)
        return self.norm(x), tuple(tails)


class TCNModel(nn.Module):
    """Input projection -> causal dilated TCN -> RNNModel's head (fed by the TCN output and the projection skip).

    The state is the per-block tail of the last d block activations, so TBPTT chunks carry the exact convolution
    history. Trunk parameters are ``inp.*`` and ``tcn.*``: head-only FT is ``--freeze inp,tcn``.
    ``wnn.tcn_ref.TCNStream`` is the row-at-a-time reference of the same function.
    """

    def __init__(self, n_in: int, channels: int = 128, layers: int = 12, n_out: int = 2, mix: bool = False):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, channels)
        self.tcn = TCN(channels, layers, mix)
        self.head = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.GELU(), nn.Linear(channels, n_out)
        )

    def forward(self, x, h=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.tcn(z, h)
        return self.head(torch.cat([o, z], dim=-1)), h


A6 = 110  # raw input column of a6; rows with a6 at ENDPOINT are the quote-move rows where targets can change


class TwoClockModel(nn.Module):
    """Row-clock GRU plus an event-clock GRU that steps only on event rows (raw a6 == float32 lower endpoint).

    At an event row e the event GRU reads [o(e), mean of o over the rows since the previous event up to e,
    log1p(their count)], o = row GRU output; between events its state is carried unchanged. The head reads
    [o, event state, projection skip z, log1p(rows since the last event)]. State ``(h_row, h_event, seg_sum,
    seg_rows)`` (sum and count of o over the rows after the last event) carries across TBPTT chunks exactly.
    Trunk parameters are inp.*, gru.* (row clock) and gru_event.*: ``--freeze inp,gru`` freezes the whole trunk.
    ``wnn.twoclock_ref.TwoClockStream`` is the row-at-a-time reference.
    """

    def __init__(self, n_in: int, hidden: int = 192, layers: int = 2, event_hidden: int = 128, n_out: int = 2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.gru_event = nn.GRU(2 * hidden + 1, event_hidden, batch_first=True)
        self.event_hidden = event_hidden
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + event_hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )

    def clocks(self, x, state=None):
        """Row output o, event state held at each row, rows since the last event, projection z, new state."""
        b, t, _ = x.shape
        events = x[..., A6].float() == ENDPOINT  # raw input, before standardisation
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h_row = self.gru(z, None if state is None else state[0])
        acc = torch.promote_types(o.dtype, torch.float32)
        if state is None:
            h_event = z.new_zeros(1, b, self.event_hidden)
            seg_sum = o.new_zeros(b, o.shape[-1], dtype=acc)
            seg_rows = torch.zeros(b, device=x.device, dtype=torch.long)
        else:
            h_event, seg_sum, seg_rows = state[1:]
        rows = torch.arange(t, device=x.device).expand(b, t)
        neg = torch.full_like(rows, -1)
        last = torch.cummax(torch.where(events, rows, neg), dim=1).values  # last event at or before the row
        prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)  # last event strictly before the row
        elapsed = torch.where(last >= 0, rows - last, seg_rows[:, None] + rows + 1)
        cs = torch.cumsum(o.to(acc), dim=1)

        def rows_of(v, index):
            return torch.gather(v, 1, index[..., None].expand(-1, -1, v.shape[-1]))

        held = h_event[0][:, None, :].expand(b, t, self.event_hidden)
        counts = events.sum(1)
        maxc = int(counts.max())
        if maxc > 0:
            order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)[:, :maxc]  # event rows in order
            p = torch.gather(prev, 1, order)
            cs_e = rows_of(cs, order)
            seg = torch.where((p >= 0)[..., None], cs_e - rows_of(cs, p.clamp_min(0)), cs_e + seg_sum[:, None])
            n = torch.where(p >= 0, order - p, seg_rows[:, None] + order + 1).to(acc)[..., None]
            ev_in = torch.cat([rows_of(o, order).to(acc), seg / n, torch.log1p(n)], dim=-1)
            ev_out, _ = self.gru_event(ev_in.to(h_event.dtype), h_event)
            ev_out = ev_out.to(held.dtype)
            n_upto = torch.cumsum(events.long(), 1)
            held = torch.where((n_upto > 0)[..., None], rows_of(ev_out, (n_upto - 1).clamp_min(0)), held)
            h_last = ev_out[torch.arange(b, device=x.device), (counts - 1).clamp_min(0)]
            h_event = torch.where((counts > 0)[:, None], h_last, h_event[0])[None]
        end = last[:, -1]
        tail = cs[:, -1] - rows_of(cs, end.clamp_min(0)[:, None])[:, 0]
        seg_sum = torch.where((end >= 0)[:, None], tail, seg_sum + cs[:, -1])
        return o, held, elapsed, z, (h_row, h_event, seg_sum, elapsed[:, -1])

    def forward(self, x, state=None):
        o, held, elapsed, z, state = self.clocks(x, state)
        since = torch.log1p(elapsed.to(torch.promote_types(z.dtype, torch.float32)))[..., None]
        return self.head(torch.cat([o, held, z, since], dim=-1)), state


def token_groups(n_in: int):
    """Column groups of the (112 raw + 99 event) input used as attention tokens."""
    raw = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112)]
    ev = [(112, 134), (134, 156), (156, 178), (178, 189), (189, 211)]
    return [g for g in raw + ev if g[1] <= n_in]


def weighted_corr_loss(pred, target, valid):
    """1 - mean weighted Pearson over (batch, target) with metric weights |clip(y)|.

    pred, target: (B, T, K); valid: (B, T) bool. Predictions are not clipped
    (keeps gradients); submission output scale is set separately.
    """
    pred = pred.float()
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * valid[..., None]
    wsum = w.sum(dim=1).clamp_min(1e-8)
    my = (w * y).sum(1) / wsum
    mp = (w * pred).sum(1) / wsum
    yc = y - my[:, None]
    pc = pred - mp[:, None]
    cov = (w * yc * pc).sum(1) / wsum
    vy = (w * yc * yc).sum(1) / wsum
    vp = (w * pc * pc).sum(1) / wsum
    rho = cov / torch.sqrt(vy * vp + 1e-12)
    ok = (wsum > 1e-6) & (vy > 1e-8)
    return 1.0 - (rho * ok).sum() / ok.sum().clamp_min(1)


def weighted_pooled_corr_loss(pred, target, valid):
    """1 - mean over targets of the POOLED |clip(y)|-weighted Pearson over every row of every sequence in the batch:
    the competition metric's form (global, not per sequence), so between-sequence levels count too."""
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    p = pred.float()
    w = y.abs() * valid[..., None]
    k = y.shape[-1]
    w, y, p = w.reshape(-1, k), y.reshape(-1, k), p.reshape(-1, k)
    ws = w.sum(0).clamp_min(1e-8)
    my, mp = (w * y).sum(0) / ws, (w * p).sum(0) / ws
    yc, pc = y - my, p - mp
    cov = (w * yc * pc).sum(0) / ws
    vy, vp = (w * yc * yc).sum(0) / ws, (w * pc * pc).sum(0) / ws
    return 1.0 - (cov / torch.sqrt(vy * vp + 1e-12)).mean()


class PooledCorrMemory:
    """Pooled |y|-weighted Pearson over the current chunk plus a decayed memory of earlier chunks.

    ``weighted_pooled_corr_loss`` recentres and rescales every 16-sequence chunk on its own, so offsets and
    scales shared by a whole batch are invisible to it. Here the six weighted sums of earlier chunks are kept
    (detached, float64) and added to the current chunk's sums before the correlation is formed; gradients flow
    only through the current rows, i.e. each step follows the correlation of the whole recent pool. The loss is
    divided by the current chunk's share of the pooled weight so its gradient scale matches the per-chunk loss.
    """

    def __init__(self, decay: float = 0.98):
        self.decay = decay
        self.memory = None

    def __call__(self, pred, target, valid):
        y = target.double().clamp(-METRIC_CLIP, METRIC_CLIP)
        p = pred.double()
        w = y.abs() * valid[..., None].double()
        k = y.shape[-1]
        w, y, p = w.reshape(-1, k), y.reshape(-1, k), p.reshape(-1, k)
        sums = torch.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * y).sum(0), (w * p * p).sum(0),
                            (w * y * p).sum(0)])
        pooled = sums if self.memory is None else sums + self.memory
        n, sy, sp, syy, spp, syp = pooled
        n = n.clamp_min(1e-12)
        cov = syp / n - (sy / n) * (sp / n)
        vy = (syy / n - (sy / n) ** 2).clamp_min(0.0)
        vp = (spp / n - (sp / n) ** 2).clamp_min(0.0)
        rho = cov / torch.sqrt(vy * vp + 1e-18)
        share = (sums[0] / n).detach().clamp_min(1e-6)
        self.memory = self.decay * pooled.detach()
        return ((1.0 - rho) / share).mean().float()


def weighted_mse_loss(pred, target, valid):
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * valid[..., None]
    return (w * (pred.float() - y) ** 2).sum() / w.sum().clamp_min(1e-8)


def weighted_huber_loss(pred, target, valid, delta: float = 0.5):
    """Huber on clip(y) with metric weights |clip(y)|; delta in target units."""
    y = target.float().clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * valid[..., None]
    return (w * F.huber_loss(pred.float(), y, reduction="none", delta=delta)).sum() / w.sum().clamp_min(1e-8)


class DualClockModel(nn.Module):
    """Row-clock GRU plus an event-clock GRU that steps only on i0 price events.

    ``forward(x, events, state)`` with x (B, T, n_in), events (B, T) bool.
    State ``(h_row, h_event, gap)`` carries across TBPTT chunks; ``gap`` counts
    rows since the last event. The event GRU input at an event row is the
    projected row input plus log1p(rows since the previous event). Rows between
    events see the event state from the latest event at or before them.
    """

    def __init__(self, n_in: int, hidden: int = 256, layers: int = 2, event_hidden: int = 128,
                 dropout: float = 0.0, n_out: int = 2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.event_gru = nn.GRU(hidden + 1, event_hidden, num_layers=1, batch_first=True)
        self.event_hidden = event_hidden
        self.head = nn.Sequential(nn.Linear(2 * hidden + event_hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, x, events, state=None):
        b, t, _ = x.shape
        dev = x.device
        z = F.gelu(self.inp((x - self.mean) / self.std))
        if state is None:
            h_row = None
            h_event = torch.zeros(b, self.event_hidden, device=dev, dtype=z.dtype)
            gap_in = torch.zeros(b, device=dev, dtype=torch.long)
        else:
            h_row, h_event, gap_in = state
        o, h_row = self.gru(z, h_row)

        rows = torch.arange(t, device=dev).expand(b, t)
        neg = torch.full_like(rows, -1)
        last = torch.cummax(torch.where(events, rows, neg), dim=1).values
        prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)
        gap = torch.where(prev >= 0, rows - prev, rows + 1 + gap_in[:, None])
        counts = events.sum(1)
        n_upto = torch.cumsum(events.long(), 1)

        held = h_event[:, None, :].expand(b, t, self.event_hidden)
        if int(counts.max()) > 0:
            maxc = int(counts.max())
            order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)[:, :maxc]
            ev_in = torch.cat([
                torch.gather(z, 1, order[..., None].expand(b, maxc, z.shape[-1])),
                torch.log1p(torch.gather(gap, 1, order).to(z.dtype))[..., None],
            ], dim=-1)
            ev_out, _ = self.event_gru(ev_in.to(h_event.dtype), h_event[None].contiguous())
            ev_out = ev_out.to(z.dtype)
            idx = (n_upto - 1).clamp_min(0)
            gathered = torch.gather(ev_out, 1, idx[..., None].expand(b, t, self.event_hidden))
            held = torch.where((n_upto > 0)[..., None], gathered, held)
            last_idx = (counts - 1).clamp_min(0)
            h_last = ev_out[torch.arange(b, device=dev), last_idx]
            h_event = torch.where((counts > 0)[:, None], h_last, h_event)

        last_row = last[:, -1]
        gap_out = torch.where(last_row >= 0, t - 1 - last_row, gap_in + t)
        out = self.head(torch.cat([o, held, z], dim=-1))
        return out, (h_row, h_event, gap_out)
