"""Recurrent models and metric-aligned losses (research side, torch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

METRIC_CLIP = 2.0


class RNNModel(nn.Module):
    """Input projection -> stacked GRU -> head fed by GRU output and the projection skip."""

    def __init__(
        self, n_in: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0, n_out: int = 2,
        cell: str = "gru",
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        # Attribute name kept as ``gru`` so existing checkpoints and exporters stay valid.
        self.gru = rnn(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )

    def forward(self, x, h=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.gru(z, h)
        return self.head(torch.cat([o, z], dim=-1)), h


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
