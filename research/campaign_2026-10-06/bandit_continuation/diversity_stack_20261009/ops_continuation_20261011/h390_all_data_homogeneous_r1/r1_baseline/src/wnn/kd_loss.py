"""Distillation-loss variants for the cached-teacher branch of the trainer (v10_kd).

All options default to the v9 objective; ``kd_distill_loss`` with a default ``KDConfig`` evaluates exactly the v9
expression ``(w * (out - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)``.

Conventions (``out``, ``tgt``, ``w`` are (B, T, 2) float tensors; ``w`` is the row weight already used by v9):

* Channel split: s = (p0 + p1) / 2, d = (p0 - p1) / 2 for the student and the target. Since
  e0^2 + e1^2 = 2 (es^2 + ed^2), the MSE term is ``2 (wS es^2 + wD ed^2)`` weighted with the per-target weights
  (the two columns share a weight only when ``w[...,0] == w[...,1]``; v9 weights are |clip(y_c)| per column, so the
  split uses the column mean ``wm = (w0 + w1) / 2`` for the s/d term). wS = wD = 1 therefore reproduces the v9 MSE
  exactly when the columns share weights, and the default (1, 1) skips the split entirely (bit-identical v9 path).
* Pearson: 1 - weighted Pearson(out[..., c], tgt[..., c]) over all rows of the chunk with weights ``w[..., c]``,
  averaged over c in {0, 1}. Scale free; ``mse+pearson`` adds ``pearson_weight`` times it to the MSE term.
* Run level: runs are maximal row segments starting at an event row (``event_rows``, a bool (B, T) mask; row 0 always
  starts a run) and ending before the next event row. For each (sequence, run) the weighted mean (weights ``w``) of the
  student output and of the target are formed; the term is sum_r W_r (mean_out_r - mean_tgt_r)^2 / sum_r W_r,
  summed over the 2 columns by averaging w across them (W_r = sum of w over rows of the run, per column).
  Rows with zero weight (unscored) do not contribute. Added as ``runlevel_weight`` times this term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

EPS = 1e-8


@dataclass(frozen=True)
class KDConfig:
    sum_weight: float = 1.0
    diff_weight: float = 1.0
    loss: str = "mse"  # mse | pearson | mse+pearson
    pearson_weight: float = 1.0
    runlevel_weight: float = 0.0

    @property
    def is_default(self) -> bool:
        return (self.sum_weight == 1.0 and self.diff_weight == 1.0 and self.loss == "mse"
                and self.runlevel_weight == 0.0)


def parse_channel_weights(text: str) -> tuple[float, float]:
    parts = [p for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("--distill-channel-weights expects SUM,DIFF")
    values = tuple(float(p) for p in parts)
    if not all(v == v and abs(v) != float("inf") and v >= 0 for v in values):
        raise ValueError("--distill-channel-weights must be finite and nonnegative")
    return values  # type: ignore[return-value]


def weighted_pearson_loss(out: Tensor, tgt: Tensor, w: Tensor) -> Tensor:
    """Mean over the last axis (targets) of 1 - weighted Pearson correlation over all (B, T) rows."""
    w = w.float()
    o, t = out.float(), tgt.float()
    ws = w.sum((0, 1)).clamp_min(EPS)
    mo = (w * o).sum((0, 1)) / ws
    mt = (w * t).sum((0, 1)) / ws
    do, dt = o - mo, t - mt
    cov = (w * do * dt).sum((0, 1))
    var_o = (w * do * do).sum((0, 1))
    var_t = (w * dt * dt).sum((0, 1))
    corr = cov / torch.sqrt((var_o + EPS) * (var_t + EPS))
    return (1.0 - corr).mean()


def run_ids(event_rows: Tensor) -> Tensor:
    """(B, T) bool -> (B, T) long run index within each sequence; row 0 always begins run 0."""
    starts = event_rows.clone()
    starts[:, 0] = True
    return starts.long().cumsum(1) - 1


def runlevel_loss(out: Tensor, tgt: Tensor, w: Tensor, event_rows: Tensor) -> Tensor:
    b, t, c = out.shape
    rid = run_ids(event_rows) + torch.arange(b, device=out.device)[:, None] * t  # unique (sequence, run) id
    flat = rid.reshape(-1)
    n = b * t
    wf = w.float().reshape(-1, c)
    sw = torch.zeros(n, c, device=out.device).index_add_(0, flat, wf)
    so = torch.zeros(n, c, device=out.device).index_add_(0, flat, wf * out.float().reshape(-1, c))
    st = torch.zeros(n, c, device=out.device).index_add_(0, flat, wf * tgt.float().reshape(-1, c))
    safe = sw.clamp_min(EPS)
    diff = so / safe - st / safe
    return (sw * diff ** 2).sum() / sw.sum().clamp_min(EPS)


def kd_distill_loss(out: Tensor, tgt: Tensor, w: Tensor, cfg: KDConfig, event_rows: Tensor | None = None) -> Tensor:
    out = out.float()
    if cfg.loss == "pearson":
        loss = weighted_pearson_loss(out, tgt, w)
    else:
        if cfg.sum_weight == 1.0 and cfg.diff_weight == 1.0:
            mse = (w * (out - tgt) ** 2).sum() / w.sum().clamp_min(EPS)
        else:
            e = out - tgt
            es, ed = (e[..., 0] + e[..., 1]) / 2, (e[..., 0] - e[..., 1]) / 2
            wm = w.mean(-1)
            mse = (2 * wm * (cfg.sum_weight * es ** 2 + cfg.diff_weight * ed ** 2)).sum() / w.sum().clamp_min(EPS)
        loss = mse
        if cfg.loss == "mse+pearson":
            loss = mse + cfg.pearson_weight * weighted_pearson_loss(out, tgt, w)
    if cfg.runlevel_weight > 0:
        if event_rows is None:
            raise ValueError("run-level term needs event_rows")
        loss = loss + cfg.runlevel_weight * runlevel_loss(out, tgt, w, event_rows)
    return loss
