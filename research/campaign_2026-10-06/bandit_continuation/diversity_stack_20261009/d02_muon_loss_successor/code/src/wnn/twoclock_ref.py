"""Row-at-a-time numpy reference of ``wnn.nn.TwoClockModel``: the spec for a native streaming port (research only).

Per sequence call ``reset()``, then ``step(row)`` for every row in order; ``row`` is the model input after
``train_rnn.build_inputs`` (112 raw + event features), before normalisation. For row t:

    z = GELU(W_inp (row - mean) / std + b_inp)
    o = row GRU stack (layers x GRU(H)) on z                    # every row
    seg_sum += o;  seg_rows += 1                                 # rows since the last event, this row included
    if float32(row[110]) == ENDPOINT (a6 at its lower endpoint):  # event row
        s = GRU_event([o; seg_sum / seg_rows; log1p(seg_rows)], s)
        seg_sum = 0;  seg_rows = 0
    out = W_h2 GELU(W_h0 [o; s; z; log1p(seg_rows)] + b_h0) + b_h2

so the head sees log1p(0) = 0 on event rows and log1p(rows since the last event, or since the sequence start)
otherwise; s is carried unchanged on non-event rows. GRU gates follow torch (r, z, n; n = tanh(W_in x + b_in +
r * (W_hn h + b_hn))), GELU is the exact erf form.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from wnn.event_content import ENDPOINT
from wnn.nn import A6


def gelu(v: np.ndarray) -> np.ndarray:
    return 0.5 * v * (1.0 + erf(v / np.sqrt(2.0)))


def sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def gru_cell(x: np.ndarray, h: np.ndarray, w_ih, w_hh, b_ih, b_hh) -> np.ndarray:
    n = h.shape[0]
    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh
    r = sigmoid(gi[:n] + gh[:n])
    u = sigmoid(gi[n : 2 * n] + gh[n : 2 * n])
    c = np.tanh(gi[2 * n :] + r * gh[2 * n :])
    return (1.0 - u) * c + u * h


class TwoClockStream:
    def __init__(self, state_dict, dtype=np.float64):
        p = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=dtype)
             for k, v in state_dict.items()}
        self.mean, self.std = p["mean"], p["std"]
        self.w_in, self.b_in = p["inp.weight"], p["inp.bias"]
        self.row_layers = []
        while f"gru.weight_ih_l{len(self.row_layers)}" in p:
            k = len(self.row_layers)
            self.row_layers.append(tuple(p[f"gru.{n}_l{k}"] for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")))
        self.event_layer = tuple(p[f"gru_event.{n}_l0"] for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
        self.w_h0, self.b_h0 = p["head.0.weight"], p["head.0.bias"]
        self.w_h2, self.b_h2 = p["head.2.weight"], p["head.2.bias"]
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        hidden, event_hidden = self.w_in.shape[0], self.event_layer[1].shape[1]
        self.h = [np.zeros(hidden, dtype=self.dtype) for _ in self.row_layers]
        self.s = np.zeros(event_hidden, dtype=self.dtype)
        self.seg_sum = np.zeros(hidden, dtype=self.dtype)
        self.seg_rows = 0

    def step(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row)
        z = gelu(self.w_in @ ((row.astype(self.dtype) - self.mean) / self.std) + self.b_in)
        o = z
        for k, layer in enumerate(self.row_layers):
            self.h[k] = gru_cell(o, self.h[k], *layer)
            o = self.h[k]
        self.seg_sum = self.seg_sum + o
        self.seg_rows += 1
        if np.float32(row[A6]) == np.float32(ENDPOINT):
            n = float(self.seg_rows)
            self.s = gru_cell(np.concatenate([o, self.seg_sum / n, [np.log1p(n)]]), self.s, *self.event_layer)
            self.seg_sum = np.zeros_like(self.seg_sum)
            self.seg_rows = 0
        a = gelu(self.w_h0 @ np.concatenate([o, self.s, z, [np.log1p(self.seg_rows)]]) + self.b_h0)
        return self.w_h2 @ a + self.b_h2
