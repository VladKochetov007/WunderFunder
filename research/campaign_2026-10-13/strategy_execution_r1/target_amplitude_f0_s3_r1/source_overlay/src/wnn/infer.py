"""Submission-side stateful inference in plain numpy (no torch at test time).

``NumpyRNN`` mirrors ``wnn.nn.RNNModel`` one row at a time. Weights are exported
with ``export_weights`` into a flat ``.npz``.
"""

from __future__ import annotations

import math

import numpy as np


def _gelu(x):
    # torch.nn.GELU default: exact erf form.
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


_erf = np.frompyfunc(math.erf, 1, 1)


def _erf_vec(x):
    return _erf(x).astype(np.float32)


def export_weights(model) -> dict[str, np.ndarray]:
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    return sd


class NumpyRNN:
    def __init__(self, weights: dict[str, np.ndarray]):
        w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.mean, self.inv_std = w["mean"], 1.0 / w["std"]
        self.w_in, self.b_in = w["inp.weight"], w["inp.bias"]
        self.layers = []
        k = 0
        while f"gru.weight_ih_l{k}" in w:
            self.layers.append(
                (
                    w[f"gru.weight_ih_l{k}"],
                    w[f"gru.weight_hh_l{k}"],
                    w[f"gru.bias_ih_l{k}"],
                    w[f"gru.bias_hh_l{k}"],
                )
            )
            k += 1
        self.hidden = self.w_in.shape[0]
        self.w1, self.b1 = w["head.0.weight"], w["head.0.bias"]
        self.w2, self.b2 = w["head.2.weight"], w["head.2.bias"]
        self.h = np.zeros((len(self.layers), self.hidden), dtype=np.float32)

    def reset(self):
        self.h.fill(0.0)

    def step(self, x: np.ndarray) -> np.ndarray:
        z = self.w_in @ ((x - self.mean) * self.inv_std) + self.b_in
        z = 0.5 * z * (1.0 + _erf_vec(z / np.float32(math.sqrt(2.0))))
        inp = z
        n = self.hidden
        for k, (wih, whh, bih, bhh) in enumerate(self.layers):
            gi = wih @ inp + bih
            gh = whh @ self.h[k] + bhh
            r = 1.0 / (1.0 + np.exp(-(gi[:n] + gh[:n])))
            u = 1.0 / (1.0 + np.exp(-(gi[n : 2 * n] + gh[n : 2 * n])))
            c = np.tanh(gi[2 * n :] + r * gh[2 * n :])
            self.h[k] = (1.0 - u) * c + u * self.h[k]
            inp = self.h[k]
        a = self.w1 @ np.concatenate([inp, z]) + self.b1
        a = 0.5 * a * (1.0 + _erf_vec(a / np.float32(math.sqrt(2.0))))
        return self.w2 @ a + self.b2
