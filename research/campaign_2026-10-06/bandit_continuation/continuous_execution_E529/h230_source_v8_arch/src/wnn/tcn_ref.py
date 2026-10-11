"""Row-at-a-time numpy reference of ``wnn.nn.TCNModel``: the spec for a native streaming port (research only).

Per sequence call ``reset()``, then ``step(row)`` for every row in order; ``row`` is the model input after
``train_rnn.build_inputs`` (112 raw + event features), before normalisation. For row t:

    z = GELU(W_inp (row - mean) / std + b_inp);  x = z
    block i (dilation d = 2**i):  a = GELU(LayerNorm_i(x))
                                  past = ring_i[t mod d]          # a(t - d); zeros while t < d
                                  y = W_i[:, :C] past + W_i[:, C:] a + b_i   (optional mix: y = M_i GELU(y) + m_i)
                                  ring_i[t mod d] = a;  x = x + y
    o = LayerNorm_out(x);  out = W_h2 GELU(W_h0 [o; z] + b_h0) + b_h2

GELU is the exact erf form, LayerNorm uses the biased variance and eps 1e-5. Weights: 2C^2 + 3C per block (+ C^2 + C
with mix); the rings hold sum(d) = 2^L - 1 rows of C activations, of which each row reads and writes one per block.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def gelu(v: np.ndarray) -> np.ndarray:
    return 0.5 * v * (1.0 + erf(v / np.sqrt(2.0)))


def layer_norm(v: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mu = v.mean()
    var = ((v - mu) ** 2).mean()
    return (v - mu) / np.sqrt(var + eps) * weight + bias


class TCNStream:
    def __init__(self, state_dict, eps: float = 1e-5, dtype=np.float64):
        p = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=dtype)
             for k, v in state_dict.items()}
        self.eps = eps
        self.mean, self.std = p["mean"], p["std"]
        self.w_in, self.b_in = p["inp.weight"], p["inp.bias"]
        c = self.w_in.shape[0]
        n_blocks = 1 + max(int(k.split(".")[2]) for k in p if k.startswith("tcn.blocks."))
        self.blocks = []
        for i in range(n_blocks):
            q = f"tcn.blocks.{i}."
            w = p[q + "conv.weight"]
            self.blocks.append({
                "d": 2**i, "g": p[q + "norm.weight"], "beta": p[q + "norm.bias"],
                "w_past": w[:, :c], "w_now": w[:, c:], "b": p[q + "conv.bias"],
                "mix_w": p.get(q + "mix.1.weight"), "mix_b": p.get(q + "mix.1.bias"),
            })
        self.g_out, self.beta_out = p["tcn.norm.weight"], p["tcn.norm.bias"]
        self.w_h0, self.b_h0 = p["head.0.weight"], p["head.0.bias"]
        self.w_h2, self.b_h2 = p["head.2.weight"], p["head.2.bias"]
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        c = self.w_in.shape[0]
        self.rings = [np.zeros((blk["d"], c), dtype=self.dtype) for blk in self.blocks]
        self.t = 0

    def step(self, row: np.ndarray) -> np.ndarray:
        xn = (np.asarray(row, dtype=self.dtype) - self.mean) / self.std
        z = gelu(self.w_in @ xn + self.b_in)
        x = z.copy()
        for blk, ring in zip(self.blocks, self.rings, strict=True):
            a = gelu(layer_norm(x, blk["g"], blk["beta"], self.eps))
            slot = self.t % blk["d"]
            y = blk["w_past"] @ ring[slot] + blk["w_now"] @ a + blk["b"]
            if blk["mix_w"] is not None:
                y = blk["mix_w"] @ gelu(y) + blk["mix_b"]
            ring[slot] = a
            x = x + y
        o = layer_norm(x, self.g_out, self.beta_out, self.eps)
        h = gelu(self.w_h0 @ np.concatenate([o, z]) + self.b_h0)
        self.t += 1
        return self.w_h2 @ h + self.b_h2
