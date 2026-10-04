"""Rewrite the organizer ONNX baseline (GRU 112->128, GRU 128->128, linear head) as an exact
``wnn.nn.RNNModel`` state dict so the verified native Predictor can run it.

    .venv/bin/python scripts/convert_baseline.py submissions/_parts/baseline_rnnmodel.npz

The RNNModel input projection becomes identity on the 112 raw inputs plus a constant shift
C (GELU(u + C) == u + C in float32 for u + C >= ~10), with the shift removed through the first
GRU input bias. The head MLP is an identity on the last GRU output (same shift trick), then the
baseline linear layer with the shift folded into its bias. Event-feature inputs get zero weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from wnn.features import N_EVENT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
SHIFT = 16.0
H = 128
N_RAW = 112


def _torch_order(w):
    """ONNX GRU gate blocks (z, r, h) -> PyTorch (r, z, n) along axis 0."""
    z, r, h = np.split(w, 3, axis=0)
    return np.concatenate([r, z, h], axis=0)


def convert() -> dict[str, np.ndarray]:
    m = onnx.load(ROOT / "baseline" / "baseline.onnx")
    init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
    grus = [n for n in m.graph.node if n.op_type == "GRU"]
    n_in = N_RAW + N_EVENT_FEATURES
    sd = {"mean": np.zeros(n_in), "std": np.ones(n_in)}
    w_in = np.zeros((H, n_in))
    w_in[np.arange(N_RAW), np.arange(N_RAW)] = 1.0
    sd["inp.weight"], sd["inp.bias"] = w_in, np.full(H, SHIFT)
    for k, node in enumerate(grus):
        W, R, B = (init[node.input[j]][0] for j in (1, 2, 3))
        w_ih = _torch_order(W)
        b_ih, b_hh = _torch_order(B[: 3 * H]), _torch_order(B[3 * H:])
        if k == 0:
            full = np.zeros((3 * H, H))
            full[:, :N_RAW] = w_ih
            w_ih = full
            b_ih = b_ih - SHIFT * w_ih.sum(axis=1)  # z = x + SHIFT on every projected unit
        sd[f"gru.weight_ih_l{k}"], sd[f"gru.weight_hh_l{k}"] = w_ih, _torch_order(R)
        sd[f"gru.bias_ih_l{k}"], sd[f"gru.bias_hh_l{k}"] = b_ih, b_hh
    w_out = init[[n for n in m.graph.node if n.op_type == "MatMul"][0].input[1]].T  # (2, H)
    b_out = init["reg_head.bias"]
    head0 = np.zeros((H, 2 * H))
    head0[np.arange(H), np.arange(H)] = 1.0
    sd["head.0.weight"], sd["head.0.bias"] = head0, np.full(H, SHIFT)
    sd["head.2.weight"], sd["head.2.bias"] = w_out, b_out - SHIFT * w_out.sum(axis=1)
    return {k: v.astype(np.float32) for k, v in sd.items()}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **convert())
    print("wrote", out)
