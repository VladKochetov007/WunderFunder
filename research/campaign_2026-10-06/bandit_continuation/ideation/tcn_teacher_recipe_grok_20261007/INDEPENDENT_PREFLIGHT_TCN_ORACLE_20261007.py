"""Long-lag, chunk-state, causal and within-chunk gradient oracle for TCNModel.

Synthetic inputs only. This is a research preflight, not production code or a fit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from wnn.nn import TCNModel
from wnn.tcn_ref import TCNStream


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20261007)
    started = time.monotonic()

    batch, steps, n_in, channels, layers, chunk = 1, 4101, 211, 128, 12, 2000
    model = TCNModel(n_in, channels, layers, n_out=2, mix=True).eval()
    x = torch.randn(batch, steps, n_in, generator=torch.Generator().manual_seed(718204))

    with torch.inference_mode():
        full, _ = model(x)
        state, pieces = None, []
        for start in range(0, steps, chunk):
            out, state = model(x[:, start : start + chunk], state)
            state = tuple(t.detach() for t in state)
            pieces.append(out)
        chunked = torch.cat(pieces, dim=1)
        chunk_max_abs = float((chunked - full).abs().max())
        torch.testing.assert_close(chunked, full, rtol=0.0, atol=1e-5)
        tail_lengths = [int(t.shape[1]) for t in state]
        assert tail_lengths == [2**i for i in range(layers)], tail_lengths

        # A future perturbation must not affect the strict prefix before that row.
        future_x = x.clone()
        future_x[:, 3000, :] += 5.0
        future_out, _ = model(future_x)
        prefix_unchanged = torch.equal(full[:, :3000], future_out[:, :3000])
        assert prefix_unchanged

        # Input row zero reaches output row 2048 through the largest (2**11)
        # dilation; the full-length fixture also crosses the 2000-row chunk edge.
        lag_x = x.clone()
        lag_x[:, 0, :] += 10.0
        lag_out, _ = model(lag_x)
        lag_2048_max_abs = float((lag_out[:, 2048] - full[:, 2048]).abs().max())
        assert lag_2048_max_abs > 1e-8, lag_2048_max_abs

    # Explicit stream reset after contaminating the state with unrelated rows.
    stream = TCNStream(model.state_dict())
    for row in x[0, -4:].numpy():
        stream.step(row)
    stream.reset()
    reset_first = stream.step(x[0, 0].numpy())
    reset_max_abs = float(np.max(np.abs(reset_first - full[0, 0].numpy())))
    np.testing.assert_allclose(reset_first, full[0, 0].numpy(), rtol=0.0, atol=1e-5)

    # Detached-TBPTT smoke test: only the current 2000-row chunk participates
    # in autograd. No equality with full-sequence BPTT gradients is expected.
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, carry = model(x[:, :chunk])
        carry = tuple(t.detach() for t in carry)
    current = x[:, chunk : 2 * chunk].clone().requires_grad_(True)
    pred, _ = model(current, carry)
    pred[:, 100:600].square().mean().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    grad_norm = float(current.grad.norm())
    assert grad_norm > 0.0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    result = {
        "status": "PASS",
        "synthetic_only": True,
        "no_fit_or_labels": True,
        "shape": {
            "batch": batch,
            "timesteps": steps,
            "input_features": n_in,
            "channels": channels,
            "layers": layers,
            "mix": True,
            "chunk_length": chunk,
        },
        "checks": {
            "full_vs_chunked_forward": "PASS",
            "chunked_max_abs_difference": chunk_max_abs,
            "tail_lengths": tail_lengths,
            "future_perturbation_prefix_exact": prefix_unchanged,
            "row0_to_lag2048_max_abs_difference": lag_2048_max_abs,
            "stream_reset_first_row_max_abs_difference": reset_max_abs,
            "detached_tbptt_within_chunk_gradients_finite": True,
            "within_chunk_input_gradient_norm": grad_norm,
            "full_bptt_gradient_equality_required": False,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pid": os.getpid(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
