"""Check row-chunked feature-group attention math or stress its full row shape."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import impact81.representations as representations
import torch
from impact81.representations import (
    GROUP_ATTENTION_ROW_CHUNK,
    GROUP_TOKEN_WIDTH,
    N_FEATURES,
    GroupInputProjection,
    _row_chunked_attention,
)
from torch import nn


def _cpu_parity(rows: int, chunk_rows: int) -> dict[str, Any]:
    """Compare direct and chunked MHA outputs, input gradients and parameter gradients."""
    if rows <= 0 or chunk_rows <= 0:
        raise ValueError("CPU parity rows and chunk size must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(810207)
    direct_attention = nn.MultiheadAttention(
        GROUP_TOKEN_WIDTH, 4, dropout=0.0, batch_first=True
    ).cpu()
    chunked_attention = copy.deepcopy(direct_attention)
    values = torch.randn(rows, 16, GROUP_TOKEN_WIDTH)
    direct_values = values.detach().clone().requires_grad_(True)
    chunked_values = values.detach().clone().requires_grad_(True)
    loss_weights = torch.randn_like(values)

    direct = direct_attention(direct_values, direct_values, direct_values, need_weights=False)[0]
    direct_loss = (direct * loss_weights).sum() / direct.numel()
    direct_loss.backward()

    chunked = _row_chunked_attention(chunked_attention, chunked_values, chunk_rows)
    chunked_loss = (chunked * loss_weights).sum() / chunked.numel()
    chunked_loss.backward()

    torch.testing.assert_close(chunked, direct, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(chunked_values.grad, direct_values.grad, rtol=3e-5, atol=3e-7)
    max_parameter_gradient_delta = 0.0
    for (direct_name, direct_parameter), (chunked_name, chunked_parameter) in zip(
        direct_attention.named_parameters(), chunked_attention.named_parameters(), strict=True
    ):
        if direct_name != chunked_name:
            raise AssertionError("attention parameter names differ")
        if direct_parameter.grad is None or chunked_parameter.grad is None:
            raise AssertionError(f"missing attention gradient for {direct_name}")
        torch.testing.assert_close(
            chunked_parameter.grad,
            direct_parameter.grad,
            rtol=3e-5,
            atol=3e-7,
        )
        max_parameter_gradient_delta = max(
            max_parameter_gradient_delta,
            float((chunked_parameter.grad - direct_parameter.grad).abs().max().detach()),
        )
    if not bool(torch.isfinite(chunked).all() and torch.isfinite(chunked_values.grad).all()):
        raise FloatingPointError("chunked CPU attention produced nonfinite outputs or gradients")
    if float(direct_values.grad.abs().sum()) == 0.0:
        raise FloatingPointError("CPU parity fixture did not exercise a nonzero input gradient")

    torch.manual_seed(810208)
    direct_projection = GroupInputProjection(nn.Linear(N_FEATURES, 32)).cpu()
    with torch.no_grad():
        direct_projection.output_projection.weight.normal_(mean=0.0, std=0.02)
    chunked_projection = copy.deepcopy(direct_projection)
    projection_input = torch.randn(3, 5, N_FEATURES)
    direct_projection_input = projection_input.detach().clone().requires_grad_(True)
    chunked_projection_input = projection_input.detach().clone().requires_grad_(True)
    projection_weights = torch.randn(3, 5, 32)
    with patch.object(representations, "GROUP_ATTENTION_ROW_CHUNK", 1000):
        direct_projection_output = direct_projection(direct_projection_input)
    (direct_projection_output * projection_weights).sum().backward()
    with patch.object(representations, "GROUP_ATTENTION_ROW_CHUNK", 7):
        chunked_projection_output = chunked_projection(chunked_projection_input)
    (chunked_projection_output * projection_weights).sum().backward()
    torch.testing.assert_close(
        chunked_projection_output, direct_projection_output, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        chunked_projection_input.grad, direct_projection_input.grad, rtol=3e-5, atol=3e-7
    )
    max_projection_gradient_delta = 0.0
    for (direct_name, direct_parameter), (
        chunked_name,
        chunked_parameter,
    ) in zip(
        direct_projection.named_parameters(), chunked_projection.named_parameters(), strict=True
    ):
        if direct_name != chunked_name:
            raise AssertionError("group projection parameter names differ")
        if direct_parameter.grad is None or chunked_parameter.grad is None:
            raise AssertionError(f"missing group projection gradient for {direct_name}")
        torch.testing.assert_close(
            chunked_parameter.grad,
            direct_parameter.grad,
            rtol=3e-5,
            atol=3e-7,
        )
        max_projection_gradient_delta = max(
            max_projection_gradient_delta,
            float((chunked_parameter.grad - direct_parameter.grad).abs().max().detach()),
        )
    return {
        "mode": "cpu-parity",
        "device": "cpu",
        "rows": rows,
        "tokens_per_row": 16,
        "width": GROUP_TOKEN_WIDTH,
        "chunk_rows": chunk_rows,
        "chunks": math.ceil(rows / chunk_rows),
        "max_output_abs_delta": float((chunked - direct).abs().max().detach()),
        "max_input_gradient_abs_delta": float(
            (chunked_values.grad - direct_values.grad).abs().max().detach()
        ),
        "max_parameter_gradient_abs_delta": max_parameter_gradient_delta,
        "full_projection_rows": 15,
        "full_projection_forced_chunk_rows": 7,
        "full_projection_max_output_abs_delta": float(
            (chunked_projection_output - direct_projection_output).abs().max().detach()
        ),
        "full_projection_max_input_gradient_abs_delta": float(
            (chunked_projection_input.grad - direct_projection_input.grad).abs().max().detach()
        ),
        "full_projection_max_parameter_gradient_abs_delta": max_projection_gradient_delta,
        "all_outputs_and_gradients_finite": True,
    }


def _cuda_full_shape() -> dict[str, Any]:
    """Run one component-only B=256,T=256 forward/backward on an available CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("cuda-full requires an available CUDA device")
    torch.manual_seed(810207)
    device = torch.device("cuda", torch.cuda.current_device())
    projection = GroupInputProjection(nn.Linear(N_FEATURES, 768)).to(device)
    # Make the new branch active so backward exercises its attention parameters.
    with torch.no_grad():
        projection.output_projection.weight.normal_(mean=0.0, std=0.02)
        projection.output_projection.bias.zero_()
    if projection.attention.dropout != 0.0:
        raise RuntimeError("component stress requires the production dropout=0 attention")

    values = torch.randn(256, 256, N_FEATURES, device=device, dtype=torch.float32)
    values.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = projection(values)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    attention_gradient = projection.attention.in_proj_weight.grad
    if attention_gradient is None or not bool(torch.isfinite(attention_gradient).all()):
        raise FloatingPointError("full-shape attention backward did not produce finite gradients")
    if float(attention_gradient.abs().sum()) <= 0:
        raise FloatingPointError("full-shape attention gradient is unexpectedly zero")
    if values.grad is None or not bool(torch.isfinite(values.grad).all()):
        raise FloatingPointError("full-shape input backward did not produce finite gradients")
    if not bool(torch.isfinite(output).all() and torch.isfinite(loss)):
        raise FloatingPointError("full-shape group projection produced nonfinite values")

    rows = 256 * 256
    return {
        "mode": "cuda-full-component-only",
        "device": torch.cuda.get_device_name(device),
        "shape": [256, 256, N_FEATURES],
        "flattened_rows": rows,
        "row_chunk_limit": GROUP_ATTENTION_ROW_CHUNK,
        "attention_heads": projection.attention.num_heads,
        "attention_chunk_count": math.ceil(rows / GROUP_ATTENTION_ROW_CHUNK),
        "attention_dropout": projection.attention.dropout,
        "autocast_dtype": "bfloat16",
        "loss": float(loss.detach()),
        "attention_gradient_l1": float(attention_gradient.abs().sum().detach()),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "all_outputs_and_gradients_finite": True,
        "full_transformer_or_dataset_used": False,
    }


def main() -> None:
    """Run one bounded component check and write a new JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-parity", "cuda-full"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=73)
    parser.add_argument("--chunk-rows", type=int, default=11)
    args = parser.parse_args()
    result = (
        _cpu_parity(args.rows, args.chunk_rows) if args.mode == "cpu-parity" else _cuda_full_shape()
    )
    result["schema"] = "impact81.group-attention-chunk-stress.v1"
    result["torch_version"] = torch.__version__
    result["row_chunk_limit_in_production"] = GROUP_ATTENTION_ROW_CHUNK
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
