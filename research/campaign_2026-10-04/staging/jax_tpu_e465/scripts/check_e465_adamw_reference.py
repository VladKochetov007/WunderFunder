"""Bounded scalar parity check for dynamic-beta AdamW against PyTorch.

This fixture imports PyTorch only when executed. It does not import JAX, build
a model, load data, or execute a training batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from wnn_jax.e465_host_contract import adamw_scalar_step  # noqa: E402
from wnn_jax.onecycle import onecycle_point  # noqa: E402


def main() -> None:
    """Compare every scalar update and moment to production torch.optim.AdamW."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=7_460)
    args = parser.parse_args()
    if args.updates != 7_460:
        parser.error("registered fixture must cover all 7460 used scheduler indices")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    parameter = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1e-3 / 25.0,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    first = 0.0
    second = 0.0
    reference_parameter = float(parameter.detach())
    max_parameter_error = 0.0
    max_first_error = 0.0
    max_second_error = 0.0
    checkpoints: list[dict[str, float | int]] = []

    for index in range(args.updates):
        point = onecycle_point(index, args.updates)
        gradient = math.sin(index * 0.017) + 0.1 * math.cos(index * 0.0031)
        optimizer.param_groups[0]["lr"] = point.learning_rate
        optimizer.param_groups[0]["betas"] = (point.beta1, point.beta2)
        parameter.grad = torch.tensor(gradient, dtype=torch.float32)
        optimizer.step()

        reference_parameter, first, second = adamw_scalar_step(
            reference_parameter,
            gradient,
            first,
            second,
            index + 1,
            learning_rate=point.learning_rate,
            beta1=point.beta1,
            beta2=point.beta2,
            weight_decay=0.01,
            epsilon=1e-8,
        )
        state = optimizer.state[parameter]
        actual_parameter = float(parameter.detach())
        actual_first = float(state["exp_avg"])
        actual_second = float(state["exp_avg_sq"])
        errors = (
            abs(actual_parameter - reference_parameter),
            abs(actual_first - first),
            abs(actual_second - second),
        )
        max_parameter_error = max(max_parameter_error, errors[0])
        max_first_error = max(max_first_error, errors[1])
        max_second_error = max(max_second_error, errors[2])
        if index in {0, 371, 372, 3_729, 7_458, 7_459}:
            checkpoints.append(
                {
                    "index": index,
                    "learning_rate": point.learning_rate,
                    "beta1": point.beta1,
                    "parameter_error": errors[0],
                    "first_moment_error": errors[1],
                    "second_moment_error": errors[2],
                }
            )

    # PyTorch stores moments and parameters in FP32; the host reference uses
    # Python doubles, so the tolerances bound only accumulated FP32 rounding.
    limits = {"parameter": 2e-5, "first_moment": 2e-6, "second_moment": 2e-6}
    if (
        max_parameter_error > limits["parameter"]
        or max_first_error > limits["first_moment"]
        or max_second_error > limits["second_moment"]
    ):
        raise AssertionError(
            f"dynamic-beta AdamW mismatch: parameter={max_parameter_error}, "
            f"m={max_first_error}, v={max_second_error}"
        )
    report = {
        "format": "wnn.e465-dynamic-beta-adamw-reference",
        "status": "passed",
        "updates": args.updates,
        "torch_version": torch.__version__,
        "jax_imported": False,
        "model_or_data_loaded": False,
        "maximum_absolute_error": {
            "parameter": max_parameter_error,
            "first_moment": max_first_error,
            "second_moment": max_second_error,
        },
        "acceptance_limits": limits,
        "selected_steps": checkpoints,
        "bias_correction": "current beta1 raised to one-based step; beta2 likewise",
        "optimizer": "torch.optim.AdamW foreach=False fused=False",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
