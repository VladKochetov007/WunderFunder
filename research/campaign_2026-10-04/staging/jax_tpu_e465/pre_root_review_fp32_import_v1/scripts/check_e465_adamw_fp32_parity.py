"""Compare the actual staged JAX FP32 AdamW update to Torch's FP32 update.

Run only in the sealed private CPU/JAX environment. This file is source-only
until root seals it; do not import JAX or execute it in the local environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

import numpy as np  # noqa: E402
from wnn_jax.onecycle import onecycle_point  # noqa: E402
from wnn_jax.rnn import adamw_init, adamw_update  # noqa: E402

UPDATES = 7_460
# These are explicitly FP32 roundoff limits, distinct from the 1e-10 formula test.
ABSOLUTE_LIMITS = {"parameter": 5e-6, "first_moment": 1e-6, "second_moment": 1e-5}
RNN_SOURCE = STAGE_ROOT / "src" / "wnn_jax" / "rnn.py"
ONECYCLE_SOURCE = STAGE_ROOT / "src" / "wnn_jax" / "onecycle.py"


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run one compiled scalar/multi-parameter scan and compare every state step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import torch

    torch.set_num_threads(1)
    if jax.default_backend() != "cpu":
        raise RuntimeError("reference fixture requires CPU JAX")
    schedule = [onecycle_point(index, UPDATES) for index in range(UPDATES)]
    learning_rate_values = np.asarray([point.learning_rate for point in schedule], dtype=np.float32)
    beta1_values = np.asarray([point.beta1 for point in schedule], dtype=np.float32)
    beta2_values = np.asarray([point.beta2 for point in schedule], dtype=np.float32)
    learning_rates = jnp.asarray(learning_rate_values)
    beta1s = jnp.asarray(beta1_values)
    beta2s = jnp.asarray(beta2_values)
    gradients = np.asarray(
        [
            [
                math.sin(index * 0.017) + 0.1 * math.cos(index * 0.0031),
                math.cos(index * 0.011) - 0.2 * math.sin(index * 0.0023),
            ]
            for index in range(UPDATES)
        ],
        dtype=np.float32,
    )
    jax_gradients = jnp.asarray(gradients)
    initial_params = {
        "weight_a": jnp.asarray(0.37, dtype=jnp.float32),
        "weight_b": jnp.asarray(-0.125, dtype=jnp.float32),
        "mean": jnp.asarray([0.125, -0.5], dtype=jnp.float32),
        "std": jnp.asarray([1.25, 0.75], dtype=jnp.float32),
    }
    initial_state = adamw_init(initial_params)

    def update(carry, values):
        params, state = carry
        learning_rate, beta1, beta2, gradient = values
        grads = {"weight_a": gradient[0], "weight_b": gradient[1]}
        next_params, next_state = adamw_update(
            params,
            grads,
            state,
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta1=beta1,
            beta2=beta2,
            epsilon=1e-8,
        )
        observed = (
            next_params["weight_a"],
            next_params["weight_b"],
            next_state.first["weight_a"],
            next_state.first["weight_b"],
            next_state.second["weight_a"],
            next_state.second["weight_b"],
        )
        return (next_params, next_state), observed

    compiled_scan = jax.jit(
        lambda params, state, lr, b1, b2, grads: jax.lax.scan(
            update, (params, state), (lr, b1, b2, grads)
        )
    )
    (final_params, final_state), trajectory = compiled_scan(
        initial_params, initial_state, learning_rates, beta1s, beta2s, jax_gradients
    )
    jax_values = np.stack([np.asarray(value, dtype=np.float32) for value in trajectory], axis=1)

    parameters = [
        torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor(-0.125, dtype=torch.float32)),
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=schedule[0].learning_rate,
        betas=(schedule[0].beta1, schedule[0].beta2),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    torch_values = np.empty((UPDATES, 6), dtype=np.float32)
    for index in range(UPDATES):
        optimizer.param_groups[0]["lr"] = float(learning_rate_values[index])
        optimizer.param_groups[0]["betas"] = (
            float(beta1_values[index]),
            float(beta2_values[index]),
        )
        for parameter, gradient in zip(parameters, gradients[index], strict=True):
            parameter.grad = torch.tensor(float(gradient), dtype=torch.float32)
        optimizer.step()
        for parameter_index, parameter in enumerate(parameters):
            state = optimizer.state[parameter]
            torch_values[index, parameter_index] = float(parameter.detach())
            torch_values[index, 2 + parameter_index] = float(state["exp_avg"])
            torch_values[index, 4 + parameter_index] = float(state["exp_avg_sq"])

    labels = (
        "weight_a",
        "weight_b",
        "first_a",
        "first_b",
        "second_a",
        "second_b",
    )
    error_by_value = np.max(np.abs(jax_values - torch_values), axis=0)
    maxima = {
        key: float(np.max(error_by_value[indices]))
        for key, indices in {
            "parameter": (0, 1),
            "first_moment": (2, 3),
            "second_moment": (4, 5),
        }.items()
    }
    if any(maxima[key] > ABSOLUTE_LIMITS[key] for key in maxima):
        raise AssertionError(f"actual FP32 JAX/Torch AdamW parity failed: {maxima}")
    if int(final_state.count) != UPDATES:
        raise AssertionError(f"JAX AdamW count is {int(final_state.count)}, expected {UPDATES}")
    for name in ("mean", "std"):
        if not np.array_equal(np.asarray(final_params[name]), np.asarray(initial_params[name])):
            raise AssertionError(f"AdamW modified frozen normalization buffer {name}")
    if set(final_state.first) != {"weight_a", "weight_b"} or set(final_state.second) != {
        "weight_a",
        "weight_b",
    }:
        raise AssertionError("AdamW moments included frozen normalization buffers")
    report = {
        "format": "wnn.e465-dynamic-beta-adamw-fp32-parity.v1",
        "status": "passed",
        "updates": UPDATES,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "torch_version": torch.__version__,
        "jax_imported": True,
        "model_or_data_loaded": False,
        "comparison": "compiled lax.scan around actual staged rnn.adamw_update vs scalar Torch AdamW",
        "parameter_moment_order": list(labels),
        "maximum_absolute_error": maxima,
        "acceptance_limits": ABSOLUTE_LIMITS,
        "jax_update_count": int(final_state.count),
        "frozen_buffers_unchanged": True,
        "frozen_buffers": ["mean", "std"],
        "dtype": "FP32 parameters, gradients, and moments in both implementations",
        "learning_rate_beta_schedule": (
            "same FP32-quantized standard-library OneCycle values for used indices 0..7459"
        ),
        "source_sha256": {
            "rnn.py": sha256(RNN_SOURCE),
            "onecycle.py": sha256(ONECYCLE_SOURCE),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
