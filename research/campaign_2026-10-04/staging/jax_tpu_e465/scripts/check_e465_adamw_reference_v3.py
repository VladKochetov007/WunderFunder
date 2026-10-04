"""Compare the AdamW formula in binary64 without claiming JAX FP32 parity.

This diagnostic reuses the standard-library host recurrence and compares it to
PyTorch's FP64 scalar implementation. A separate FP32 JAX/Torch fixture is
required for production-dtype numerical parity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT / "src"))

from wnn_jax.e465_host_contract import adamw_scalar_step  # noqa: E402
from wnn_jax.onecycle import onecycle_point  # noqa: E402

UPDATES = 7_460
ABSOLUTE_LIMITS = {"parameter": 1e-10, "first_moment": 1e-10, "second_moment": 1e-10}
JAX_SOURCE = STAGE_ROOT / "src" / "wnn_jax" / "rnn.py"
HOST_SOURCE = STAGE_ROOT / "src" / "wnn_jax" / "e465_host_contract.py"
ONECYCLE_SOURCE = STAGE_ROOT / "src" / "wnn_jax" / "onecycle.py"


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_ast_digest(path: Path, function_name: str) -> tuple[str, str]:
    """Return exact function text and normalized AST digests without importing it."""
    source = path.read_text()
    module = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    lines = source.splitlines(keepends=True)
    function_text = "".join(lines[function.lineno - 1 : function.end_lineno])
    normalized = ast.dump(function, include_attributes=False)
    return (
        hashlib.sha256(function_text.encode()).hexdigest(),
        hashlib.sha256(normalized.encode()).hexdigest(),
    )


def assert_formula_binding() -> dict[str, object]:
    """Bind scalar recurrence expressions to the reviewed JAX function source."""
    jax_source = JAX_SOURCE.read_text()
    host_source = HOST_SOURCE.read_text()
    jax_required = (
        "correction1 = 1.0 - beta1**count",
        "correction2 = 1.0 - beta2**count",
        "moment1 = beta1 * state.first[name] + (1.0 - beta1) * gradient",
        "moment2 = beta2 * state.second[name] + (1.0 - beta2) * gradient**2",
        "normalized = (moment1 / correction1) / (jnp.sqrt(moment2 / correction2) + epsilon)",
        "updated[name] = value - learning_rate * (normalized + weight_decay * value)",
    )
    host_required = (
        "m = beta1 * first_moment + (1.0 - beta1) * gradient",
        "v = beta2 * second_moment + (1.0 - beta2) * gradient * gradient",
        "m_hat = m / (1.0 - beta1**step)",
        "v_hat = v / (1.0 - beta2**step)",
        "updated = parameter - learning_rate * (",
    )
    missing = [line for line in jax_required if line not in jax_source]
    missing.extend(line for line in host_required if line not in host_source)
    if missing:
        raise RuntimeError(f"AdamW source formula binding changed: {missing}")
    return {
        "jax_file_sha256": sha256(JAX_SOURCE),
        "jax_function_sha256": function_ast_digest(JAX_SOURCE, "adamw_update")[0],
        "jax_function_ast_sha256": function_ast_digest(JAX_SOURCE, "adamw_update")[1],
        "host_file_sha256": sha256(HOST_SOURCE),
        "host_function_sha256": function_ast_digest(HOST_SOURCE, "adamw_scalar_step")[0],
        "host_function_ast_sha256": function_ast_digest(HOST_SOURCE, "adamw_scalar_step")[1],
        "onecycle_file_sha256": sha256(ONECYCLE_SOURCE),
        "onecycle_function_sha256": function_ast_digest(ONECYCLE_SOURCE, "onecycle_point")[0],
        "onecycle_function_ast_sha256": function_ast_digest(ONECYCLE_SOURCE, "onecycle_point")[1],
        "jax_formula_lines": list(jax_required),
        "host_formula_lines": list(host_required),
        "claim": "same AdamW moment, bias-correction, normalization, and decoupled-decay algebra; not numerical JAX FP32 parity",
    }


def main() -> None:
    """Run all production schedule indices against Torch and host binary64."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    formula_binding = assert_formula_binding()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    parameter = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float64))
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

    for index in range(UPDATES):
        point = onecycle_point(index, UPDATES)
        gradient = math.sin(index * 0.017) + 0.1 * math.cos(index * 0.0031)
        optimizer.param_groups[0]["lr"] = point.learning_rate
        optimizer.param_groups[0]["betas"] = (point.beta1, point.beta2)
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
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

    maxima = {
        "parameter": max_parameter_error,
        "first_moment": max_first_error,
        "second_moment": max_second_error,
    }
    limits_passed = all(maxima[key] <= ABSOLUTE_LIMITS[key] for key in maxima)
    dtype_matches = parameter.dtype == torch.float64
    report = {
        "format": "wnn.e465-dynamic-beta-adamw-formula-reference.v2",
        "status": "passed" if limits_passed and dtype_matches else "failed",
        "updates": UPDATES,
        "torch_version": torch.__version__,
        "jax_imported": False,
        "model_or_data_loaded": False,
        "torch_parameter_dtype": str(parameter.dtype),
        "torch_parameter_dtype_matches_expected": dtype_matches,
        "host_reference_arithmetic": "Python binary64 floats",
        "maximum_absolute_error": maxima,
        "acceptance_limits": ABSOLUTE_LIMITS,
        "selected_steps": checkpoints,
        "formula_binding": formula_binding,
        "source_diagnostic_only": (
            "does not establish FP32 JAX/Torch parity; that is a separate production-dtype check"
        ),
        "optimizer": "torch.optim.AdamW foreach=False fused=False",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not limits_passed or not dtype_matches:
        raise AssertionError(
            f"binary64 dynamic-beta AdamW formula mismatch; report={args.output}, "
            f"maximum_absolute_error={maxima}, dtype_matches={dtype_matches}"
        )


if __name__ == "__main__":
    main()
