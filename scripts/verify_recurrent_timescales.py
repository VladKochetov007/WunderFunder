"""Check constant-gate memory and truncated initializer gradients in a scalar toy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def write_gate_bias(timescale_rows: float) -> float:
    """Return b such that 1-sigmoid(b) = exp(-1/tau) at zero gate input."""
    if not math.isfinite(timescale_rows) or timescale_rows < 1:
        raise ValueError("timescale must be finite and at least one row")
    return math.log(math.expm1(1.0 / timescale_rows))


def initial_state_gradient(retention: float, start: int, stop: int) -> float:
    """Gradient of sum(h[start:stop]) for h[t]=retention*h[t-1], prior h0."""
    return math.fsum(retention ** (row + 1) for row in range(start, stop))


def evaluate() -> dict[str, object]:
    """Compare analytic decay, recurrence and finite differences without fitting."""
    records = []
    for timescale in (1 / math.log(2), 10.0, 100.0, 1000.0, 5000.0, 10000.0):
        bias = write_gate_bias(timescale)
        write = 1 / (1 + math.exp(-bias))
        retention = 1 - write
        assert math.isclose(retention, math.exp(-1 / timescale), abs_tol=2e-16)
        gradient_full = initial_state_gradient(retention, 99, 20000)
        gradient_first_chunk = initial_state_gradient(retention, 99, 2000)
        # The toy loss is linear. Each post-detach chunk has zero derivative
        # with respect to the initial state, even though its value still depends on it.
        epsilon = 1e-3
        losses = []
        for initial in (1 - epsilon, 1 + epsilon):
            hidden = initial
            values = []
            for row in range(20000):
                hidden *= retention
                if row >= 99:
                    values.append(hidden)
            losses.append(math.fsum(values))
        finite_difference = (losses[1] - losses[0]) / (2 * epsilon)
        relative_error = abs(finite_difference - gradient_full) / gradient_full
        assert relative_error < 1e-9
        records.append(
            {
                "timescale_rows": timescale,
                "write_gate_bias": bias,
                "write_fraction_at_zero_input": write,
                "initial_perturbation_after_rows": {
                    str(rows): retention**rows for rows in (100, 200, 2000, 10000, 20000)
                },
                "full_initializer_gradient": gradient_full,
                "first_chunk_gradient_fraction": gradient_first_chunk / gradient_full,
                "finite_difference_relative_error": relative_error,
            }
        )
    return {
        "scope": "scalar constant-gate synthetic algebra only; no model/data quality result",
        "gate_convention": "minGRU write z; retention 1-z (opposite PyTorch GRU update convention)",
        "loss": "sum hidden over positions99..19999; no candidate forcing",
        "tbptt_chunk_rows": 2000,
        "records": records,
    }


def main() -> None:
    """Save the reproducible algebra receipt without replacing earlier evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checked_timescales": len(result["records"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
