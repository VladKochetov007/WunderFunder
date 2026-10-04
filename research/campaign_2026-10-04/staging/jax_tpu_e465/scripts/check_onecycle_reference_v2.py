"""Bounded scalar comparison against the repository's PyTorch scheduler."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

STAGE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(STAGE / "src"))
sys.path.insert(0, str(ROOT / "src"))

from wnn_jax.onecycle import onecycle_point  # noqa: E402

LR_REL_TOL = 2e-13
LR_ABS_TOL = 1e-16
BETA_REL_TOL = 2e-13
BETA_ABS_TOL = 1e-15


def _expect_error(callable_: Callable[[], object], exception: type[Exception]) -> None:
    """Require an invalid-input call to fail with the specified exception."""
    try:
        callable_()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def _check_input_contract() -> None:
    """Exercise boundary, nonfinite, and type validation without Torch."""
    defaults = {"step": 0, "total_steps": 7460}
    for name in (
        "max_lr",
        "pct_start",
        "div_factor",
        "final_div_factor",
        "base_momentum",
        "max_momentum",
        "beta2",
    ):
        values = dict(defaults)
        values[name] = math.nan
        _expect_error(lambda values=values: onecycle_point(**values), ValueError)
    _expect_error(lambda: onecycle_point(True, 7460), TypeError)
    _expect_error(lambda: onecycle_point(0, 7460.0), TypeError)
    _expect_error(lambda: onecycle_point(-1, 7460), ValueError)
    _expect_error(lambda: onecycle_point(7461, 7460), ValueError)
    _expect_error(lambda: onecycle_point(0, 1), ValueError)

    start = onecycle_point(0, 7460)
    peak = onecycle_point(372, 7460)
    floor = onecycle_point(7459, 7460)
    unused = onecycle_point(7460, 7460)
    assert math.isclose(start.learning_rate, 0.001 / 25.0, rel_tol=LR_REL_TOL, abs_tol=LR_ABS_TOL)
    assert math.isclose(start.beta1, 0.95, rel_tol=BETA_REL_TOL, abs_tol=BETA_ABS_TOL)
    assert math.isclose(peak.learning_rate, 0.001, rel_tol=LR_REL_TOL, abs_tol=LR_ABS_TOL)
    assert math.isclose(peak.beta1, 0.85, rel_tol=BETA_REL_TOL, abs_tol=BETA_ABS_TOL)
    assert math.isclose(
        floor.learning_rate,
        (0.001 / 25.0) / 100.0,
        rel_tol=LR_REL_TOL,
        abs_tol=LR_ABS_TOL,
    )
    assert math.isclose(floor.beta1, 0.95, rel_tol=BETA_REL_TOL, abs_tol=BETA_ABS_TOL)
    assert unused.learning_rate > floor.learning_rate
    assert unused.beta1 < floor.beta1


def run_referencecheck(output: Path) -> dict[str, object]:
    """Compare all scheduled values against the production builder."""
    _check_input_contract()

    import torch

    from wnn.training_dynamics import build_scheduler

    total_steps = 7460
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    args = SimpleNamespace(
        warmup_frac=None,
        sched="onecycle",
        lr=0.001,
        head_lr_mult=1.0,
        optim="adamw",
        wsd_decay=0.2,
        wsd_final=0.01,
        restarts=1,
    )
    scheduler = build_scheduler(optimizer, args, total_steps)
    max_lr_error = 0.0
    max_beta1_error = 0.0
    max_beta2_error = 0.0
    compared = 0

    for index in range(total_steps + 1):
        reference_lr = float(scheduler.get_last_lr()[0])
        reference_beta1, reference_beta2 = optimizer.param_groups[0]["betas"]
        port = onecycle_point(index, total_steps)
        max_lr_error = max(max_lr_error, abs(reference_lr - port.learning_rate))
        max_beta1_error = max(max_beta1_error, abs(reference_beta1 - port.beta1))
        max_beta2_error = max(max_beta2_error, abs(reference_beta2 - port.beta2))
        if not math.isclose(
            reference_lr, port.learning_rate, rel_tol=LR_REL_TOL, abs_tol=LR_ABS_TOL
        ):
            raise AssertionError(
                f"LR mismatch at index {index}: {reference_lr} != {port.learning_rate}"
            )
        if not math.isclose(
            reference_beta1, port.beta1, rel_tol=BETA_REL_TOL, abs_tol=BETA_ABS_TOL
        ):
            raise AssertionError(
                f"beta1 mismatch at index {index}: {reference_beta1} != {port.beta1}"
            )
        if not math.isclose(reference_beta2, port.beta2, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(
                f"beta2 mismatch at index {index}: {reference_beta2} != {port.beta2}"
            )
        compared += 1
        if index < total_steps:
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()

    if compared != 7461 or scheduler.last_epoch != 7460:
        raise AssertionError("reference did not cover indices 0..7460")
    unexpected = {name for name in sys.modules if name == "jax" or name.startswith("jax.")}
    if unexpected or "train_rnn" in sys.modules or "wnn.data" in sys.modules:
        raise AssertionError(f"reference imported prohibited model/data stack: {unexpected}")
    if output.exists():
        raise FileExistsError(output)
    result: dict[str, object] = {
        "status": "passed",
        "torch_version": torch.__version__,
        "optimizer": "scalar AdamW with zero gradients; no model or data imports",
        "total_optimizer_updates": total_steps,
        "compared_scheduler_indices": [0, total_steps],
        "compared_value_count": compared,
        "warmup_peak_index": 372,
        "final_used_index": 7459,
        "post_last_unused_index": 7460,
        "maximum_absolute_error": {
            "learning_rate": max_lr_error,
            "beta1": max_beta1_error,
            "beta2": max_beta2_error,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the CPU reference and write one no-overwrite result file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_referencecheck(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
