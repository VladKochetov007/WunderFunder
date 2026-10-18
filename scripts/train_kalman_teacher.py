from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch

from wnn.kalman_local_trend import make_kalman_model
from wnn.resources import configure_strict_fp32
from wnn.training.environment import TrainingEnvironment
from wnn.training.runner import run_training


def main() -> None:
    """Run the shared trainer with the focused Kalman-GRU model factory."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kalman-arm", choices=("fixed", "learned"), default="learned")
    parser.add_argument("--synthetic-smoke", action="store_true")
    options, trainer_args = parser.parse_known_args()
    if options.synthetic_smoke:
        run_synthetic_smoke(options.kalman_arm)
        return
    precision = "fp32"
    for index, token in enumerate(trainer_args):
        if token.startswith("--precision="):
            precision = token.partition("=")[2]
            break
        if token == "--precision":
            if index + 1 == len(trainer_args):
                parser.error("--precision needs a value")
            precision = trainer_args[index + 1]
            break
    if precision != "fp32":
        parser.error("the Kalman-GRU scientific adapter requires --precision fp32")
    if "--precision" not in trainer_args and not any(
        token.startswith("--precision=") for token in trainer_args
    ):
        trainer_args.extend(("--precision", "fp32"))
    configure_strict_fp32()
    environment = TrainingEnvironment()
    environment.entrypoint_override = Path(__file__).resolve()
    environment.make_model = partial(make_kalman_model, arm=options.kalman_arm)
    run_training(trainer_args, environment=environment)


def run_synthetic_smoke(arm: str) -> None:
    """Check causal prefixes, reset reproducibility, finite loss, and Q/R gradients."""
    model = make_kalman_model(
        {"hidden": 16, "layers": 1, "dropout": 0.0}, n_in=56, n_out=2, arm=arm
    )
    model.eval()
    inputs = torch.randn(2, 24, 56, dtype=torch.float32)
    changed = inputs.clone()
    changed[:, 12:] += 3.0
    with torch.enable_grad():
        baseline, _ = model(inputs)
        repeated, _ = model(inputs)
        future_changed, _ = model(changed)
        torch.testing.assert_close(baseline, repeated)
        torch.testing.assert_close(baseline[:, :12], future_changed[:, :12])
        loss = baseline.square().mean()
        loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(baseline).all():
        raise RuntimeError("synthetic Kalman-GRU smoke produced nonfinite values")
    if arm == "learned":
        gradient = model.kalman.log_q_level.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("learned process variance did not receive a finite gradient")
    print(f"KALMAN_SMOKE arm={arm} causal_prefix=PASS reset=PASS finite=PASS")


if __name__ == "__main__":
    main()
