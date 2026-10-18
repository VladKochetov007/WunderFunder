from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.kalman_local_trend import EMA_ALPHA, channels_for_scope, make_kalman_model
from wnn.resources import configure_strict_fp32
from wnn.training.environment import TrainingEnvironment
from wnn.training.runner import run_training


def main() -> None:
    """Run the shared trainer with the focused Kalman-GRU model factory."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--kalman-arm", choices=("raw", "ema", "fixed", "learned"), default="learned"
    )
    parser.add_argument("--kalman-channels", choices=("pilot4", "all112"), default="pilot4")
    parser.add_argument("--synthetic-smoke", action="store_true")
    options, trainer_args = parser.parse_known_args()
    if options.synthetic_smoke:
        run_synthetic_smoke(options.kalman_arm, options.kalman_channels)
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
    channels = channels_for_scope(options.kalman_channels)
    algorithm = {
        "raw": "capacity_matched_zero_features",
        "ema": "steady_state_ema",
        "fixed": "fixed_local_trend",
        "learned": "learned_local_trend",
    }[options.kalman_arm]
    environment.model_metadata = {
        "kind": "causal_smoother_gru",
        "version": 1,
        "arm": options.kalman_arm,
        "algorithm": algorithm,
        "channel_scope": options.kalman_channels,
        "raw_channels": list(channels),
        "q_r_tying": (
            "not_applicable"
            if options.kalman_arm in ("raw", "ema")
            else "one scalar per local-trend state family shared across selected channels"
        ),
        "feature_order": (
            ["constant_zero_padding"]
            if options.kalman_arm == "raw"
            else [
                "posterior_level",
                "posterior_slope",
                "standardized_innovation",
                "log_innovation_variance",
            ]
        ),
        "feature_width": len(channels) * 4,
        "covariance_update": (
            "Joseph form" if options.kalman_arm in ("fixed", "learned") else "not_applicable"
        ),
        "ema_alpha": EMA_ALPHA if options.kalman_arm == "ema" else None,
    }
    environment.make_model = partial(make_kalman_model, arm=options.kalman_arm, channels=channels)
    run_training(trainer_args, environment=environment)


def run_synthetic_smoke(arm: str, channel_scope: str) -> None:
    """Check causal prefixes, reset reproducibility, finite loss, and Q/R gradients."""
    channels = channels_for_scope(channel_scope)
    model = make_kalman_model(
        {"hidden": 16, "layers": 1, "dropout": 0.0},
        n_in=N_FEATURES + N_EVENT_FEATURES,
        n_out=2,
        arm=arm,
        channels=channels,
    )
    model.eval()
    inputs = torch.randn(2, 24, N_FEATURES + N_EVENT_FEATURES, dtype=torch.float32)
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
        gradient = model.features.log_q_level.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("learned process variance did not receive a finite gradient")
    print(f"KALMAN_SMOKE arm={arm} causal_prefix=PASS reset=PASS finite=PASS")


if __name__ == "__main__":
    main()
