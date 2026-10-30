"""Checkpoint-derived causal input configuration for generic prediction exports."""

from __future__ import annotations

from collections.abc import Mapping

from wnn.training.environment import TrainingEnvironment


def configure_prediction_inputs(
    environment: TrainingEnvironment, checkpoint_args: Mapping[str, object]
) -> None:
    """Select the checkpoint's causal input schema without changing trainer globals."""
    if not isinstance(checkpoint_args, Mapping):
        raise TypeError("checkpoint args must be a mapping")
    environment.inputs.tiedlead = checkpoint_args.get("enc", "linear") == "tiedlead"
