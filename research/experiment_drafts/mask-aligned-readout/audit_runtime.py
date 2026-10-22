"""B300 hardware observation used by receiver-side fit and profile gates."""

from __future__ import annotations

# Any represents dynamic runtime telemetry that is validated field by field.
from typing import Any


def require_b300_runtime() -> dict[str, Any]:
    """Capture current observer hardware; this is not trainer-process evidence."""
    import torch

    from wnn.training.runtime_provenance import capture_execution_runtime

    if not torch.cuda.is_available():
        raise RuntimeError("B300 receiver requires an available CUDA device")
    return capture_execution_runtime(torch.device("cuda"), None, "receiver_observer")


def require_same_device(training: dict[str, Any], observed: dict[str, Any]) -> None:
    """Require a receiver observation to match the trainer's recorded GPU."""
    fields = (
        "device",
        "device_uuid",
        "nvidia_smi_device_uuid",
        "driver_version",
        "compute_capability",
        "torch_version",
        "torch_cuda_version",
    )
    if any(training.get(key) != observed.get(key) for key in fields):
        raise ValueError("trainer-process hardware differs from the receiver's observed B300")
