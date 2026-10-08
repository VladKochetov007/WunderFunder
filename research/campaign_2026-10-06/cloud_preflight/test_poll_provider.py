"""Focused validation for the finite Kaggle provider target resource labels."""

from __future__ import annotations

import pytest
from poll_provider import VALID_RESOURCES


def test_cpu_kernel_can_be_monitored_without_gpu_attribution() -> None:
    """CPU-only kernel targets must not be mislabeled as GPU workers."""
    assert "cpu" in VALID_RESOURCES


@pytest.mark.parametrize("resource", ["gpu", "tpu"])
def test_accelerator_resources_remain_supported(resource: str) -> None:
    """Existing GPU and TPU provider targets keep their resource labels."""
    assert resource in VALID_RESOURCES


def test_unknown_resource_is_rejected() -> None:
    """A typo must not silently enter the active polling set."""
    assert "accelerator" not in VALID_RESOURCES
