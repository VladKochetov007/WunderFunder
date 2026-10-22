"""Unit coverage for the PR36 command and B300 profile admission contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/experiment_drafts/mask-aligned-readout"
sys.path.insert(0, str(PACKET))

from b300_profile import _record  # noqa: E402
from receiver import Cell, fit_command  # noqa: E402


def test_paired_fit_commands_preserve_all_fit_settings_except_mask_arm() -> None:
    """Keep paired commands byte-equivalent outside run name and mask flag."""
    cell = Cell("F0", 0, 1001, "oldz4_h230_parent_f0_s3_r3")
    control = fit_command(cell, "all_valid", Path("fold0.json"))
    treatment = fit_command(cell, "is_scored_weighted", Path("fold0.json"))

    def _without_arm(command: list[str]) -> list[str]:
        """Remove only the arm-specific marker and artifact name."""
        result = list(command)
        if "--mask-weight" in result:
            result.remove("--mask-weight")
        name = result.index("--name")
        del result[name : name + 2]
        return result

    assert _without_arm(control) == _without_arm(treatment)
    assert "--mask-weight" not in control
    assert "--mask-weight" in treatment
    assert "--propensity=" in control and "--valid-true-mask=-1" in control
    assert control[control.index("--precision") + 1] == "fp32"
    assert control[control.index("--checkpoint-epoch") + 1] == "0"


def _profile_output(*, free: int = 150, reserved: int = 800) -> str:
    metrics = {
        "step_wall": 1.2,
        "seq_per_s": 20.0,
        "cuda_peak_allocated_bytes": 700,
        "cuda_peak_reserved_bytes": reserved,
        "cuda_free_memory_bytes": free,
        "cuda_total_memory_bytes": 1000,
    }
    evidence = {
        "valid_for_throughput": True,
        "optimizer_attempts": 4,
        "optimizer_completed_updates": 4,
        "amp_skipped_updates": 0,
        "gradient_finite": True,
        "gradient_stats_steps": 4,
        "loss_checks": 4,
    }
    sample = {
        "device": "NVIDIA B300",
        "device_uuid": "gpu-1234",
        "driver_version": "580.00",
        "precision": "fp32",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast_active_at_capture": False,
        "torch_cuda_version": "12.9",
    }
    runtime = {"process_start": sample, "process_end": sample}
    return "\n".join(
        (
            "PROFILE " + json.dumps(metrics),
            "PROFILE_UPDATE_EVIDENCE " + json.dumps(evidence),
            "PROFILE_EXECUTION_RUNTIME " + json.dumps(runtime),
        )
    )


def test_profile_parser_requires_ten_percent_cuda_headroom() -> None:
    """Reject a bounded profile whose CUDA memory leaves under ten percent free."""
    metrics, evidence, _ = _record(_profile_output())
    assert metrics["cuda_total_memory_bytes"] == 1000
    assert evidence["optimizer_completed_updates"] == 4

    with pytest.raises(ValueError, match="headroom"):
        _record(_profile_output(free=90, reserved=910))
