"""Unit coverage for the PR36 command and B300 profile admission contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/experiment_drafts/mask-aligned-readout"
sys.path.insert(0, str(PACKET))

from audit_validation import validate_fixed_endpoint  # noqa: E402
from b300_profile import _record, profile_command  # noqa: E402
from receiver import Cell, fit_command  # noqa: E402
from receiver_assets import RECIPE_PATH, load_json, recipe_cells, select_cells  # noqa: E402
from receiver_lock import claim_receiver_stage  # noqa: E402


def test_paired_fit_commands_preserve_all_fit_settings_except_mask_arm() -> None:
    """Keep paired commands byte-equivalent outside run name and mask flag."""
    cell = Cell("F0", 0, 1001, "oldz4_h230_parent_f0_s3_r3", 3, 2)
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
    assert control[control.index("--epochs") + 1] == "3"
    assert control[control.index("--checkpoint-epoch") + 1] == "2"
    assert "_r2_f0_s1001_e3_ep2_all_valid" in control[control.index("--name") + 1]


def test_profile_command_uses_recipe_endpoint_and_bounded_profile_flags() -> None:
    """Keep the bounded profile on the amended fit command and its fixed endpoint."""
    cell = recipe_cells(load_json(RECIPE_PATH))[0]
    command = profile_command(cell, Path("fold0.json"), "profile_name")

    assert command[command.index("--epochs") + 1] == "3"
    assert command[command.index("--checkpoint-epoch") + 1] == "2"
    assert command[command.index("--profile-steps") + 1] == "4"
    assert command[command.index("--profile-warmup") + 1] == "1"
    assert "--grad-stats" in command


def test_recipe_selects_independent_paired_fit_workers() -> None:
    """Expose one complete control/treatment cell to each receiver worker."""
    cells = recipe_cells(load_json(RECIPE_PATH))

    assert [cell.fold for cell in select_cells(cells, "F0")] == ["F0"]
    assert [cell.fold for cell in select_cells(cells, "F2")] == ["F2"]
    assert select_cells(cells, None) == cells
    with pytest.raises(ValueError, match="not present"):
        select_cells(cells, "F1")


def test_receiver_cell_lock_rejects_duplicate_worker_and_releases_on_exit(tmp_path: Path) -> None:
    """Prevent two cell workers from racing on the same receipts and run names."""
    with (
        claim_receiver_stage(tmp_path, "fit", "F0"),
        pytest.raises(FileExistsError, match="lock exists"),
        claim_receiver_stage(tmp_path, "fit", "F0"),
    ):
        pass

    with claim_receiver_stage(tmp_path, "fit", "F0"):
        pass


def test_fixed_endpoint_requires_all_three_complete_epochs(tmp_path: Path) -> None:
    """Reject a fixed epoch-2 checkpoint without three full no-eval epochs."""
    cell = recipe_cells(load_json(RECIPE_PATH))[0]
    history_path = tmp_path / "artifacts" / "runs" / cell.run_name("all_valid") / "history.json"
    history_path.parent.mkdir(parents=True)
    attempts_per_epoch = 20
    completed_history = [
        {
            "epoch": epoch,
            "evaluation": "withheld",
            "fit_summary": {
                "optimizer_windows": attempts_per_epoch,
                "optimizer_updates": {
                    "optimizer_attempts": attempts_per_epoch,
                    "optimizer_completed_updates": attempts_per_epoch,
                    "amp_skipped_updates": 0,
                },
            },
        }
        for epoch in range(3)
    ]
    history_path.write_text(json.dumps(completed_history), encoding="utf-8")
    split = SimpleNamespace(valid_groups=list(range(64)))
    assets = SimpleNamespace(root=tmp_path, splits={cell.fold: split})
    checkpoint = {
        "epoch": 2,
        "args": {
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 2,
            "epochs": 3,
            "max_batches": 0,
            "profile_steps": 0,
            "no_eval_during_fit": True,
        },
    }
    diagnostics = [
        {
            "epoch_index": epoch,
            "normalization_identity": "parent-norm",
            "epoch": {
                "optimizer_attempts": attempts_per_epoch,
                "optimizer_completed_updates": attempts_per_epoch,
                "amp_skipped_updates": 0,
                "completed_batches": 2,
            },
        }
        for epoch in range(3)
    ]
    provenance = {
        "selected_epoch": 2,
        "training_clocks": {"update_rows": 2000},
        "normalization_identity": "parent-norm",
        "update_diagnostics": diagnostics,
    }

    result = validate_fixed_endpoint(cell, "all_valid", assets, checkpoint, provenance)
    assert result["epochs"] == 3
    assert result["checkpoint_epoch"] == 2
    assert result["expected_optimizer_attempts_per_epoch"] == attempts_per_epoch
    assert result["expected_total_optimizer_attempts"] == 3 * attempts_per_epoch

    incomplete = {**provenance, "update_diagnostics": diagnostics[:2]}
    with pytest.raises(ValueError, match="lacks all 3 fit epochs"):
        validate_fixed_endpoint(cell, "all_valid", assets, checkpoint, incomplete)

    history_path.write_text(json.dumps(completed_history[:2]), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain all 3 epochs"):
        validate_fixed_endpoint(cell, "all_valid", assets, checkpoint, provenance)


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
