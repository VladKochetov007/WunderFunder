"""B300 parity gate for the isolated Transformer and GRU control engines."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from impact81.receiver_loop_support import (
    TOLERANCE,
    MemoryRankLog,
    SyntheticDataset,
    compare,
    compare_update_record,
)

# Canonical runtime values and config dictionaries are dynamically imported after pin admission.
SEED = 204101


def _setup(
    artifact: dict[str, Any], artifact_sha256: str, device: torch.device, family: str
) -> tuple[dict[str, Any], Any, Callable[[], tuple[nn.Module, torch.optim.Optimizer, Any]], int]:
    """Load the exact card04 control recipe and make a small B300-only probe config."""
    from impact81.bindings import PACKET, load_cell
    from impact81.modeling import build_model
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_distributed import DistributedContext
    from wnn.training.sequence_fit_utils import make_optimizer, seed_worker, total_updates
    from wnn.training.sequence_precision import configure_precision_mode

    cell_name = "ip09_control_s204101.json" if family == "gru" else "ip04_control_s204101.json"
    cell, native_config = load_cell(PACKET / "configs" / cell_name)
    config = load_config(native_config)
    config.update(
        {
            "seed": SEED,
            "batch_size": 2,
            "chunk_rows": 64,
            "accumulate_chunks": 2,
            "gradient_horizon_rows": 64,
            "epochs": 4,
            "required_world_size": 1,
            "required_device": "B300",
            "optimizer_update_diagnostics_every": None,
        }
    )
    config["model"].update(
        {
            "hidden": 48,
            "layers": 2,
            "heads": 4,
            "ff_multiplier": 2,
            "context_rows": 16,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "compile_blocks": False,
            "attention_backend": "efficient",
        }
    )
    config["impact81"] = {
        "schema": "impact81.model.v1",
        "cell": cell,
        "preprocessing": artifact,
        "preprocessing_sha256": artifact_sha256,
    }
    if family == "gru":
        config.update({"chunk_rows": 1000, "gradient_horizon_rows": 1000, "accumulate_chunks": 1})
        config["model"].update({"hidden": 32, "attention_backend": "math"})
    configure_precision_mode(str(config["precision_mode"]))
    context = DistributedContext(0, int(device.index or 0), 1, device, False)
    schedule_updates = total_updates(config, 2, 1)

    def construct() -> tuple[nn.Module, torch.optim.Optimizer, Any]:
        """Build a matched small model and canonical optimizer from the common seed."""
        seed_worker(SEED, 0)
        model = build_model(config, device)
        optimizer, scheduler = make_optimizer(model, config, schedule_updates)
        return model, optimizer, scheduler

    return config, context, construct, schedule_updates


def _run_one(
    train_epoch: Callable[..., Any],
    dataset: SyntheticDataset,
    config: dict[str, Any],
    context: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[Any, MemoryRankLog]:
    """Capture the result and rank log from a single bounded canonical update."""
    from wnn.training.sequence_fit_utils import seed_worker

    seed_worker(SEED, 0)
    log = MemoryRankLog()
    result = train_epoch(dataset, model, optimizer, scheduler, config, context, log, 0, 0, 1)
    torch.cuda.synchronize(context.device)
    return result, log


def run_loop_checks(
    artifact: dict[str, Any],
    artifact_sha256: str,
    device: torch.device,
    family: str = "transformer",
) -> dict[str, object]:
    """Compare stock and experiment control updates for one pinned family on B300.

    The gate builds the same card04 plain control twice, executes one optimizer update
    through each epoch driver, and compares losses, row counts, logs, parameters,
    optimizer state and scheduler state. It is receiver evidence only, not a quality
    score or a full-fit/profile approval.
    """
    if family not in ("transformer", "gru"):
        raise ValueError("unknown control-engine receiver family")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("control-engine parity requires the actual receiver CUDA device")
    device_name = torch.cuda.get_device_name(device)
    if "B300" not in device_name.upper():
        raise RuntimeError(f"control-engine parity requires B300, got {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("control-engine parity requires CUDA BF16 support")
    if family == "transformer" and os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise RuntimeError("CUBLAS_EMULATE_SINGLE_PRECISION=1 is required for BF16x9")
    if family == "gru" and os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") not in (None, "0"):
        raise RuntimeError("GRU FP32 requires cuBLAS emulation disabled")
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ValueError("preprocessing SHA-256 must be lowercase hexadecimal")

    from impact81.auxiliary_loop import train_epoch as experiment_train_epoch
    from wnn.training.sequence_loop import train_epoch as stock_train_epoch

    config, context, construct, schedule_updates = _setup(artifact, artifact_sha256, device, family)
    dataset = SyntheticDataset()
    stock_model, stock_optimizer, stock_scheduler = construct()
    experiment_model, experiment_optimizer, experiment_scheduler = construct()
    initial_diff = compare(
        experiment_model.state_dict(), stock_model.state_dict(), "initial model state"
    )
    if len(dataset) != 2 or dataset.valid_rows != 39_802:
        raise AssertionError(
            "synthetic receiver data must be two full sequences with 99 warmup rows"
        )

    stock_result, stock_log = _run_one(
        stock_train_epoch,
        dataset,
        config,
        context,
        stock_model,
        stock_optimizer,
        stock_scheduler,
    )
    experiment_result, experiment_log = _run_one(
        experiment_train_epoch,
        dataset,
        config,
        context,
        experiment_model,
        experiment_optimizer,
        experiment_scheduler,
    )
    if stock_result.updates != 1 or experiment_result.updates != 1:
        raise AssertionError("both control engines must complete exactly one optimizer update")
    if stock_result.complete or experiment_result.complete:
        raise AssertionError("bounded one-update control runs must stop before an epoch boundary")
    if len(stock_log.records) != 1 or len(experiment_log.records) != 1:
        raise AssertionError("each control engine must emit exactly one update record")
    if stock_result.objective_totals is None or experiment_result.objective_totals is None:
        raise AssertionError("control parity requires canonical standalone objective totals")

    result_diff = max(
        compare(experiment_result.totals, stock_result.totals, "epoch sufficient sums"),
        compare(
            experiment_result.objective_totals,
            stock_result.objective_totals,
            "objective sufficient sums",
        ),
    )
    expected_valid, expected_rows = (1802, 2000) if family == "gru" else (58, 256)
    if int(stock_result.totals[3].item()) != expected_valid:
        raise AssertionError("synthetic update post-warmup row count differs")
    if (
        int(stock_result.totals[4].item()) != expected_rows
        or int(stock_result.totals[5].item()) != expected_rows
    ):
        raise AssertionError("synthetic update forwarded row count differs")
    record_diff = compare_update_record(stock_log.records[0], experiment_log.records[0])
    model_diff = compare(
        experiment_model.state_dict(), stock_model.state_dict(), "updated model state"
    )
    optimizer_diff = compare(
        experiment_optimizer.state_dict(), stock_optimizer.state_dict(), "optimizer state"
    )
    scheduler_diff = compare(
        experiment_scheduler.state_dict(), stock_scheduler.state_dict(), "scheduler state"
    )
    maximum = max(
        initial_diff, result_diff, record_diff, model_diff, optimizer_diff, scheduler_diff
    )
    return {
        "status": "passed",
        "scope": "one synthetic update; stock sequence loop versus the selected family control engine",
        "device": device_name,
        "precision_mode": config["precision_mode"],
        "family": family,
        "model": config["model"],
        "optimizer": config["optimizer"],
        "schedule_total_updates": schedule_updates,
        "sequences": len(dataset),
        "full_rows_per_sequence": 20_000,
        "updates_per_engine": 1,
        "valid_rows": int(stock_result.totals[3].item()),
        "forwarded_rows": int(stock_result.totals[4].item()),
        "real_rows": int(stock_result.totals[5].item()),
        "primary_loss": float(stock_log.records[0]["total_loss"]),
        "primary_loss_by_engine": {
            "stock": float(stock_log.records[0]["total_loss"]),
            "experiment_control": float(experiment_log.records[0]["total_loss"]),
        },
        "max_abs_differences": {
            "initial_model": initial_diff,
            "sufficient_sums": result_diff,
            "update_record": record_diff,
            "updated_model": model_diff,
            "optimizer_state": optimizer_diff,
            "scheduler_state": scheduler_diff,
            "overall": maximum,
        },
        "tolerance_atol": TOLERANCE,
        "quality_score": None,
    }
