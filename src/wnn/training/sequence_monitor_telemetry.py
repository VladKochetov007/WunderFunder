from __future__ import annotations

from collections.abc import Mapping

from wnn.training.sequence_monitor_events import MonitorEventWriter


def record_train_step(
    writer: MonitorEventWriter,
    record: Mapping[str, object],
    epoch_index: int,
    epoch_position: float,
    device_type: str,
    cadence: int,
    mask_comparison: str,
) -> None:
    """Append existing objective, preclip, learning-rate, and throughput diagnostics."""
    metrics: dict[str, int | float] = {
        "train/loss": float(record["total_loss"]),
        "train/label_weighted_mse": float(record["label_loss"]),
        "train/kd_loss": float(record["kd_loss"]),
        "train/hint_loss": float(record["hint_loss"]),
        "optimizer/gradient_norm_preclip": float(record["gradient_norm_preclip"]),
        "optimizer/gradient_clip_threshold": float(record["gradient_clip_threshold"]),
        "optimizer/was_clipped": float(record["gradient_clipped"]),
        "timing/window_seconds": float(record["window_seconds"]),
        "timing/optimizer_seconds": float(record["optimizer_seconds"]),
        "throughput/objective_rows_per_second": float(record["valid_rows_per_second"]),
        "throughput/forwarded_rows_per_second": float(record["forwarded_rows_per_second"]),
        "train/selected_rows": int(record["valid_rows"]),
        "train/forwarded_rows": int(record["forwarded_rows"]),
        "train/real_rows": int(record["real_rows"]),
    }
    rates = record.get("learning_rates", {})
    if isinstance(rates, Mapping):
        metrics.update(
            {
                f"optimizer/lr/{str(name).replace('/', '_')}": float(value)
                for name, value in rates.items()
            }
        )
    writer.append(
        "train_step",
        int(record["step"]),
        epoch_position,
        metrics,
        {
            "loss_scope": "online optimizer window on the recipe training mask",
            "epoch_index": epoch_index + 1,
            "comparison": f"train/loss is online; {mask_comparison}",
            "training_telemetry_every_steps": cadence,
            "device_type": device_type,
        },
    )
