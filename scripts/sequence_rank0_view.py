from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from scripts.collect_sequence_monitor import CHUNK_BYTES
from scripts.sequence_monitor_mirror_io import _archive_journal, atomic_bytes

METRICS = {
    "total_loss": "train/loss",
    "label_loss": "train/native_label_objective",
    "kd_loss": "train/kd_loss",
    "hint_loss": "train/hint_loss",
    "composite_wmse": "train/composite_wmse",
    "blend_objective_alpha": "train/blend_alpha",
    "gradient_norm_preclip": "optimizer/gradient_norm_preclip",
    "gradient_clipped": "optimizer/was_clipped",
    "optimizer_seconds": "timing/optimizer_seconds",
    "window_seconds": "timing/window_seconds",
    "valid_rows_per_second": "throughput/objective_rows_per_second",
    "forwarded_rows_per_second": "throughput/forwarded_rows_per_second",
}
NATIVE_OBJECTIVE_METRICS = {
    "objective_wmse": "train/objective_wmse_raw",
    "objective_correlation_loss": "train/objective_correlation_loss_raw",
    "wmse_objective_weight": "train/objective_weight/wmse",
    "correlation_objective_weight": "train/objective_weight/correlation",
    "microbatch_correlation_surrogate_t0": "train/objective_correlation_surrogate/t0",
    "microbatch_correlation_surrogate_t1": "train/objective_correlation_surrogate/t1",
    "correlation_prediction_std_floor": "train/correlation_prediction_std_floor",
}


def _updates_per_epoch(metadata: dict[str, object]) -> int:
    """Reuse the declared padded schedule arithmetic from native fit provenance."""
    config, provenance = metadata["config"], metadata["provenance"]
    data = provenance["data"]
    if data["input_schema"] != "raw112-v1":
        raise ValueError("rank-zero view requires the declared full-sequence raw112 schema")
    members = len(data["fit_members"])
    world = int(provenance["world_size"])
    batch, chunk, accumulation = (
        int(config[key]) for key in ("batch_size", "chunk_rows", "accumulate_chunks")
    )
    if min(members, world, batch, chunk, accumulation) < 1:
        raise ValueError("rank-zero schedule fields must be positive")
    return math.ceil(math.ceil(members / world) / batch) * math.ceil(
        math.ceil(20_000 / chunk) / accumulation
    )


def update_rank0_view(native: Path, view: Path, previous: dict[str, object]) -> dict[str, object]:
    """Convert actual B300 rank-zero scalars without inventing per-update wall times.

    The separate stream affects charts only, never liveness. Its timestamp is the
    actual sampled source-file mtime and elapsed-time plots explicitly omit it.
    """
    source, metadata_path = native / "rank0.jsonl", native / "run_metadata.json"
    target = view / "monitor_rank0.jsonl"
    if source.is_symlink() or metadata_path.is_symlink() or target.is_symlink():
        raise ValueError("rank-zero mirror paths cannot be symlinks")
    if not source.is_file() or not metadata_path.is_file():
        return previous
    metadata = json.loads(metadata_path.read_bytes())
    updates_per_epoch = _updates_per_epoch(metadata)
    info = source.stat()
    identity = f"{info.st_dev}:{info.st_ino}"
    offset = int(previous.get("offset", 0))
    with source.open("rb") as stream:
        stream.seek(max(0, offset - 64))
        marker = stream.read(min(64, offset)).hex()
        same = (
            previous.get("identity") == identity
            and marker == previous.get("marker")
            and offset <= info.st_size
        )
        if not same:
            offset = 0
        stream.seek(offset)
        payload = stream.read(CHUNK_BYTES)
        payload = payload[: payload.rfind(b"\n") + 1]
        stop = offset + len(payload)
        stream.seek(max(0, stop - 64))
        next_marker = stream.read(min(64, stop)).hex()
    if not same:
        _archive_journal(target)
        atomic_bytes(target, b"")
    events: list[bytes] = []
    for line in payload.splitlines():
        row = json.loads(line)
        metrics = {
            metric: float(row[key])
            for key, metric in METRICS.items()
            if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
        }
        metrics.update(_native_objective_metrics(row, metadata["config"]))
        for name, rate in row.get("learning_rates", {}).items():
            if isinstance(rate, (int, float)) and math.isfinite(float(rate)):
                metrics[f"optimizer/lr/{str(name).replace('/', '_')}"] = float(rate)
        event = {
            "schema": "wnn.monitor.v1",
            "kind": "train_step",
            "step": int(row["step"]),
            "epoch": int(row["step"]) / updates_per_epoch,
            "wall_time": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
            "metrics": metrics,
            "context": {
                "source": "unchanged native rank0.jsonl",
                "source_byte_offset": offset,
                "wall_time_is_update_time": False,
                "wall_time_basis": "actual source file mtime; per-update wall time unavailable",
                "epoch_basis": "native fit-members/config/world-size padded update schedule",
                "loss_scope": "native label or curvature-scaled composite objective, as configured",
                "native_objective": _native_objective_context(row, metadata["config"]),
            },
        }
        events.append((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode())
    if events:
        with target.open("ab") as stream:
            stream.write(b"".join(events))
    return {"identity": identity, "offset": stop, "marker": next_marker}


def _native_objective_metrics(
    row: dict[str, object], config: dict[str, object]
) -> dict[str, float]:
    """Expose native raw components and their weights as separate numeric series."""
    metrics = {
        metric: float(row[source])
        for source, metric in NATIVE_OBJECTIVE_METRICS.items()
        if isinstance(row.get(source), (int, float)) and math.isfinite(float(row[source]))
    }
    kd_weight = config.get("kd_weight", 0.0)
    if isinstance(kd_weight, (int, float)) and math.isfinite(float(kd_weight)):
        metrics["train/objective_weight/kd"] = float(kd_weight)
    hint_weight = config.get("hint_weight", 0.0)
    if isinstance(hint_weight, (int, float)) and math.isfinite(float(hint_weight)):
        metrics["train/objective_weight/hint"] = float(hint_weight)
    return metrics


def _native_objective_context(
    row: dict[str, object], config: dict[str, object]
) -> dict[str, object]:
    """Retain nonnumeric objective semantics needed to interpret native components."""
    fields = (
        "objective_mode",
        "objective_aggregation",
        "label_loss_is_curvature_scaled",
        "correlation_scope",
    )
    context = {key: row[key] for key in fields if isinstance(row.get(key), (str, bool))}
    floor = row.get("correlation_prediction_std_floor")
    if isinstance(floor, (int, float)) and math.isfinite(float(floor)):
        context["correlation_prediction_std_floor"] = float(floor)
    label_scaled = row.get("label_loss_is_curvature_scaled") is True
    context["component_semantics"] = {
        "train/native_label_objective": (
            "curvature-scaled label term" if label_scaled else "native label loss as logged"
        ),
        "train/objective_wmse_raw": "raw WMSE component; apply train/objective_weight/wmse",
        "train/objective_correlation_loss_raw": (
            "raw correlation component; apply train/objective_weight/correlation"
        ),
        "train/kd_loss": f"raw KD term; configured kd_weight={config.get('kd_weight', 0)}",
        "train/hint_loss": f"raw hint term; configured hint_weight={config.get('hint_weight', 0)}",
        "train/loss": "native total_loss; not recomputed from displayed components",
    }
    return context
