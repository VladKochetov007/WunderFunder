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
            },
        }
        events.append((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode())
    if events:
        with target.open("ab") as stream:
            stream.write(b"".join(events))
    return {"identity": identity, "offset": stop, "marker": next_marker}
