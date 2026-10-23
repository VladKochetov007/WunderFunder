"""Separate optimizer-phase and complete-window timing on the receiving host."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def timing_summary(
    directory: Path, config: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    """Summarize heterogeneous JSON receipts without importing model code."""
    windows, optimizers, files = [], [], {}
    for rank in range(8):
        path = directory / "profile-eight" / f"rank{rank}.jsonl"
        if path.stat().st_size > 32_000_000:
            raise ValueError("Oversized engineering rank log")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["step"] for row in records] != list(range(1, 13)):
            raise ValueError("Incomplete rank progress")
        if not all(
            row["rank"] == rank and math.isfinite(row["gradient_norm_preclip"]) for row in records
        ):
            raise ValueError("Invalid profile gradients")
        window = [row["window_seconds"] for row in records[2:]]
        optimizer = [row["optimizer_seconds"] for row in records[2:]]
        if not all(math.isfinite(value) and value > 0 for value in window + optimizer):
            raise ValueError("Missing or invalid CUDA optimizer/window timings")
        windows.append(statistics.median(window))
        optimizers.append(statistics.median(optimizer))
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    count = len(profile["identity"]["data"]["fit_members"])
    steps = config["epochs"] * math.ceil(math.ceil(count / 8) / config["batch_size"])
    steps *= math.ceil(math.ceil(20_000 / config["chunk_rows"]) / config["accumulate_chunks"])
    return {
        "planned_updates": steps,
        "warmup_updates": math.ceil(config["warmup_fraction"] * steps),
        "warmed_median_window_seconds_by_rank": windows,
        "warmed_median_optimizer_seconds_by_rank": optimizers,
        "optimizer_time_extrapolation_hours": max(optimizers) * steps / 3600,
        "training_window_time_extrapolation_hours": max(windows) * steps / 3600,
        "exclusions": "Startup, data reads/transfers, saves, recovery, validation, exports and long-run drift; not an ETA.",
        "rank_logs_sha256": files,
    }
