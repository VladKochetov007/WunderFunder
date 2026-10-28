from __future__ import annotations

import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packet_contract import file_hash, normalized_config, read, require, write_new
from packet_lineage import validate_cache, validate_identity

# Runtime receipts contain heterogeneous nested JSON from the canonical trainer.


def positive(value: object) -> bool:
    """Accept only finite positive real resource counters."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def profile(root: Path, cell: str) -> dict[str, Any]:
    """Validate twelve real updates and their immutable per-rank diagnostic logs."""
    directory = root / cell / "profile"
    value = read(directory / "profile_receipt.json")
    require(value.get("receipt_schema") == "wnn.sequence-profile.v1", "Wrong profile schema")
    require(
        value.get("status") == "profile_complete" and value.get("updates") == 12,
        "Need twelve completed updates",
    )
    require(value.get("approved") is False, "Raw profile must remain unmodified and unapproved")
    require(value.get("errors") == [] and value.get("error_count") == 0, "Profile has errors")
    require(value.get("scientific_checkpoint_written") is False, "Profile wrote scientific weights")
    require(
        all(
            positive(value.get(key))
            for key in ("elapsed_seconds", "forwarded_rows", "real_rows", "valid_rows")
        ),
        "Incomplete profile work",
    )
    require(
        value["forwarded_rows"] >= value["real_rows"] >= value["valid_rows"], "Invalid work counts"
    )
    losses = value.get("losses", {})
    require(set(losses) == {"label", "distillation", "hint", "total"}, "Missing profile losses")
    require(
        all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in losses.values()),
        "Invalid profile losses",
    )
    identity = value["identity"]
    validate_identity(root, cell, identity)
    require(
        value.get("model_parameters") == identity.get("model_parameters"),
        "Parameter identity differs",
    )
    if cell.startswith("kd_"):
        require(
            identity["data"].get("teacher") == validate_cache(root, cell),
            "Profile used another teacher",
        )
    config = normalized_config(cell)
    runtime = identity["runtime"]
    devices = runtime.get("devices", [])
    require(
        len(devices) == 2 and all("B300" in str(item.get("name", "")).upper() for item in devices),
        "Two B300 GPUs required",
    )
    precision = runtime.get("precision", {})
    require(
        precision.get("mode") == "bf16_bf16x9"
        and precision.get("cublas_emulate_single_precision") == "1",
        "Wrong precision runtime",
    )
    for name in (
        "parameter_dtype",
        "gradient_dtype",
        "optimizer_state_dtype",
        "activation_cache_dtype",
    ):
        require(precision.get(name) == "float32", "FP32 master/state/cache required")
    require(
        runtime.get("matmul_tf32") is False and runtime.get("cudnn_tf32") is False,
        "TF32 must remain disabled",
    )
    require(runtime.get("matmul_precision") == "highest", "Wrong matmul control")
    require(
        runtime.get("autocast") is False and runtime.get("amp_scaler") is False,
        "Unexpected global AMP state",
    )
    require(runtime.get("cublas_workspace_config") == ":4096:8", "Wrong cuBLAS workspace")
    require(
        runtime.get("attention_backend") == config["model"]["attention_backend"],
        "Wrong attention runtime",
    )
    require(
        runtime.get("compiler", {}).get("enabled") == config["model"]["compile_blocks"],
        "Wrong compiler runtime",
    )
    for rank in range(2):
        records = [
            json.loads(line) for line in (directory / f"rank{rank}.jsonl").read_text().splitlines()
        ]
        require([row["step"] for row in records] == list(range(1, 13)), "Incomplete rank logs")
        require(
            all(
                positive(row["learning_rate"]) and math.isfinite(row["gradient_norm_preclip"])
                for row in records
            ),
            "Invalid rank LR/gradient",
        )
    return value


def admit(
    root: Path, cell: str, reviewer: str, host_ram_gib: float, max_vram_fraction: float
) -> None:
    """Approve a fresh copy after explicit resource review; preserve the raw receipt.

    Args:
        root: Prepared external run root.
        cell: One frozen scientific fit identifier.
        reviewer: Person who inspected the exact profile and resource budgets.
        host_ram_gib: RAM actually available to this two-rank job, excluding other jobs.
        max_vram_fraction: Maximum per-rank reserved fraction, at most 0.8.
    """
    require(bool(reviewer.strip()), "Name the receiving reviewer")
    require(positive(host_ram_gib), "Specify the actual allocated host RAM")
    require(
        positive(max_vram_fraction) and max_vram_fraction <= 0.8, "Keep at least 20% VRAM headroom"
    )
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    budget = host_ram_gib * 2**30
    require(budget <= physical, "Declared job RAM exceeds physical host RAM")
    value = profile(root, cell)
    memory = value["memory_by_rank"]
    devices = value["identity"]["runtime"]["devices"]
    require([item.get("rank") for item in memory] == [0, 1], "Missing rank memory records")
    for rank, item in enumerate(memory):
        require(
            all(
                positive(item.get(key))
                for key in (
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "device_total_memory_bytes",
                    "host_peak_rss_bytes",
                )
            ),
            "Invalid resource counter",
        )
        require(
            item["peak_allocated_bytes"]
            <= item["peak_reserved_bytes"]
            <= max_vram_fraction * item["device_total_memory_bytes"],
            "Insufficient GPU memory headroom",
        )
        require(
            devices[rank].get("rank") == rank
            and devices[rank].get("total_memory_bytes") == item["device_total_memory_bytes"],
            "Device receipt differs",
        )
    parameters = value["model_parameters"]["total"]
    require(type(parameters) is int and parameters > 0, "Missing parameter count")
    host_floor = sum(item["host_peak_rss_bytes"] for item in memory) + 12 * parameters * 2
    require(host_floor <= 0.8 * budget, "RAM cannot cover conservative two-rank resume staging")
    data = value["identity"]["data"]
    prediction_members = len(data["evaluation_members"])
    if cell.startswith("teacher_"):
        prediction_members += len(data["fit_members"])
    scratch_floor = 32 * parameters + 160_000 * prediction_members + 10 * 2**30
    require(
        shutil.disk_usage(root).free >= scratch_floor,
        "Insufficient scratch for safe checkpoints and exports",
    )
    directory = root / cell
    raw = directory / "profile/profile_receipt.json"
    approved = value | {
        "approved": True,
        "admission": {
            "schema": "wnn.source-kd-profile-admission.v1",
            "cell": cell,
            "reviewer": reviewer.strip(),
            "reviewed_at": datetime.now(UTC).isoformat(),
            "raw_profile_sha256": file_hash(raw),
            "rank_log_sha256": {
                str(rank): file_hash(directory / f"profile/rank{rank}.jsonl") for rank in range(2)
            },
            "packet_sha256": file_hash(root / "packet.json"),
            "host_ram_budget_bytes": budget,
            "physical_host_ram_bytes": physical,
            "resume_host_floor_bytes": host_floor,
            "max_vram_fraction": max_vram_fraction,
            "scratch_floor_bytes": scratch_floor,
            "limits": "Engineering admission only. No quality, long-run stability or significance claim.",
        },
    }
    write_new(directory / "approved_profile.json", approved)


def verify_approval(root: Path, cell: str) -> None:
    """Recheck raw evidence and exact approval binding immediately before a fit."""
    raw = profile(root, cell)
    approved = read(root / cell / "approved_profile.json")
    require(approved.get("approved") is True, "Explicit review required")
    admission = approved["admission"]
    require(admission.get("cell") == cell, "Approval belongs to another cell")
    require(
        admission.get("raw_profile_sha256")
        == file_hash(root / cell / "profile/profile_receipt.json"),
        "Raw profile changed",
    )
    require(
        admission.get("packet_sha256") == file_hash(root / "packet.json"), "Preparation changed"
    )
    require(
        {key: value for key, value in approved.items() if key not in ("approved", "admission")}
        == {key: value for key, value in raw.items() if key != "approved"},
        "Approved copy differs from raw profile",
    )
    for rank in range(2):
        require(
            admission["rank_log_sha256"][str(rank)]
            == file_hash(root / cell / f"profile/rank{rank}.jsonl"),
            "Rank profile log changed",
        )
    require(
        shutil.disk_usage(root).free >= admission["scratch_floor_bytes"],
        "Scratch budget no longer fits",
    )
