from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import (
    LaunchError,
    atomic_write_json,
    owned_process_state,
    read_json,
    read_jsonl_tail,
    sha256_file,
    utc_now,
)

RECEIPT_SCHEMA = "wnn.galam-l4-quality8-launch.v1"


def refresh_summary(receipt_path: Path) -> dict[str, Any]:
    """Write a terminal summary from fit artifacts and receipt-owned process identities."""
    if receipt_path.parent.is_symlink():
        raise LaunchError("launch receipt directory cannot be a symlink")
    receipt = read_json(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise LaunchError("unsupported fixed-eight launch receipt")
    launch_root = receipt_path.parent.resolve(strict=True)
    raw_cells = receipt.get("cells")
    if not isinstance(raw_cells, dict) or len(raw_cells) != 8:
        raise LaunchError("launch receipt does not own exactly eight panel cells")
    cells: dict[str, dict[str, Any]] = {}
    for cell_id, record in raw_cells.items():
        if not isinstance(record, dict):
            raise LaunchError(f"{cell_id}: malformed process receipt")
        cells[cell_id] = summarize_cell(cell_id, record, launch_root)
    states = {record["state"] for record in cells.values()}
    if states.issubset({"completed", "early_stopped"}):
        overall = "completed"
    elif "running" in states:
        overall = "running"
    else:
        overall = "failed"
    summary = {
        "schema": "wnn.galam-l4-quality8-summary.v1",
        "updated_at": utc_now(),
        "launch_id": receipt.get("launch_id"),
        "status": overall,
        "panel_sha256": receipt.get("panel_sha256"),
        "source_sha256": receipt.get("source_sha256"),
        "launcher_sha256": receipt.get("launcher_sha256"),
        "split_sha256": receipt.get("split_sha256"),
        "train_eval_mask": "all_after_warmup",
        "official_valid_mask": "is_scored",
        "cells": cells,
    }
    atomic_write_json(launch_root / "summary.json", summary)
    return summary


def summarize_cell(cell_id: str, record: dict[str, Any], launch_root: Path) -> dict[str, Any]:
    """Summarize one cell without following paths outside this launch or signaling PIDs."""
    if type(record.get("pid")) is not int:
        return {
            "gpu": record.get("gpu"),
            "gpu_uuid": record.get("gpu_uuid"),
            "pid": None,
            "process_state": "not_started",
            "state": "failed",
            "output": record.get("output"),
            "log": record.get("log"),
            "integrity_error": record.get("start_error", "fit process was not started"),
        }
    output = Path(str(record.get("output", "")))
    if (launch_root / "runs").is_symlink():
        raise LaunchError("runs directory cannot be a symlink")
    expected_output = launch_root / "runs" / cell_id
    if output.resolve(strict=False) != expected_output.resolve(strict=False):
        raise LaunchError(f"{cell_id}: run output path differs from its receipt-owned cell path")
    process_state = owned_process_state(record)
    if output.is_symlink():
        raise LaunchError(f"{cell_id}: run output cannot be a symlink")
    if not output.exists():
        if process_state == "running":
            return {
                "gpu": record.get("gpu"),
                "gpu_uuid": record.get("gpu_uuid"),
                "pid": record.get("pid"),
                "start_ticks": record.get("start_ticks"),
                "process_group": record.get("process_group"),
                "process_state": process_state,
                "state": "running",
                "output": str(output),
                "log": record.get("log"),
                "terminal_status": None,
                "checkpoint_pointers": None,
                "latest_epoch_metrics": None,
                "latest_official_validation": None,
            }
        return {
            "gpu": record.get("gpu"),
            "gpu_uuid": record.get("gpu_uuid"),
            "pid": record.get("pid"),
            "start_ticks": record.get("start_ticks"),
            "process_group": record.get("process_group"),
            "process_state": process_state,
            "state": "failed",
            "output": str(output),
            "log": record.get("log"),
            "integrity_error": "fit exited before creating its reserved run directory",
        }
    try:
        resolved_output = output.resolve(strict=True)
        resolved_output.relative_to(launch_root)
    except (OSError, ValueError):
        raise LaunchError(f"{cell_id}: run output path escapes the launch receipt") from None
    if output.is_symlink() or not resolved_output.is_dir():
        raise LaunchError(f"{cell_id}: run output is missing or unsafe")
    terminal = _optional_json(resolved_output / "terminal_status.json")
    pointers = _optional_json(resolved_output / "checkpoint_pointers.json")
    latest_metric = read_jsonl_tail(resolved_output / "metrics.jsonl")
    latest_valid = read_jsonl_tail(resolved_output / "monitor_evaluations.jsonl")
    status_error: str | None = None
    state = "running" if process_state == "running" else "failed"
    if terminal is not None:
        try:
            _validate_terminal(terminal, record, pointers, latest_valid, resolved_output)
            state = str(terminal["status"])
        except LaunchError as error:
            status_error = str(error)
            state = "failed"
    if latest_valid is not None:
        context = latest_valid.get("context")
        valid_context = context.get("valid") if isinstance(context, dict) else None
        train_context = context.get("train_eval") if isinstance(context, dict) else None
        if (
            not isinstance(valid_context, dict)
            or valid_context.get("row_mask") != "is_scored"
            or not isinstance(train_context, dict)
            or train_context.get("row_mask") != "all_after_warmup"
        ):
            status_error = "latest evaluation event does not identify official is_scored valid mask"
            state = "failed"
    result = {
        "gpu": record.get("gpu"),
        "gpu_uuid": record.get("gpu_uuid"),
        "pid": record.get("pid"),
        "start_ticks": record.get("start_ticks"),
        "process_group": record.get("process_group"),
        "process_state": process_state,
        "state": state,
        "output": str(resolved_output),
        "log": record.get("log"),
        "terminal_status": terminal,
        "checkpoint_pointers": pointers,
        "latest_epoch_metrics": latest_metric,
        "latest_official_validation": latest_valid,
    }
    if status_error is not None:
        result["integrity_error"] = status_error
    return result


def _validate_terminal(
    terminal: dict[str, Any],
    record: dict[str, Any],
    pointers: dict[str, Any] | None,
    latest_valid: dict[str, Any] | None,
    output: Path,
) -> None:
    """Bind terminal state, epoch pointers, and pooled official-valid report to the profile."""
    identity = terminal.get("identity")
    if not isinstance(identity, dict) or _canonical_sha256(identity) != record.get("identity_sha256"):
        raise LaunchError("terminal run identity differs from the admitted profile")
    status = terminal.get("status")
    if status not in ("completed", "early_stopped"):
        raise LaunchError("terminal status is not a declared successful fit result")
    completed = terminal.get("completed_epochs")
    requested = terminal.get("requested_epochs")
    if type(completed) is not int or type(requested) is not int or requested != 20 or not 5 <= completed <= 20:
        raise LaunchError("terminal epoch counts violate the twenty-epoch stop policy")
    expected_kind = "declared_early_stop" if status == "early_stopped" else "max_epochs"
    if terminal.get("completion_kind") != expected_kind:
        raise LaunchError("terminal completion kind disagrees with fit status")
    if status == "completed" and completed != 20:
        raise LaunchError("completed fits must reach the configured twenty-epoch maximum")
    if type(terminal.get("global_updates")) is not int or terminal["global_updates"] < 1:
        raise LaunchError("terminal fit update count is invalid")
    if not _is_sha256(terminal.get("model_sha256")):
        raise LaunchError("terminal model digest is invalid")
    model_path = output / "model.pt"
    if model_path.is_symlink() or not model_path.is_file():
        raise LaunchError("terminal model artifact is missing or unsafe")
    if sha256_file(model_path) != terminal["model_sha256"]:
        raise LaunchError("terminal model digest does not match model.pt")
    if not isinstance(pointers, dict):
        raise LaunchError("completed fit is missing checkpoint_pointers.json")
    if (
        pointers.get("run_identity_sha256") != record.get("identity_sha256")
        or pointers.get("completed_epochs") != completed
        or pointers.get("status") != status
        or not _finite_number(pointers.get("best_raw_wp"))
    ):
        raise LaunchError("checkpoint pointers do not match terminal fit identity or epoch")
    best, last = pointers.get("best"), pointers.get("last")
    if not isinstance(best, dict) or not isinstance(last, dict):
        raise LaunchError("best and last checkpoint pointers are missing")
    if (
        type(best.get("epoch")) is not int
        or not 1 <= best["epoch"] <= completed
        or last.get("epoch") != completed
        or not _finite_number(best.get("valid_wp"))
        or not _finite_number(last.get("valid_wp"))
        or not _is_sha256(best.get("sha256"))
        or not _is_sha256(last.get("sha256"))
    ):
        raise LaunchError("best or last checkpoint pointer is malformed")
    if best.get("valid_wp") != pointers.get("best_raw_wp"):
        raise LaunchError("best pointer does not track the strict raw-WP maximum")
    early = pointers.get("early_stopping")
    if not isinstance(early, dict) or (
        early.get("min_epoch"), early.get("patience"), early.get("min_delta")
    ) != (5, 4, 0.0005):
        raise LaunchError("checkpoint pointers omit the approved 5/4/0.0005 policy")
    bad_count = early.get("bad_count")
    if type(bad_count) is not int or bad_count < 0:
        raise LaunchError("checkpoint pointers contain an invalid early-stop bad count")
    if (status == "early_stopped" and (completed < 5 or bad_count < 4)) or (
        status == "completed" and bad_count >= 4
    ):
        raise LaunchError("terminal status is inconsistent with the declared early-stop policy")
    if not isinstance(latest_valid, dict) or (
        latest_valid.get("kind") != "evaluation"
        or latest_valid.get("epoch") != completed
        or not isinstance(latest_valid.get("context"), dict)
        or not isinstance(latest_valid["context"].get("valid"), dict)
        or latest_valid["context"]["valid"].get("row_mask") != "is_scored"
        or not isinstance(latest_valid["context"].get("train_eval"), dict)
        or latest_valid["context"]["train_eval"].get("row_mask") != "all_after_warmup"
        or not isinstance(latest_valid.get("metrics"), dict)
        or not _finite_number(latest_valid["metrics"].get("valid/wp"))
    ):
        raise LaunchError("terminal fit is missing its final pooled official-valid evaluation")
    if not math.isclose(
        float(last["valid_wp"]),
        float(latest_valid["metrics"]["valid/wp"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise LaunchError("last checkpoint WP differs from final pooled official-valid event")


def _optional_json(path: Path) -> dict[str, Any] | None:
    """Return one present artifact while rejecting symlinks and malformed JSON."""
    if path.is_symlink():
        raise LaunchError(f"fit artifact cannot be a symlink: {path.name}")
    if not path.exists():
        return None
    return read_json(path)


def _canonical_sha256(value: object) -> str:
    """Hash the same JSON identity representation as sequence provenance."""
    from wnn.training.sequence_provenance import canonical_sha256

    return canonical_sha256(value)


def _is_sha256(value: object) -> bool:
    """Accept one lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    """Accept finite JSON numbers while rejecting booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
