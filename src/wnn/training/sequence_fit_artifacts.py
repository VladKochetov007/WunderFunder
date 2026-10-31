from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_checkpoint import atomic_write_json, save_final_model
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_loop import EpochResult
from wnn.training.sequence_precision import precision_identity
from wnn.training.sequence_provenance import file_sha256, json_value


def epoch_summary(
    result: EpochResult, config: Mapping[str, object]
) -> dict[str, int | float | bool]:
    """Convert global sufficient sums into losses, work counts, and epoch duration."""
    totals = result.totals
    label_denom, hint_denom = float(totals[6].item()), float(totals[7].item())
    label_loss = float(totals[0].item()) / label_denom if label_denom > 0 else 0.0
    kd_loss = float(totals[1].item()) / label_denom if label_denom > 0 else 0.0
    hint_loss = float(totals[2].item()) / hint_denom if hint_denom > 0 else 0.0
    return {
        "updates": result.updates,
        "complete": result.complete,
        "label_loss": label_loss,
        "kd_loss": kd_loss,
        "hint_loss": hint_loss,
        "total_loss": label_loss
        + float(config["kd_weight"]) * kd_loss
        + float(config["hint_weight"]) * hint_loss,
        "valid_rows": int(totals[3].item()),
        "forwarded_rows": int(totals[4].item()),
        "real_rows": int(totals[5].item()),
        "elapsed_seconds": result.elapsed_seconds,
    }


def write_final_outputs(
    output: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    history: list[dict[str, object]],
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    profile_receipt: Path,
    profile_hash: str,
    global_step: int,
    epoch_index_sha256: str | None = None,
) -> bool:
    """Write the last completed model and terminal status, preserving prior artifacts."""
    epochs = int(config["epochs"])
    completed_epochs = len(history)
    if not 1 <= completed_epochs <= epochs:
        raise ValueError("final outputs require a nonempty completed epoch prefix")
    early_stopped = bool(history[-1].get("early_stopped", False))
    if completed_epochs < epochs and not early_stopped:
        raise ValueError("a shortened fit must end through the declared early-stop policy")
    retention = config.get("checkpoint_retention", "final_only")
    if retention == "all_epochs":
        if not _is_sha256(epoch_index_sha256):
            raise ValueError("all-epoch retention requires the final epoch-index digest")
    elif retention != "final_only" or epoch_index_sha256 is not None:
        raise ValueError("fixed-final retention received an invalid epoch-index binding")
    model_path = output / "model.pt"
    provenance: dict[str, object] = {
        "schema": "wnn.sequence-provenance.v1",
        "identity": identity,
        "normalization": normalization,
        "profile_receipt": {"path": str(profile_receipt), "sha256": profile_hash},
        "history": json_value(history),
        "fixed_checkpoint_policy": (
            "last-completed-epoch-after-declared-early-stop"
            if early_stopped
            else "configured-final-epoch"
        ),
        "completed_epochs": completed_epochs,
        "requested_epochs": epochs,
        "global_updates": global_step,
        "precision": precision_identity(
            str(config.get("precision_mode", "fp32")),
            str(config["model"].get("attention_backend", "math")),
        ),
    }
    monitoring_plan = data_provenance.get("monitoring_plan")
    if isinstance(monitoring_plan, Mapping):
        provenance["evaluation_annotation"] = {
            "no_eval_during_fit": False,
            "evaluation_reporting": monitoring_plan.get(
                "evaluation_reporting", "full official valid after each epoch; exposed report"
            ),
            "monitoring_plan": json_value(monitoring_plan),
        }
    if epoch_index_sha256 is not None:
        provenance["epoch_index_sha256"] = epoch_index_sha256
    if model_path.exists():
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        saved_provenance = saved.get("provenance") if isinstance(saved, dict) else None
        if (
            not isinstance(saved, dict)
            or saved.get("checkpoint_schema") != "wnn.sequence-fit.v1"
            or saved.get("completed_epochs") != completed_epochs
            or not isinstance(saved_provenance, dict)
            or saved_provenance.get("identity") != identity
            or saved_provenance.get("epoch_index_sha256") != epoch_index_sha256
        ):
            raise FileExistsError("an incompatible model.pt already exists; refusing to replace it")
    else:
        save_final_model(
            model_path,
            model.state_dict(),
            config,
            completed_epochs - 1,
            completed_epochs,
            data_provenance,
            provenance,
            allow_early_stop=early_stopped,
        )
    status_path = output / "terminal_status.json"
    status = {
        "status": "early_stopped" if early_stopped else "completed",
        "completion_kind": "declared_early_stop" if early_stopped else "max_epochs",
        "completed_epochs": completed_epochs,
        "requested_epochs": epochs,
        "global_updates": global_step,
        "model_sha256": file_sha256(model_path),
        "identity": identity,
    }
    if epoch_index_sha256 is not None:
        status["epoch_index_sha256"] = epoch_index_sha256
    if status_path.exists():
        saved_status = json.loads(status_path.read_text(encoding="utf-8"))
        if json_value(saved_status) != json_value(status):
            raise FileExistsError("an incompatible terminal status already exists")
    else:
        atomic_write_json(status, status_path)
    return True


def _is_sha256(value: object) -> bool:
    """Accept a lowercase hexadecimal SHA-256 digest only."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def record_epoch_metrics(
    recorder: RunRecorder | None,
    summary: Mapping[str, int | float | bool],
    epoch: int,
    global_step: int,
    learning_rates: Mapping[str, float],
) -> bool:
    """Append one aggregate epoch row after training and before its safe checkpoint."""
    if recorder is None:
        return True
    elapsed = float(summary["elapsed_seconds"])
    forwarded = float(summary["forwarded_rows"])
    valid = float(summary["valid_rows"])
    recorder.record_metrics(
        step=global_step,
        epoch=epoch + 1,
        loss=float(summary["total_loss"]),
        learning_rates=dict(learning_rates),
        throughput={
            "valid_rows_per_second": valid / max(elapsed, 1e-12),
            "forwarded_rows_per_second": forwarded / max(elapsed, 1e-12),
        },
        timing={"epoch_seconds": elapsed},
    )
    return True


def metrics_offset(output: Path, context: DistributedContext) -> int:
    """Return the rank-zero aggregate log size for the current safe boundary."""
    if not context.is_primary:
        return 0
    path = output / "metrics.jsonl"
    return path.stat().st_size if path.exists() else 0
