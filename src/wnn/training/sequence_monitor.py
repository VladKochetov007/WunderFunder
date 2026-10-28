from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch

from wnn.gmetric import METRIC_ID
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe
from wnn.training.sequence_monitor_evaluation import evaluate_monitoring_sets
from wnn.training.sequence_monitor_events import (
    MONITOR_EVENT_FILENAME,
    MonitorEventWriter,
    resource_metrics,
)
from wnn.training.sequence_monitor_resources import HostResourceSampler


class SequenceMonitor:
    """Produce opt-in online events for fixed train diagnostics and full valid reports."""

    def __init__(
        self,
        output: Path,
        config: Mapping[str, object],
        train_eval_members: list[tuple[str, int]],
        valid_members: list[tuple[str, int]],
        split_sha256: str,
        run_identity_sha256: str,
        *,
        resume: bool,
    ) -> None:
        """Bind event streams to exact source-qualified membership and run metadata."""
        self.diagnostics_every_steps = int(config["diagnostics_every_steps"])
        self.resource_sampler = HostResourceSampler(output)
        self.writer = MonitorEventWriter(output / MONITOR_EVENT_FILENAME, resume=resume)
        self._train_eval_members = train_eval_members
        self._valid_members = valid_members
        self._train_eval_context = {
            "evaluation_scope": "fixed source-qualified fitting cohort; diagnostic only",
            "member_count": len(train_eval_members),
            "member_sha256": _membership_sha256(train_eval_members),
        }
        self._valid_context = {
            "evaluation_scope": "full official valid; report-only; validation remains out of fitting",
            "split_sha256": split_sha256,
            "member_count": len(valid_members),
            "member_sha256": _membership_sha256(valid_members),
        }
        self._metric_context = {
            "metric_id": METRIC_ID,
            "run_identity_sha256": run_identity_sha256,
            "mask": "official is_scored mask after row 99 warmup",
            "loss_definition": (
                "weighted MSE of raw predictions against clip(target,-2,2), "
                "weighted by abs(clipped target)"
            ),
            "wp_definition": "one global pooled clipped weighted Pearson over the whole cohort",
            "run_metadata": "run_metadata.json",
        }

    def fit_status_context(self, resume: bool) -> dict[str, object]:
        """Return the exact small cohort membership once for durable run navigation."""
        return {
            "resume": resume,
            "no_eval_during_fit": False,
            "validation_reporting": (
                "full official valid after each epoch; report-only; no checkpoint selection"
            ),
            "train_eval_members": [list(member) for member in self._train_eval_members],
            "train_eval_member_sha256": self._train_eval_context["member_sha256"],
            "valid_member_count": self._valid_context["member_count"],
            "valid_member_sha256": self._valid_context["member_sha256"],
            "split_sha256": self._valid_context["split_sha256"],
        }

    def offsets(self) -> dict[str, int]:
        """Return the event and evaluation journal positions for a safe checkpoint."""
        return self.writer.offsets()

    def restore_offsets(self, offsets: Mapping[str, object], global_step: int) -> None:
        """Retain orphaned tails, truncate to the safe checkpoint, and reload reports."""
        self.writer.restore_offsets(offsets, global_step)

    def diagnostic_probe(self, model: torch.nn.Module) -> DiagnosticProbe:
        """Create one ephemeral hook set for the current sampled optimizer window."""
        return DiagnosticProbe(model)

    def train_step(
        self,
        record: Mapping[str, object],
        epoch_index: int,
        epoch_position: float,
        device: torch.device,
    ) -> None:
        """Append optimizer and online training-window telemetry."""
        step = int(record["step"])
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
            "throughput/scored_rows_per_second": float(record["valid_rows_per_second"]),
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
        self.writer.append(
            "train_step",
            step,
            epoch_position,
            metrics,
            {
                "loss_scope": "online optimizer window on the recipe training mask",
                "epoch_index": epoch_index + 1,
                "comparison": (
                    "train/loss is online; train_eval/loss and valid/loss use official masks "
                    "and weighted MSE"
                ),
                "device_type": device.type,
            },
        )

    def resources(
        self, step: int, epoch_index: int, epoch_position: float, device: torch.device
    ) -> None:
        """Append the rank-zero process and local device readings that are available."""
        self.writer.append(
            "resources",
            step,
            epoch_position,
            resource_metrics(device) | self.resource_sampler.sample(_gpu_identifier(device)),
            {
                "resource_scope": "rank zero local process and device",
                "device_type": device.type,
                "epoch_index": epoch_index + 1,
                "cpu_utilization_convention": (
                    "cpu/process_utilization_pct is percent of one logical CPU and can exceed 100"
                ),
            },
        )

    def diagnostics(
        self,
        step: int,
        epoch_index: int,
        epoch_position: float,
        metrics: dict[str, int | float],
    ) -> None:
        """Append one sampled diagnostic map, leaving unobserved branches absent."""
        self.writer.append(
            "diagnostics",
            step,
            epoch_position,
            metrics,
            {
                "sampling": f"every {self.diagnostics_every_steps} optimizer steps",
                "scope": "rank-zero sampled forward/backward optimizer window",
                "missing_metrics": "unavailable model branches are omitted",
                "epoch_index": epoch_index + 1,
            },
        )

    def evaluate_epoch(
        self,
        model: torch.nn.Module,
        train_eval: SequenceDataset,
        valid: SequenceDataset,
        chunk_rows: int,
        evaluation_batch_size: int,
        precision_mode: str,
        context: DistributedContext,
        epoch: int,
        step: int,
    ) -> None:
        """Append one paired report event with train-eval and complete official-valid scores."""
        _validate_dataset_members(train_eval, self._train_eval_members, "train_eval")
        _validate_dataset_members(valid, self._valid_members, "valid")
        train_metrics, valid_metrics = evaluate_monitoring_sets(
            model,
            train_eval,
            valid,
            chunk_rows,
            precision_mode,
            context,
            evaluation_batch_size,
        )
        if train_metrics["train_eval/sequence_count"] != len(self._train_eval_members):
            raise ValueError(
                "train-eval evaluation did not cover its fixed source-qualified cohort"
            )
        if valid_metrics["valid/sequence_count"] != len(self._valid_members):
            raise ValueError("valid evaluation did not cover the full official membership")
        train_metrics["train_eval/score"] = train_metrics["train_eval/wp"]
        train_metrics["train_eval/score/t0"] = train_metrics["train_eval/wp/t0"]
        train_metrics["train_eval/score/t1"] = train_metrics["train_eval/wp/t1"]
        valid_metrics["valid/score"] = valid_metrics["valid/wp"]
        valid_metrics["valid/score/t0"] = valid_metrics["valid/wp/t0"]
        valid_metrics["valid/score/t1"] = valid_metrics["valid/wp/t1"]
        self.writer.append(
            "evaluation",
            step,
            epoch + 1,
            train_metrics | valid_metrics,
            self._metric_context
            | {
                "train_eval": self._train_eval_context
                | {"coverage_complete": True, "batch_size": evaluation_batch_size},
                "valid": self._valid_context
                | {
                    "coverage_complete": True,
                    "selected_rows": valid_metrics["valid/selected_rows"],
                    "batch_size": evaluation_batch_size,
                },
            },
        )

    def status(
        self, state: str, step: int, epoch: int, context: dict[str, object] | None = None
    ) -> None:
        """Append run status without rewriting fit metadata or earlier event lines."""
        self.writer.append("status", step, epoch, {}, {"status": state, **(context or {})})


def _membership_sha256(members: list[tuple[str, int]]) -> str:
    """Hash exact source-qualified membership without sorting identifiers."""
    encoded = json.dumps([list(member) for member in members], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_dataset_members(
    dataset: SequenceDataset, expected: list[tuple[str, int]], scope: str
) -> None:
    """Check canonical SequenceRecords membership when the dataset exposes it."""
    actual = getattr(dataset, "members", None)
    if actual is not None and list(actual) != expected:
        raise ValueError(f"{scope} dataset membership differs from its recorded cohort")
    if scope == "train_eval" and getattr(dataset, "training_mask", "is_scored") != "is_scored":
        raise ValueError("train-eval diagnostic cohort must use the official is_scored mask")


def _gpu_identifier(device: torch.device) -> str | None:
    """Resolve the physical UUID for this CUDA worker without guessing visible ordinals."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        identifier = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    except (AssertionError, RuntimeError, ValueError):
        return None
    if isinstance(identifier, bytes):
        identifier = identifier.decode("ascii", errors="ignore")
    elif identifier is not None and not isinstance(identifier, str):
        identifier = str(identifier)
    if not isinstance(identifier, str) or not identifier:
        return None
    if re.fullmatch(
        r"[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}",
        identifier,
    ):
        return f"GPU-{identifier}"
    return identifier
