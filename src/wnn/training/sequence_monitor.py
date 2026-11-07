from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from wnn.gmetric import METRIC_ID
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_utils import SequenceDataset
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe
from wnn.training.sequence_monitor_evaluation import (
    EvaluationEventContext,
    evaluate_and_record_epoch,
)
from wnn.training.sequence_monitor_events import (
    MONITOR_EVENT_FILENAME,
    MonitorEventWriter,
)
from wnn.training.sequence_monitor_identity import (
    gpu_identifier as _gpu_identifier,
)
from wnn.training.sequence_monitor_identity import (
    mask_comparison as _mask_comparison,
)
from wnn.training.sequence_monitor_identity import (
    mask_definition as _mask_definition,
)
from wnn.training.sequence_monitor_identity import (
    membership_sha256 as _membership_sha256,
)
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_monitor_resources import HostResourceSampler, resource_metrics
from wnn.training.sequence_monitor_telemetry import record_train_step


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
        input_feature_mode: str = "raw112",
        evaluation_role: str = "valid",
        evaluation_mask: str = "is_scored",
        primary_split_sha256: str | None = None,
    ) -> None:
        """Bind event streams to exact source-qualified membership and run metadata."""
        self.diagnostics_every_steps = int(config["diagnostics_every_steps"])
        self.training_telemetry_every_steps = int(config.get("training_telemetry_every_steps", 1))
        self.evaluation_batch_size = int(config.get("eval_batch_size", 1))
        self.train_eval_mask = str(config.get("train_eval_mask", "is_scored"))
        self.input_feature_mode = input_feature_mode
        if evaluation_role not in ("valid", "control"):
            raise ValueError("unsupported monitored evaluation role")
        self.evaluation_role = evaluation_role
        self.evaluation_mask = evaluation_mask
        self.primary_split_sha256 = primary_split_sha256
        self.early_stopping = config.get("early_stopping")
        self.resource_sampler = HostResourceSampler(output)
        self.writer = MonitorEventWriter(
            output / MONITOR_EVENT_FILENAME,
            resume=resume,
            extended_limits="training_telemetry_every_steps" in config,
        )
        self.progress = MonitorProgress(output)
        self._train_eval_members = train_eval_members
        self._evaluation_members = valid_members
        evaluation_label = (
            "full official valid" if evaluation_role == "valid" else "inner train control"
        )
        self._monitoring_plan = {"evaluation_reporting": f"{evaluation_label} after each epoch"}
        self._train_eval_context = {
            "evaluation_scope": "fixed source-qualified fitting cohort; diagnostic only",
            "row_mask": self.train_eval_mask,
            "mask_definition": _mask_definition(self.train_eval_mask),
            "member_count": len(train_eval_members),
            "member_sha256": _membership_sha256(train_eval_members),
            "input_feature_mode": input_feature_mode,
        }
        self._evaluation_context = {
            "evaluation_scope": (
                "full official valid; development-exposed report"
                if evaluation_role == "valid"
                else "auxiliary control within official train; fitting-only proxy, not chronological validation"
            ),
            "role": evaluation_role,
            "scope": "primary_valid" if evaluation_role == "valid" else "inner_train_proxy",
            "source": valid_members[0][0] if valid_members else None,
            "row_mask": evaluation_mask,
            "mask_definition": _mask_definition(evaluation_mask),
            "split_sha256": split_sha256,
            "primary_split_sha256": primary_split_sha256,
            "member_count": len(valid_members),
            "member_sha256": _membership_sha256(valid_members),
            "input_feature_mode": input_feature_mode,
        }
        self._metric_context = {
            "metric_id": METRIC_ID,
            "run_identity_sha256": run_identity_sha256,
            "loss_definition": (
                "weighted MSE of raw predictions against clip(target,-2,2), "
                "weighted by abs(clipped target)"
            ),
            "wp_definition": "one global pooled clipped weighted Pearson over the whole cohort",
            "mask_comparison": _mask_comparison(
                self.train_eval_mask, self.evaluation_mask, self.evaluation_role
            ),
            "run_metadata": "run_metadata.json",
        }

    def fit_status_context(self, resume: bool) -> dict[str, object]:
        """Return the exact small cohort membership once for durable run navigation."""
        status: dict[str, object] = {
            "resume": resume,
            "no_eval_during_fit": False,
            "evaluation_reporting": self._monitoring_plan["evaluation_reporting"],
            "checkpoint_selection": (
                "role-specific evaluation is reported per epoch; control remains a fitting-only proxy"
            ),
            "early_stopping": self.early_stopping,
            "train_eval_mask": self.train_eval_mask,
            "evaluation_role": self.evaluation_role,
            "evaluation_mask": self.evaluation_mask,
            "input_feature_mode": self.input_feature_mode,
            "train_eval_members": [list(member) for member in self._train_eval_members],
            "train_eval_member_sha256": self._train_eval_context["member_sha256"],
            "evaluation_member_count": self._evaluation_context["member_count"],
            "evaluation_member_sha256": self._evaluation_context["member_sha256"],
            "split_sha256": self._evaluation_context["split_sha256"],
        }
        if self.evaluation_role == "valid":
            status["validation_reporting"] = self._monitoring_plan["evaluation_reporting"]
        return status

    def profile_status_context(self) -> dict[str, object]:
        """Describe profile-only telemetry without claiming validation was evaluated."""
        return {
            "no_eval_during_fit": True,
            "evaluation_reporting": "not evaluated during bounded engineering profile",
            "intended_evaluation_reporting": self._monitoring_plan.get("evaluation_reporting"),
            "train_eval_mask": self.train_eval_mask,
            "evaluation_role": self.evaluation_role,
            "evaluation_mask": self.evaluation_mask,
            "input_feature_mode": self.input_feature_mode,
            "train_eval_members": [list(member) for member in self._train_eval_members],
            "train_eval_member_sha256": self._train_eval_context["member_sha256"],
            "evaluation_member_count": self._evaluation_context["member_count"],
            "evaluation_member_sha256": self._evaluation_context["member_sha256"],
            "split_sha256": self._evaluation_context["split_sha256"],
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
        record_train_step(
            self.writer,
            record,
            epoch_index,
            epoch_position,
            device.type,
            self.training_telemetry_every_steps,
            str(self._metric_context["mask_comparison"]),
        )
        self.progress.flush()

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
                "parameter_norm_sampling": "first 4096 trainable scalars in model parameter order",
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
    ) -> dict[str, int | float]:
        """Append source-qualified train-eval and configured evaluation metrics."""
        self.progress.phase("evaluation", step, epoch + 1)
        result = evaluate_and_record_epoch(
            model,
            train_eval,
            valid,
            chunk_rows,
            evaluation_batch_size or self.evaluation_batch_size,
            precision_mode,
            context,
            epoch,
            step,
            EvaluationEventContext(
                self.writer,
                self._train_eval_members,
                self._evaluation_members,
                self.train_eval_mask,
                self.evaluation_role,
                self.evaluation_mask,
                self._metric_context,
                self._train_eval_context,
                self._evaluation_context,
            ),
        )
        self.progress.phase("checkpoint", step, epoch + 1)
        return result

    def status(
        self, state: str, step: int, epoch: int, context: dict[str, object] | None = None
    ) -> None:
        """Append run status without rewriting fit metadata or earlier event lines."""
        self.writer.append("status", step, epoch, {}, {"status": state, **(context or {})})
        self.progress.phase("training" if state == "running" else state, step, epoch)
