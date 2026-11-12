from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from wnn.training.sequence_data import SequenceData, SequenceRecords
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_config import (
    RAW_STUDENT_MODES,
    STANDALONE_LABELS_KD_MODE,
    STANDALONE_LABELS_MODE,
)
from wnn.training.sequence_monitor_evaluation import evaluate_monitoring_sets
from wnn.training.sequence_monitor_events import MONITOR_EVENT_FILENAME, MonitorEventWriter
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_reference_cache import ReferenceCache
from wnn.training.sequence_reference_pins import PRIMARY_SPLIT_SHA256


def select_train_eval_members(
    members: list[tuple[str, int]], count: int, selection: str
) -> list[tuple[str, int]]:
    """Select whole fitting sequences by legacy order or stable source-qualified hash rank."""
    if not members or not 1 <= count <= len(members):
        raise ValueError("train-eval cohort size must fit the fitting membership")
    if any(source != "train" for source, _ in members):
        raise ValueError("train-eval cohort selection accepts official train members only")
    if len(members) != len(set(members)):
        raise ValueError(
            "train-eval fitting membership contains duplicate source-qualified members"
        )
    if selection == "prefix":
        return members[:count]
    if selection != "hash_rank_v1":
        raise ValueError(f"unsupported train-eval cohort selection: {selection}")

    def rank(member: tuple[str, int]) -> tuple[bytes, str, int]:
        """Create a seed-independent stable rank for one source-qualified member."""
        payload = json.dumps(
            ["wnn.train_eval_selection.hash_rank_v1", member[0], int(member[1])],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).digest(), member[0], int(member[1])

    return sorted(members, key=rank)[:count]


def validate_complement_monitor(config: Mapping[str, object]) -> None:
    """Admit only the report-only official-valid monitor used by the receiving panel."""
    monitor = config.get("monitoring")
    if monitor is None:
        return
    allowed = {"train_eval_sequences", "eval_batch_size", "train_eval_selection"}
    if (
        not isinstance(monitor, dict)
        or not set(monitor) <= allowed
        or not {
            "train_eval_sequences",
            "eval_batch_size",
        }
        <= set(monitor)
    ):
        raise ValueError("complement monitoring requires train_eval_sequences and eval_batch_size")
    for key, maximum in (("train_eval_sequences", 256), ("eval_batch_size", 32)):
        if type(monitor[key]) is not int or not 1 <= monitor[key] <= maximum:
            raise ValueError(f"monitoring.{key} must be in [1, {maximum}]")
    if monitor.get("train_eval_selection", "prefix") not in ("prefix", "hash_rank_v1"):
        raise ValueError("monitoring.train_eval_selection must be prefix or hash_rank_v1")
    mode = config.get("objective_mode")
    if mode == STANDALONE_LABELS_MODE:
        if config.get("blend_objective_alpha", 0) != 0 or config["kd_weight"] != 0:
            raise ValueError("standalone-label monitoring requires raw z without blend or KD")
        return
    if mode == STANDALONE_LABELS_KD_MODE:
        if config.get("blend_objective_alpha", 0) != 0 or config["hint_weight"] != 0:
            raise ValueError("standalone label+KD monitoring forbids blend and hidden hints")
        return
    if config.get("blend_objective_alpha", 0.1) != 0.1 or config["kd_weight"] != 0:
        raise ValueError("receiving monitoring reports the fixed alpha=0.1 endpoint without KD")


class ComplementMonitor:
    """Report the composed endpoint each epoch without selecting or stopping a fit."""

    def __init__(
        self,
        data: SequenceData,
        reference_root: Path | None,
        config: Mapping[str, object],
        output: Path,
        *,
        resume: bool,
        primary: bool,
        report_epochs: bool = True,
    ) -> None:
        if data.split.sha256 != PRIMARY_SPLIT_SHA256 or data.split.valid_groups:
            raise ValueError("complement monitor requires the original train-only official split")
        members = data.fit.members
        evaluation = data.evaluation.members
        if len(members) != 10325 or len(evaluation) != 1873:
            raise ValueError("complement monitor requires the complete official memberships")
        raw_student = config.get("objective_mode") in RAW_STUDENT_MODES
        standalone_kd = config.get("objective_mode") == STANDALONE_LABELS_KD_MODE
        if raw_student and reference_root is not None:
            if config.get("objective_mode") == STANDALONE_LABELS_MODE:
                raise ValueError(
                    "standalone-label monitoring rejects --monitor-reference-root; "
                    "use an offline fixed readout for TL3 diagnostics"
                )
            raise ValueError("standalone label+KD monitoring rejects --monitor-reference-root")
        if not raw_student and reference_root is None:
            raise ValueError("historical complement monitoring requires --monitor-reference-root")
        settings = config["monitoring"]
        self.batch_size = int(settings["eval_batch_size"])
        fit_reference = None
        valid_reference = None
        if reference_root is not None:
            fit_reference = ReferenceCache(reference_root / "fit", data.split, members)
            valid_reference = ReferenceCache(
                reference_root / "evaluation", data.split, evaluation, membership="evaluation"
            )
            if (
                fit_reference.identity["reference_sha256"]
                != valid_reference.identity["reference_sha256"]
            ):
                raise ValueError("monitor fit and valid references differ")
        trained_reference = data.provenance.get("reference")
        if config.get("objective_mode") == STANDALONE_LABELS_MODE and trained_reference is not None:
            raise ValueError("standalone labels cannot have training reference ancestry")
        if standalone_kd and trained_reference is not None:
            raise ValueError("standalone label+KD monitoring cannot have blend-reference ancestry")
        trained_teacher = data.provenance.get("teacher")
        if standalone_kd and trained_teacher is None:
            raise ValueError(
                "standalone label+KD monitoring requires the validated fit teacher cache"
            )
        if config.get("objective_mode") == STANDALONE_LABELS_MODE and trained_teacher is not None:
            raise ValueError("standalone label-only monitoring rejects teacher-cache ancestry")
        if (
            not raw_student
            and trained_reference is not None
            and (fit_reference is None or trained_reference != fit_reference.identity)
        ):
            raise ValueError("monitoring reference differs from the training objective reference")
        selection = str(settings.get("train_eval_selection", "prefix"))
        train_members = select_train_eval_members(
            members, int(settings["train_eval_sequences"]), selection
        )
        root = data.fit.sources["train"].environment.root
        self.train_eval = SequenceRecords(
            root, train_members, 0, cache=fit_reference, training_mask="all_after_warmup"
        )
        self.valid = SequenceRecords(root, evaluation, 0, cache=valid_reference)
        self.context = {
            "split_sha256": data.split.sha256,
            "train_eval_members": [list(member) for member in train_members],
            "train_eval_member_sha256": membership_sha256(train_members),
            "train_eval_selection": selection,
            "train_eval_population": "official fitting membership only",
            "train_eval_cohort_size": len(train_members),
            "evaluation_member_sha256": membership_sha256(evaluation),
            "evaluation_member_count": len(evaluation),
            "evaluation_population": "complete official validation membership",
            "train_eval_mask": "all_after_warmup",
            "evaluation_mask": "is_scored after warmup",
            "generalization_gap_comparable": False,
            "generalization_gap_comparability_reason": (
                "in-sample fitting subset and complete independent validation differ in "
                "population and row mask"
            ),
            "metric_id": "global_wp_clip2_v1",
            "endpoint": (
                "raw student z; label plus fit-cache KD; no reference or blend"
                if standalone_kd
                else "raw standalone z; no reference or blend"
                if config.get("objective_mode") == STANDALONE_LABELS_MODE
                else "0.9 * clean_tl3 + 0.1 * z; no component preclipping"
            ),
            "checkpoint_selection": f"fixed final epoch {config['epochs']}; monitoring is report only",
            "evaluation_exposure": "development exposed; official source boundary",
        }
        if not raw_student:
            assert fit_reference is not None and valid_reference is not None
            self.context.update(
                reference_sha256=fit_reference.identity["reference_sha256"],
                fit_reference_manifest_sha256=fit_reference.identity["manifest_sha256"],
                evaluation_reference_manifest_sha256=valid_reference.identity["manifest_sha256"],
            )
        # Both ranks bind reporting inputs before run identity is computed. They do
        # not become inputs to either raw-student objective or fit normalization.
        data.provenance["report_only_monitor"] = self.context
        self.writer = (
            MonitorEventWriter(output / MONITOR_EVENT_FILENAME, resume=resume)
            if primary and report_epochs
            else None
        )
        self.progress = MonitorProgress(output) if primary else None

    def evaluate(
        self,
        model: torch.nn.Module,
        config: Mapping[str, object],
        context: DistributedContext,
        epoch: int,
        step: int,
    ) -> bool:
        """Evaluate raw master weights on rank zero, restoring mode, buffers, and all RNG."""
        if self.writer is None:
            return True
        if self.progress is not None:
            self.progress.phase("evaluation", step, epoch)
        train, valid = evaluate_monitoring_sets(
            model,
            self.train_eval,
            self.valid,
            int(config["chunk_rows"]),
            str(config["precision_mode"]),
            context,
            self.batch_size,
            blend_alpha=None if config.get("objective_mode") in RAW_STUDENT_MODES else 0.1,
            report_standalone=config.get("objective_mode") is not None,
            since_diagnostics=float(config.get("since_weight", 0.0)) > 0,
        )
        if valid["valid/sequence_count"] != 1873 or valid["valid/selected_rows"] != 3_528_273:
            raise RuntimeError("full official valid monitoring coverage is incomplete")
        self.writer.append("evaluation", step, epoch + 1, train | valid, self.context)
        return True

    def offsets(self) -> dict[str, int]:
        """Return primary monitor offsets for the exact safe epoch checkpoint."""
        return self.writer.offsets() if self.writer is not None else {}

    def restore_offsets(self, offsets: Mapping[str, object], step: int) -> bool:
        """Restore the report journal with the same safe-boundary resume contract."""
        if self.writer is not None:
            self.writer.restore_offsets(offsets, step)
        return True
