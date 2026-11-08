"""Verify final-fit teacher checkpoints and source-index prediction exports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from wnn.distillation_cache import SEQUENCE_ROWS, file_sha256, row_groups_sha256
from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.final_submit_bridge import (
    validate_final_fit_artifact,
    validate_final_fit_parent,
)

_RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _audit_teacher_exports(
    run_args: Mapping[str, object],
    provenance: Mapping[str, object],
    membership: FinalFitMembership,
    root: Path,
    seen: set[str],
) -> list[dict[str, object]]:
    """Check each direct teacher prediction and recurse into its fixed checkpoint lineage."""
    specifications = [
        value.partition("*")[0] for value in str(run_args.get("distill") or "").split(",") if value
    ]
    records = provenance.get("distill_teacher_exports", [])
    if not specifications:
        if records:
            raise ValueError("final-fit provenance has teacher exports without a recipe")
        return []
    if not isinstance(records, list):
        raise ValueError("final-fit teacher export provenance is malformed")
    by_pair = {
        (record.get("run"), record.get("split")): record
        for record in records
        if isinstance(record, dict)
    }
    expected_pairs = {(name, split) for name in specifications for split in ("train", "valid")}
    if len(by_pair) != len(records) or set(by_pair) != expected_pairs:
        raise ValueError("final-fit teacher export records differ from the recipe sources")
    expected_groups = {
        "train": list(membership.train_groups),
        "valid": list(membership.valid_groups),
    }
    closure: list[dict[str, object]] = []
    for name in specifications:
        if not _RUN_NAME.fullmatch(name):
            raise ValueError("final-fit teacher recipe has an unsafe source name")
        for split_name in ("train", "valid"):
            record = by_pair[(name, split_name)]
            sidecar_path = root / "cache" / f"pred_{name}_{split_name}.json"
            prediction_path = sidecar_path.with_suffix(".npy")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            checkpoint_file = sidecar.get("checkpoint_file")
            if checkpoint_file not in ("model.pt", f"model_ep{sidecar.get('epoch')}.pt"):
                raise ValueError(f"final-fit teacher {name} has an invalid checkpoint reference")
            checkpoint_path = root / "artifacts" / "runs" / name / checkpoint_file
            checkpoint_sha256 = file_sha256(checkpoint_path)
            run_provenance_path = checkpoint_path.parent / "provenance.json"
            run_provenance_sha256 = file_sha256(run_provenance_path)
            if (
                sidecar.get("run") != name
                or sidecar.get("split") != split_name
                or sidecar.get("row_groups") != expected_groups[split_name]
                or sidecar.get("final_fit_manifest_sha256") != membership.manifest_sha256
                or sidecar.get("final_fit_members_sha256") != membership.fit_members_sha256
                or sidecar.get("source_validation_sha256") != membership.source_validation_sha256
                or sidecar.get("campaign_export_scope") != "final-fit-source-index-feature-only"
                or sidecar.get("feature_only") is not True
                or sidecar.get("prediction_role")
                != "fitting-only teacher features; no independent score"
                or sidecar.get("independent_score_claim") != "forbidden"
                or sidecar.get("checkpoint_sha256") != checkpoint_sha256
                or sidecar.get("run_provenance_sha256") != run_provenance_sha256
                or sidecar.get("row_groups_sha256")
                != row_groups_sha256(expected_groups[split_name])
                or record.get("checkpoint_sha256") != checkpoint_sha256
                or record.get("prediction_sha256") != sidecar.get("prediction_sha256")
                or record.get("final_fit_manifest_sha256") != membership.manifest_sha256
                or record.get("source_validation_sha256") != membership.source_validation_sha256
                or record.get("prediction_role")
                != "fitting-only teacher features; no independent score"
                or record.get("independent_score_claim") != "forbidden"
            ):
                raise ValueError(f"final-fit teacher {name}/{split_name} export identity differs")
            if file_sha256(prediction_path) != sidecar.get("prediction_sha256"):
                raise ValueError(f"final-fit teacher {name}/{split_name} bytes differ")
            values = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(expected_groups[split_name]), SEQUENCE_ROWS, 2):
                raise ValueError(f"final-fit teacher {name}/{split_name} shape differs")
            closure.extend(
                _audit_final_fit_ancestors(name, checkpoint_file, membership, root, seen.copy())
            )
    return closure


def _audit_final_fit_ancestors(
    run: str,
    checkpoint_file: str,
    membership: FinalFitMembership,
    root: Path,
    seen: set[str] | None = None,
) -> list[dict[str, object]]:
    """Verify initializer and distillation ancestors use the same final-fit identity."""
    visited = set() if seen is None else seen
    if run in visited or not _RUN_NAME.fullmatch(run):
        raise ValueError("final-fit teacher ancestry has a cycle or unsafe run name")
    if checkpoint_file != "model.pt" and re.fullmatch(r"model_ep\d+\.pt", checkpoint_file) is None:
        raise ValueError("final-fit ancestor checkpoint name is unsafe")
    visited.add(run)
    run_root = root / "artifacts" / "runs" / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / checkpoint_file
    if not provenance_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"final-fit ancestor {run} lacks its selected artifacts")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if provenance.get("checkpoint_sha256", {}).get(checkpoint_file) != checkpoint_sha256:
        raise ValueError(f"final-fit ancestor {run} checkpoint hash differs from provenance")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_final_fit_parent(checkpoint, membership)
    run_args = provenance.get("args")
    if not isinstance(run_args, dict):
        raise ValueError(f"final-fit ancestor {run} has no recorded arguments")
    validate_final_fit_artifact(
        run_args, provenance, membership, checkpoint_epoch=checkpoint["epoch"]
    )
    if run_args.get("propensity") or run_args.get("prop_input"):
        raise ValueError("final-fit teacher ancestry cannot use a learned propensity model")
    closure: list[dict[str, object]] = [
        {
            "run": run,
            "checkpoint_file": checkpoint_file,
            "checkpoint_sha256": checkpoint_sha256,
            "final_fit_manifest_sha256": membership.manifest_sha256,
            "final_fit_members_sha256": membership.fit_members_sha256,
            "source_validation_sha256": membership.source_validation_sha256,
        }
    ]
    initializer = str(run_args.get("init") or "")
    if initializer:
        parent, separator, epoch_text = initializer.partition("@ep")
        if not _RUN_NAME.fullmatch(parent) or (separator and not epoch_text.isdecimal()):
            raise ValueError(f"final-fit ancestor {run} has an unsafe initializer reference")
        parent_file = f"model_ep{int(epoch_text)}.pt" if separator else "model.pt"
        parent_path = root / "artifacts" / "runs" / parent / parent_file
        parent_sha256 = file_sha256(parent_path)
        if provenance.get("initial_checkpoint_sha256") != parent_sha256:
            raise ValueError(f"final-fit ancestor {run} initializer hash differs")
        closure.extend(
            _audit_final_fit_ancestors(parent, parent_file, membership, root, visited.copy())
        )
    closure.extend(_audit_teacher_exports(run_args, provenance, membership, root, visited.copy()))
    return closure


def audit_final_fit_ancestor_closure(
    run: str, checkpoint_file: str, membership: FinalFitMembership, root: Path
) -> list[dict[str, object]]:
    """Return a verified initializer and teacher closure for one final-fit checkpoint."""
    return _audit_final_fit_ancestors(run, checkpoint_file, membership, root)


def bind_final_fit_teachers(
    teachers: list[tuple[str, float]], membership: FinalFitMembership, root: Path
) -> list[dict[str, object]]:
    """Verify direct teacher checkpoints and feature exports against final-fit membership."""
    bound: list[dict[str, object]] = []
    expected_groups = {
        "train": list(membership.train_groups),
        "valid": list(membership.valid_groups),
    }
    for split_name, groups in expected_groups.items():
        with np.load(root / "cache" / f"{split_name}_targets.npz", allow_pickle=False) as targets:
            source_order = np.asarray(targets["row_group"])
        if source_order.ndim != 1 or source_order.dtype.kind not in "iu":
            raise ValueError(f"final-fit {split_name} target index is malformed")
        if source_order.tolist() != groups:
            raise ValueError(f"final-fit {split_name} target order differs from source membership")
    for name, scale in teachers:
        train_sidecar = json.loads(
            (root / "cache" / f"pred_{name}_train.json").read_text(encoding="utf-8")
        )
        checkpoint_file = train_sidecar.get("checkpoint_file")
        if checkpoint_file not in ("model.pt", f"model_ep{train_sidecar.get('epoch')}.pt"):
            raise ValueError(f"final-fit teacher {name} has an invalid selected checkpoint")
        ancestor_closure = _audit_final_fit_ancestors(name, checkpoint_file, membership, root)
        for split_name in ("train", "valid"):
            sidecar_path = root / "cache" / f"pred_{name}_{split_name}.json"
            prediction_path = sidecar_path.with_suffix(".npy")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            source_path = root / "artifacts" / "runs" / name / "provenance.json"
            checkpoint_path = source_path.parent / sidecar.get("checkpoint_file", "")
            checkpoint_sha256 = file_sha256(checkpoint_path)
            if (
                sidecar.get("run") != name
                or sidecar.get("split") != split_name
                or sidecar.get("row_groups") != expected_groups[split_name]
                or sidecar.get("final_fit_manifest_sha256") != membership.manifest_sha256
                or sidecar.get("final_fit_members_sha256") != membership.fit_members_sha256
                or sidecar.get("source_validation_sha256") != membership.source_validation_sha256
                or sidecar.get("campaign_export_scope") != "final-fit-source-index-feature-only"
                or sidecar.get("feature_only") is not True
                or sidecar.get("prediction_role")
                != "fitting-only teacher features; no independent score"
                or sidecar.get("independent_score_claim") != "forbidden"
                or sidecar.get("checkpoint_file") != checkpoint_file
                or sidecar.get("checkpoint_sha256") != checkpoint_sha256
                or sidecar.get("run_provenance_sha256") != file_sha256(source_path)
                or sidecar.get("row_groups_sha256")
                != row_groups_sha256(expected_groups[split_name])
                or file_sha256(prediction_path) != sidecar.get("prediction_sha256")
            ):
                raise ValueError(f"final-fit teacher {name}/{split_name} export identity differs")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_final_fit_parent(checkpoint, membership)
            values = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(expected_groups[split_name]), SEQUENCE_ROWS, 2):
                raise ValueError(f"final-fit teacher {name}/{split_name} shape differs")
            if values.dtype != np.float32:
                raise ValueError(f"final-fit teacher {name}/{split_name} must be float32")
            bound.append(
                {
                    "run": name,
                    "split": split_name,
                    "scale": scale,
                    "epoch": checkpoint["epoch"],
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_sha256": checkpoint_sha256,
                    "prediction_sha256": sidecar["prediction_sha256"],
                    "final_fit_manifest_sha256": membership.manifest_sha256,
                    "source_validation_sha256": membership.source_validation_sha256,
                    "prediction_role": "fitting-only teacher features; no independent score",
                    "independent_score_claim": "forbidden",
                    "ancestor_closure": ancestor_closure,
                }
            )
    return bound
