from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np

from wnn.privileged import validate_crossfit_cache
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, CampaignSplit, file_sha256
from wnn.training.chronological_teacher import (
    validate_chronological_teacher_checkpoint,
    validate_chronological_train_index_order,
)
from wnn.training.environment import TrainingEnvironment


def bind_fold_teachers(
    teachers: list[tuple[str, float]] | None,
    campaign_split: CampaignSplit,
    splits: Iterable[str],
    *,
    environment: TrainingEnvironment | None = None,
) -> list[dict[str, Any]]:
    """Distillation targets under a clean fold must come from teachers fitted on that same fold.

    Each cached prediction needs a sidecar naming the teacher run, split and the student's split-manifest hash,
    and its bytes must match the recorded prediction hash; the evaluation fold never reached the teacher's fit.
    """
    environment = environment if environment is not None else TrainingEnvironment()
    bound = []
    for name, scale in teachers:
        for split in splits:
            pred = environment.root / "cache" / f"pred_{name}_{split}.npy"
            side = json.loads(pred.with_suffix(".json").read_text())
            if side.get("run") != name or side.get("split") != split:
                raise ValueError(f"teacher sidecar does not describe {name}/{split}")
            if side.get("split_manifest_sha256") != campaign_split.sha256:
                raise ValueError(f"teacher {name} was not fitted on the student's clean fold")
            with np.load(
                environment.root / "cache" / f"{split}_targets.npz", allow_pickle=False
            ) as targets:
                order = np.asarray(targets["row_group"])
            if order.ndim != 1 or order.dtype.kind not in "iu":
                raise ValueError("teacher target index must be a one-dimensional integer array")
            if list(map(int, side.get("row_groups", []))) != order.tolist():
                raise ValueError(f"teacher {name}/{split} rows are not in cache target order")
            digest = file_sha256(pred)
            if digest != side.get("prediction_sha256"):
                raise ValueError(f"teacher prediction bytes changed since export: {pred}")
            if campaign_split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
                if (
                    split != "train"
                    or side.get("campaign_export_scope")
                    != "chronological-full-train-index-feature-only"
                    or side.get("campaign_split_manifest_sha256") != campaign_split.sha256
                    or side.get("feature_only") is not True
                    or side.get("evaluation_membership_sha256")
                    != hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
                ):
                    raise ValueError(
                        f"teacher {name}/{split} is not a full-index train-only export"
                    )
                validate_chronological_train_index_order(
                    environment.root, order.tolist(), campaign_split, name
                )
                checkpoint_file = side.get("checkpoint_file")
                provenance_path = environment.root / "artifacts" / "runs" / name / "provenance.json"
                if (
                    not isinstance(checkpoint_file, str)
                    or checkpoint_file not in ("model.pt", f"model_ep{side.get('epoch')}.pt")
                    or side.get("run_provenance_sha256") != file_sha256(provenance_path)
                ):
                    raise ValueError(f"teacher {name} export lacks bound checkpoint provenance")
                checkpoint_path = environment.root / "artifacts" / "runs" / name / checkpoint_file
                checkpoint_sha256 = file_sha256(checkpoint_path)
                if side.get("checkpoint_sha256") != checkpoint_sha256:
                    raise ValueError(f"teacher {name} checkpoint bytes changed since export")
                run_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                import torch

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                validate_chronological_teacher_checkpoint(
                    checkpoint,
                    run_provenance,
                    campaign_split,
                    checkpoint_file,
                    checkpoint_sha256,
                    name,
                )
                if side.get("epoch") != checkpoint.get("epoch"):
                    raise ValueError(f"teacher {name} export epoch differs from checkpoint")
                values = np.load(pred, mmap_mode="r", allow_pickle=False)
                if values.shape != (len(order), environment.sequence_length, 2):
                    raise ValueError(f"teacher {name}/{split} prediction schema differs")
                if values.dtype != np.float32:
                    raise ValueError(f"teacher {name}/{split} predictions must be float32")
            if "crossfit_format" in side:
                validate_crossfit_cache(side, campaign_split, environment.root, split)
            record = {
                "run": name,
                "split": split,
                "scale": scale,
                "epoch": side.get("epoch"),
                "checkpoint_sha256": side.get("checkpoint_sha256"),
                "prediction_sha256": digest,
            }
            if campaign_split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
                record.update(
                    {
                        "checkpoint_file": checkpoint_file,
                        "run_provenance_sha256": side["run_provenance_sha256"],
                        "campaign_export_scope": side["campaign_export_scope"],
                        "row_groups_sha256": hashlib.sha256(
                            order.astype("<i8").tobytes()
                        ).hexdigest(),
                    }
                )
            bound.append(record)
    return bound


def prop_cache_path(
    name: str, split: str, *, environment: TrainingEnvironment | None = None
) -> Path:
    """Propensity cache written by scripts/cache_propensity.py; refuses a cache built from other weights."""
    environment = environment if environment is not None else TrainingEnvironment()
    path = environment.root / "cache" / f"propcache_{name}_{split}.npy"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/cache_propensity.py {name}")
    model_path = environment.root / "artifacts" / "runs" / name / "model.pt"
    want = file_sha256(model_path)
    metadata_size = meta.stat().st_size
    if metadata_size <= 0 or metadata_size > 64000:
        raise ValueError("propensity cache sidecar exceeds its 64,000-byte read bound")
    metadata_raw = meta.read_bytes()
    if len(metadata_raw) != metadata_size:
        raise ValueError("propensity cache sidecar changed while being read")
    got = json.loads(metadata_raw)
    if not isinstance(got, dict):
        raise ValueError("propensity cache sidecar must be a JSON object")
    if got.get("model_sha256") != want:
        raise RuntimeError(f"{path}: propensity weights changed since the cache was built")
    return path
