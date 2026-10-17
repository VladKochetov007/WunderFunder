from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np

from wnn.privileged import validate_crossfit_cache
from wnn.splits import CampaignSplit, file_sha256
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
            order = np.load(environment.root / "cache" / f"{split}_targets.npz")["row_group"]
            if list(map(int, side.get("row_groups", []))) != order.tolist():
                raise ValueError(f"teacher {name}/{split} rows are not in cache target order")
            digest = file_sha256(pred)
            if digest != side.get("prediction_sha256"):
                raise ValueError(f"teacher prediction bytes changed since export: {pred}")
            if "crossfit_format" in side:
                validate_crossfit_cache(side, campaign_split, environment.root, split)
            bound.append(
                {
                    "run": name,
                    "split": split,
                    "scale": scale,
                    "epoch": side.get("epoch"),
                    "checkpoint_sha256": side.get("checkpoint_sha256"),
                    "prediction_sha256": digest,
                }
            )
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
