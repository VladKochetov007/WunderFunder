"""Fold-teacher cache bindings already validated by the source audit helper."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any  # Any covers schema-checked artifacts and split membership adapters.

import numpy as np


def _same_membership(actual: Any, expected: Any, label: str) -> None:
    """Require duplicate-free exact fit membership, regardless of list order."""
    values = list(map(int, actual)) if isinstance(actual, list) else []
    if (
        not isinstance(actual, list)
        or len(values) != len(set(values))
        or set(values) != set(expected)
    ):
        raise ValueError(f"{label} fit membership differs from the full clean pool")


def _little_endian_groups(groups: Any) -> bytes:
    """Encode ordered campaign groups as the canonical sidecar does."""
    return np.asarray(groups, dtype="<i8").tobytes()


def _architecture_args(record: dict[str, Any]) -> dict[str, Any]:
    """Pin canonical direct-teacher/student widths and tied-lead adapter choice."""
    teacher = record["kind"] == "teacher"
    return {
        "arch": "gru",
        "cell": "gru",
        "hidden": 256,
        "layers": 2,
        "encoder_width": None,
        "recurrent_width": None,
        "readout_width": None,
        "head_layers": 1,
        "enc": "tiedlead" if teacher else "linear",
        "research_features": "tl3_lead" if teacher else "none",
        "readout_activation": "gelu",
    }


def verify_teacher_caches(
    cell_id: str,
    project_root: Path,
    split: Any,
    targets: list[dict[str, Any]],
    digest: Callable[[Path], str],
    read_json: Callable[[Path], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind audited teacher predictions to current sidecar bytes and exact roles."""
    if len(targets) != 2 or [item.get("run") for item in targets] != [
        f"tl3kd_r1_{cell_id.lower()}_teacher_a",
        f"tl3kd_r1_{cell_id.lower()}_teacher_b",
    ]:
        raise ValueError(f"{cell_id} does not have the two prescribed teachers")
    files = []
    for target in targets:
        if target.get("coefficient") != 0.5:
            raise ValueError(f"{cell_id} teacher cache coefficients changed")
        for source in ("train", "valid"):
            path = project_root / "cache" / f"pred_{target['run']}_{source}.npy"
            sidecar = path.with_suffix(".json")
            targets_path = project_root / "cache" / f"{source}_targets.npz"
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"{target['run']}/{source} teacher cache is missing")
            if not targets_path.is_file():
                raise FileNotFoundError(f"{source} target ordering is missing")
            with np.load(targets_path, allow_pickle=False) as target_archive:
                target_order = list(map(int, target_archive["row_group"]))
            fitting_groups = list(split.train_groups if source == "train" else split.valid_groups)
            membership_sha = hashlib.sha256(
                np.asarray(target_order, dtype="<i8").tobytes()
            ).hexdigest()
            metadata = read_json(sidecar)
            sidecar_sha = digest(sidecar)
            prediction_sha = target.get(f"{source}_prediction_sha256")
            if (
                metadata.get("run") != target["run"]
                or metadata.get("split") != source
                or metadata.get("split_manifest_sha256") != split.sha256
                or metadata.get("epoch") != 3
                or metadata.get("checkpoint_sha256") != target["checkpoint_sha256"]
                or metadata.get("prediction_sha256") != prediction_sha
                or metadata.get("row_groups") != target_order
                or metadata.get("evaluation_membership_sha256") != membership_sha
                or metadata.get("fitting_only_groups") != fitting_groups
                or metadata.get("campaign_export_scope") != "fit-only-scattered-to-full-index"
                or metadata.get("feature_only") is not True
                or metadata.get("inference_precision") != "fp32"
                or metadata.get("zeros_outside_fit") is not True
            ):
                raise ValueError(f"{target['run']}/{source} cache sidecar changed")
            files.append(
                {
                    "run": target["run"],
                    "split": source,
                    "epoch": 3,
                    "checkpoint_sha256": target["checkpoint_sha256"],
                    "prediction_path": str(path),
                    "prediction_sha256": prediction_sha,
                    "sidecar_path": str(sidecar),
                    "sidecar_sha256": sidecar_sha,
                }
            )
    return targets, files
