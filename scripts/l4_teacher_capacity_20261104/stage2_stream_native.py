"""Reuse the audited canonical GRU teacher checkpoint/export validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_distill_cache import _audit_ancestor_closure, _inspect_teacher
from scripts.l4_teacher_capacity_20261104.stage2_stream_common import GRU_SHAPES, read_json
from wnn.distillation_cache import file_sha256
from wnn.splits import CampaignSplit


def verify_native_gru(
    run: str, split: CampaignSplit, project_root: Path, target_order: list[int]
) -> tuple[dict[str, Any], np.ndarray]:
    """Verify one fixed native GRU checkpoint and full chronological train export."""
    hidden, layers, seed = GRU_SHAPES[run]
    checkpoint_root = project_root / "artifacts" / "runs"
    record, arrays, _ = _inspect_teacher(
        run,
        split,
        project_root / "cache",
        checkpoint_root,
        {"train": target_order},
        3,
        "model_ep3.pt",
    )
    if (record["hidden"], record["layers"], record["seed"], record["cell"]) != (
        hidden,
        layers,
        seed,
        "gru",
    ):
        raise ValueError(f"native GRU checkpoint recipe differs for {run}")
    checkpoint_path = checkpoint_root / run / "model_ep3.pt"
    provenance_path = checkpoint_root / run / "provenance.json"
    provenance = read_json(provenance_path)
    args = provenance.get("args", {})
    if args.get("init") not in (None, "") or args.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"native GRU teacher is not fresh scratch on the official split: {run}")
    record["ancestor_closure"] = _audit_ancestor_closure(
        run,
        split,
        checkpoint_root,
        "model_ep3.pt",
        record["checkpoint_sha256"],
        data_root=project_root,
    )
    record["prediction_sidecar_sha256"] = record["predictions"]["train"]["sidecar_sha256"]
    record["prediction_sha256"] = record["predictions"]["train"]["prediction_sha256"]
    record["prediction_path"] = str(project_root / "cache" / f"pred_{run}_train.npy")
    record["provenance_sha256"] = file_sha256(provenance_path)
    record["checkpoint_path"] = str(checkpoint_path)
    return record, arrays["train"]
