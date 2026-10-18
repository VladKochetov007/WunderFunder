"""Freeze and validate seed-4 feature-only predictions before official grading."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
RUNS = (
    "kalman_groupfast_active_f0_s4_r1",
    "kalman_groupfast_zero_f0_s4_r1",
)
EXPECTED_SHAPE = (468, 20_000, 2)
CHUNK_GROUPS = 32


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prediction(name: str, groups: list[int], split_sha256: str) -> dict[str, Any]:
    """Validate one feature-only prediction and its evaluation membership sidecar."""
    path = ROOT / "cache" / f"pred_{name}_valid.npy"
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_membership = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    if (
        sidecar.get("campaign_export_scope") != "evaluation"
        or sidecar.get("feature_only") is not True
    ):
        raise ValueError(f"{name}: export is not feature-only evaluation inference")
    if sidecar.get("split_manifest_sha256") != split_sha256:
        raise ValueError(f"{name}: split hash differs from the frozen F0 manifest")
    if sidecar.get("row_groups") != groups:
        raise ValueError(f"{name}: row-group order differs from the frozen evaluation list")
    if sidecar.get("evaluation_membership_sha256") != expected_membership:
        raise ValueError(f"{name}: evaluation membership hash differs")
    if sidecar.get("prediction_sha256") != sha256_file(path):
        raise ValueError(f"{name}: prediction sidecar hash differs from the file")

    prediction = np.load(path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != EXPECTED_SHAPE or prediction.dtype != np.float32:
        raise ValueError(f"{name}: expected float32 prediction shape {EXPECTED_SHAPE}")
    for start in range(0, len(groups), CHUNK_GROUPS):
        if not np.isfinite(prediction[start : start + CHUNK_GROUPS]).all():
            raise FloatingPointError(f"{name}: non-finite prediction values")

    checkpoint_path = ROOT / "artifacts/runs" / name / "model_ep0.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if sidecar.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{name}: prediction sidecar checkpoint hash differs")
    return {
        "run": name,
        "prediction_path": path.relative_to(ROOT).as_posix(),
        "prediction_sha256": sha256_file(path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "checkpoint_sha256": checkpoint_sha256,
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": True,
        "feature_only_evaluation_export": True,
        "evaluation_membership_sha256": expected_membership,
    }


def main() -> None:
    """Write a label-blind receipt binding both predictions to one F0 panel."""
    split_bytes = SPLIT_PATH.read_bytes()
    split = json.loads(split_bytes)
    groups = [int(group) for group in split["evaluation"]["valid"]]
    if len(groups) != EXPECTED_SHAPE[0]:
        raise ValueError(f"expected {EXPECTED_SHAPE[0]} evaluation groups, found {len(groups)}")

    outputs = [
        validate_prediction(name, groups, hashlib.sha256(split_bytes).hexdigest()) for name in RUNS
    ]
    if outputs[0]["evaluation_membership_sha256"] != outputs[1]["evaluation_membership_sha256"]:
        raise ValueError("active and control predictions do not share evaluation membership")
    receipt = {
        "schema": "wnn.kalman_group_recurrent_seed4_prediction_freeze.v1",
        "experiment_id": "E-KALMAN-GROUP-RECURRENT-SEED4-20261018-R1",
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "evaluation_group_count": len(groups),
        "evaluation_membership_sha256": outputs[0]["evaluation_membership_sha256"],
        "predictions": outputs,
        "targets_read": False,
        "masks_read": False,
        "status": "frozen_feature_only_predictions_ready_for_official_grade",
        "source_sha256": {
            "script": sha256_file(Path(__file__)),
            "split": sha256_file(SPLIT_PATH),
        },
    }
    output_path = PACKET / "prediction_freeze_seed4_r1.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Wrote label-blind prediction freeze receipt to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
