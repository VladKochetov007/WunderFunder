"""Validate the regenerated organizer baseline export and bind its provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREDICTION_PATH = ROOT / "cache/pred_baseline_valid.npy"
SIDECAR_PATH = ROOT / "cache/pred_baseline_valid.json"
TARGET_CACHE_PATH = ROOT / "cache/valid_targets.npz"
MODEL_PATH = ROOT / "baseline/baseline.onnx"
PREDICTOR_PATH = ROOT / "scripts/predict_baseline.py"


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory use."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Verify every output row and write one immutable prediction sidecar."""
    if SIDECAR_PATH.exists():
        raise FileExistsError(f"refusing to replace frozen receipt: {SIDECAR_PATH}")

    with np.load(TARGET_CACHE_PATH, allow_pickle=False) as targets:
        row_groups = targets["row_group"].astype(np.int64, copy=False)

    prediction = np.load(PREDICTION_PATH, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(row_groups), 20_000, 2)
    if prediction.shape != expected_shape or prediction.dtype != np.float32:
        raise ValueError(
            f"unexpected baseline array: shape={prediction.shape}, dtype={prediction.dtype}"
        )
    for start in range(0, len(row_groups), 8):
        if not np.isfinite(prediction[start : start + 8]).all():
            raise ValueError(f"non-finite baseline output in rows starting at {start}")

    row_group_sha256 = hashlib.sha256(row_groups.astype("<i8").tobytes()).hexdigest()
    receipt = {
        "source": "organizer_baseline_onnx_valid",
        "split": "valid",
        "producer_attempt": "e514-baseline-prediction-v1",
        "prediction_path": str(PREDICTION_PATH.relative_to(ROOT)),
        "prediction_sha256": sha256_file(PREDICTION_PATH),
        "shape": list(map(int, prediction.shape)),
        "dtype": str(prediction.dtype),
        "row_groups": list(map(int, row_groups)),
        "row_groups_sha256_little_endian_int64": row_group_sha256,
        "valid_targets_sha256": sha256_file(TARGET_CACHE_PATH),
        "baseline_onnx_sha256": sha256_file(MODEL_PATH),
        "predictor_sha256": sha256_file(PREDICTOR_PATH),
        "receipt_writer_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": datetime.now(UTC).isoformat(),
        "validation": "complete target-cache row order and all outputs finite",
    }
    SIDECAR_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "producer_attempt", "prediction_sha256", "shape", "row_groups_sha256_little_endian_int64",
        "valid_targets_sha256", "baseline_onnx_sha256", "predictor_sha256",
        "receipt_writer_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
