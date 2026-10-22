"""Profile causality and batch/stream parity on one fitting-side input sequence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from wnn.data import SequenceFile  # noqa: E402
from wnn.kalman_group_features import (  # noqa: E402
    FEATURE_WIDTH,
    StreamingKalmanGroupFeatures,
    features_numpy,
    features_torch,
    features_torch_step,
    initial_stream_state_torch,
)

SOURCE_GROUP = 6768
SOURCE_COMPONENT = 2406
REFERENCE_DIR = ROOT / "research/campaign_2026-10-18/online_features_r1/group_fast_slow_r1"
sys.path.insert(0, str(REFERENCE_DIR))
from group_features import extract_online_features  # noqa: E402


def _stream_numpy(sequence: np.ndarray) -> np.ndarray:
    state = StreamingKalmanGroupFeatures()
    return np.stack([state.update(row) for row in sequence])


def _stream_torch(sequence: np.ndarray) -> np.ndarray:
    state = initial_stream_state_torch()
    outputs = []
    for row in sequence:
        features, state = features_torch_step(torch.from_numpy(row[None].copy()), state)
        outputs.append(features[0].numpy())
    return np.stack(outputs)


def main() -> None:
    source = ROOT / "datasets/train.parquet"
    sequence = SequenceFile(source).read(SOURCE_GROUP, targets=False, mask=False).features
    batch = torch.from_numpy(sequence[None].copy())

    numpy_features = features_numpy(sequence)
    _, _, reference_features = extract_online_features(sequence)
    batch_features = features_torch(batch)[0].numpy()
    streamed_numpy = _stream_numpy(sequence)
    streamed_torch = _stream_torch(sequence)

    prefix_end = 10_000
    perturbed = sequence.copy()
    perturbed[prefix_end:] = perturbed[prefix_end:] + np.float32(0.25)
    future_features = features_numpy(perturbed)
    repeat_stream = StreamingKalmanGroupFeatures()
    first_pass = np.stack([repeat_stream.update(row) for row in sequence])
    repeat_stream.reset()
    reset_pass = np.stack([repeat_stream.update(row) for row in sequence])

    batch_delta = np.abs(batch_features - numpy_features)
    batch_max_index = np.unravel_index(int(np.argmax(batch_delta)), batch_delta.shape)
    torch_stream_delta = np.abs(streamed_torch - batch_features)
    torch_stream_max_index = np.unravel_index(
        int(np.argmax(torch_stream_delta)), torch_stream_delta.shape
    )
    detail_row = int(batch_max_index[0])
    detail_group = int(batch_max_index[1] % 13)
    detail_indices = (detail_group, 13 + detail_group, 26 + detail_group)
    metrics = {
        "schema": "wnn.kalman_group_feature_preflight.v1",
        "source": {
            "split": "train",
            "row_group": SOURCE_GROUP,
            "component": SOURCE_COMPONENT,
            "rows": int(len(sequence)),
            "features": int(sequence.shape[1]),
            "targets_read": False,
            "masks_read": False,
            "features_sha256": hashlib.sha256(sequence.tobytes()).hexdigest(),
        },
        "feature_width": FEATURE_WIDTH,
        "implementation_sha256": hashlib.sha256(
            (ROOT / "src/wnn/kalman_group_features.py").read_bytes()
        ).hexdigest(),
        "reference_implementation_sha256": hashlib.sha256(
            (REFERENCE_DIR / "group_features.py").read_bytes()
        ).hexdigest(),
        "finite": bool(np.isfinite(numpy_features).all() and np.isfinite(batch_features).all()),
        "reference_recipe_max_abs": float(np.max(np.abs(reference_features - numpy_features[99:]))),
        "batch_vs_numpy_max_abs": float(np.max(np.abs(batch_features - numpy_features))),
        "batch_vs_numpy_max_index": [int(index) for index in batch_max_index],
        "batch_vs_numpy_values": [
            float(batch_features[batch_max_index]),
            float(numpy_features[batch_max_index]),
        ],
        "batch_vs_numpy_detail_row_features": {
            str(index): [
                float(batch_features[detail_row, index]),
                float(numpy_features[detail_row, index]),
            ]
            for index in detail_indices
        },
        "batch_vs_numpy_max_by_feature_block": [
            float(np.max(batch_delta[:, start:end])) for start, end in ((0, 13), (13, 26), (26, 39))
        ],
        "numpy_stream_vs_batch_max_abs": float(np.max(np.abs(streamed_numpy - numpy_features))),
        "torch_stream_vs_batch_max_abs": float(np.max(np.abs(streamed_torch - batch_features))),
        "torch_stream_vs_batch_max_index": [int(index) for index in torch_stream_max_index],
        "future_perturbation_prefix_max_abs": float(
            np.max(np.abs(future_features[:prefix_end] - numpy_features[:prefix_end]))
        ),
        "reset_repeat_max_abs": float(np.max(np.abs(reset_pass - first_pass))),
        "warmup_prefix_nonzero_count": int(np.count_nonzero(numpy_features[:99])),
        "all_zero_control_nonzero_count": int(
            np.count_nonzero(features_numpy(sequence, control="all"))
        ),
    }
    result = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    (Path(__file__).parent / "preflight_result_r1.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
