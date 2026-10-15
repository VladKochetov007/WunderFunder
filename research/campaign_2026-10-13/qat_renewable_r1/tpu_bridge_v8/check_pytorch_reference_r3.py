"""Localize TPU/native differences with the updated EMA and production features."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel


def main() -> None:
    """Compare the same exported parameters across the three forward paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve(strict=True)
    packet = root / "research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8"
    destination = packet / "PYTORCH_REFERENCE_R3.json"
    if destination.exists():
        raise FileExistsError(destination)
    base = root / "artifacts/kaggle_tpu_v8_terminal_20261015_r1/cloud_preflight_tpu_v8"
    receipt = json.loads((base / "realdata_update_receipt.json").read_text())
    model_path = base / "realdata_reload/ema_after_one_chunk_fp32.npz"
    prediction_path = base / "realdata_reload/updated_ema_valid_fit_group_prediction.npz"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == receipt["checkpoint"]["ema_sha256"]
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == receipt["prediction_export"]["sha256"]
    with np.load(model_path, allow_pickle=False) as archive:
        weights = {name: np.array(archive[name], dtype=np.float32) for name in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        tpu_prediction = np.array(archive["predictions"], dtype=np.float32)
    sequence = SequenceFile(root / "datasets/valid.parquet").read(254, targets=False, mask=False)
    torch.set_num_threads(1)
    raw = torch.from_numpy(sequence.features).unsqueeze(0)
    inputs = torch.cat((raw, event_features_torch(raw)), dim=-1)
    assert inputs.shape == (1, 20_000, 211)
    input_hash = hashlib.sha256(inputs.contiguous().numpy().tobytes()).hexdigest()
    model = RNNModel(211, hidden=256, layers=2, n_out=23).eval()
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    chunks = []
    state = None
    with torch.no_grad():
        for start in range(0, 20_000, 2000):
            out, state = model(inputs[:, start:start + 2000], state)
            chunks.append(out.numpy())
    pytorch_prediction = np.concatenate(chunks, axis=1)
    library = root / "artifacts/native_qat_r1"
    binary_hash = hashlib.sha256((library / "wnn_native.abi3.so").read_bytes()).hexdigest()
    assert binary_hash == "a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108"
    sys.path.insert(0, str(library))
    native = importlib.import_module("wnn_native")
    weights["head.2.weight"] = np.ascontiguousarray(weights["head.2.weight"][:2])
    weights["head.2.bias"] = np.ascontiguousarray(weights["head.2.bias"][:2])
    predictor = native.Predictor(weights, precision="f32", fast_activations=False)
    native_prediction = np.stack([predictor.step(row) for row in sequence.features])[None]

    def errors(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        difference = np.abs(first - second)
        return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}

    result = {"utc": datetime.now(UTC).isoformat(), "source_qualified_group": ["valid", 254],
              "input_sha256": input_hash, "producer_input_sha256": receipt["prediction_export"]["input_sha256"],
              "input_identity_exact": input_hash == receipt["prediction_export"]["input_sha256"],
              "pytorch_vs_tpu_all23": errors(pytorch_prediction, tpu_prediction),
              "pytorch_vs_tpu_main2": errors(pytorch_prediction[..., :2], tpu_prediction[..., :2]),
              "native_vs_pytorch_main2": errors(native_prediction, pytorch_prediction[..., :2]),
              "all_outputs_finite": bool(np.isfinite(pytorch_prediction).all() and np.isfinite(native_prediction).all()),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "fitting-sequence diagnostic; localizes forward mismatch, no score"}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
