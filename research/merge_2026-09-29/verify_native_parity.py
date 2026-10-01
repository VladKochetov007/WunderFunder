"""Compare the Rust streaming port with its fixed-epoch PyTorch checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wnn_native
from data import VALID_PATH, SequenceReader
from model import ModelConfig, Predictor
from predict_oof import evaluation_groups


def main() -> None:
    """Measure prediction parity and one-row latency on development sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sequences", type=int, default=3)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Predictor(ModelConfig(**checkpoint["config"]), checkpoint["mean"], checkpoint["std"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(args.weights) as arrays:
        native = wnn_native.MergedPredictor({name: arrays[name] for name in arrays.files})

    groups = evaluation_groups(args.fold)[: args.sequences]
    reader = SequenceReader(VALID_PATH)
    torch_predictions = []
    native_predictions = []
    elapsed = 0.0
    rows = 0
    with torch.inference_mode():
        for group in groups:
            sequence = reader.read(int(group))
            state = model.initial_state(1)
            native.reset()
            current = np.empty((len(sequence.features), 2), dtype=np.float32)
            start = time.perf_counter()
            for row_index, row in enumerate(sequence.features):
                if row_index and row_index % 2000 == 0:
                    state = model.initial_state(1)
                    native.reset()
                native_prediction = native.step(row)
                current[row_index] = native_prediction
            elapsed += time.perf_counter() - start

            reference = np.empty_like(current)
            for block_start in range(0, len(sequence.features), 1000):
                if block_start and block_start % 2000 == 0:
                    state = model.initial_state(1)
                block = torch.from_numpy(sequence.features[block_start : block_start + 1000])
                output, state = model(block.unsqueeze(0), state)
                reference[block_start : block_start + len(block)] = output[0, :, :2].numpy()
            native_predictions.append(current)
            torch_predictions.append(reference)
            rows += len(sequence.features)

    native_all = np.concatenate(native_predictions)
    torch_all = np.concatenate(torch_predictions)
    max_abs = float(np.max(np.abs(native_all - torch_all)))
    correlations = [
        float(np.corrcoef(native_all[:, target], torch_all[:, target])[0, 1]) for target in range(2)
    ]
    print(
        {
            "fold": args.fold,
            "sequences": len(groups),
            "rows": rows,
            "max_abs_error": max_abs,
            "prediction_correlations": correlations,
            "rust_microseconds_per_row": 1e6 * elapsed / rows,
        }
    )


if __name__ == "__main__":
    main()
