from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wnn.features import StreamingEventFeatures, event_features_torch
from wnn.nn import RNNModel
from wnn.research_features import StreamingResearchFeatures, research_features_torch
from wnn.resources import configure_torch


def predict(checkpoint: Path, raw: np.ndarray, device: str) -> tuple[np.ndarray, dict]:
    """Compare complete and row-streaming inference for supported research checkpoints."""
    configure_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["args"]
    if config.get("arch", "gru") != "gru" or config.get("enc", "linear") != "linear":
        raise ValueError("This streaming reference supports the linear-encoder GRU only")
    if any(config.get(key) for key in ("online_norm", "heuristics", "prop_input", "raw_only")):
        raise ValueError("Reference does not support extra non-volume input transforms")
    mode = config.get("research_features", "none")
    model = RNNModel(
        payload["n_in"],
        hidden=config["hidden"],
        layers=config.get("layers", 2),
        n_out=payload["model"]["head.2.weight"].shape[0],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    tensor = torch.as_tensor(raw[None], dtype=torch.float32, device=device)
    inputs = torch.cat(
        (tensor, event_features_torch(tensor), research_features_torch(tensor, mode)), -1
    )
    state = None
    event, volume = StreamingEventFeatures(), StreamingResearchFeatures(mode)
    rows = []
    with torch.no_grad():
        batch, _ = model(inputs)
        for row in raw:
            features = np.concatenate((row, event.update(row), volume.update(row)))
            output, state = model(torch.as_tensor(features[None, None], device=device), state)
            rows.append(output[0, 0, :2].cpu().numpy())
    result = np.stack(rows)
    expected = batch[0, :, :2].cpu().numpy()
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)
    return result, {
        "mode": mode,
        "rows": len(raw),
        "max_abs": float(np.abs(result - expected).max()),
        "scope": "Python float32 parity; native feature implementation still required",
    }


def main() -> None:
    """Write predictions only after batch/streaming equivalence passes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result, report = predict(args.checkpoint, np.load(args.raw, allow_pickle=False), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
