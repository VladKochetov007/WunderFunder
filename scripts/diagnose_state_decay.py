"""Measure finite state-perturbation decay on one declared fitting sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel

CHECKPOINT = "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt"
CHECKPOINT_SHA = "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda"
BINS = ((99, 200), (200, 1000), (1000, 5000), (5000, 10000), (10000, 20000))


def file_sha256(path: Path) -> str:
    """Hash a file with bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run a CPU inference diagnostic; this neither fits nor ranks initializers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(1000, 20000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint_path = args.root / CHECKPOINT
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("diagnostic checkpoint differs from the pinned clean F0 control")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint["model"]
    model = RNNModel(211, hidden=256, layers=2, n_out=weights["head.2.weight"].shape[0])
    model.load_state_dict(weights, strict=True)
    model.eval()
    sequence = SequenceFile(args.root / "datasets/train.parquet").read(37)
    if sequence.seq_ix != 7598721:
        raise ValueError("declared fitting row-group 37 sequence identity changed")
    raw = torch.from_numpy(sequence.features[: args.rows].copy()).unsqueeze(0)
    inputs = torch.cat((raw, event_features_torch(raw)), dim=-1)
    generator = torch.Generator().manual_seed(734)
    delta = torch.randn(2, 1, 256, generator=generator)
    delta = delta * (0.1 / delta.square().mean().sqrt())
    states = {
        "control": torch.zeros_like(delta),
        "row0": delta.clone(),
        "row99": torch.zeros_like(delta),
    }
    bins = {
        arm: {str(lo): {"count": 0, "square_sum": 0.0, "max_abs": 0.0} for lo, _ in BINS}
        for arm in ("row0", "row99")
    }
    observations = []
    checkpoints = {1, 10, 50, 99, 100, 110, 150, 200, 1000, 5000, 10000, 20000}
    boundaries = sorted(
        {0, args.rows, *range(100, args.rows, 100), *(p for p in checkpoints if p < args.rows)}
    )
    with torch.inference_mode():
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if start == 99:
                states["row99"] = states["row99"] + delta
            predictions = {}
            for arm, hidden in states.items():
                output, new_hidden = model(inputs[:, start:end], hidden)
                if not torch.isfinite(output).all() or not torch.isfinite(new_hidden).all():
                    raise FloatingPointError(f"nonfinite output/state in {arm} at {end}")
                states[arm] = new_hidden
                predictions[arm] = output[..., :2]
            for arm in ("row0", "row99"):
                difference = (predictions[arm] - predictions["control"]).double()
                for lo, hi in BINS:
                    left, right = max(lo, start), min(hi, end)
                    if right > left:
                        selected = difference[:, left - start : right - start]
                        record = bins[arm][str(lo)]
                        record["count"] += selected.numel()
                        record["square_sum"] += selected.square().sum().item()
                        record["max_abs"] = max(record["max_abs"], selected.abs().max().item())
                if end in checkpoints:
                    difference_h = (states[arm] - states["control"]).double()
                    observations.append(
                        {
                            "arm": arm,
                            "after_row": end - 1,
                            "hidden_rms_by_layer": difference_h.square()
                            .mean(dim=(1, 2))
                            .sqrt()
                            .tolist(),
                            "prediction_abs_max_last_row": difference[:, -1].abs().max().item(),
                        }
                    )
    for records in bins.values():
        for record in records.values():
            record["rms"] = (
                (record["square_sum"] / record["count"]) ** 0.5 if record["count"] else None
            )
    result = {
        "format": "wnn.initial-state-finite-perturbation-diagnostic",
        "source_script_sha256": file_sha256(Path(__file__)),
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "data": {"source": "train", "row_group": 37, "seq_ix": sequence.seq_ix, "rows": args.rows},
        "feature_row_sha256": hashlib.sha256(
            np.asarray(sequence.features, dtype="<f4").tobytes()
        ).hexdigest(),
        "perturbation": {"seed": 734, "global_rms": 0.1, "same_vector_at_rows": [0, 99]},
        "observations": observations,
        "prediction_difference_bins": bins,
        "elapsed_seconds": time.perf_counter() - started,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "torch_version": torch.__version__,
        "quality_claim": False,
        "limitations": "One fitted sequence, one checkpoint and one random perturbation direction. Not a trained initializer, Jacobian bound, score gain or serving check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
