"""Export a two-stream CPU-float32 reference from the exact C1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import train_rnn

from wnn.data import SequenceFile
from wnn.propensity import load_propensity
from wnn.resources import configure_torch
from wnn.sentinel import SentinelPreprocessor
from wnn.splits import file_sha256

ROOT = Path(__file__).resolve().parents[4]
RUNS = ("bd26_c1v2_s3_f0", "bd26_c1v2_s4_f0")
DATA = ROOT / "cache/valid_subset20.parquet"
OUT = ROOT / "research/campaign_2026-10-06/bandit_continuation/submission_C1_E524"


def configure_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load an own checkpoint using the project's ordinary inference setup."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    state = checkpoint["model"]
    configure_torch()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU reference unexpectedly initialized CUDA")
    model = train_rnn.make_model(args, checkpoint["n_in"], state["head.2.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing")
        else None
    )
    train_rnn.RAW_ONLY = bool(args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(args.get("online_norm"))
    train_rnn.HEURISTICS = bool(args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = args.get("research_features", "none")
    prop_input = args.get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts/runs" / prop_input / "model.pt", "cpu")
        if prop_input
        else None
    )
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [tensor.to("cpu") for tensor in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = bool(checkpoint.get("readout_tilted", False))
    if args.get("arch") == "gated":
        model.gate_mode = "soft"
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("CPU reference model parameters are not float32")
    return model, checkpoint


def cpu_predictions() -> tuple[np.ndarray, list[int], dict[str, str]]:
    """Run the two exact package members on the same two complete raw streams."""
    device = torch.device("cpu")
    sequences = SequenceFile(DATA)
    predictions: list[np.ndarray] = []
    sequence_ids: list[int] = []
    checkpoint_hashes: dict[str, str] = {}
    for run in RUNS:
        checkpoint_path = ROOT / "artifacts/runs" / run / "model_ep1.pt"
        model, checkpoint = configure_model(checkpoint_path)
        checkpoint_hashes[run] = file_sha256(checkpoint_path)
        current: list[np.ndarray] = []
        for index in range(2):
            sequence = sequences.read(index, targets=False, mask=False)
            if run == RUNS[0]:
                sequence_ids.append(sequence.seq_ix)
            elif sequence_ids[index] != sequence.seq_ix:
                raise ValueError("sequence order changed between C1 checkpoint exports")
            features = torch.from_numpy(sequence.features).unsqueeze(0)
            values = train_rnn.predict(model, features, 4000, device)[0]
            current.append(np.asarray(values, dtype=np.float32))
        predictions.append(np.stack(current))
        del checkpoint, model
    return np.stack(predictions), sequence_ids, checkpoint_hashes


def main() -> None:
    """Persist CPU references and exact checkpoint/input provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "torch_cpu_reference_first2.npz")
    parser.add_argument("--metadata", type=Path, default=OUT / "torch_cpu_reference_first2.json")
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU reference began with CUDA already initialized")
    predictions, sequence_ids, checkpoint_hashes = cpu_predictions()
    np.savez_compressed(args.output, pred=predictions, seq_ix=np.asarray(sequence_ids))
    metadata = {
        "purpose": "independent CPU float32 model-export reference for C1 ZIP parity",
        "runs": list(RUNS),
        "checkpoint_sha256": checkpoint_hashes,
        "data": DATA.relative_to(ROOT).as_posix(),
        "data_sha256": file_sha256(DATA),
        "sequence_ids": sequence_ids,
        "rows_per_sequence": 20_000,
        "output_shape": list(predictions.shape),
        "output_dtype": str(predictions.dtype),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "reference_source": "scripts/train_rnn.py make_model and predict, eval mode, CPU/no autocast",
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
