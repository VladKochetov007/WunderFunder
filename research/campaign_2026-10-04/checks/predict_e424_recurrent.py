"""Export fixed-epoch E424 specialist and prefix-adapter VALID predictions.

This isolated entry point mirrors ``scripts/predict_run.py`` while inferring
the output width from each supported architecture's actual state-dict head.
It always loads epoch 3, runs FP32 inference over every VALID target-cache
sequence, and writes a content-hashed prediction sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _weight_width(state: Mapping[str, object], key: str) -> int:
    """Read a linear output width without importing Torch."""
    weight = state.get(key)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) <= 0:
        raise ValueError(f"checkpoint is missing a matrix head tensor {key!r}")
    return int(shape[0])


def infer_output_width(arch: str, state: Mapping[str, object]) -> int:
    """Infer and cross-check model output width for supported E424 architectures."""
    if arch == "specialist":
        width = _weight_width(state, "universal_head.2.weight")
        expert_weights = sorted(
            (key, value)
            for key, value in state.items()
            if key.startswith("expert_heads.") and key.endswith(".weight")
        )
        if not expert_weights:
            raise ValueError("specialist checkpoint has no expert output heads")
        for key, _ in expert_weights:
            if _weight_width(state, key) != width:
                raise ValueError("specialist expert head widths disagree")
        return width
    if arch == "prefix-adapter":
        return _weight_width(state, "head.2.weight")
    raise ValueError(f"unsupported E424 recurrent architecture {arch!r}")


def _predict(args: argparse.Namespace) -> None:
    """Run the complete fixed-epoch VALID prediction export."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import train_rnn
    from torch.utils.data import DataLoader
    from train_rnn import Sequences, make_model, predict

    from wnn.data import SEQUENCE_LENGTH as DATA_SEQUENCE_LENGTH
    from wnn.propensity import load_propensity
    from wnn.resources import configure_torch, dataloader_budget
    from wnn.sentinel import SentinelPreprocessor
    from wnn.splits import file_sha256

    if args.split != "valid":
        raise ValueError("the E424 scorer expects all VALID rows")
    if args.epoch != 3:
        raise ValueError("E424 predictions are fixed to checkpoint epoch 3")
    if args.inference_precision != "fp32":
        raise ValueError("E424 paired predictions require common FP32 inference")
    if not args.name.replace("_", "").isalnum():
        raise ValueError("run name must contain only letters, digits, and underscores")
    output_name = args.output_name or args.name
    if not output_name.replace("_", "").isalnum():
        raise ValueError("output name must contain only letters, digits, and underscores")

    checkpoint_path = ROOT / "artifacts" / "runs" / args.name / "model_ep3.pt"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(checkpoint_args, dict) or not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks its training args or model state")
    if checkpoint.get("epoch") != 3:
        raise ValueError("model_ep3.pt does not contain fixed epoch 3")
    if checkpoint_args.get("checkpoint_policy") != "fixed":
        raise ValueError("E424 prediction requires the fixed checkpoint policy")
    if checkpoint_args.get("checkpoint_epoch") != 3:
        raise ValueError("E424 checkpoint metadata must fix epoch 3")
    arch = str(checkpoint_args.get("arch", ""))
    if arch not in {"specialist", "prefix-adapter"}:
        raise ValueError(f"expected specialist or prefix-adapter checkpoint, got {arch!r}")
    if checkpoint_args.get("privileged_mode", "none") != "none":
        raise ValueError("privileged teacher inputs cannot be exported as causal predictions")
    if (
        checkpoint_args.get("quote_cache_root")
        and checkpoint.get("provenance", {}).get("quote_cache") is None
    ):
        raise ValueError("checkpoint lacks quote-cache provenance")
    if checkpoint.get("quantization") is not None or checkpoint.get("factorization") is not None:
        raise ValueError("E424 recurrent wrapper does not accept quantized/factorized checkpoints")

    output_width = infer_output_width(arch, state)
    if output_width != 23:
        raise ValueError(f"E424 auxiliary checkpoints require 23 outputs, got {output_width}")
    configure_torch()
    device = torch.device("cuda")
    model = make_model(checkpoint_args, checkpoint["n_in"], output_width).to(device)
    model.load_state_dict(state, strict=True)
    model._inference_fp32 = True

    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing")
        else None
    )
    train_rnn.RAW_ONLY = bool(checkpoint_args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(checkpoint_args.get("online_norm"))
    train_rnn.HEURISTICS = bool(checkpoint_args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = checkpoint_args.get("research_features", "none")
    input_propensity = checkpoint_args.get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts" / "runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [tensor.to(device) for tensor in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = checkpoint.get("readout_tilted", False)

    groups = np.load(ROOT / "cache" / "valid_targets.npz", allow_pickle=False)["row_group"]
    dataset = Sequences("valid", groups)
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
    )
    final = ROOT / "cache" / f"pred_{output_name}_valid.npy"
    sidecar_path = final.with_suffix(".json")
    if final.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite prediction artifact: {final}")
    temporary = final.with_name(final.name + ".partial")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), DATA_SEQUENCE_LENGTH, 2),
    )
    written = 0
    for features, _, _ in loader:
        prediction = predict(model, features, 4000, device)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite prediction values near VALID sequence {written}")
        output[written : written + len(prediction)] = prediction[..., :2]
        written += len(prediction)
    if written != len(groups):
        raise RuntimeError(f"wrote {written} of {len(groups)} VALID sequences")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("source checkpoint changed during prediction export")
    output.flush()
    del output
    temporary.replace(final)
    stale_sums = ROOT / "cache" / f"seqsums_{output_name}.npz"
    if stale_sums.exists():
        stale_sums.unlink()

    quote_cache = checkpoint.get("provenance", {}).get("quote_cache")
    sidecar = {
        "run": args.name,
        "split": "valid",
        "epoch": 3,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": checkpoint_args.get("split_manifest_sha256"),
        "quote_cache": quote_cache,
        "gate_mode": "soft",
        "inference_precision": "fp32",
        "quantization_preset": None,
        "recurrent_rank": None,
        "sentinel_preprocessing": checkpoint.get("sentinel_preprocessing"),
        "row_groups": groups.tolist(),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")


def main() -> None:
    """Parse the narrow E424 inference interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid", choices=("valid",))
    parser.add_argument("--epoch", type=int, choices=(3,), default=3)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--inference-precision", choices=("fp32",), default="fp32")
    args = parser.parse_args()
    _predict(args)


if __name__ == "__main__":
    main()
