"""GPU predictions of a trained run on all validation sequences (cache row order).

    scripts/limited .venv/bin/python scripts/predict_run.py RUN [valid|train]
Writes cache/pred_<RUN>_<split>.npy (B, T, 2) float32 as a memmap, rows in cache/<split>_targets order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import train_rnn
from torch.utils.data import DataLoader
from train_rnn import Sequences, make_model, predict

from wnn.propensity import load_propensity
from wnn.resources import configure_torch, dataloader_budget
from wnn.splits import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--inference-precision", choices=["checkpoint", "fp32"], default="checkpoint",
                        help="force common fp32 evaluation for paired fp16/fp32 training controls")
    parser.add_argument("--gate-mode", choices=["soft", "hard", "confident"], default="soft")
    args = parser.parse_args()
    name, split = args.name, args.split
    checkpoint_name = f"model_ep{args.epoch}.pt" if args.epoch is not None else "model.pt"
    checkpoint_path = ROOT / "artifacts" / "runs" / name / checkpoint_name
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ck["args"].get("privileged_mode", "none") != "none":
        raise ValueError("privileged teachers require component-crossfit extraction, not causal prediction export")
    quote_cache = ck.get("provenance", {}).get("quote_cache")
    if ck["args"].get("quote_cache_root") and quote_cache is None:
        raise ValueError("checkpoint lacks quote cache provenance")
    configure_torch()
    device = torch.device("cuda")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    if (args.inference_precision == "fp32" or ck["args"].get("precision") == "fp32"
            or ck.get("quantization") is not None):
        model._inference_fp32 = True
    if ck["args"].get("arch") == "gated":
        model.gate_mode = args.gate_mode
    elif args.gate_mode != "soft":
        parser.error("--gate-mode requires a gated checkpoint")
    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))
    train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
    train_rnn.HEURISTICS = bool(ck["args"].get("heuristics"))
    train_rnn.RESEARCH_FEATURES = ck["args"].get("research_features", "none")
    input_propensity = ck["args"].get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts" / "runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if ck.get("readout") is not None:
        train_rnn.READOUT = [t.to(device) for t in ck["readout"]]
        train_rnn.READOUT_TILTED = ck.get("readout_tilted", False)
    groups = np.load(ROOT / "cache" / f"{split}_targets.npz")["row_group"]
    dataset = Sequences(split, groups)
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=workers,
        prefetch_factor=prefetch if workers else None,
    )
    final = ROOT / "cache" / f"pred_{args.output_name or name}_{split}.npy"
    tmp = final.with_name(final.name + ".partial")  # renamed only after every sequence is written
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(groups), 20_000, 2)
    )
    i = 0
    for x, _, _ in loader:
        p = predict(model, x, 4000, device)
        out[i : i + len(p)] = p[..., :2]
        i += len(p)
    out.flush()
    del out
    if i != len(groups):
        raise RuntimeError(f"wrote {i} of {len(groups)} sequences")
    stale = (
        ROOT / "cache" / f"seqsums_{args.output_name or name}.npz"
    )  # cached metric sums of an older prediction file
    if stale.exists():
        stale.unlink()
    tmp.replace(final)
    metadata = {
        "run": name,
        "split": split,
        "epoch": ck.get("epoch"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": ck["args"].get("split_manifest_sha256"),
        "quote_cache": quote_cache,
        "gate_mode": args.gate_mode,
        "inference_precision": "fp32" if getattr(model, "_inference_fp32", False) else "fp16",
        "quantization_preset": ck.get("quantization", {}).get("preset"),
        "row_groups": groups.tolist(),
    }
    final.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
