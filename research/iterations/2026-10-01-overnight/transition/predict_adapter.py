"""Prediction-only fixed checkpoint adapter; isolated output, no outcome query."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import train_rnn as base
from wnn.data import SequenceFile
from features import RAW_SLICE, batch_features, SCHEMA
from model import FeatureHeadModel


class InputSequences(Dataset):
    def __init__(self, groups, metadata):
        self.groups = list(groups)
        self.metadata = metadata
        self.reader = None
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, index):
        if self.reader is None:
            self.reader = SequenceFile(ROOT/"datasets/valid.parquet")
        x = self.reader.read(int(self.groups[index]), targets=False, mask=False).features
        extra = batch_features(x[:, RAW_SLICE], self.metadata["nuisance"], self.metadata["arm"])
        return torch.from_numpy(np.concatenate([x, extra], axis=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck["args"]
    metadata = config["transition_features"]
    if metadata["schema"] != SCHEMA or ck.get("epoch") != 1:
        raise ValueError("expected fixed epoch1 transition checkpoint")
    base.configure_torch()
    model = FeatureHeadModel(ck["n_in"], config["hidden"], config["layers"], config["dropout"],
                             ck["model"]["head.2.weight"].shape[0]).cuda().eval()
    model.load_state_dict(ck["model"])
    groups = np.load(ROOT/"cache/valid_targets.npz")["row_group"]
    cf = np.load(ROOT/"cache/chain_folds.npz")
    fold = int(metadata["source_fold"])
    included_groups = cf["row_group"][(cf["split"] == "valid") & (cf["fold"] == fold)] if fold >= 0 else groups
    included_positions = np.flatnonzero(np.isin(groups, included_groups))
    selected_groups = groups[included_positions]
    loader = DataLoader(InputSequences(selected_groups, metadata), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, prefetch_factor=1 if args.workers else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(groups), 20000, 2))
    out[:] = np.nan
    written = 0
    for x in loader:
        pred = base.predict(model, x, 4000, "cuda")
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite forecasts")
        out[included_positions[written:written+len(pred)]] = pred
        written += len(pred)
    out.flush()
    del out
    if written != len(selected_groups):
        raise RuntimeError("incomplete predictions")
    temporary.replace(args.output)
    receipt = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "output": str(args.output), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "sequences": written, "epoch": ck["epoch"], "row_groups": selected_groups.tolist(),
               "cache_row_groups_sha256": hashlib.sha256(groups.tobytes()).hexdigest(),
               "included_positions": included_positions.tolist(), "outside_fold": "NaN",
               "feature_metadata": metadata, "WP_queries": 0, "targets_read": False, "mask_read": False}
    with args.output.with_suffix(".json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sequences": written, "epoch": ck["epoch"], "WP_queries": 0}))

if __name__ == "__main__":
    main()
