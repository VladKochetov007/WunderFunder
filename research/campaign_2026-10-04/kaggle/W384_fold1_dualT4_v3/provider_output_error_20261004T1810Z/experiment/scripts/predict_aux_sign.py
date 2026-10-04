"""Export the unused next-change sign logit from fixed fold checkpoints."""

from __future__ import annotations

import argparse

import decide
import numpy as np
import torch
import train_rnn
from torch.utils.data import DataLoader
from train_rnn import Sequences, build_inputs, forward_chunk, make_model


def main() -> None:
    """Write one logit per row, aligned to valid-target sequence order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()

    checkpoint = torch.load(
        decide.ROOT / "artifacts" / "runs" / args.run / f"model_ep{args.epoch}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["args"]
    if config.get("aux_next", 0) <= 0 or config.get("aux_next_k"):
        raise ValueError("expected one next-change auxiliary sign head")
    n_out = checkpoint["model"]["head.2.weight"].shape[0]
    device = torch.device("cuda")
    model = make_model(config, checkpoint["n_in"], n_out).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    train_rnn.RAW_ONLY = bool(config.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(config.get("online_norm"))

    groups = np.load(decide.ROOT / "cache" / "valid_targets.npz")["row_group"]
    _, fold = decide._chains()
    idx = np.flatnonzero(fold == args.fold)
    loader = DataLoader(Sequences("valid", groups[idx]), batch_size=8, num_workers=2)
    path = decide.ROOT / "cache" / f"auxsign_{args.run}_ep{args.epoch}_valid.npy"
    tmp = path.with_name(path.name + ".partial")
    output = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(groups), 20_000)
    )
    offset = 0
    with torch.no_grad():
        for x, _, _ in loader:
            inputs = build_inputs(x.to(device, non_blocking=True))
            state = None
            batch_idx = idx[offset : offset + len(x)]
            for start in range(0, 20_000, 4000):
                with torch.autocast("cuda", dtype=torch.float16):
                    out, state = forward_chunk(model, inputs, slice(start, start + 4000), state)
                output[batch_idx, start : start + out.shape[1]] = out[..., -1].float().cpu().numpy()
            offset += len(x)
    if offset != len(idx):
        raise RuntimeError(f"wrote {offset} of {len(idx)} sequences")
    output.flush()
    del output
    tmp.replace(path)
    print(f"wrote {path} for {offset} fold-{args.fold} sequences")


if __name__ == "__main__":
    main()
