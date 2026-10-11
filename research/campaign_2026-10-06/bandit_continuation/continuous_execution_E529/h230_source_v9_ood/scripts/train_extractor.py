"""Train the small next-step GRU feature extractor used by train_rnn.py --frozen-extractor.

    WNN_PROJECT_ROOT=<project> python scripts/train_extractor.py --out artifacts/extractor_s0.pt \
        --groups fit_groups.npy --split train --epochs 2

Reads features of the listed fitting row groups only (never labels, never evaluation components) and predicts
the next row's 104 raw price/volume channels. Input to the extractor is the standard causal raw + event view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rnn  # noqa: E402
from wnn.data import SequenceFile  # noqa: E402
from wnn.extractor import BASE_WIDTH, NextStepExtractor, fit_extractor  # noqa: E402


class FeatureRows(Dataset):
    """Features only, for an explicit list of fitting row groups."""

    def __init__(self, split: str, groups: list[int]):
        self.path, self.groups, self._file = train_rnn.ROOT / "datasets" / f"{split}.parquet", groups, None

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i):
        if self._file is None:
            self._file = SequenceFile(self.path)
        return torch.from_numpy(self._file.read(self.groups[i], targets=False, mask=False).features)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", required=True, help="npy of fitting row groups (no evaluation components)")
    ap.add_argument("--split", choices=["train", "valid"], default="train")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    groups = np.load(args.groups).astype(int).tolist()
    loader = DataLoader(FeatureRows(args.split, groups), batch_size=args.batch, shuffle=True, drop_last=True)

    def base_views():
        for x in loader:
            yield train_rnn.build_inputs(x.to(device))[..., :BASE_WIDTH]

    model = NextStepExtractor(hidden=args.hidden).to(device)
    sample = torch.cat([b.reshape(-1, BASE_WIDTH)[::7] for _, b in zip(range(2), base_views())])
    model.mean.copy_(sample.mean(0))
    model.std.copy_(sample.std(0).clamp_min(1e-3))
    losses = fit_extractor(model, base_views, device, args.epochs, args.chunk, args.lr)
    torch.save({"model": model.state_dict(), "dims": model.dims, "fit_groups": len(groups), "split": args.split,
                "losses": losses, "seed": args.seed}, args.out)
    print(json.dumps({"out": args.out, "losses": losses}))


if __name__ == "__main__":
    main()
