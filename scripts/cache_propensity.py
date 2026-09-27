"""Precompute a frozen propensity net over every sequence once (train and valid), for --prop-cache.

    scripts/limited .venv/bin/python scripts/cache_propensity.py prop_v1 [--check 16]
Writes cache/propcache_<name>_<split>.npy of shape (row groups, 20000) float32 plus a .json sidecar
with the sha256 of the propensity weights; train_rnn.py refuses a cache whose sha differs. The values
are the same computation the training loop runs per batch (build_inputs -> propensity, fp16 autocast,
chunk 4000), so --check N recomputes N random sequences in a training-shaped batch and reports the
max abs difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn as T  # noqa: E402
from wnn.data import N_FEATURES, SequenceFile  # noqa: E402
from wnn.features import N_EVENT_FEATURES  # noqa: E402
from wnn.propensity import load_propensity, propensity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--check", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda")
    wpath = ROOT / "artifacts" / "runs" / a.name / "model.pt"
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    model = load_propensity(wpath, dev)
    for split in ("train", "valid"):
        n = len(SequenceFile(ROOT / "datasets" / f"{split}.parquet"))
        out_path = ROOT / "cache" / f"propcache_{a.name}_{split}.npy"
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 20000))
        idx = np.arange(n)
        if split == "train":
            idx = np.array([g for g in idx if g not in T.CORRUPT_TRAIN_ROW_GROUPS])
        ds = T.Sequences(split, idx)
        loader = DataLoader(ds, batch_size=a.batch, num_workers=4, prefetch_factor=2, pin_memory=True)
        t0, pos = time.time(), 0
        for x, _, _ in loader:
            inputs = T.build_inputs(x.to(dev, non_blocking=True))
            p = propensity(model, inputs[..., : N_FEATURES + N_EVENT_FEATURES]).cpu().numpy()
            out[idx[pos:pos + len(p)]] = p
            pos += len(p)
        out.flush()
        del out
        json.dump({"model": a.name, "model_sha256": sha, "split": split, "rows": int(n), "n_filled": int(len(idx)), "seconds": round(time.time() - t0, 1)},
                  open(out_path.with_suffix(".json"), "w"))
        print(split, "rows", n, "filled", len(idx), f"{time.time() - t0:.0f}s", flush=True)
        # parity: recompute random sequences in a fresh batch of a different composition
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, min(a.check, len(idx)), replace=False)
        cache = np.load(out_path, mmap_mode="r")
        x = torch.stack([torch.from_numpy(T.Sequences(split, [g])[0][0].numpy()) for g in pick])
        fresh = propensity(model, T.build_inputs(x.to(dev))[..., : N_FEATURES + N_EVENT_FEATURES]).cpu().numpy()
        d = np.abs(fresh - np.asarray(cache[pick]))
        print(f"{split} parity on {len(pick)} random sequences: max abs diff {d.max():.3g}, mean {d.mean():.3g}", flush=True)


if __name__ == "__main__":
    main()
