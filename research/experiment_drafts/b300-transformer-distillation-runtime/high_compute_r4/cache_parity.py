"""Whole-cache single-rank versus eight-rank export acceptance on the receiver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_cache import TeacherCache


def compare_caches(
    one: Path, eight: Path, data_root: Path, split_manifest: Path, output: Path
) -> None:
    """Compare every fitting row from the same checkpoint and bind both manifests."""
    split = load_campaign_split(split_manifest, data_root)
    members = [("train", g) for g in split.train_groups] + [
        ("valid", g) for g in split.valid_groups
    ]
    left, right = TeacherCache(one, split, members), TeacherCache(eight, split, members)
    if left.metadata["producer"] != right.metadata["producer"]:
        raise ValueError("cache parity requires the same complete checkpoint ancestry")
    if left.metadata["world_size"] != 1 or right.metadata["world_size"] != 8:
        raise ValueError("cache parity requires actual one- and eight-rank exports")
    if left.metadata["hint_dim"] != 0 or right.metadata["hint_dim"] != 0:
        raise ValueError("R4 output-distillation caches must have no hidden hints")
    manifests = {
        "one": file_sha256(one / "manifest.json"),
        "eight": file_sha256(eight / "manifest.json"),
    }
    maximum = 0.0
    for member in members:
        first = np.load(one / left.entries[member]["file"], mmap_mode="r", allow_pickle=False)
        second = np.load(eight / right.entries[member]["file"], mmap_mode="r", allow_pickle=False)
        if any(
            value.dtype != np.float32 or value.shape != (20_000, 2) for value in (first, second)
        ):
            raise ValueError("invalid parity array shape or dtype")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("nonfinite parity arrays")
        maximum = max(maximum, float(np.max(np.abs(first - second))))
        if not np.allclose(first, second, rtol=1e-5, atol=1e-6):
            raise ValueError(f"one/eight-rank export parity failed at {member}")
        for root, cache in ((one, left), (eight, right)):
            entry = cache.entries[member]
            if file_sha256(root / entry["file"]) != entry["sha256"]:
                raise ValueError("cache payload changed during parity verification")
    if manifests != {
        "one": file_sha256(one / "manifest.json"),
        "eight": file_sha256(eight / "manifest.json"),
    }:
        raise ValueError("cache manifest changed during parity verification")
    receipt = {
        "schema": "wnn.sequence-cache-parity.v1",
        "passed": True,
        "split_sha256": split.sha256,
        "checkpoint_sha256": left.metadata["producer"]["checkpoint_sha256"],
        "manifest_sha256": manifests,
        "sequences": len(members),
        "max_abs_error": maximum,
        "rtol": 1e-5,
        "atol": 1e-6,
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Dispatch whole-cache acceptance without running a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("one", "eight", "data-root", "split-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare_caches(args.one, args.eight, args.data_root, args.split_manifest, args.output)


if __name__ == "__main__":
    main()
