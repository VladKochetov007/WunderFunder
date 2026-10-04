"""Cache train-only-decoded class-3 future quote-path auxiliary labels."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile
from wnn.ticks import CLASS3_QUANTUM, reconstruct

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (8, 24)
SEQ_LEN = 20_000


def event_labels(features: np.ndarray, moves: dict, spreads: dict) -> tuple[np.ndarray, int, int]:
    """Return row-aligned future event-average displacements in target units."""
    events = np.r_[0, np.flatnonzero(np.any(features[1:, :22] != features[:-1, :22], axis=1)) + 1]
    ask, bid, unknown = reconstruct(features[events], moves, spreads)
    mid = 0.5 * (ask + bid)
    prefix = np.r_[0.0, np.cumsum(mid)]
    values = np.full((len(events), len(HORIZONS)), np.nan, dtype=np.float32)
    for column, horizon in enumerate(HORIZONS):
        stop = len(events) - horizon
        if stop <= 0:
            continue
        future_mean = (prefix[horizon + 1 :] - prefix[1:stop + 1]) / horizon
        values[:stop, column] = np.clip(
            CLASS3_QUANTUM * (future_mean - mid[:stop]), -2.0, 2.0
        )
    event_id = np.searchsorted(events, np.arange(len(features)), side="right") - 1
    return values[event_id], unknown, len(events) - 1


def main() -> None:
    """Write one mmap-compatible label array, indexed by parquet row group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "valid"))
    args = parser.parse_args()
    classes = np.load(ROOT / "research/artifacts/market-classes-v1.npz")[args.split]
    with (ROOT / "cache/class3_moves_full.pkl").open("rb") as handle:
        moves, spreads = pickle.load(handle)
    source = SequenceFile(ROOT / "datasets" / f"{args.split}.parquet")
    destination = ROOT / "cache" / f"quote_path_{args.split}.npy"
    partial = destination.with_name(destination.name + ".partial")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(len(classes), SEQ_LEN, len(HORIZONS)),
    )
    output[:] = 0
    unknown_events = total_events = 0
    for count, index in enumerate(np.flatnonzero(classes == 3), 1):
        features = source.read(int(index), targets=False, mask=False).features
        if len(features) != SEQ_LEN:
            raise ValueError(f"row group {index} has {len(features)} rows")
        labels, unknown, n_events = event_labels(features, moves, spreads)
        output[index] = labels.astype(np.float16)
        unknown_events += unknown
        total_events += n_events
        if count % 250 == 0:
            print(f"{args.split}: {count} class-3 sequences cached", flush=True)
    output.flush()
    del output
    partial.replace(destination)
    metadata = {
        "split": args.split,
        "shape": [int(len(classes)), SEQ_LEN, len(HORIZONS)],
        "horizons_book_events": list(HORIZONS),
        "class3_sequences": int(np.sum(classes == 3)),
        "unknown_transitions": int(unknown_events),
        "total_transitions": int(total_events),
        "decoder": "cache/class3_moves_full.pkl (train fitted)",
        "target_units": "clipped class-3 target quantum displacement from current decoded mid",
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
