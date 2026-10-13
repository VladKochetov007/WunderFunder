"""Export exact native member predictions for a fixed final-package diagnostic panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS, GlobalAccumulator, validate_sequence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ("weights", "native", "split", "data"):
        if digest(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"{name} content differs from frozen identity")
    from wnn.splits import load_campaign_split

    split = load_campaign_split(args.split, ROOT)
    groups = list(split.evaluation_groups)
    sys.path.insert(0, str(args.native.parent.resolve()))
    import wnn_native

    codes = None
    if args.packed:
        from wnn.quantization_export import read_packed_quantized_export

        weights, codes, _ = read_packed_quantized_export(args.weights)
    else:
        with np.load(args.weights, allow_pickle=False) as archive:
            weights = {
                name: np.ascontiguousarray(archive[name], dtype=np.float32)
                for name in archive.files
            }
    model = wnn_native.EnsemblePredictor(
        [weights],
        [np.float32(1)],
        fast_activations=True,
        precision="q8vnni",
        q8_codes=None if codes is None else [codes],
        budget_s=None,
    )
    args.output.mkdir(parents=True)
    output_path = args.output / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(groups), 20_000, 2)
    )
    parquet = pq.ParquetFile(args.data)
    total = GlobalAccumulator()
    callback_seconds = 0.0
    first: tuple[np.ndarray, np.ndarray] | None = None
    with (args.output / "progress.jsonl").open("x") as progress:
        for index, group in enumerate(groups):
            table = parquet.read_row_group(group, use_threads=False)
            _, need = validate_sequence(table)
            if table.num_rows != 20_000:
                raise ValueError("diagnostic panel requires complete 20,000-row sequences")
            features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(
                np.float32
            )
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(
                np.float32
            )
            mask = need & table["is_scored"].to_numpy().astype(bool)
            model.reset()
            started = time.monotonic()
            values = np.asarray([model.step(row) for row in features], dtype=np.float32)
            callback_seconds += time.monotonic() - started
            if values.shape != (20_000, 2) or not np.isfinite(values).all():
                raise ValueError("invalid native predictions")
            predictions[index] = values
            total.add(targets, values, mask)
            if first is None:
                first = features.copy(), values.copy()
            progress.write(json.dumps({"index": index, "group": group}) + "\n")
            progress.flush()
            if (index + 1) % 8 == 0:
                predictions.flush()
                predictions._mmap.madvise(mmap.MADV_DONTNEED)
                with output_path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert first is not None
    model.reset()
    replay = np.asarray([model.step(row) for row in first[0]], dtype=np.float32)
    if not np.array_equal(replay, first[1]):
        raise ValueError("A-to-B-to-A full-sequence reset replay differs")
    predictions.flush()
    del predictions
    result = {
        "scope": "final fitted-data diagnostic; no held-out or public-score claim",
        "weights": str(args.weights),
        "weights_sha256": args.weights_sha256,
        "native_sha256": args.native_sha256,
        "split_sha256": args.split_sha256,
        "data_sha256": args.data_sha256,
        "groups": groups,
        "score": total.result(),
        "callback_seconds": callback_seconds,
        "reset_replay_bitwise": True,
        "prediction_sha256": digest(output_path),
        "source_sha256": digest(Path(__file__)),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
