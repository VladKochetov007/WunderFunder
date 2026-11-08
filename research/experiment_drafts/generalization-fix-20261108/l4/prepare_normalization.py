from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wnn.training.sequence_config import load_config
from wnn.training.sequence_data import build_data
from wnn.training.sequence_features import numpy_sequence_reader, sequence_input_width
from wnn.training.sequence_normalization import compute_population_moments


def prepare(args: argparse.Namespace) -> dict[str, object]:
    """Time a bounded feature-only sample or publish canonical complete fit-only moments."""
    config = load_config(args.config)
    data = build_data(config, args.data_root, args.split_manifest)
    if len(data.fit) != 10325 or len(data.evaluation) != 1873 or data.split.valid_groups:
        raise ValueError("normalization requires exact train-only/full-valid membership")
    started = time.monotonic()
    if args.sample_sequences:
        count = args.sample_sequences
        if not 1 <= count <= 32:
            raise ValueError("normalization timing probe is bounded to32 fitting sequences")

        def raw_reader(source: str, group: int) -> np.ndarray:
            """Read features alone; labels/masks and validation members never enter moments."""
            records = data.fit.sources[source]
            if records._file is None:
                records._file = records.environment.sequence_file(records.path)
            return records._file.read(group, targets=False, mask=False).features

        reader = numpy_sequence_reader(raw_reader, data.fit.input_feature_mode)
        mean, std = compute_population_moments(
            data.fit.members[:count],
            reader,
            feature_count=sequence_input_width(data.fit.input_feature_mode),
        )
        elapsed = time.monotonic() - started
        return {
            "schema": "wnn.normalization-timing-probe.v1",
            "sample_sequences": count,
            "elapsed_seconds": elapsed,
            "linear_full_fit_estimate_seconds": elapsed * 10325 / count,
            "finite": bool(np.isfinite(mean).all() and np.isfinite(std).all()),
            "cache_or_training_moments_written": False,
            "limitation": "small fitting-prefix timing sample; I/O/cache contention can change full time",
        }
    mean, std = data.normalization()
    return {
        "schema": "wnn.generalization-normalization.v1",
        "status": "passed",
        "input_feature_mode": data.fit.input_feature_mode,
        "feature_count": mean.numel(),
        "normalization": data.provenance["normalization"],
        "split_sha256": data.split.sha256,
        "fit_train": 10325,
        "fit_valid": 0,
        "elapsed_seconds": time.monotonic() - started,
        "finite": bool(np.isfinite(mean.numpy()).all() and np.isfinite(std.numpy()).all()),
    }


def main() -> None:
    """Prepare feature statistics only; never construct a model or use an accelerator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-sequences", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
