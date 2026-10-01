"""Convert the cross-fitted train propensity cache to the teammate loader format."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cache" / "propcache_prop_v1_train.npy"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "prop_v1_train.f16"
SHAPE = (10_607, 20_000)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path) -> dict[str, object]:
    """Write a contiguous float16 propensity memmap and a provenance record."""
    values = np.load(source, mmap_mode="r")
    if values.shape != SHAPE:
        raise ValueError(f"expected propensity shape {SHAPE}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("propensity cache contains non-finite values")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("propensity values must lie in [0, 1]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    target = np.memmap(partial, mode="w+", dtype=np.float16, shape=SHAPE)
    for start in range(0, SHAPE[0], 64):
        target[start : start + 64] = values[start : start + 64]
    target.flush()
    del target
    partial.replace(destination)
    record = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "output": str(destination.relative_to(ROOT)),
        "output_sha256": sha256(destination),
        "shape": list(SHAPE),
        "dtype": "float16",
        "conversion": "row-batched float32 to float16; STEP_MASK applied by SequenceReader",
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    """Create the teacher-training mask in the binary format expected by data.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
