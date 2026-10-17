"""Append a forty-channel causal dictionary suffix to the already-pinned pilot arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wnn.moment_geometry.discrete import DISCRETE_WIDTH, discrete_features, fit_dictionary
from wnn.moment_geometry.provenance import validate_storage


def stage_dictionary(source: Path, output: Path) -> dict[str, object]:
    """Fit dictionary/scaler on fitting rows and retain immutable base arrays as hardlinks."""
    validate_storage(output, required_bytes=2_500_000_000)
    output.mkdir(parents=True, exist_ok=False)
    fit_raw = np.load(source / "fit_raw.npy", mmap_mode="r")
    vocabulary = fit_dictionary(fit_raw)
    total, sums, squares = 0, np.zeros(DISCRETE_WIDTH), np.zeros(DISCRETE_WIDTH)
    for raw in fit_raw:
        values = discrete_features(raw, vocabulary).astype(np.float64)
        total += len(values)
        sums += values.sum(axis=0)
        squares += (values * values).sum(axis=0)
    mean = sums / total
    scale = np.maximum(np.sqrt(np.maximum(squares / total - mean * mean, 0)), 0.05)
    for role in ("fit", "eval"):
        base = np.load(source / f"{role}_x.npy", mmap_mode="r")
        raw = np.load(source / f"{role}_raw.npy", mmap_mode="r")
        result = np.lib.format.open_memmap(
            output / f"{role}_x.npy", mode="w+", dtype=np.float32, shape=(*base.shape[:2], 271)
        )
        for index, sequence in enumerate(raw):
            result[index, :, :231] = base[index]
            result[index, :, 231:] = (discrete_features(sequence, vocabulary) - mean) / scale
        result.flush()
        del result, base, raw
        for name in ("y", "mask", "component", "fold"):
            os.link(source / f"{role}_{name}.npy", output / f"{role}_{name}.npy")
    changed = np.array(fit_raw[0], copy=True)
    changed[512:] += np.float32(0.731)
    original = discrete_features(fit_raw[0], vocabulary)[:512]
    if not np.array_equal(original, discrete_features(changed, vocabulary)[:512]):
        raise ValueError("dictionary suffix mutation invariant failed")
    if not np.array_equal(original, discrete_features(fit_raw[0, :512], vocabulary)):
        raise ValueError("dictionary prefix-only invariant failed")
    receipt = {
        "vocabulary": vocabulary.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "source": str(source),
        "width": 271,
        "causal_invariance": "exact",
        "order": "base231/currentcategory16/rowoccupancy16/eventdominant4/dwell4",
        "fit_sequences": len(fit_raw),
        "fitting_only": True,
    }
    (output / "dictionary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Stage the preregistered dictionary comparison on approved physical disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_dictionary(args.source, args.output)


if __name__ == "__main__":
    main()
