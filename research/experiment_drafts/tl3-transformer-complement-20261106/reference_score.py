"""Score frozen TL3 reference predictions with the unchanged official accumulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.data import WARMUP, SequenceFile
from wnn.evaluation import paired_component_moments
from wnn.gmetric import GlobalAccumulator, _merge_moments
from wnn.splits import load_campaign_split
from wnn.training.sequence_reference_cache import ReferenceCache


def main() -> None:
    """Replay the full reference without loading or executing its models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_campaign_split(args.split_manifest, args.data_root)
    members = [("valid", group) for group in split.evaluation_groups]
    cache = ReferenceCache(args.reference_cache, split, members, membership="evaluation")
    source = SequenceFile(args.data_root / "datasets/valid.parquet")
    component_map = {(s, g): c for s, g, c in split.components}
    accumulator = GlobalAccumulator()
    centered: dict[int, np.ndarray] = {}
    for source_name, group in members:
        values, _ = cache.read(source_name, group, 0)
        values = values.numpy()
        seq = source.read(group)
        if seq.scored is None or seq.targets is None:
            raise ValueError("official valid mask/targets unavailable")
        mask = seq.scored.copy()
        mask[:WARMUP] = False
        component = component_map[(source_name, group)]
        accumulator.add(seq.targets, values, mask, group=component)
        if mask.any():
            moments, _ = paired_component_moments(
                seq.targets[None], values[None], values[None], mask[None]
            )
            previous = centered.get(component, np.zeros((2, 6), dtype=np.float64))
            centered[component] = np.stack(
                [_merge_moments(previous[k], moments[0, k]) for k in range(2)]
            )
    cache.verify_unchanged()
    result = {
        "schema": "wnn.tl3-reference-score.v1",
        "score": accumulator.result(),
        "split_sha256": split.sha256,
        "predictions_manifest_sha256": cache.identity["manifest_sha256"],
        "producer": cache.identity,
        "sequences": len(members),
        "component_centered_moments": {str(k): v.tolist() for k, v in centered.items()},
        "interpretation": "Exposed official valid; frozen clean reference, no new independent fit.",
    }
    # A wrong native route or ancestor must fail before four expensive fits.
    if abs(result["score"]["wp"] - 0.6923774245652725) > 2e-6:
        raise ValueError("reference replay differs from the pinned PR49 Torch baseline")
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
