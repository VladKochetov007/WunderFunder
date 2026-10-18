"""Profile one exact fit-side sequence through the frozen feature and ridge path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
from ridge_model import WeightedRidge, WeightedStats
from run_cv import ARMS, _feature_arms, _verify_prefix_invariants
from source_utils import (
    ROOT,
    load_members_and_folds,
    require_physical_scratch,
    verify_recipe_sources,
    verify_train_footer,
    write_json,
)

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile


def run(root: Path, recipe_path: Path, output_path: Path) -> dict[str, Any]:
    """Measure actual single-sequence extraction and fit accumulation, without scoring."""
    started = time.perf_counter()
    output = require_physical_scratch(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing profile: {output}")
    output.mkdir(parents=True)

    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    source_hashes = verify_recipe_sources(root, recipe)
    members, fold_by_component, input_hashes = load_members_and_folds(root, recipe)
    selection = recipe["selection"]
    train_path = root / selection["data_path"]
    train_identity = verify_train_footer(train_path, selection["data_identity"])

    source = SequenceFile(train_path)
    if len(source) != int(selection["row_groups"]):
        raise ValueError("train Parquet row-group count differs from frozen recipe")
    member = members[0]
    sequence = source.read(member["row_group"], targets=True, mask=False)
    if sequence.seq_ix != member["seq_ix"] or sequence.targets is None:
        raise ValueError("profile sequence identity or fit labels do not match the frozen sample")
    if sequence.features.shape[0] != SEQUENCE_LENGTH:
        raise ValueError("profile sequence does not have the expected complete length")

    designs = _feature_arms(sequence.features)
    if any(len(values) != SEQUENCE_LENGTH - WARMUP for values in designs.values()):
        raise ValueError("profile feature rows do not align after the frozen warmup")
    _verify_prefix_invariants(sequence.features, designs)

    config = recipe["model"]
    stride = int(config["fit_row_stride"])
    target = np.clip(
        sequence.targets[WARMUP::stride],
        -float(config["target_clip"]),
        float(config["target_clip"]),
    )
    model_scales: dict[str, list[dict[str, float]]] = {}
    for arm in ARMS:
        sampled = designs[arm][::stride].astype(np.float64, copy=False)
        model_scales[arm] = []
        for target_index in range(2):
            target_values = target[:, target_index].astype(np.float64, copy=False)
            accumulator = WeightedStats.empty(sampled.shape[1])
            accumulator.update(sampled, target_values, np.abs(target_values))
            fitted = WeightedRidge.fit(accumulator, float(config["ridge_alpha"]))
            if not np.isfinite(fitted.coefficients).all():
                raise FloatingPointError(f"profile ridge coefficients are non-finite for {arm}")
            model_scales[arm].append(
                {
                    "feature_width": sampled.shape[1],
                    "minimum_fit_scale": float(fitted.scale_x.min()),
                    "median_fit_scale": float(np.median(fitted.scale_x)),
                    "maximum_fit_scale": float(fitted.scale_x.max()),
                }
            )

    result = {
        "experiment_id": recipe["experiment_id"],
        "profile_id": "one_complete_fit_component_exact_feature_ridge_path",
        "status": "completed_resource_profile_only",
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "source_hashes": source_hashes,
        "sample_source_hashes": input_hashes,
        "data_identity": train_identity,
        "member": {**member, "fold": fold_by_component[member["component"]]},
        "rows": SEQUENCE_LENGTH,
        "targets_read": "one predeclared fold0.fit.train component only for stable-ridge shape profile",
        "masks_read": False,
        "oof_predictions_created": False,
        "official_metric": None,
        "quality_claim": None,
        "model_scale_diagnostics": model_scales,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
    }
    receipt_path = output / "profile_receipt.json"
    write_json(receipt_path, result)
    print(json.dumps({"status": result["status"], "receipt": str(receipt_path)}), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    """Parse the immutable recipe and physical-scratch profile path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / Path(__file__).relative_to(ROOT).parent / "recipe.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the one-component profile without grading or prediction output."""
    arguments = parse_args()
    run(arguments.root.resolve(), arguments.recipe.resolve(), arguments.output)


if __name__ == "__main__":
    main()
