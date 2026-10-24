"""Run only the frozen F1 ancestry verifier over source metadata and checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from f1_grade_preflight_r2 import (  # noqa: E402
    _read_json,
    _resolve,
    _source_manifest,
    _verify_f1_ancestry,
)
from online_relationship_grade_core import sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse the frozen R2 recipe and create-only metadata report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate recursive F1 source ancestry without opening prediction or target arrays."""
    args = _parse_args()
    recipe_path = _resolve(str(args.recipe))
    recipe_sha256 = sha256_file(recipe_path)
    if recipe_sha256 != args.recipe_sha256:
        raise ValueError("R2 recipe SHA-256 differs from the requested identity")
    recipe = _read_json(recipe_path)
    source_manifest = _source_manifest(recipe)
    subset = _read_json(_resolve(recipe["subset_manifest"]["path"]))
    split = _read_json(_resolve(recipe["split_manifest"]["path"]))
    groups = [int(group) for group in subset["evaluation_groups"]]
    ancestry = _verify_f1_ancestry(recipe, split, groups)
    result: dict[str, Any] = {
        "schema": "wnn.f1-mean-shrink-ancestry-metadata-check.v2",
        "status": "passed",
        "recipe_sha256": recipe_sha256,
        "source_manifest_sha256": recipe["source_manifest"]["sha256"],
        "source_manifest_file_count": len(source_manifest["files"]),
        "split_manifest_sha256": recipe["split_manifest"]["sha256"],
        "evaluation_group_count": len(groups),
        "ancestry": ancestry,
        "access_policy": {
            "prediction_payloads_opened_or_hashed": False,
            "target_cache_opened_or_hashed": False,
            "prediction_values_read": False,
            "target_values_read": False,
            "model_checkpoint_sha256_verified": True,
            "model_checkpoints_imported_or_deserialized": False,
        },
    }
    output_path = _resolve(str(args.output))
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
