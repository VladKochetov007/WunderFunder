"""Prepare one fold's shared sparse input dictionary on CPU."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def _project_path(root: Path, value: str, *, confined: bool = True) -> Path:
    """Resolve a project path, optionally allowing an external manifest."""
    path = Path(value)
    path = (root / path if not path.is_absolute() else path).resolve()
    if confined and not path.is_relative_to(root):
        raise ValueError("manifest and outputs must remain under WNN_PROJECT_ROOT")
    return path


def main() -> int:
    """Fit or reuse the fixed train-only dictionary and write a compact receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--cache", required=True, help="create-only dictionary artifact path")
    parser.add_argument("--output", required=True, help="create-only compact JSON receipt path")
    args = parser.parse_args()
    root = Path(os.environ["WNN_PROJECT_ROOT"]).resolve()
    manifest, cache, output = (
        _project_path(root, args.split_manifest, confined=False),
        _project_path(root, args.cache),
        _project_path(root, args.output),
    )
    if cache.exists() or output.exists() or cache == output:
        raise FileExistsError("cache and receipt paths must be distinct and absent")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from wnn.training.input_representation_job import run_preparation_job

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = run_preparation_job(root, manifest, cache)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    logging.info("prepared representation cache %s", cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
