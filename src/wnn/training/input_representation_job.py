"""Bounded orchestration and metadata for the canonical representation fitter."""

from __future__ import annotations

import os
import signal
from pathlib import Path


def run_preparation_job(root: Path, manifest: Path, cache: Path) -> dict[str, object]:
    """Prepare the create-only shared cache and return compact provenance metadata."""
    import numpy as np
    import torch

    from wnn.splits import file_sha256, load_campaign_split
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.input_representation import input_representation_sha256
    from wnn.training.input_representation_dictionary import (
        REPRESENTATION_RECIPE,
        representation_recipe_sha256,
    )
    from wnn.training.input_representation_fit import prepare_input_representation

    root, manifest, cache = root.resolve(), manifest.resolve(), cache.resolve()
    if not cache.is_relative_to(root):
        raise ValueError("dictionary cache must remain under the data root")
    if cache.exists():
        raise FileExistsError(f"dictionary cache already exists: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    environment = TrainingEnvironment(root=root)
    signal.alarm(600)
    try:
        split = load_campaign_split(manifest, root)
        fitted = prepare_input_representation(
            np.asarray(split.train_groups),
            split,
            environment,
            torch.device("cpu"),
            cache_path=cache,
            output_path=cache.parent,
        )
    finally:
        signal.alarm(0)
    metadata = fitted.fit_metadata
    source_root = Path(environment.source_root).resolve()
    source_files = (
        "src/wnn/data.py",
        "src/wnn/splits.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/input_representation.py",
        "src/wnn/training/input_representation_dictionary.py",
        "src/wnn/training/input_representation_fit.py",
        "src/wnn/training/input_representation_job.py",
        "scripts/fit_input_representation.py",
    )
    return {
        "schema": "wnn.input_representation_preparation.v1",
        "split_manifest_sha256": split.sha256,
        "source_root": str(source_root),
        "data_root": str(root),
        "source_sha256": {name: file_sha256(source_root / name) for name in source_files},
        "recipe": REPRESENTATION_RECIPE,
        "recipe_sha256": representation_recipe_sha256(),
        "membership": {
            "members": metadata["fit_members"],
            "sha256": metadata["fit_members_sha256"],
        },
        "counts": {
            "fit_sequences": len(metadata["fit_members"]),
            "fit_rows": metadata["fit_row_count"],
            "successful_updates": metadata["dictionary_training"][
                "attempted_and_successful_updates"
            ],
        },
        "representation_sha256": input_representation_sha256(fitted),
        "cache": {"path": str(cache), "sha256": file_sha256(cache), "bytes": cache.stat().st_size},
        "runtime": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
