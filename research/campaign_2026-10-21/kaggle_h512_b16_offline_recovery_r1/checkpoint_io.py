"""Hash-bound loading, copying and canonical trajectory mean helpers."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recovery_contract import (
    DERIVED_SHA,
    EXPECTED_NORMALIZATION,
    FIT_NAME,
    MEAN_SOURCE_SHA,
    JsonObject,
    require,
    sha256,
    verify_pinned,
)


def canonical_mean_state(
    project_root: Path,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return the pinned canonical equal-parameter mean helper after checking its source."""
    helper_path = project_root / "scripts/average_trajectory_checkpoints.py"
    verify_pinned(helper_path, MEAN_SOURCE_SHA, "canonical checkpoint mean helper")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.average_trajectory_checkpoints import mean_state

    return mean_state


def load_source_checkpoints(fit_dir: Path) -> list[JsonObject]:
    """Load only the three small hash-bound checkpoints and verify common ancestry."""
    import torch

    checkpoints = []
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        for epoch in (3, 4, 5):
            path = fit_dir / f"model_ep{epoch}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            require(
                checkpoint.get("epoch") == epoch and checkpoint.get("n_in") == 211,
                f"CP{epoch} architecture identity changed",
            )
            require(
                checkpoint.get("args", {}).get("name") == FIT_NAME, f"CP{epoch} run name changed"
            )
            require(
                checkpoint.get("provenance", {}).get("split_manifest_sha256") == DERIVED_SHA,
                f"CP{epoch} split identity changed",
            )
            require(
                checkpoint.get("provenance", {}).get("initial_checkpoint_sha256") is None,
                f"CP{epoch} unexpectedly has a parent checkpoint",
            )
            require(
                "initial_parameter_sha256" not in checkpoint.get("provenance", {}),
                f"CP{epoch} unexpectedly contains a reconstructed initialization identity",
            )
            require(
                "initial_parameter_sha256" not in checkpoint,
                f"CP{epoch} has an unexpected top-level initialization identity",
            )
            require(
                checkpoint.get("provenance", {}).get("normalization_identity")
                == EXPECTED_NORMALIZATION,
                f"CP{epoch} normalization identity changed",
            )
            checkpoints.append(checkpoint)
    args = [checkpoint.get("args") for checkpoint in checkpoints]
    require(args[0] == args[1] == args[2], "CP3/4/5 trainer arguments differ")
    provenance = [checkpoint.get("provenance", {}) for checkpoint in checkpoints]
    for field in (
        "split_manifest_sha256",
        "fit_membership_identity",
        "normalization_identity",
        "initial_checkpoint_sha256",
    ):
        require(
            provenance[0].get(field) == provenance[1].get(field) == provenance[2].get(field),
            f"CP3/4/5 ancestry differs at {field}",
        )
    return checkpoints


def copy_file(source: Path, destination: Path, expected_hash: str | None = None) -> str:
    """Copy one immutable source artifact and verify that the destination is byte-identical."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = sha256(destination)
    if expected_hash is not None:
        require(digest == expected_hash, f"Copied artifact identity differs: {source.name}")
    return digest
