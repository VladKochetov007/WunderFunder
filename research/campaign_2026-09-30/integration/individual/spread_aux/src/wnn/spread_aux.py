from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from torch import Tensor

UNKNOWN = 255


def file_sha256(path: Path) -> str:
    """Hash a cache artifact without loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_spread_cache(
    root: Path,
    split_sha256: str,
    train: Iterable[int],
    valid: Iterable[int],
    evaluation: Iterable[int],
) -> dict:
    """Reject incomplete, changed, or differently partitioned supervised caches."""
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete" or metadata.get("supervision_only") is not True:
        raise ValueError("spread cache is incomplete or not supervision-only")
    if set(metadata.get("files", {})) != {"train", "valid"} or metadata.get("N_SP_AUX") != 12:
        raise ValueError("spread cache file set/head dimension differs")
    if metadata["split_manifest_sha256"] != split_sha256:
        raise ValueError("spread cache split differs")
    for name, groups in [
        ("fit_train_groups", train),
        ("fit_valid_groups", valid),
        ("evaluation_valid_groups", evaluation),
    ]:
        if len(set(metadata[name])) != len(metadata[name]) or set(metadata[name]) != set(
            map(int, groups)
        ):
            raise ValueError(f"spread cache {name} differs")
    if set(metadata["fit_valid_groups"]) & set(metadata["evaluation_valid_groups"]):
        raise ValueError("spread cache fitting/evaluation VALID membership overlaps")
    if metadata["unknown"] != UNKNOWN or metadata["dtype"] != "uint8":
        raise ValueError("unknown-label encoding changed")
    for path_key, expected in [
        ("selection.json", metadata["selection_sha256"]),
        ("interior_code_table.json", metadata["map_sha256"]),
    ]:
        if file_sha256(root / path_key) != expected:
            raise ValueError(f"spread provenance hash differs: {path_key}")
    for split, entry in metadata["files"].items():
        if entry["file"] != f"spread_{split}.npy" or entry["shape"] != [
            len(metadata[f"fit_{split}_groups"]),
            20_000,
        ]:
            raise ValueError("spread cache file path/shape differs")
        if entry["row_groups"] != metadata[f"fit_{split}_groups"]:
            raise ValueError("cache row order differs from its exact group index")
        if file_sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError(f"spread cache array hash differs: {split}")
        values = np.load(root / entry["file"], mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint8 or list(values.shape) != entry["shape"]:
            raise ValueError("spread cache array dtype/shape differs")
        for first in range(0, len(values), 128):
            block = values[first : first + 128]
            if np.any((block >= 12) & (block != UNKNOWN)):
                raise ValueError("spread cache contains invalid bin/unknown values")
    return {
        "metadata_sha256": file_sha256(metadata_path),
        "map_sha256": metadata["map_sha256"],
        "selection_sha256": metadata["selection_sha256"],
        "array_sha256": {split: entry["sha256"] for split, entry in metadata["files"].items()},
        "supervision_only": True,
        "unknown": UNKNOWN,
    }


class PackedSpreadCache:
    """Sparse row-group label cache; outside fitting membership is an error."""

    def __init__(self, root: Path, split: str) -> None:
        self.metadata = json.loads((root / "metadata.json").read_text())
        if self.metadata.get("status") != "complete":
            raise ValueError("incomplete spread cache")
        entry = self.metadata["files"][split]
        self.position = {int(group): index for index, group in enumerate(entry["row_groups"])}
        self.values = np.load(root / entry["file"], mmap_mode="r", allow_pickle=False)
        if list(self.values.shape) != entry["shape"] or self.values.dtype != np.uint8:
            raise ValueError("spread cache shape/type differs")

    def read(self, group: int) -> np.ndarray:
        """Return copied encoded supervision for exactly the requested FIT group."""
        if int(group) not in self.position:
            raise ValueError(f"row group{group} is outside spread cache FIT membership")
        return np.array(self.values[self.position[int(group)]], dtype=np.uint8, copy=True)


def structural_loss(
    output: Tensor,
    spread: Tensor,
    branch: Tensor,
    classes: Tensor,
    weights: Tensor,
    start: int,
    spread_bins: int,
    encoded_spread: Tensor | None = None,
) -> Tensor:
    """Use original weighting; mask only the spread numerator on unknown codes.

    Args:
        output: Existing full-head logits for one TBPTT chunk.
        spread: Original proxy or cached-current spread labels, already selected.
        branch: Unmodified original branch labels for this chunk.
        classes: Unmodified original class labels for this chunk.
        weights: Original `sw` for this chunk; denominator remains `weights.sum`.
        start: Existing structural output-head start.
        spread_bins: Existing N_SP_AUX, not a new output dimension.
        encoded_spread: Shared uint8 cache (255unknown), used solely as loss mask.

    Returns:
        Structural CE before the existing aux_struct coefficient.
    """
    logits = output.float()
    row_weights = weights.reshape(-1)
    ce = 0.0
    for lo, hi, label in [
        (start, start + spread_bins, spread),
        (start + spread_bins, start + spread_bins + 3, branch),
        (start + spread_bins + 3, start + spread_bins + 7, classes),
    ]:
        term = F.cross_entropy(
            logits[..., lo:hi].reshape(-1, hi - lo), label.reshape(-1), reduction="none"
        )
        if lo == start and encoded_spread is not None:
            term = term * (encoded_spread.reshape(-1) != UNKNOWN)
        ce = ce + term
    return (ce * row_weights).sum() / row_weights.sum().clamp_min(1e-8)
