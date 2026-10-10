from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from wnn.data import N_FEATURES, SequenceFile
from wnn.sentinel import SENTINEL, SENTINEL_TOLERANCE, missing_numpy
from wnn.splits import connected_components, file_sha256, load_campaign_split


def fit_safe_statistics(samples: Iterable[np.ndarray]) -> dict:
    """Accumulate nonmissing-only feature moments in float64, then freeze FP32 constants."""
    count = np.zeros(N_FEATURES, np.int64)
    mean, m2 = np.zeros(N_FEATURES, np.float64), np.zeros(N_FEATURES, np.float64)
    for sample in samples:
        values = np.asarray(sample, np.float32)
        if values.ndim != 2 or values.shape[1] != N_FEATURES or not np.isfinite(values).all():
            raise ValueError("sentinel calibration expects finite raw matrices of width112")
        keep = ~missing_numpy(values)
        added = keep.sum(axis=0)
        batch_mean = np.where(keep, values, 0).sum(axis=0, dtype=np.float64) / np.maximum(added, 1)
        delta = batch_mean - mean
        total = count + added
        centered = np.where(keep, values.astype(np.float64) - batch_mean, 0)
        m2 += (centered * centered).sum(axis=0) + delta * delta * count * added / np.maximum(
            total, 1
        )
        mean += delta * added / np.maximum(total, 1)
        count = total
    std = np.sqrt(np.maximum(m2 / np.maximum(count - 1, 1), 0)).clip(1e-3)
    mean = np.where(count > 0, mean, 0)
    std = np.where(count > 1, std, 1)
    return {
        "safe_mean": mean.astype(np.float32).tolist(),
        "safe_std": std.astype(np.float32).tolist(),
        "observed_count": count.tolist(),
        "fallback": "count0: mean0/std1; count1: meanObserved/std1; std floor1e-3",
    }


def dataset_identity(root: Path, split: str) -> dict:
    """Bind full dataset contents, honoring the shared portable-source registry."""
    source = root / "datasets" / f"{split}.parquet"
    registry_name = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if registry_name:
        from wnn.portability import verify_portable_source

        registry = Path(registry_name)
        name = f"datasets/{split}.parquet"
        entry = json.loads(registry.read_text())["sources"][name]
        verify_portable_source(source, name, entry["original_identity"], registry)
        digest = entry["sha256"]
    else:
        digest = file_sha256(source)
    return {"bytes": source.stat().st_size, "sha256": digest}


def calibrate(
    parent_path: Path,
    root: Path,
    sequence_count: int = 64,
    row_stride: int = 7,
    seed: int = 81081,
    split_manifest: Path | None = None,
) -> tuple[dict, dict]:
    """Read only declared parent-fit features; bind components, rows and sources before reuse."""
    if sequence_count < 1 or row_stride < 1:
        raise ValueError("calibration sequence count and row stride must be positive")
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if (
        args.get("arch", "gru") != "gru"
        or args.get("cell", "gru") != "gru"
        or args.get("enc", "linear") != "linear"
        or args.get("privileged_mode", "none") != "none"
        or any(
            checkpoint.get(key)
            for key in ("sentinel_preprocessing", "factorization", "quantization")
        )
    ):
        raise ValueError(
            "sentinel calibration requires an unmodified plain dense causal GRU parent"
        )
    groups = checkpoint.get("provenance", {}).get("fit_row_groups")
    if not groups or set(groups) != {"train", "valid"} or not any(groups.values()):
        raise ValueError("parent checkpoint needs explicit train/valid fit membership")
    groups = {split: sorted(map(int, selected)) for split, selected in groups.items()}
    if any(len(set(selected)) != len(selected) for selected in groups.values()):
        raise ValueError("parent fit membership contains duplicate row groups")
    expected_manifest = args.get("split_manifest_sha256")
    if bool(expected_manifest) != (split_manifest is not None):
        raise ValueError("clean parent calibration requires its matching split manifest")
    if split_manifest is not None:
        outer = load_campaign_split(split_manifest, root)
        if outer.sha256 != expected_manifest:
            raise ValueError("sentinel split manifest differs from parent")
        outer.assert_fit_groups(groups["train"], groups["valid"])
    chain_path = root / "cache/chains.npz"
    with np.load(chain_path, allow_pickle=False) as graph:
        components = connected_components(graph["next"])
        component_of = {
            (str(split), int(group)): int(component)
            for split, group, component in zip(
                graph["split"], graph["row_group"], components, strict=True
            )
        }
    pool = []
    for split, selected in groups.items():
        repeats = int(args.get("valid_repeat", 1)) if split == "valid" and groups["train"] else 1
        for repetition in range(repeats):
            pool.extend((split, group, repetition) for group in selected)
    if not all((split, group) in component_of for split, group, _ in pool):
        raise ValueError("parent membership is outside the bound component graph")
    rng = np.random.default_rng(seed)
    sampled = [pool[index] for index in rng.permutation(len(pool))[:sequence_count]]
    selected_rows, source_rows = [], {}
    readers = {
        split: SequenceFile(root / "datasets" / f"{split}.parquet")
        for split in groups
        if groups[split]
    }

    def samples() -> Iterable[np.ndarray]:
        for split, group, repetition in sampled:
            raw = readers[split].read(group, targets=False, mask=False).features
            rows = np.arange(0, len(raw), row_stride, dtype=np.int64)
            source_rows[f"{split}:{group}"] = len(raw)
            selected_rows.append(
                {
                    "split": split,
                    "row_group": group,
                    "repetition": repetition,
                    "component": component_of[(split, group)],
                    "row_count": len(rows),
                    "row_index_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                }
            )
            yield raw[rows]

    statistics = fit_safe_statistics(samples())
    calibration = {
        "version": 1,
        "parent_sha256": file_sha256(parent_path),
        "split_manifest_sha256": expected_manifest,
        "fit_row_groups": groups,
        "fit_components": sorted({component_of[(split, group)] for split, group, _ in pool}),
        "chains_sha256": file_sha256(chain_path),
        "datasets": {split: dataset_identity(root, split) for split in readers},
        "valid_repeat": args.get("valid_repeat", 1),
        "seed": seed,
        "requested_sequences": sequence_count,
        "selected_sequences": selected_rows,
        "row_stride": row_stride,
        "source_sequence_lengths": source_rows,
        "parent_mean": checkpoint["model"]["mean"][:N_FEATURES].tolist(),
        "parent_std": checkpoint["model"]["std"][:N_FEATURES].tolist(),
        "sentinel": float(SENTINEL),
        "absolute_tolerance": float(SENTINEL_TOLERANCE),
        "scope": "raw-level block only; events/history/propensity remain original raw",
        "component_claim": "clean outer exclusion"
        if expected_manifest
        else "legacy parent membership; no fresh exclusion claim",
        **statistics,
    }
    return checkpoint, calibration


def validate_fit(
    calibration: dict, root: Path, fit_groups: dict, split_manifest_sha256: str | None
) -> None:
    """Reject fitting that consumes constants calibrated outside its declared population."""
    if calibration["split_manifest_sha256"] != split_manifest_sha256:
        raise ValueError("sentinel calibration split differs from fitting split")
    for sample in calibration["selected_sequences"]:
        if sample["row_group"] not in fit_groups[sample["split"]]:
            raise ValueError("sentinel calibration sample is outside current fit membership")
    if file_sha256(root / "cache/chains.npz") != calibration["chains_sha256"]:
        raise ValueError("sentinel calibration component graph changed")
    for split, identity in calibration["datasets"].items():
        if dataset_identity(root, split) != identity:
            raise ValueError("sentinel calibration dataset content changed")
