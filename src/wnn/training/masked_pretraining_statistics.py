from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.splits import CampaignSplit, file_sha256, load_campaign_split
from wnn.training.environment import TrainingEnvironment
from wnn.training.masked_group_pretraining import (
    BASE_INPUT_WIDTH,
    fit_input_statistics,
    fitting_members,
)


def load_pretraining_membership(
    root: Path, split_path: Path
) -> tuple[TrainingEnvironment, CampaignSplit, tuple[tuple[str, int], ...]]:
    """Validate the fold manifest and return only its fitting row groups."""
    environment = TrainingEnvironment(root=root.resolve())
    split = load_campaign_split(split_path, environment.root)
    members = fitting_members(split)
    if not members:
        raise ValueError("campaign split contains no feature-only fitting groups")
    return environment, split, members


def _statistics_sample(
    members: tuple[tuple[str, int], ...], *, seed: int, sample_sequences: int
) -> list[int]:
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(len(members), size=min(sample_sequences, len(members)), replace=False)
    ).tolist()


def _statistics_sources(environment: TrainingEnvironment) -> dict[str, str]:
    names = (
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/splits.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/masked_group_pretraining.py",
        "src/wnn/training/masked_pretraining_statistics.py",
    )
    return {name: file_sha256(environment.source_root / name) for name in names}


def prepare_statistics(args: argparse.Namespace) -> None:
    """Write or verify deterministic fit-only canonical input statistics."""
    if args.cpu_threads <= 0 or args.sample_sequences <= 0 or args.row_stride <= 0:
        raise ValueError("CPU threads, statistics sample count, and row stride must be positive")
    torch.set_num_threads(args.cpu_threads)
    environment, split, members = load_pretraining_membership(args.root, args.split_manifest)
    selected = _statistics_sample(members, seed=args.seed, sample_sequences=args.sample_sequences)
    if args.verify_only:
        _verify_statistics(args, split, members, selected, environment)
        return
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing statistics; verify or archive them first"
        )
    mean, std = fit_input_statistics(
        members,
        selected,
        root=environment.root,
        environment=environment,
        row_stride=args.row_stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = args.output.with_suffix(f".tmp-{os.getpid()}.npz")
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(stream, mean=mean, std=std)
    temporary_archive.replace(args.output)
    metadata = {
        "schema": "causal_group_mask_stats_v1",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split.sha256,
        "fold": split.fold,
        "sampling_seed": args.seed,
        "fit_member_count": len(members),
        "stats_sample_members": [members[index] for index in selected],
        "row_stride": args.row_stride,
        "input_width": BASE_INPUT_WIDTH,
        "mean_std_floor": 1e-3,
        "statistics_sha256": file_sha256(args.output),
        "source_sha256": _statistics_sources(environment),
    }
    temporary_metadata = metadata_path.with_suffix(f".tmp-{os.getpid()}.json")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    print(json.dumps(metadata, indent=2))


def _verify_statistics(
    args: argparse.Namespace,
    split: CampaignSplit,
    members: tuple[tuple[str, int], ...],
    selected: list[int],
    environment: TrainingEnvironment,
) -> None:
    """Reject a cached statistics archive from a different fold or source."""
    if not args.output.is_file():
        raise FileNotFoundError(f"statistics archive is missing: {args.output}")
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "causal_group_mask_stats_v1",
        "split_manifest_sha256": split.sha256,
        "fold": split.fold,
        "sampling_seed": args.seed,
        "fit_member_count": len(members),
        "stats_sample_members": [list(members[index]) for index in selected],
        "row_stride": args.row_stride,
        "input_width": BASE_INPUT_WIDTH,
        "statistics_sha256": file_sha256(args.output),
        "source_sha256": _statistics_sources(environment),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("existing statistics archive differs from the frozen fold recipe")


def load_statistics(
    path: Path, split: CampaignSplit, environment: TrainingEnvironment
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load hash-bound normalization and verify its exact 64-sequence sample."""
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    members = fitting_members(split)
    expected_sample = _statistics_sample(members, seed=880301 + split.fold, sample_sequences=64)
    expected_members = [list(members[index]) for index in expected_sample]
    if metadata.get("schema") != "causal_group_mask_stats_v1":
        raise ValueError("statistics metadata schema is unsupported")
    if metadata.get("split_manifest_sha256") != split.sha256:
        raise ValueError("statistics belong to a different campaign split")
    if metadata.get("statistics_sha256") != file_sha256(path):
        raise ValueError("statistics archive hash differs from its metadata")
    if metadata.get("source_sha256") != _statistics_sources(environment):
        raise ValueError("statistics were produced by different feature or data code")
    if metadata.get("sampling_seed") != 880301 + split.fold:
        raise ValueError("statistics do not use the frozen fold-specific sampling seed")
    if (
        metadata.get("fit_member_count") != len(members)
        or metadata.get("stats_sample_members") != expected_members
        or len(expected_sample) != 64
    ):
        raise ValueError("statistics sample membership differs from the frozen 64-sequence sample")
    if metadata.get("row_stride") != 64 or metadata.get("input_width") != BASE_INPUT_WIDTH:
        raise ValueError("statistics schema differs from the frozen recipe")
    with np.load(path, allow_pickle=False) as arrays:
        mean = np.asarray(arrays["mean"], dtype=np.float32)
        std = np.asarray(arrays["std"], dtype=np.float32)
    if mean.shape != (BASE_INPUT_WIDTH,) or std.shape != mean.shape:
        raise ValueError("statistics width differs from canonical inputs")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("statistics contain nonfinite means or invalid standard deviations")
    return torch.from_numpy(mean), torch.from_numpy(std), metadata
