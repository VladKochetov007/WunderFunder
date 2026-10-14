#!/usr/bin/env python3
"""Profile source-qualified F0 batches across the fixed eight-shape ladder.

This is a bounded resource and runtime probe. It strictly validates the
historical clean-T3 cache contract, then checks a separate matched profile
contract before training-style FP32 updates. It does not grade quality or fit
an epoch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from wnn.student_frontier_order import (
    members_sha256,
    positions_sha256,
    seeded_epoch_positions,
    source_qualified_members,
    source_qualified_target_positions,
)

SHAPE_PROFILE_ORDER = ((192, 2), (256, 1), (256, 2), (384, 2),
                       (512, 1), (640, 1), (512, 2), (448, 3))
PROFILE_BATCH_CHOICES = (1, 4, 8, 16)
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
MAX_SPLIT_MANIFEST_BYTES = 2_000_000
MAX_CACHE_METADATA_BYTES = 512_000
MAX_SMALL_JSON_BYTES = 64_000
N_EVAL_GROUPS_F0 = 468
SEQUENCE_ROWS = 20_000
CHUNK_ROWS = 2_000
BATCH_SIZE = 16
SEED = 3
EPOCH = 0
ALPHA = 0.9
NORM_MEAN_SHA256 = "bb85c72b87656979f97c1786400aec6bc9e43acd337078000eb482c09c5a2da8"
NORM_STD_SHA256 = "b730600851e52e246fc638230844d6e4f2828e4972b45db1f0128cce4e4454c5"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_CACHE_METADATA_SHA256 = "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8"
EXPECTED_PROPENSITY_RUN = "c20260929_prop_f0"
EXPECTED_PROPENSITY_MODEL_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
EXPECTED_PROPENSITY_PROVENANCE_SHA256 = "f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812"
EXPECTED_FIRST_BATCH_POSITIONS = [
    10562, 8725, 6305, 1586, 2134, 11477, 8128, 1560,
    2922, 8340, 4913, 2805, 4517, 9259, 2632, 2203,
]
EXPECTED_FIRST_BATCH_MEMBERS = [
    ["valid", 37], ["train", 9839], ["train", 1639], ["train", 7977],
    ["train", 7548], ["valid", 1263], ["train", 3788], ["train", 9480],
    ["train", 2478], ["train", 1094], ["train", 7529], ["train", 10227],
    ["train", 2131], ["train", 9134], ["train", 1832], ["train", 5224],
]
EXPECTED_FIRST_BATCH_SEQ_IX = [
    44346154, 1997333359, 342781553, 1629092802,
    1546977674, 1466505621, 774192677, 1930716618,
    514227336, 230774826, 1544636377, 2077965454,
    444849898, 1864890861, 382374263, 1074328294,
]
EXPECTED_DROPPED_TAIL_POSITIONS = [5638, 4941, 3775, 2355]
EXPECTED_DROPPED_TAIL_MEMBERS = [
    ["train", 3449], ["train", 2425], ["train", 879], ["train", 7224],
]
EXPECTED_TRAIN_ORDER_SHA256 = "98a3e8da248f52e307af61efaeab845a76cf8e89e005ea5630c0af9a133508a9"
EXPECTED_SAMPLER_POSITIONS_SHA256 = "0ebdb64780b034ced17e78bd95cb0f3b94e86cd6575b5ee2d6bc0680ee820c69"
EXPECTED_SAMPLER_SOURCE_ROWS_SHA256 = "124edb9c19924911d173c1082c49f53b26ad281995c6300266ab46d402c827e2"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded 4 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash an in-memory value."""
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a normalization buffer using canonical little-endian float32 bytes."""
    array = value.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    return sha256_bytes(array.tobytes())


def bounded_json(path: Path, maximum_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    """Read a JSON object only after enforcing a hard file-size bound."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{label} exceeds its {maximum_bytes}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{label} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def cgroup_memory_current() -> int | None:
    """Read this process's cgroup-v2 memory charge when available."""
    try:
        entries = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::")]
        if len(entries) != 1:
            return None
        path = (Path("/sys/fs/cgroup") / entries[0].lstrip("/")).resolve()
        root = Path("/sys/fs/cgroup").resolve()
        if not path.is_relative_to(root):
            return None
        value = (path / "memory.current").read_text().strip()
        return int(value) if value.isascii() and value.isdigit() else None
    except (OSError, ValueError):
        return None


def nvidia_smi_used_bytes() -> int:
    """Return a conservative sum of current GPU usage reported by nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values or any(not item.isascii() or not item.isdigit() for item in values):
        raise RuntimeError(f"unparseable nvidia-smi memory output: {result.stdout!r}")
    return sum((int(item) + 1) * 1024**2 for item in values)


def source_manifest(script: Path, repo: Path) -> dict[str, str]:
    """Bind this harness to the production data, objective, cache and model code."""
    paths = (
        script,
        repo / "scripts/train_rnn.py",
        repo / "src/wnn/data.py",
        repo / "src/wnn/distillation_cache.py",
        repo / "src/wnn/features.py",
        repo / "src/wnn/nn.py",
        repo / "src/wnn/propensity.py",
        repo / "src/wnn/resources.py",
        repo / "src/wnn/splits.py",
        repo / "src/wnn/student_frontier.py",
        repo / "src/wnn/student_frontier_order.py",
        repo / "src/wnn/training_controls.py",
        repo / "src/wnn/training_dynamics.py",
        repo / "src/wnn/training_sampling.py",
    )
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def load_bound_norm(root: Path, contract: dict[str, Any], split: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy only the pinned H256 control's common 211-feature mean/std buffers."""
    source = contract["normalization_source"]
    checkpoint_path = root / source["checkpoint"]
    provenance_path = root / source["provenance"]
    if (sha256_file(checkpoint_path) != source["checkpoint_sha256"]
            or sha256_file(provenance_path) != source["provenance_sha256"]):
        raise ValueError("frozen normalization checkpoint or provenance changed")
    provenance, _ = bounded_json(provenance_path, 8_000_000, "normalization provenance")
    if (provenance.get("split_manifest_sha256") != split.sha256
            or set(provenance.get("fit_row_groups", {}).get("train", [])) != set(split.train_groups)
            or set(provenance.get("fit_row_groups", {}).get("valid", [])) != set(split.valid_groups)
            or set(provenance.get("evaluation_row_groups", [])) != set(split.evaluation_groups)):
        raise ValueError("normalization provenance does not use the same source-qualified clean F0 split")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 5:
        raise ValueError("normalization source is not fixed checkpoint epoch 5")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not {"mean", "std"}.issubset(state):
        raise ValueError("normalization checkpoint has no mean/std model buffers")
    mean = state["mean"].detach().to(device="cpu", dtype=torch.float32).clone()
    std = state["std"].detach().to(device="cpu", dtype=torch.float32).clone()
    del state, checkpoint
    if (mean.shape != (211,) or std.shape != (211,)
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or torch.any(std <= 0)
            or tensor_sha256(mean) != NORM_MEAN_SHA256
            or tensor_sha256(std) != NORM_STD_SHA256):
        raise ValueError("shared H256 normalization buffers differ from the frozen 211-feature identities")
    return mean, std


def validate_cache_rows(cache_metadata: dict[str, Any], split: Any) -> dict[tuple[str, int], int]:
    """Bind every cache row to a source-local identity and verify source-qualified targets."""
    train = cache_metadata["files"]["main_homogeneous_train"]["row_groups"]
    valid = cache_metadata["files"]["main_homogeneous_valid"]["row_groups"]
    mapping = source_qualified_target_positions(train, valid, split.train_groups, split.valid_groups)
    if len(mapping) != len(split.train_groups) + len(split.valid_groups):
        raise ValueError("not every homogeneous cache row mapped to one source-qualified fit group")
    return mapping


def validate_order_and_sources(
    root: Path, contract: dict[str, Any], trainer: Any
) -> tuple[Any, dict[str, Any], dict[tuple[str, int], int], dict[str, Any]]:
    """Strictly validate fit data, full cache payloads and all source-qualified row identities."""
    split_path = root / contract["split"]["path"]
    bounded_json(split_path, MAX_SPLIT_MANIFEST_BYTES, "campaign split manifest")
    split = trainer.load_campaign_split(split_path, root)
    if split.sha256 != EXPECTED_SPLIT_SHA256 or split.sha256 != contract["split"]["sha256"]:
        raise ValueError("clean F0 source-qualified split identity changed")
    if (len(split.train_groups), len(split.valid_groups), len(split.evaluation_groups)) != (10535, 1405, 468):
        raise ValueError("clean F0 fitting/evaluation group counts changed")
    group_identities = {
        "train": split.train_groups,
        "valid_fit": split.valid_groups,
        "valid_evaluation": split.evaluation_groups,
    }
    for name, groups in group_identities.items():
        identity = {
            "count": len(groups),
            "ordered_int64_le_sha256": sha256_bytes(np.asarray(groups, dtype="<i8").tobytes()),
        }
        if identity != contract["split"]["source_qualified_groups"][name]:
            raise ValueError(f"source-qualified clean F0 group set changed: {name}")

    seed = int(contract["sample_order"]["seed"])
    train_order = np.random.default_rng(seed).permutation(np.asarray(split.train_groups, dtype=np.int64))
    train_order = train_order[: int(contract["sample_order"]["train_seqs_argument"])]
    if len(train_order) != len(split.train_groups) or set(map(int, train_order)) != set(split.train_groups):
        raise ValueError("frontier recipe must include every clean-F0 train group before batch tail dropping")
    valid_order = np.asarray(split.valid_groups, dtype=np.int64)
    split.assert_fit_groups(train_order, valid_order)
    combined_size = len(train_order) + len(valid_order)
    positions = seeded_epoch_positions(combined_size, seed, BATCH_SIZE, EPOCH)
    members = source_qualified_members(train_order, valid_order, positions)
    if (positions[:BATCH_SIZE] != EXPECTED_FIRST_BATCH_POSITIONS
            or positions[:BATCH_SIZE] != contract["sample_order"]["first_batch_positions"]):
        raise ValueError("epoch-0 sampler's first 16 positions differ from the frozen actual-data test")
    first_batch_members = [[source, group] for source, group in members[:BATCH_SIZE]]
    if (first_batch_members != EXPECTED_FIRST_BATCH_MEMBERS
            or first_batch_members != contract["sample_order"]["first_batch_members"]):
        raise ValueError("epoch-0 first batch's source-qualified memberships differ from the frozen actual-data test")
    tail_start = len(positions) - len(positions) % BATCH_SIZE
    dropped_tail = source_qualified_members(train_order, valid_order, positions[tail_start:])
    dropped_tail_members = [[source, group] for source, group in dropped_tail]
    if (positions[tail_start:] != EXPECTED_DROPPED_TAIL_POSITIONS
            or dropped_tail_members != EXPECTED_DROPPED_TAIL_MEMBERS
            or positions[tail_start:] != contract["sample_order"]["dropped_tail_positions"]
            or dropped_tail_members != contract["sample_order"]["dropped_tail_members"]):
        raise ValueError("drop_last tail differs from the frozen four source-qualified omissions")
    if len(positions) != 11_940 or len(positions) // BATCH_SIZE != 746:
        raise ValueError("one-epoch matched sampler must retain exactly 746 complete B16 batches")

    split_manifest, _ = bounded_json(split_path, MAX_SPLIT_MANIFEST_BYTES, "campaign split manifest")
    source_seq_ix: dict[tuple[str, int], int] = {}
    for source_name, sequences in split_manifest["sequences"].items():
        for item in sequences:
            key = (str(source_name), int(item["row_group"]))
            if key in source_seq_ix:
                raise ValueError(f"source split repeats a source-qualified group: {key}")
            source_seq_ix[key] = int(item["seq_ix"])

    cache_root = root / contract["cache"]["root"]
    metadata_path = cache_root / "metadata.json"
    metadata, metadata_raw = bounded_json(metadata_path, MAX_CACHE_METADATA_BYTES, "clean-T3 cache metadata")
    metadata_sha = sha256_bytes(metadata_raw)
    if metadata_sha != EXPECTED_CACHE_METADATA_SHA256 or metadata_sha != contract["cache"]["metadata_sha256"]:
        raise ValueError("clean-T3 cache metadata hash changed")
    if metadata.get("student_contract") != contract["cache"]["historical_student_contract"]:
        raise ValueError("immutable producer's historical student contract was changed")

    # This is the unchanged producer validator: do not skip it or relax its H256 contract.
    cache_validation = trainer.validate_distillation_cache(
        cache_root, root, split, "homogeneous", require_families=False
    )
    if cache_validation["metadata_sha256"] != metadata_sha:
        raise ValueError("strict producer validator returned another metadata identity")
    expected_files = contract["cache"]["verified_targets"]
    if cache_validation["verified_files"] != expected_files:
        raise ValueError("strict producer validator did not verify both pinned full target files")
    cache_rows = validate_cache_rows(metadata, split)
    for source_row in cache_rows:
        if source_row not in source_seq_ix:
            raise ValueError(f"homogeneous cache row lacks source-qualified split ancestry: {source_row}")

    first_members = members[:BATCH_SIZE]
    first_seq_ix = []
    for source, group in first_members:
        sequence_ix = source_seq_ix.get((source, group))
        if sequence_ix is None:
            raise ValueError(f"first batch row has no source-qualified seq_ix: {(source, group)}")
        sequence = trainer.SequenceFile(root / "datasets" / f"{source}.parquet").read(group)
        if int(sequence.seq_ix) != sequence_ix:
            raise ValueError(f"parquet seq_ix disagrees with split identity for {(source, group)}")
        first_seq_ix.append(sequence_ix)
    if (first_seq_ix != EXPECTED_FIRST_BATCH_SEQ_IX
            or first_seq_ix != contract["sample_order"]["first_batch_seq_ix"]):
        raise ValueError("first epoch-0 Parquet identities differ from the frozen source-qualified sample order")
    order_hashes = {
        "train_order_sha256": sha256_bytes(np.asarray(train_order, dtype="<i8").tobytes()),
        "sampler_positions_sha256": positions_sha256(positions),
        "sampler_source_rows_sha256": members_sha256(members),
    }
    expected_hashes = {
        "train_order_sha256": EXPECTED_TRAIN_ORDER_SHA256,
        "sampler_positions_sha256": EXPECTED_SAMPLER_POSITIONS_SHA256,
        "sampler_source_rows_sha256": EXPECTED_SAMPLER_SOURCE_ROWS_SHA256,
    }
    for name, digest in order_hashes.items():
        if digest != expected_hashes[name] or digest != contract["sample_order"][name]:
            raise ValueError(f"full source-qualified order digest differs from frozen contract: {name}")

    return split, cache_validation, cache_rows, {
        "source_qualified_group_identities": {
            name: {
                "count": len(groups),
                "ordered_int64_le_sha256": sha256_bytes(np.asarray(groups, dtype="<i8").tobytes()),
            }
            for name, groups in group_identities.items()
        },
        **order_hashes,
        "first_batch_positions": positions[:BATCH_SIZE],
        "first_batch_members": [[source, group] for source, group in first_members],
        "first_batch_seq_ix": first_seq_ix,
        "dropped_tail_positions": positions[tail_start:],
        "dropped_tail_members": [[source, group] for source, group in dropped_tail],
        "fit_groups_before_drop_last": len(train_order) + len(valid_order),
        "fit_group_exposures_after_drop_last": (len(positions) // BATCH_SIZE) * BATCH_SIZE,
        "omitted_fit_group_count_per_epoch": len(dropped_tail),
        "batches_per_epoch": len(positions) // BATCH_SIZE,
        "chunks_per_sequence": SEQUENCE_ROWS // CHUNK_ROWS,
        "optimizer_updates_per_epoch": (len(positions) // BATCH_SIZE) * (SEQUENCE_ROWS // CHUNK_ROWS),
        "all_cache_rows_source_qualified": len(cache_rows),
        "strict_cache_verified_files": cache_validation["verified_files"],
        "strict_cache_metadata_sha256": cache_validation["metadata_sha256"],
    }


def verify_ancestry(root: Path, contract: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Check teacher, propensity and common-normalization checkpoint ancestry."""
    if metadata.get("target_contract") != contract["cache"]["target_contract"]:
        raise ValueError("clean-T3 homogeneous target formula differs from the frozen profile contract")
    expected_teachers = contract["cache"]["teachers"]
    source_records = metadata.get("sources", [])
    actual_by_run = {item.get("run"): item for item in source_records if isinstance(item, dict)}
    if set(actual_by_run) != {item["run"] for item in expected_teachers}:
        raise ValueError("clean-T3 teacher roster differs from the frozen consumer recipe")
    teachers = []
    for expected in expected_teachers:
        actual = actual_by_run[expected["run"]]
        if (actual.get("family") != "strong" or actual.get("epoch") != 2
                or abs(float(actual.get("coefficient", 0.0)) - 0.1) > 1e-7
                or actual.get("checkpoint_sha256") != expected["checkpoint_sha256"]):
            raise ValueError(f"clean-T3 teacher metadata changed: {expected['run']}")
        checkpoint = root / expected["checkpoint"]
        provenance = root / expected["provenance"]
        if (sha256_file(checkpoint) != expected["checkpoint_sha256"]
                or sha256_file(provenance) != expected["provenance_sha256"]):
            raise ValueError(f"clean-T3 teacher bytes or provenance changed: {expected['run']}")
        teachers.append(expected)

    propensity = contract["propensity"]
    model_path = root / propensity["checkpoint"]
    provenance_path = root / propensity["provenance"]
    if (sha256_file(model_path) != propensity["checkpoint_sha256"]
            or sha256_file(provenance_path) != propensity["provenance_sha256"]):
        raise ValueError("F0 propensity checkpoint/provenance changed")
    propensity_receipts = {}
    for source, expected in propensity["caches"].items():
        cache_path = root / expected["path"]
        sidecar_path = root / expected["sidecar"]
        sidecar, sidecar_raw = bounded_json(sidecar_path, MAX_SMALL_JSON_BYTES,
                                            f"{source} propensity cache sidecar")
        if (sha256_file(cache_path) != expected["sha256"]
                or cache_path.stat().st_size != expected["bytes"]
                or sha256_bytes(sidecar_raw) != expected["sidecar_sha256"]
                or sidecar.get("model") != propensity["run"]
                or sidecar.get("model_sha256") != propensity["checkpoint_sha256"]
                or sidecar.get("split") != source
                or sidecar.get("rows") != expected["rows"]
                or sidecar.get("n_filled") != expected["n_filled"]):
            raise ValueError(f"F0 propensity cache or sidecar changed: {source}")
        propensity_receipts[source] = {
            "path": expected["path"], "sha256": expected["sha256"],
            "bytes": expected["bytes"], "sidecar_sha256": expected["sidecar_sha256"],
        }
    return {"teacher_sources": teachers,
            "propensity": {"run": propensity["run"],
                           "checkpoint_sha256": propensity["checkpoint_sha256"],
                           "provenance_sha256": propensity["provenance_sha256"],
                           "caches": propensity_receipts}}


def build_first_batch(root: Path, contract: dict[str, Any], trainer: Any,
                      split: Any, cache_validation: dict[str, Any],
                      cache_rows: dict[tuple[str, int], int], order_receipt: dict[str, Any]
                      ) -> tuple[Any, dict[tuple[str, int], int], float, int]:
    """Load the exact production first epoch-0 batch with the same source ordering."""
    seed = int(contract["recipe"]["seed"])
    train_order = np.random.default_rng(seed).permutation(np.asarray(split.train_groups, dtype=np.int64))
    train_order = train_order[: int(contract["sample_order"]["train_seqs_argument"])]
    train_groups = [int(group) for group in train_order]
    valid_groups = [int(group) for group in split.valid_groups]
    propensity = contract["propensity"]
    prop_cache = {}
    for source in ("train", "valid"):
        sidecar_path = root / propensity["caches"][source]["sidecar"]
        bounded_json(sidecar_path, MAX_SMALL_JSON_BYTES, f"{source} propensity sidecar")
        prop_cache[source] = trainer.prop_cache_path(propensity["run"], source)
    train_ds = trainer.Sequences(
        "train", train_groups, prop_cache=prop_cache["train"],
        distill_cache_root=root / contract["cache"]["root"],
        distill_cache_variant="homogeneous", distill_cache_metadata=cache_validation,
    )
    valid_ds = trainer.Sequences(
        "valid", valid_groups, prop_cache=prop_cache["valid"],
        distill_cache_root=root / contract["cache"]["root"],
        distill_cache_variant="homogeneous", distill_cache_metadata=cache_validation,
    )
    dataset = ConcatDataset([train_ds, valid_ds])
    sampler = trainer.EpochOrderSampler(len(dataset), seed, batch_size=BATCH_SIZE, drop_last=True)
    sampler.set_epoch(EPOCH)
    start = time.perf_counter()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True,
                        num_workers=0, prefetch_factor=None, pin_memory=True)
    batch = next(iter(loader))
    load_seconds = time.perf_counter() - start
    x, targets, scored, prop = batch
    if (tuple(x.shape) != (BATCH_SIZE, SEQUENCE_ROWS, 112)
            or tuple(targets.shape) != (BATCH_SIZE, SEQUENCE_ROWS, 4)
            or tuple(scored.shape) != (BATCH_SIZE, SEQUENCE_ROWS)
            or tuple(prop.shape) != (BATCH_SIZE, SEQUENCE_ROWS)):
        raise ValueError("production Sequences loader returned a different batch schema")
    arrays = {
        "train": np.load(root / contract["cache"]["root"] / "main_homogeneous_train.npy",
                          mmap_mode="r", allow_pickle=False),
        "valid": np.load(root / contract["cache"]["root"] / "main_homogeneous_valid.npy",
                          mmap_mode="r", allow_pickle=False),
    }
    first_members = [tuple(item) for item in order_receipt["first_batch_members"]]
    if len(first_members) != BATCH_SIZE:
        raise ValueError("profile requires exactly the first 16 source-qualified sampler rows")
    for batch_index, (source, group) in enumerate(first_members):
        expected_row = cache_rows[(source, group)]
        row = np.asarray(arrays[source][expected_row], dtype=np.float32)
        observed = targets[batch_index, :, 2:4].numpy()
        if row.shape != (SEQUENCE_ROWS, 2) or not np.isfinite(row).all() or not np.array_equal(row, observed):
            raise ValueError(f"source-qualified target payload mismatch for {(source, group)}")
    batch_tensor_bytes = max(
        train_ds.batch_tensor_bytes(BATCH_SIZE), valid_ds.batch_tensor_bytes(BATCH_SIZE)
    )
    del arrays, train_ds, valid_ds, dataset, loader, sampler
    return batch, cache_rows, load_seconds, batch_tensor_bytes


def profile_shape(
    root: Path, trainer: Any, batch: Any, mean: torch.Tensor, std: torch.Tensor,
    propensity_model: torch.nn.Module, shape: tuple[int, int], total_updates: int,
    profile_contract_sha256: str, profile_batch_size: int,
) -> dict[str, Any]:
    """Run three actual chunk updates at the resource-profile batch size."""
    hidden, layers = shape
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args_dict = {
        "arch": "gru", "cell": "gru", "enc": "linear", "hidden": hidden,
        "layers": layers, "dropout": 0.0, "readout_activation": "gelu",
        "numeric_map": "none", "ple_intervals": 8,
    }
    model = trainer.make_model(args_dict, 211, 2).to(device=device, dtype=torch.float32)
    model.mean.copy_(mean.to(device=device, dtype=torch.float32))
    model.std.copy_(std.to(device=device, dtype=torch.float32))
    model.train()
    model._inference_fp32 = True
    optimizer_args = argparse.Namespace(
        head_lr_mult=1.0, zoneout=0.0, momentum=0.0, optim_switch="", optim="adamw",
        muon_scope="rnn", lr=1.0e-4, beta2=0.999, wd=1.0e-2, fused_adam=True,
    )
    optimizer = trainer.build_optimizer(model, optimizer_args)
    scheduler_args = argparse.Namespace(
        warmup_frac=None, sched="onecycle", lr=1.0e-4, head_lr_mult=1.0,
        optim="adamw", wsd_decay=0.2, wsd_final=0.01, wsd_shape="sqrt", restarts=1,
    )
    scheduler = trainer.build_scheduler(optimizer, scheduler_args, total_updates)

    x_cpu, target_cpu, scored_cpu, prop_cpu = (tensor[:profile_batch_size] for tensor in batch)
    x = x_cpu.to(device, non_blocking=True)
    target_pack = target_cpu.to(device, non_blocking=True)
    scored = scored_cpu.to(device, non_blocking=True)
    prop_cached = prop_cpu.to(device, non_blocking=True)
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None
    inputs = trainer.build_inputs(x)
    labels = target_pack[..., :2]
    teacher = target_pack[..., 2:4]
    valid = torch.arange(SEQUENCE_ROWS, device=device)[None].expand(profile_batch_size, -1) >= trainer.WARMUP
    row_weight = trainer.main_row_weights(
        inputs, labels, valid, scored, propensity_model, prop_cached,
        argparse.Namespace(
            mask_weight=False, prop_power=2.0, prop_eps=0.03, class_weight="",
            valid_true_mask=0.03, sync_free=True, since_weight=0.0, episode_weight=0.0,
            episode_floor=0.5, since_floor=0.25,
        ), x,
    )
    clipped_labels = labels.float().clamp(-2.0, 2.0)
    weights = clipped_labels.abs() * row_weight[..., None].float()
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("the exact first fit batch has no finite positive main loss weight")

    torch.cuda.reset_peak_memory_stats(device)
    state = None
    step_times: list[float] = []
    losses: list[float] = []
    parameter_update = False
    grad_norms: list[float] = []
    for chunk_index in range(3):
        sl = slice(chunk_index * CHUNK_ROWS, (chunk_index + 1) * CHUNK_ROWS)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        out, state = trainer.forward_chunk(model, inputs, sl, state)
        state = trainer.detach_state(state)
        target = trainer.blended_target(labels, teacher, None, sl, ALPHA, 0, 0.0)
        weight = weights[:, sl]
        loss = (weight * (out[..., :2].float() - target) ** 2).sum() / weight.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite FP32 loss for shape {shape}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm) or not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"non-finite FP32 gradient for shape {shape}")
        before = [parameter.detach().clone() for parameter in model.parameters()] if chunk_index == 0 else None
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(norm.detach().cpu()))
        if before is not None:
            parameter_update = any(not torch.equal(old, new.detach())
                                   for old, new in zip(before, model.parameters(), strict=True))
            del before
    if not parameter_update:
        raise FloatingPointError(f"first actual B16 update changed no parameters for shape {shape}")

    memory_allocated = int(torch.cuda.max_memory_allocated(device))
    memory_reserved = int(torch.cuda.max_memory_reserved(device))
    if memory_reserved > CUDA_ALLOCATOR_CAP_BYTES:
        raise RuntimeError("PyTorch CUDA allocator exceeded the explicit 3.5 GB bound")
    if nvidia_smi_used_bytes() > OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM exceeded the fixed 5 GB ceiling")

    model.eval()
    sample_raw = x_cpu[:1]
    torch.cuda.synchronize(device)
    export_start = time.perf_counter()
    sample_prediction = trainer.predict(model, sample_raw, CHUNK_ROWS, device)
    export_seconds = time.perf_counter() - export_start
    if sample_prediction.shape != (1, SEQUENCE_ROWS, 2) or not np.isfinite(sample_prediction).all():
        raise FloatingPointError(f"single-sequence causal export failed for shape {shape}")
    prediction_sha256 = sha256_bytes(np.asarray(sample_prediction[0], dtype="<f4").tobytes())
    per_sequence_export_seconds = export_seconds
    projected_eval_export_seconds = per_sequence_export_seconds * N_EVAL_GROUPS_F0
    projected_fit_update_seconds = (
        step_times[0] + float(np.median(step_times[1:])) * (total_updates - 1)
        if profile_batch_size == BATCH_SIZE else None
    )
    output_bytes = int(sample_prediction.nbytes)
    receipt = {
        "shape": [hidden, layers],
        "input_features": 211,
        "normalization_sha256": {"mean": tensor_sha256(mean), "std": tensor_sha256(std)},
        "input_projection": "linear GELU with the current projection skip into the two-output GELU head",
        "recurrent_cell": "dense stacked GRU",
        "profile_batch_size": profile_batch_size,
        "chunk_rows": CHUNK_ROWS,
        "precision": "float32; TF32 disabled",
        "updates_measured": 3,
        "optimizer_attempts": 3,
        "optimizer_completed_updates": 3,
        "optimizer": "train_rnn.build_optimizer: fused AdamW, lr=1e-4, beta=(.9,.999), wd=.01",
        "schedule": "train_rnn.build_scheduler: OneCycleLR, 7460 total steps, pct_start=.05, final_div_factor=100",
        "gradient_clip_norm": 1.0,
        "alpha": 0.9,
        "row_weight": "abs(clamp(label,-2,2))*main_row_weights; F0 propensity cache p^2+0.03 and valid_true_mask .03",
        "loss": "sum(weight*(prediction-((.1*clamp(label)+.9*clamp(teacher))))^2)/sum(weight), by each 2000-row chunk",
        "loss_finite": True,
        "loss_values_diagnostic_only": losses,
        "gradient_norms_before_clip": grad_norms,
        "parameter_update_observed": parameter_update,
        "chunk_update_seconds": step_times,
        "first_chunk_update_seconds": step_times[0],
        "steady_update_seconds_median": float(np.median(step_times[1:])),
        "one_epoch_updates": total_updates if profile_batch_size == BATCH_SIZE else None,
        "projected_one_epoch_update_wall_seconds": projected_fit_update_seconds,
        "projected_one_epoch_update_hours": (
            projected_fit_update_seconds / 3600.0 if projected_fit_update_seconds is not None else None
        ),
        "projection_scope": (
            "three updates on the exact first B16 batch; update-only linear extrapolation excludes batch loads, "
            "cache verification, checkpoint and prediction export"
            if profile_batch_size == BATCH_SIZE else
            "three updates on a prefix of the exact first B16 batch; resource-only, no full-fit time extrapolation"
        ),
        "torch_cuda_peak_allocated_bytes": memory_allocated,
        "torch_cuda_peak_reserved_bytes": memory_reserved,
        "worker_cgroup_memory_current_bytes": cgroup_memory_current(),
        "single_sequence_causal_export_seconds": per_sequence_export_seconds,
        "single_sequence_prediction_shape": [SEQUENCE_ROWS, 2],
        "single_sequence_prediction_bytes": output_bytes,
        "single_sequence_prediction_sha256": prediction_sha256,
        "projected_f0_eval_prediction_seconds": projected_eval_export_seconds,
        "projected_f0_eval_group_count": N_EVAL_GROUPS_F0,
        "prediction_projection_scope": "one fitted-data sequence, extrapolated linearly to the 468 F0 evaluation groups; excludes data reads and metric aggregation",
        "quality_claim": None,
        "profile_contract_sha256": profile_contract_sha256,
    }
    del sample_prediction, sample_raw, inputs, row_weight, weights, clipped_labels, target, out, state
    del optimizer, scheduler, model, x, target_pack, scored, prop_cached
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    return receipt


def parse_args() -> argparse.Namespace:
    """Parse the bounded profile CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--profile-batch-size", type=int, choices=PROFILE_BATCH_CHOICES, required=True)
    parser.add_argument("--shape-indices", type=int, nargs="+", required=True,
                        help="zero-based indices into the frozen shape_profile_order")
    parser.add_argument("--execute", action="store_true", help="execute the contracted actual-data resource profiles")
    return parser.parse_args()


def main() -> int:
    """Verify the exact actual-data profile contract, then optionally execute it."""
    args = parse_args()
    if args.preflight_only == args.execute:
        raise SystemExit("select exactly one of --preflight-only or --execute")
    root = args.data_root.resolve(strict=True)
    if os.environ.get("WNN_PROJECT_ROOT") != str(root):
        raise ValueError("WNN_PROJECT_ROOT must point to the data/source root named by --data-root")
    contract_path = args.contract.resolve(strict=True)
    contract_size = contract_path.stat().st_size
    if contract_size <= 0 or contract_size > 128_000:
        raise ValueError("batch profile contract exceeds its 128 KB read bound")
    contract_raw = contract_path.read_bytes()
    if len(contract_raw) != contract_size:
        raise ValueError("batch profile contract changed while being read")
    contract = json.loads(contract_raw)
    contract_sha256 = sha256_bytes(contract_raw)
    if (contract.get("format") != "wnn.student-frontier-batch-profile-contract"
            or contract.get("version") != 1
            or contract.get("shape_profile_order") != [list(shape) for shape in SHAPE_PROFILE_ORDER]
            or contract.get("recipe", {}).get("seed") != SEED
            or contract.get("recipe", {}).get("batch_size") != BATCH_SIZE
            or contract.get("recipe", {}).get("chunk_rows") != CHUNK_ROWS
            or contract.get("recipe", {}).get("precision") != "float32"):
        raise ValueError("unsupported batch profile contract or settings")
    profile_work = contract["recipe"].get("profile_work", {})
    if (args.profile_batch_size not in PROFILE_BATCH_CHOICES
            or args.profile_batch_size != profile_work.get("profile_batch_size")
            or args.shape_indices != profile_work.get("shape_indices")
            or any(index < 0 or index >= len(SHAPE_PROFILE_ORDER) for index in args.shape_indices)
            or len(set(args.shape_indices)) != len(args.shape_indices)):
        raise ValueError("profile batch/shape selection differs from the explicit profile contract")
    if contract.get("data_root") != str(root):
        raise ValueError("profile contract is bound to a different data root")

    repo = Path(__file__).resolve().parents[1]
    if repo != Path(contract["source_root"]).resolve():
        raise ValueError("profile source root differs from the frozen contract")
    sources = source_manifest(Path(__file__).resolve(), repo)
    if sources != contract.get("source_sha256"):
        raise ValueError("batch profile source hashes differ from the frozen contract")

    sys.path.insert(0, str(repo / "scripts"))
    import train_rnn as trainer  # imported after WNN_PROJECT_ROOT is validated

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split, cache_validation, cache_rows, order = validate_order_and_sources(root, contract, trainer)
    profile_scope = contract.get("profile_scope")
    if profile_scope is not None:
        if (profile_scope.get("kind") != "resource_only"
                or profile_scope.get("batch_size") != args.profile_batch_size
                or profile_scope.get("shape_indices") != args.shape_indices
                or profile_scope.get("shapes") != [list(SHAPE_PROFILE_ORDER[index]) for index in args.shape_indices]
                or profile_scope.get("source_qualified_member") != order["first_batch_members"][0]
                or profile_scope.get("seq_ix") != order["first_batch_seq_ix"][0]
                or profile_scope.get("updates") != 3
                or profile_scope.get("causal_export_sequences") != 1
                or profile_scope.get("no_quality_claim") is not True
                or profile_scope.get("no_fit_walltime_extrapolation") is not True):
            raise ValueError("resource-only profile scope differs from the validated source-qualified batch")
    ancestry = verify_ancestry(root, contract, cache_validation["metadata"])
    mean, std = load_bound_norm(root, contract, split)
    output = {
        "format": "wnn.student-frontier-batch-profile-result",
        "version": 1,
        "profile_contract_sha256": contract_sha256,
        "profile_source_sha256": sources,
        "data_root": str(root),
        "profile_selection": {
            "batch_size": args.profile_batch_size,
            "shape_indices": args.shape_indices,
            "shapes": [list(SHAPE_PROFILE_ORDER[index]) for index in args.shape_indices],
        },
        "split_manifest_sha256": split.sha256,
        "source_qualified_sampler": order,
        "resource_profile_scope": profile_scope,
        "cache_ancestry": ancestry,
        "cache_metadata_sha256": cache_validation["metadata_sha256"],
        "cache_verified_files": cache_validation["verified_files"],
        "normalization_sha256": {"mean": tensor_sha256(mean), "std": tensor_sha256(std)},
        "normalization_source": {
            **contract["normalization_source"],
            "loaded_fields_only": ["mean", "std"],
            "learned_weights_loaded": False,
        },
        "host_max_rss_bytes_after_preflight": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "host_cgroup_memory_current_bytes_after_preflight": cgroup_memory_current(),
        "quality_claim": None,
        "interpretation": "strict actual-data source/order/cache validation; no fit or quality evaluation",
    }
    if output["normalization_sha256"] != contract["normalization_sha256"]:
        raise ValueError("shared H256 normalization identities differ from the profile recipe")
    matched = contract["recipe"]["matched_full_fit_budget"]
    matched_expected = {
        "epochs": 1,
        "fixed_checkpoint_epoch": 0,
        "updates": order["optimizer_updates_per_epoch"],
        "retained_batch_count": order["batches_per_epoch"],
        "exposed_sequence_count": order["fit_group_exposures_after_drop_last"],
        "eligible_fit_sequence_count": order["fit_groups_before_drop_last"],
        "drop_last": True,
        "dropped_sequence_count": order["omitted_fit_group_count_per_epoch"],
        "same_order_and_omissions_across_shapes": True,
        "evaluation_during_fit": False,
        "checkpoint_rule": "fixed terminal epoch 0; no adaptive evaluation",
    }
    if matched != matched_expected:
        raise ValueError("matched one-epoch fit budget in the contract differs from actual sampler exposure")
    if args.execute:
        if not torch.cuda.is_available():
            raise RuntimeError("contracted actual-data batch profile requested without CUDA")
        from wnn.resources import configure_torch

        if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(CUDA_ALLOCATOR_CAP_BYTES):
            raise RuntimeError("admitted batch profile requires WNN_GPU_MEMORY_BYTES=3500000000")
        configure_torch()
        device = torch.device("cuda")
        initial_vram = nvidia_smi_used_bytes()
        if initial_vram > OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError("initial observed VRAM is above the fixed 5 GB ceiling")
        output["cuda_device"] = torch.cuda.get_device_name(device)
        output["cuda_total_memory_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
        output["nvidia_smi_used_before_profile_bytes"] = initial_vram
        output["cuda_allocator_cap_bytes"] = CUDA_ALLOCATOR_CAP_BYTES
        output["observed_vram_ceiling_bytes"] = OBSERVED_VRAM_LIMIT_BYTES
        output["tf32_enabled"] = False
        output["profile_shapes"] = []
        total_updates = int(order["optimizer_updates_per_epoch"])
        profile_started = time.perf_counter()
        batch, cache_rows, batch_load_seconds, batch_tensor_bytes = build_first_batch(
            root, contract, trainer, split, cache_validation, cache_rows, order
        )
        output["first_real_B16_batch_loader_seconds"] = batch_load_seconds
        output["batch_tensor_bytes"] = batch_tensor_bytes
        output["host_cgroup_memory_current_bytes_after_batch_load"] = cgroup_memory_current()
        output["host_max_rss_bytes_after_batch_load"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        output["source_qualified_first_batch_targets_checked"] = BATCH_SIZE
        output["resource_profile_batch_size"] = args.profile_batch_size
        output["resource_profile_shape_indices"] = args.shape_indices
        output["resource_profile_shapes"] = [list(SHAPE_PROFILE_ORDER[index]) for index in args.shape_indices]
        propensity_model = trainer.load_propensity(
            root / "artifacts" / "runs" / contract["propensity"]["run"] / "model.pt", device
        )
        for shape_index in args.shape_indices:
            shape = SHAPE_PROFILE_ORDER[shape_index]
            if nvidia_smi_used_bytes() > OBSERVED_VRAM_LIMIT_BYTES:
                raise RuntimeError("observed VRAM exceeded 5 GB before the next shape")
            try:
                shape_result = profile_shape(root, trainer, batch, mean, std,
                                             propensity_model, shape,
                                             total_updates, contract_sha256,
                                             args.profile_batch_size)
            except Exception as error:
                output["failed_shape"] = list(shape)
                output["failure"] = f"{type(error).__name__}: {error}"
                output["profile_shapes_completed"] = len(output["profile_shapes"])
                output["profile_elapsed_seconds"] = time.perf_counter() - profile_started
                output["host_max_rss_bytes_at_failure"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
                output["host_cgroup_memory_current_bytes_at_failure"] = cgroup_memory_current()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                       encoding="utf-8")
                raise
            output["profile_shapes"].append(shape_result)
            if shape_result["normalization_sha256"] != output["normalization_sha256"]:
                raise ValueError("shape-specific model did not receive the pinned common normalization")
            output["completed_shape_count"] = len(output["profile_shapes"])
            output["host_max_rss_bytes_after_shape"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            output["host_cgroup_memory_current_bytes_after_shape"] = cgroup_memory_current()
            output["nvidia_smi_used_after_shape_bytes"] = nvidia_smi_used_bytes()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                   encoding="utf-8")
            print(json.dumps({"profile_result": str(args.output), "completed_shape_count": len(output["profile_shapes"]),
                              "shape": list(shape), "steady_update_seconds_median": shape_result[
                                  "steady_update_seconds_median"]}, sort_keys=True), flush=True)
        output["profile_elapsed_seconds"] = time.perf_counter() - profile_started
        output["nvidia_smi_used_after_profile_bytes"] = nvidia_smi_used_bytes()
        output["host_max_rss_bytes_after_profile"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        output["host_cgroup_memory_current_bytes_after_profile"] = cgroup_memory_current()
        del propensity_model
        if output["nvidia_smi_used_after_profile_bytes"] > OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError("final observed VRAM exceeded the fixed 5 GB ceiling")
        output["interpretation"] = (
            f"{len(args.shape_indices)} shape resource/rate probes using profile batch size "
            f"{args.profile_batch_size} from the exact source-qualified clean-F0 B16 first batch; "
            "three FP32 chunk updates per shape and one B1 causal prediction only; not a quality fit"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"profile_result": str(args.output), "strict_cache_verified": True,
                      "first_batch_members": order["first_batch_members"],
                      "profile_shapes_completed": len(output.get("profile_shapes", []))}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
