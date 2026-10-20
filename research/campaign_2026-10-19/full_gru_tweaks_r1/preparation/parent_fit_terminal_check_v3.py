"""Bounded V3 terminal integrity check for the source-68 H256x2 parent fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from propensity_terminal_support import (
    _group_identity,
    _hash_file,
    _json_file,
    _numpy_safe_globals,
    _safe_relative,
    _stable_file,
    _tensor_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
UNIT = "computer-vision-homework-fg19-parent-fit-r2-v1.service"
INVOCATION = "5186b3016ff54b1fa754c10a1f87b3e4"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
ADMISSION_SHA = "42b5f3fe7142d931b189ff86fa24031426506e60c8a41440b0483d97dc50026b"
LAUNCH_SHA = "012c7ba3b484f16f587c61cb39f880380d22a6199e0a105773dbffc423bd22ea"
LAUNCH_ACCEPTANCE_SHA = "89b8199b6431c8959fe3bab41a630632210b30ae017dd9128f9ad2491c87a3bf"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
CACHE_ACCEPTANCE_SHA = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
RNG_DIAGNOSTIC_SHA = "0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b"
RNG_DIAGNOSTIC_ACCEPTANCE_SHA = "655c5f12db5e166cfabf68c343aeb10d709f95db4abffeaf2925b413226837ec"
RNG_DIAGNOSTIC_STOP_SHA = "8e935356da08ac249e47389bb65bc5597134e6442ab9d914da1e7ec256b73c67"
RNG_DIAGNOSTIC_RAW_SHA = "2f6ced10d3812a3b3ae8354268a193323df6a22ea8523fdd2458060f4b8fcf69"
RNG_DIAGNOSTIC_SOURCE_SHA = "75f3b53762f5faf60f785126a8dec4b24e2bf52f620fce1397d7958eb25906b2"
RNG_DIAGNOSTIC_UNIT = "computer-vision-homework-fg19-parent-rng-diag-v2.service"
RNG_DIAGNOSTIC_INVOCATION = "b9dcbfc0b8054e51a49fc3792fd037a0"
RNG_DIAGNOSTIC_TORCH_VERSION = "2.14.0+cu130"
CUDA_POLICY_SOURCE_SHA = "95fdc839ff1561622ac06c465cfc9a01370130d99be1e832b3a33c79bcb37930"
EXPECTED_RNG_TENSORS = {
    "torch": {
        "type": "torch.Tensor",
        "dtype": "torch.uint8",
        "numel": 5056,
        "cardinality": 5056,
        "sha256": "78b7aa970d6550d7d03e2b99942959f766a0e7abf3a3c2f10e55294329ec575e",
        "shape": [5056],
    },
    "cuda": {
        "type": "builtins.list",
        "cardinality": 1,
        "components": [
            {
                "type": "torch.Tensor",
                "dtype": "torch.uint8",
                "numel": 16,
                "cardinality": 16,
                "sha256": "f7d8d061e24e21aa31b7afef7d6093940af44207bb0f0cd6e9f7decec4672238",
                "shape": [16],
            }
        ],
    },
}
CONTRACT_SHA = "dbcbe194f3ceb16d499f1c68bc8d862a16c32af54a13f57db75284691d8aabb9"
TRAIN_CACHE_SHA = "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194"
VALID_CACHE_SHA = "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591"
UPDATES = 29_840
EPOCH_UPDATES = 7_460
BATCHES = 746
FIT_VALID = 1_405
EVALUATION = 468
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_AGGREGATE_BYTES = 128_000_000

MODEL_SHAPES = {
    "mean": (211,),
    "std": (211,),
    "inp.weight": (256, 211),
    "inp.bias": (256,),
    "gru.weight_ih_l0": (768, 256),
    "gru.weight_hh_l0": (768, 256),
    "gru.bias_ih_l0": (768,),
    "gru.bias_hh_l0": (768,),
    "gru.weight_ih_l1": (768, 256),
    "gru.weight_hh_l1": (768, 256),
    "gru.bias_ih_l1": (768,),
    "gru.bias_hh_l1": (768,),
    "head.0.weight": (256, 512),
    "head.0.bias": (256,),
    "head.2.weight": (23, 256),
    "head.2.bias": (23,),
}
PARAMETER_SHAPES = tuple(
    shape for name, shape in MODEL_SHAPES.items() if name not in {"mean", "std"}
)
ARGS = {
    "name": RUN,
    "device": "cuda",
    "seed": 1001,
    "train_source": "train+valid",
    "valid_fold": 0,
    "train_seqs": 10535,
    "valid_repeat": 1,
    "valid_train_fraction": 1.0,
    "arch": "gru",
    "cell": "gru",
    "enc": "linear",
    "hidden": 256,
    "layers": 2,
    "head_layers": 1,
    "dropout": 0.0,
    "aux_struct": 0.3,
    "aux_next": 0.3,
    "batch": 16,
    "chunk": 2000,
    "epochs": 4,
    "checkpoint_policy": "fixed",
    "checkpoint_epoch": 3,
    "no_eval_during_fit": True,
    "loss": "wmse",
    "loss_norm": "chunk",
    "distill_cache_variant": "homogeneous",
    "distill_cache_scale": 1.0,
    "distill_alpha": 0.9,
    "distill_family_weight": 0.0,
    "lr": 0.0001,
    "wd": 0.01,
    "beta2": 0.999,
    "sched": "onecycle",
    "warmup_frac": 0.05,
    "clip": 1.0,
    "precision": "fp32",
    "optim": "adamw",
    "grad_stats": True,
    "save_training_state": True,
}
WORKER = {
    "cpu_quota": "100%",
    "memory_high": 7_500_000_000,
    "memory_max": 8_000_000_000,
    "swap_max": 0,
    "cuda_allocator_bytes": 4_294_967_296,
    "cuda_free_reserve_bytes": 1_000_000_000,
    "loader_workers": 0,
    "runtime_max_seconds": 86_400,
}
CONTRACT_KEYS = {
    "epochs",
    "total_steps",
    "options",
    "sources",
    "datasets",
    "teacher_predictions",
    "consumed_caches",
    "fit_membership",
    "normalization",
    "ancestors",
    "loader",
    "runtime",
}
TRAINING_STATE_KEYS = {
    "format",
    "version",
    "boundary",
    "contract",
    "next_epoch",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "ema",
    "swa",
    "loader",
    "auxiliary",
    "progress",
    "rng",
}
CACHE_PROVENANCE = {
    "root": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_homogeneous_cache_v1",
    "variant": "homogeneous",
    "main_target_scale": 1.0,
    "main_target_scale_applied_before_training_clip": True,
    "metadata_sha256": CACHE_METADATA_SHA,
    "verified_files": {
        "main_homogeneous_train": TRAIN_CACHE_SHA,
        "main_homogeneous_valid": VALID_CACHE_SHA,
    },
    "source_coefficients": [
        {"family": "strong", "run": f"fg19_t3_h384x2_e3_s{seed}_v1", "epoch": 2, "coefficient": 0.1}
        for seed in (3, 4, 5)
    ],
    "coefficient_sum": 0.3,
    "homogeneous_family": "strong",
    "family_order": ["strong"],
    "epoch_convention": {
        "source_zero_based": True,
        "friend_at_ep1": "second additional epoch",
        "student_epochs": 2,
        "student_checkpoint_policy": "fixed",
        "student_checkpoint_epoch": 1,
    },
}


def _eq(actual: object, expected: object, label: str) -> None:
    """Fail on any value drift without coercing JSON types."""
    if actual != expected:
        raise ValueError(f"{label} differs")


def _check_cuda_cudnn_deterministic(value: object) -> None:
    """Require the source-pinned CUDA setup to enable cuDNN determinism."""
    if value is not True:
        raise ValueError("runtime cuDNN deterministic mode differs")


def _finite_tree(value: object) -> None:
    """Reject non-finite floating values in the bounded JSON history tree."""
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("fit history contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _int(value: object, expected: int, label: str) -> None:
    """Require an exact integer type, excluding bool and numeric strings."""
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be integer {expected}")


def _safe_globals() -> list[object]:
    """Allow only NumPy RNG reconstruction and PyTorch version metadata."""
    return [*_numpy_safe_globals(np), torch.torch_version.TorchVersion]


def _load_checkpoint(path: Path, max_bytes: int) -> tuple[object, str, dict[str, int]]:
    """Hash then restricted-load one checkpoint within the parent-specific byte cap."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"not a bounded parent checkpoint: {path.name}")
        digest = hashlib.sha256()
        total = 0
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
            total += len(chunk)
        if total != before.st_size:
            raise ValueError(f"short parent checkpoint read: {path.name}")
        stream.seek(0)
        with torch.serialization.safe_globals(_safe_globals()):
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
        linked = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"parent checkpoint changed during load: {path.name}")
        if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"parent checkpoint path changed during load: {path.name}")
        return (
            payload,
            digest.hexdigest(),
            {
                "bytes": total,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mtime_ns": before.st_mtime_ns,
            },
        )


def _source_identity(
    root: Path, manifest_path: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Hash the exact 167-file staged tree against its admitted manifest."""
    manifest, digest = _json_file(manifest_path)
    _eq(digest, SOURCE_MANIFEST_SHA, "source manifest SHA")
    _eq(manifest.get("source_commit"), COMMIT, "source commit")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != 167:
        raise ValueError("source manifest must contain 167 files")
    names: set[str] = set()
    aggregate = hashlib.sha256()
    total = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("source manifest entry is malformed")
        name, size, expected = row["path"], row.get("bytes"), row.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(expected, str):
            raise ValueError("source file pin has invalid types")
        if name in names:
            raise ValueError("source manifest contains duplicate paths")
        names.add(name)
        total += size
        if total > 128_000_000:
            raise ValueError("source tree exceeds bounded hashing budget")
        lexical = root / name
        cursor = lexical
        while cursor != root:
            if cursor.is_symlink():
                raise ValueError(f"source path contains a symlink: {name}")
            cursor = cursor.parent
        path = _safe_relative(root, name)
        actual, info = _hash_file(path, size)
        _eq((actual, info["bytes"]), (expected, size), f"source file {name}")
        aggregate.update(name.encode() + b"\0" + actual.encode("ascii") + b"\n")
    _eq(admission.get("source_files_verified"), 167, "admitted source count")
    _eq(admission.get("source_manifest_sha256"), digest, "admitted source manifest")
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _eq(actual_names, names, "source inventory")
    return {
        "commit": COMMIT,
        "files": len(names),
        "manifest_sha256": digest,
        "source_set_sha256": aggregate.hexdigest(),
        "bytes_hashed": total,
    }


def _model_fingerprint(model: object) -> tuple[str, int]:
    """Validate exact finite FP32 H256x2 state and return its stable tensor digest."""
    if not isinstance(model, dict) or set(model) != set(MODEL_SHAPES):
        raise ValueError("checkpoint tensor inventory differs from H256x2 GRU")
    records: dict[str, object] = {}
    for name, shape in MODEL_SHAPES.items():
        item = _tensor_summary(model[name], torch)
        _eq((item["dtype"], tuple(item["shape"])), ("torch.float32", shape), name)
        records[name] = item
    if bool((model["std"] < 0.001).any().item()):
        raise ValueError("saved standard deviation violates source clamp")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(records)


def _check_membership(
    provenance: Mapping[str, object], split: Mapping[str, object]
) -> dict[str, int]:
    """Validate seed-ordered fit rows and component-disjoint held-out membership."""
    fit, evaluation, seqs = split.get("fit"), split.get("evaluation"), split.get("sequences")
    if not all(isinstance(item, dict) for item in (fit, evaluation, seqs)):
        raise ValueError("campaign split structure is malformed")
    base_train, valid, outer = fit.get("train"), fit.get("valid"), evaluation.get("valid")
    if not all(isinstance(rows, list) for rows in (base_train, valid, outer)):
        raise ValueError("campaign split groups are missing")
    for rows in (base_train, valid, outer):
        if any(type(group) is not int for group in rows) or len(set(rows)) != len(rows):
            raise ValueError("campaign groups must be unique integer IDs")
    expected_train = np.random.default_rng(1001).permutation(base_train).tolist()
    expected_eval = np.random.default_rng(12345).permutation(outer)[:600].tolist()
    rows = provenance.get("fit_row_groups")
    if not isinstance(rows, dict):
        raise ValueError("fit row groups are missing")
    train, fit_valid, evaluation_rows = (
        rows.get("train"),
        rows.get("valid"),
        provenance.get("evaluation_row_groups"),
    )
    _eq(
        (train, fit_valid, evaluation_rows),
        (expected_train, valid, expected_eval),
        "ordered groups",
    )
    _eq(
        provenance.get("fit_membership_identity"),
        {
            "train": _group_identity(train, np),
            "valid": _group_identity(fit_valid, np),
            "valid_repeat": 1,
            "dataset_length": 11940,
        },
        "fit membership identity",
    )
    components = {
        source: {int(item["row_group"]): int(item["component"]) for item in members}
        for source, members in seqs.items()
    }
    fit_components = {components["train"][group] for group in train}
    fit_components.update(components["valid"][group] for group in fit_valid)
    eval_components = {components["valid"][group] for group in evaluation_rows}
    if not fit_components.isdisjoint(eval_components):
        raise ValueError("fit and evaluation components overlap")
    counts = {
        "fit_train": len(train),
        "fit_valid": len(fit_valid),
        "evaluation": len(evaluation_rows),
    }
    _eq(
        counts,
        {"fit_train": 10535, "fit_valid": FIT_VALID, "evaluation": EVALUATION},
        "membership counts",
    )
    return counts


def _check_cache(
    admission: Mapping[str, object],
    cache_root: Path,
    acceptance_path: Path,
    stop_path: Path,
    split: Mapping[str, object],
) -> dict[str, object]:
    """Bind accepted cache metadata and array stats, without opening array payloads."""
    cache = admission.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("admitted cache identity is missing")
    accept, accept_sha = _json_file(acceptance_path)
    stop, stop_sha = _json_file(stop_path)
    _eq((accept_sha, stop_sha), (CACHE_ACCEPTANCE_SHA, CACHE_STOP_SHA), "cache receipt hashes")
    _eq(
        (
            accept.get("status"),
            accept.get("unit"),
            accept.get("invocation_id"),
            accept.get("metadata_sha256"),
            accept.get("source_commit"),
            accept.get("source_files_verified"),
        ),
        (
            "ACCEPTED",
            cache.get("unit"),
            cache.get("invocation_id"),
            CACHE_METADATA_SHA,
            COMMIT,
            167,
        ),
        "cache acceptance",
    )
    before = stop.get("before", {}).get(cache.get("unit"), {})
    after = stop.get("after", {}).get(cache.get("unit"), {})
    stopped = stop.get("stop", {}).get(cache.get("unit"), {})
    _eq(
        (
            before.get("InvocationID"),
            before.get("ExecMainStatus"),
            stopped.get("returncode"),
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("InvocationID"),
            stop.get("populated"),
        ),
        (cache.get("invocation_id"), "0", 0, "not-found", "inactive", "dead", "", {}),
        "cache exact-stop",
    )
    meta_path = cache_root / "metadata.json"
    metadata, meta_sha = _json_file(meta_path)
    _eq(meta_sha, CACHE_METADATA_SHA, "cache metadata")
    _eq(
        (
            metadata.get("cache_profile"),
            metadata.get("split_manifest_sha256"),
            metadata.get("homogeneous_scale"),
            metadata.get("stored_targets"),
            metadata.get("target_contract"),
            metadata.get("root_execution_contract", {}).get("sha256"),
            metadata.get("coefficient_sum"),
            metadata.get("homogeneous_family"),
            metadata.get("family_order"),
            metadata.get("epoch_convention"),
        ),
        (
            "cleanfold_homogeneous_strong_t3_v1",
            SPLIT_SHA,
            0.8125,
            ["main_homogeneous"],
            {
                "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
                "teacher_epoch": 2,
                "teacher_mean": "unweighted arithmetic mean in float32",
                "clipping": "none during cache construction",
                "coefficient_multiplier": 0.8125,
            },
            CONTRACT_SHA,
            0.3,
            "strong",
            ["strong"],
            {
                "source_zero_based": True,
                "friend_at_ep1": "second additional epoch",
                "student_epochs": 2,
                "student_checkpoint_policy": "fixed",
                "student_checkpoint_epoch": 1,
            },
        ),
        "cache formula",
    )
    fit_groups = metadata.get("fit_groups", {})
    _eq(
        (fit_groups.get("train"), fit_groups.get("valid"), metadata.get("evaluation_valid_groups")),
        (
            split.get("fit", {}).get("train"),
            split.get("fit", {}).get("valid"),
            split.get("evaluation", {}).get("valid"),
        ),
        "cache fit/evaluation membership",
    )
    if set(fit_groups["valid"]) & set(metadata["evaluation_valid_groups"]):
        raise ValueError("cache fit-valid membership overlaps outer-valid")
    coefficients = metadata.get("source_coefficients")
    expected_sources = [
        {"family": "strong", "run": f"fg19_t3_h384x2_e3_s{seed}_v1", "epoch": 2, "coefficient": 0.1}
        for seed in (3, 4, 5)
    ]
    _eq(coefficients, expected_sources, "cache teacher order and coefficients")
    files = metadata.get("files")
    outputs = accept.get("output")
    stats = cache.get("array_stats_only")
    if not all(isinstance(value, dict) for value in (files, outputs, stats)):
        raise ValueError("cache file identities are malformed")
    observed: dict[str, object] = {}
    for name, groups, size, expected_sha in (
        ("train", 10535, 1_685_600_128, TRAIN_CACHE_SHA),
        ("valid", 1405, 224_800_128, VALID_CACHE_SHA),
    ):
        filename = f"main_homogeneous_{name}.npy"
        path = cache_root / filename
        pin, entry = stats.get(name), files.get(f"main_homogeneous_{name}")
        if not isinstance(pin, dict) or not isinstance(entry, dict):
            raise ValueError(f"cache {name} metadata/stat pin is missing")
        st = path.stat(follow_symlinks=False)
        identity = {
            "bytes": st.st_size,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
        }
        _eq(identity, pin, f"cache {name} stat")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cache array path is unsafe: {filename}")
        _eq(
            (identity["bytes"], entry.get("path"), entry.get("shape"), entry.get("dtype")),
            (size, filename, [groups, 20000, 2], "float32"),
            f"cache {name} metadata",
        )
        accepted_output = outputs.get(name)
        if not isinstance(accepted_output, dict):
            raise ValueError(f"cache acceptance output is missing: {name}")
        _eq(
            (
                entry.get("sha256"),
                accepted_output.get("sha256"),
                accepted_output.get("bytes"),
                accepted_output.get("shape"),
                accepted_output.get("dtype"),
            ),
            (expected_sha, expected_sha, size, [groups, 20000, 2], "float32"),
            f"cache {name} accepted hash/shape",
        )
        observed[name] = identity
    _eq(
        (
            cache.get("acceptance_sha256"),
            cache.get("exact_stop_sha256"),
            cache.get("metadata_sha256"),
        ),
        (accept_sha, stop_sha, meta_sha),
        "admitted cache binding",
    )
    return {
        "acceptance_sha256": accept_sha,
        "exact_stop_sha256": stop_sha,
        "metadata_sha256": meta_sha,
        "arrays_stat_only": observed,
        "array_sha256": {"train": TRAIN_CACHE_SHA, "valid": VALID_CACHE_SHA},
    }


def _check_cache_provenance(provenance: Mapping[str, object], cache_root: Path) -> None:
    """Bind the trainer's complete cache lineage to both accepted array digests."""
    actual = provenance.get("distillation_cache")
    if not isinstance(actual, dict):
        raise ValueError("distillation cache provenance is missing")
    _eq(
        actual,
        {**CACHE_PROVENANCE, "root": str(cache_root)},
        "cache source, scale, output hashes, and teacher lineage",
    )


def _check_terminal(terminal: Mapping[str, object]) -> None:
    """Require typed native success for this invocation before acceptance/stop."""
    _eq(
        (terminal.get("unit"), terminal.get("invocation_id"), terminal.get("run_name")),
        (UNIT, INVOCATION, RUN),
        "native terminal identity",
    )
    _eq(
        (terminal.get("active_state"), terminal.get("sub_state")),
        ("active", "exited"),
        "native state",
    )
    _eq(
        terminal.get("schema"),
        "fg19.parent-fit-r2-native-terminal-evidence.v1",
        "native evidence schema",
    )
    for key, value in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        _int(terminal.get(key), value, key)
    raw_bytes = terminal.get("raw_bytes")
    if type(raw_bytes) is not int or raw_bytes <= 0:
        raise ValueError("native terminal raw byte count is invalid")
    raw_sha = terminal.get("raw_sha256")
    if (
        not isinstance(raw_sha, str)
        or len(raw_sha) != 64
        or any(char not in "0123456789abcdef" for char in raw_sha)
    ):
        raise ValueError("native terminal raw capture identity is missing")


def _check_worker(
    launch: Mapping[str, object], admission: Mapping[str, object], source_root: Path
) -> None:
    """Bind admitted resource caps to the immutable systemd command and worker env."""
    worker = admission.get("worker")
    if not isinstance(worker, dict) or set(worker) != set(WORKER):
        raise ValueError("admitted worker resource identity is malformed")
    for key, expected in WORKER.items():
        actual = worker.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"admitted worker resource differs: {key}")
    command = launch.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("launch command is malformed")
    properties = {
        item.removeprefix("--property=") for item in command if item.startswith("--property=")
    }
    expected_properties = {
        "Type=exec",
        "RemainAfterExit=yes",
        "Slice=wnn-campaign.slice",
        "CPUQuota=100%",
        "MemoryHigh=7500000000",
        "MemoryMax=8000000000",
        "MemorySwapMax=0",
        "RuntimeMaxSec=86400",
        "TimeoutStopSec=45",
    }
    if not expected_properties.issubset(properties):
        raise ValueError("launch systemd resource properties differ")
    environment = {
        item.removeprefix("--setenv=").partition("=")[0]: item.removeprefix("--setenv=").partition(
            "="
        )[2]
        for item in command
        if item.startswith("--setenv=")
    }
    expected_environment = {
        "PYTHONPATH": str(source_root / "src"),
        "WNN_PROJECT_ROOT": str(source_root.parent / "data"),
        "WNN_GPU_MEMORY_BYTES": str(WORKER["cuda_allocator_bytes"]),
        "WNN_LOADER_WORKERS": "0",
        "WNN_LOADER_RAM_GB": "1",
        "WNN_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944",
        "NVIDIA_TF32_OVERRIDE": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TMPDIR": str(source_root.parent / "research_tmp"),
        "WNN_TMP_ROOT": str(source_root.parent / "research_tmp"),
    }
    _eq(
        {key: environment.get(key) for key in expected_environment},
        expected_environment,
        "launch worker environment",
    )
    _eq(
        [item for item in command if item.startswith("--working-directory=")],
        [f"--working-directory={source_root}"],
        "launch working directory",
    )


def _check_optimizer(optimizer: object, scheduler_lr: float) -> dict[str, int | float]:
    """Validate single-group AdamW counters, shapes, and finite nonnegative moments."""
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("optimizer state is malformed")
    groups, states = optimizer["param_groups"], optimizer.get("state")
    if len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("parent requires one AdamW state group")
    group = groups[0]
    betas = group.get("betas")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("AdamW beta settings are malformed")
    if type(betas[0]) not in {int, float} or type(betas[1]) not in {int, float}:
        raise ValueError("AdamW beta values must be numeric scalars")
    if not np.isfinite((betas[0], betas[1])).all() or not 0 < betas[0] < 1 or betas[1] != 0.999:
        raise ValueError("AdamW terminal beta schedule differs")
    _eq((group.get("weight_decay"), group.get("eps")), (0.01, 1e-8), "AdamW hyperparameters")
    if type(group.get("lr")) not in {int, float} or not np.isfinite(group["lr"]) or group["lr"] < 0:
        raise ValueError("AdamW terminal learning rate is invalid")
    _eq(group["lr"], scheduler_lr, "AdamW/scheduler terminal learning rate")
    for key in ("amsgrad", "maximize", "capturable", "differentiable", "fused"):
        if key in group and group[key] is not False:
            raise ValueError(f"AdamW policy differs: {key}")
    ids = group.get("params")
    if not isinstance(ids, list) or len(ids) != 14 or any(type(item) is not int for item in ids):
        raise ValueError("AdamW must have 14 integer parameter identities")
    if len(set(ids)) != 14 or set(states) != set(ids):
        raise ValueError("AdamW states do not cover the parameter group exactly")
    for parameter_id, shape in zip(ids, PARAMETER_SHAPES, strict=True):
        state = states[parameter_id]
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW parameter state schema differs")
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.ndim != 0 or not torch.isfinite(step).item():
            raise ValueError("AdamW step must be a finite scalar tensor")
        if float(step.item()) != UPDATES:
            raise ValueError("AdamW step count differs from completed updates")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise ValueError(f"AdamW {key} must be finite FP32 with its parameter shape")
            if key == "exp_avg_sq" and bool((tensor < 0).any().item()):
                raise ValueError("AdamW second moment must be nonnegative")
    return {
        "groups": 1,
        "parameters": 14,
        "moment_tensors": 28,
        "step": UPDATES,
        "learning_rate": group["lr"],
    }


def _check_counters(
    history: object, state: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
    """Reconcile four no-grade epochs across history, provenance, and resumable state."""
    diagnostics = provenance.get("update_diagnostics")
    if (
        not isinstance(history, list)
        or len(history) != 4
        or not isinstance(diagnostics, list)
        or len(diagnostics) != 4
    ):
        raise ValueError("four-epoch fit history and diagnostics are required")
    _finite_tree(history)
    progress = state.get("progress")
    if not isinstance(progress, dict) or set(progress) != {
        "step_count",
        "best",
        "history",
        "cumulative_updates",
        "update_diagnostics",
        "fitting_seconds",
    }:
        raise ValueError("training progress state is missing")
    if type(progress.get("best")) is not float or progress["best"] != -1.0:
        raise ValueError("fit-only parent must retain the unscored best sentinel")
    if (
        type(progress.get("fitting_seconds")) not in {int, float}
        or not np.isfinite(progress["fitting_seconds"])
        or progress["fitting_seconds"] < 0
    ):
        raise ValueError("saved fit duration is malformed")
    _eq(
        (progress.get("history"), progress.get("update_diagnostics")),
        (history, diagnostics),
        "state/history",
    )
    for epoch, (record, item) in enumerate(zip(history, diagnostics, strict=True)):
        _eq(
            (record.get("epoch"), record.get("evaluation"), item.get("epoch_index")),
            (epoch, "withheld", epoch),
            "fit-only history order",
        )
        if "weighted_pearson" in record:
            raise ValueError("fit history contains evaluation output")
        _eq(
            record.get("update_diagnostics"),
            {key: value for key, value in item.items() if key != "epoch_index"},
            "history update counters",
        )
        nested = item.get("epoch")
        if not isinstance(nested, dict):
            raise ValueError("epoch update counters are missing")
        for key, expected in (
            ("optimizer_attempts", EPOCH_UPDATES),
            ("optimizer_completed_updates", EPOCH_UPDATES),
            ("completed_batches", BATCHES),
            ("amp_skipped_updates", 0),
        ):
            _int(nested.get(key), expected, f"epoch {epoch} {key}")
        _int(nested.get("ema_updates"), 0, "epoch EMA update count")
        cumulative = item.get("cumulative")
        if not isinstance(cumulative, dict):
            raise ValueError("epoch cumulative counters are missing")
        for key, expected in (
            ("optimizer_attempts", (epoch + 1) * EPOCH_UPDATES),
            ("optimizer_completed_updates", (epoch + 1) * EPOCH_UPDATES),
            ("completed_batches", (epoch + 1) * BATCHES),
            ("amp_skipped_updates", 0),
            ("ema_updates", 0),
        ):
            _int(cumulative.get(key), expected, f"epoch {epoch} cumulative {key}")
        grads = item.get("grad_stats")
        if not isinstance(grads, dict):
            raise ValueError("gradient statistics are missing")
        _int(grads.get("steps"), EPOCH_UPDATES, "gradient steps")
        _int(grads.get("nonfinite_steps"), 0, "nonfinite gradient steps")
    totals = provenance.get("cumulative_updates")
    if not isinstance(totals, dict):
        raise ValueError("cumulative update counters are missing")
    for key, expected in (
        ("optimizer_attempts", UPDATES),
        ("optimizer_completed_updates", UPDATES),
        ("completed_batches", BATCHES * 4),
        ("amp_skipped_updates", 0),
        ("ema_updates", 0),
    ):
        _int(totals.get(key), expected, f"total {key}")
    _int(progress.get("step_count"), UPDATES, "state step count")
    _eq(progress.get("cumulative_updates"), totals, "state cumulative counters")


def _tensor_rng_signature(tensor: object) -> dict[str, object]:
    """Summarize one small RNG byte tensor without exposing its values."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError("Torch RNG tensor must be a one-dimensional uint8 tensor")
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return {
        "type": "torch.Tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "cardinality": tensor.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cuda_rng_signature(cuda_rng: object) -> dict[str, object]:
    """Summarize the single captured visible-device RNG tensor."""
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1:
        raise ValueError("single visible CUDA RNG state is malformed")
    return {
        "type": "builtins.list",
        "cardinality": len(cuda_rng),
        "components": [_tensor_rng_signature(cuda_rng[0])],
    }


def _check_tensor_rng(tensor: object, expected: object, label: str) -> None:
    """Require a saved RNG byte tensor to match its diagnostic signature."""
    _eq(_tensor_rng_signature(tensor), expected, label)


def _check_cuda_rng(cuda_rng: object, expected: object) -> None:
    """Require the saved visible-device RNG list to match its diagnostic."""
    _eq(_cuda_rng_signature(cuda_rng), expected, "single visible CUDA RNG")


def _validate_rng_diagnostic(
    report: Mapping[str, object], state_stat: Mapping[str, int], runtime_torch_version: str
) -> dict[str, object]:
    """Check the pinned diagnostic's schema, runtime and saved-state identity."""
    _eq(
        (
            report.get("schema"),
            report.get("status"),
            report.get("run_name"),
            report.get("format"),
            report.get("next_epoch"),
            report.get("torch_version"),
            runtime_torch_version,
            report.get("state_file_stat_only"),
        ),
        (
            "fg19.parent-fit-r2-rng-diagnostic.v2",
            "DIAGNOSTIC_ONLY_NO_ACCEPTANCE",
            RUN,
            "wnn.epoch-training-state",
            4,
            RNG_DIAGNOSTIC_TORCH_VERSION,
            RNG_DIAGNOSTIC_TORCH_VERSION,
            dict(state_stat),
        ),
        "RNG diagnostic identity",
    )
    rng = report.get("rng")
    if not isinstance(rng, dict):
        raise ValueError("RNG diagnostic summary is missing")
    _eq(rng.get("keys"), ["cuda", "numpy", "python", "torch"], "diagnostic RNG key order")
    entries = rng.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("RNG diagnostic entries are missing")
    _eq(set(entries), {"python", "numpy", "torch", "cuda"}, "diagnostic RNG keys")
    for key, expected in EXPECTED_RNG_TENSORS.items():
        _eq(entries.get(key), expected, f"diagnostic {key} RNG signature")
    return entries


def _check_loader_and_rng(
    state: Mapping[str, object], diagnostic_entries: Mapping[str, object]
) -> None:
    """Verify the saved epoch-boundary loader and RNG structures without restoring them."""
    loader = state.get("loader")
    expected_loader = {
        "sampler_type": "torch.utils.data.sampler.RandomSampler",
        "batch_size": 16,
        "drop_last": True,
        "num_workers": 0,
        "dataset_length": 11940,
        "loader_generator": None,
        "sampler_generator": None,
        "sampler_state": None,
        "sampler_epoch": None,
    }
    _eq(loader, expected_loader, "epoch-boundary loader state")
    rng = state.get("rng")
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("saved RNG state schema differs")
    python_rng = rng["python"]
    if (
        not isinstance(python_rng, tuple)
        or len(python_rng) != 3
        or type(python_rng[0]) is not int
        or not isinstance(python_rng[1], tuple)
        or len(python_rng[1]) != 625
        or any(type(value) is not int for value in python_rng[1])
        or (python_rng[2] is not None and type(python_rng[2]) is not float)
    ):
        raise ValueError("Python RNG state is malformed")
    numpy_rng = rng["numpy"]
    if (
        not isinstance(numpy_rng, tuple)
        or len(numpy_rng) != 5
        or numpy_rng[0] != "MT19937"
        or not isinstance(numpy_rng[1], np.ndarray)
        or numpy_rng[1].dtype != np.uint32
        or numpy_rng[1].shape != (624,)
        or type(numpy_rng[2]) is not int
        or type(numpy_rng[3]) is not int
        or type(numpy_rng[4]) not in {float, int}
    ):
        raise ValueError("NumPy RNG state is malformed")
    torch_rng = rng["torch"]
    _check_tensor_rng(torch_rng, diagnostic_entries.get("torch"), "CPU Torch RNG")
    _check_cuda_rng(rng["cuda"], diagnostic_entries.get("cuda"))


def _pinned_json(path: Path, expected_sha: str) -> tuple[dict[str, object], str]:
    """Read one small, stable JSON receipt and require its exact digest."""
    raw, digest, _ = _stable_file(path, 16_000)
    _eq(digest, expected_sha, f"pinned metadata digest for {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"pinned JSON receipt is not an object: {path.name}")
    return value, digest


def _check_rng_diagnostic_receipts(
    acceptance: Mapping[str, object], exact_stop: Mapping[str, object], report_sha: str
) -> dict[str, object]:
    """Bind the accepted diagnostic and exact-stop receipts to the report."""
    _eq(
        (
            acceptance.get("schema"),
            acceptance.get("status"),
            acceptance.get("unit"),
            acceptance.get("invocation_id"),
            acceptance.get("report_sha256"),
            acceptance.get("raw_sha256"),
            acceptance.get("source_sha256"),
            acceptance.get("quality_score"),
            acceptance.get("native_untouched"),
        ),
        (
            "fg19.parent-rng-diagnostic-capture-acceptance.v2",
            "CAPTURE_VALIDATED_DIAGNOSTIC_ONLY_NOT_PARENT_ACCEPTANCE",
            RNG_DIAGNOSTIC_UNIT,
            RNG_DIAGNOSTIC_INVOCATION,
            RNG_DIAGNOSTIC_SHA,
            RNG_DIAGNOSTIC_RAW_SHA,
            RNG_DIAGNOSTIC_SOURCE_SHA,
            None,
            True,
        ),
        "RNG diagnostic capture acceptance",
    )
    before = exact_stop.get("before")
    after = exact_stop.get("after")
    stopped = exact_stop.get("stop")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(stopped, dict):
        raise ValueError("RNG diagnostic exact-stop receipt is incomplete")
    _eq(
        (
            exact_stop.get("unit"),
            exact_stop.get("invocation_id"),
            exact_stop.get("capture_acceptance_sha256"),
            exact_stop.get("diagnostic_report_sha256"),
            exact_stop.get("raw_capture_sha256"),
            before.get("InvocationID"),
            before.get("ActiveState"),
            before.get("SubState"),
            before.get("MainPID"),
            before.get("ExecMainCode"),
            before.get("ExecMainStatus"),
            stopped.get("returncode"),
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("InvocationID"),
            exact_stop.get("populated_before"),
            exact_stop.get("populated_after"),
        ),
        (
            RNG_DIAGNOSTIC_UNIT,
            RNG_DIAGNOSTIC_INVOCATION,
            RNG_DIAGNOSTIC_ACCEPTANCE_SHA,
            RNG_DIAGNOSTIC_SHA,
            RNG_DIAGNOSTIC_RAW_SHA,
            RNG_DIAGNOSTIC_INVOCATION,
            "active",
            "exited",
            "0",
            "1",
            "0",
            0,
            "not-found",
            "inactive",
            "dead",
            "",
            {},
            {},
        ),
        "RNG diagnostic exact stop",
    )
    return {
        "report_sha256": report_sha,
        "capture_acceptance_sha256": RNG_DIAGNOSTIC_ACCEPTANCE_SHA,
        "exact_stop_sha256": RNG_DIAGNOSTIC_STOP_SHA,
        "raw_capture_sha256": RNG_DIAGNOSTIC_RAW_SHA,
        "source_sha256": RNG_DIAGNOSTIC_SOURCE_SHA,
    }


def _check_scheduler(scheduler: object) -> float:
    """Bind the saved OneCycle schedule cursor to the complete optimizer budget."""
    if not isinstance(scheduler, dict):
        raise ValueError("scheduler state is missing")
    _eq(
        set(scheduler),
        {
            "total_steps",
            "_schedule_phases",
            "_anneal_func_type",
            "cycle_momentum",
            "use_beta1",
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_is_initial",
            "_get_lr_called_within_step",
            "_last_lr",
        },
        "OneCycle scheduler schema",
    )
    _int(scheduler.get("total_steps"), UPDATES, "scheduler total steps")
    _int(scheduler.get("last_epoch"), UPDATES, "scheduler cursor")
    _int(scheduler.get("_step_count"), UPDATES + 1, "scheduler step count")
    if scheduler.get("cycle_momentum") is not True:
        raise ValueError("AdamW OneCycle beta schedule was not retained")
    _eq(
        (
            scheduler.get("_anneal_func_type"),
            scheduler.get("use_beta1"),
            scheduler.get("_is_initial"),
            scheduler.get("_get_lr_called_within_step"),
        ),
        ("cos", True, False, False),
        "OneCycle schedule policy",
    )
    _eq(
        scheduler.get("_schedule_phases"),
        [
            {
                "end_step": 1491.0,
                "start_lr": "initial_lr",
                "end_lr": "max_lr",
                "start_momentum": "max_momentum",
                "end_momentum": "base_momentum",
            },
            {
                "end_step": 29839,
                "start_lr": "max_lr",
                "end_lr": "min_lr",
                "start_momentum": "base_momentum",
                "end_momentum": "max_momentum",
            },
        ],
        "OneCycle schedule phases",
    )
    _eq(scheduler.get("base_lrs"), [4e-06], "OneCycle base rate")
    rates = scheduler.get("_last_lr")
    if (
        not isinstance(rates, list)
        or len(rates) != 1
        or type(rates[0]) not in {int, float}
        or not np.isfinite(rates[0])
        or rates[0] < 0
    ):
        raise ValueError("OneCycle final learning rate is malformed")
    return float(rates[0])


def _check_auxiliary(auxiliary: object) -> None:
    """Validate the source-derived no-EMA/no-SWA auxiliary continuation state."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {
        "norm_ema",
        "dynamics",
        "pool_memory",
        "readout_moments",
        "input_representation",
    }:
        raise ValueError("training auxiliary-state schema differs")
    dynamics = auxiliary.get("dynamics")
    expected_dynamics = {
        "switch": None,
        "caps": {},
        "eta": 0.0,
        "clip": 1.0,
        "no_clip": False,
        "seed": 1001 + 104729,
        "is_switched": False,
        "events": [],
        "generators": {},
    }
    _eq(dynamics, expected_dynamics, "fit-only optimizer dynamics")
    _eq(
        {
            key: auxiliary.get(key)
            for key in ("norm_ema", "pool_memory", "readout_moments", "input_representation")
        },
        {
            "norm_ema": None,
            "pool_memory": None,
            "readout_moments": None,
            "input_representation": None,
        },
        "disabled auxiliary state",
    )


def _expected_launch_args(source_root: Path, split_path: Path) -> dict[str, object]:
    """Build the full parser-default contract, then apply only admitted parent values."""
    import wnn.training.cli as training_cli

    cli_path = Path(training_cli.__file__).resolve(strict=True)
    _eq(
        cli_path,
        (source_root / "src/wnn/training/cli.py").resolve(strict=True),
        "canonical trainer CLI source",
    )
    namespace = training_cli.build_parser().parse_args(["--name", RUN])
    defaults = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }
    defaults.update(ARGS)
    defaults.update(
        {
            "split_manifest": str(split_path),
            "distill_cache_root": str(
                source_root.parent / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
            ),
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": SPLIT_SHA,
        }
    )
    return defaults


def _check_paths(source_root: Path, run_dir: Path, split_path: Path, cache_root: Path) -> None:
    """Require the single admitted source, output, split and cache locations."""
    source = source_root.resolve(strict=True)
    base = source.parent
    _eq(source.name, "source-fg19-cache-68ff6829", "staged source directory")
    expected = {
        "source root": source_root,
        "run directory": run_dir,
        "split manifest": split_path,
        "cache root": cache_root,
    }
    targets = {
        "source root": base / "source-fg19-cache-68ff6829",
        "run directory": base / "data/artifacts/runs" / RUN,
        "split manifest": base / "splits/fold0.json",
        "cache root": base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1",
    }
    for label, path in expected.items():
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        _eq(path.resolve(strict=True), targets[label].resolve(strict=True), label)


def _check_contract(
    state: Mapping[str, object],
    provenance: Mapping[str, object],
    admission: Mapping[str, object],
    launch_args: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    """Bind continuation options, source/dataset maps, cache, and strict runtime."""
    contract = state.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("epoch continuation contract is missing")
    _eq(set(contract), CONTRACT_KEYS, "continuation contract schema")
    _eq((contract.get("epochs"), contract.get("total_steps")), (4, UPDATES), "contract horizon")
    excluded = {
        "name",
        "resume_state",
        "save_training_state",
        "stop_after_epoch",
        "split_manifest",
        "fit_groups_manifest",
    }
    expected_options = {key: value for key, value in launch_args.items() if key not in excluded}
    _eq(contract.get("options"), expected_options, "contract options")
    _eq(
        contract.get("fit_membership"),
        provenance.get("fit_membership_identity"),
        "contract fit membership",
    )
    _eq(
        contract.get("normalization"),
        provenance.get("normalization_identity"),
        "contract normalization",
    )
    _eq(
        contract.get("ancestors"),
        {
            "initial_checkpoint_sha256": None,
            "propensity_checkpoint_sha256": None,
            "quote_cache": None,
            "spread_aux": None,
        },
        "clean parent ancestry",
    )
    data = admission.get("data_identity_stats_only")
    if not isinstance(data, dict):
        raise ValueError("admitted dataset identity is missing")
    expected_data = {
        split: {
            "bytes": data[f"datasets/{split}.parquet"]["bytes"],
            "sha256": data[f"datasets/{split}.parquet"]["registered_sha256"],
        }
        for split in ("train", "valid")
    }
    _eq(contract.get("datasets"), expected_data, "contract dataset identities")
    _eq(contract.get("teacher_predictions"), {}, "unexpected standalone teacher predictions")
    cache_meta = provenance.get("distillation_cache", {})
    verified = cache_meta.get("verified_files", {})
    _eq(
        contract.get("consumed_caches"),
        {
            "distillation_cache_metadata": CACHE_METADATA_SHA,
            **{f"distillation_{name}": value for name, value in verified.items()},
        },
        "contract cache identities",
    )
    files = manifest.get("files")
    source_map = {item["path"]: item["sha256"] for item in files}
    _eq(
        source_map.get("src/wnn/resources.py"),
        CUDA_POLICY_SOURCE_SHA,
        "CUDA resource-policy source pin",
    )
    expected_sources = {
        name.removeprefix("src/wnn/"): digest
        for name, digest in source_map.items()
        if name.startswith("src/wnn/") and name.endswith(".py")
    }
    expected_sources["trainer"] = source_map["scripts/train_rnn.py"]
    _eq(contract.get("sources"), expected_sources, "continuation source map")
    implementation_files = {
        name: digest
        for name, digest in source_map.items()
        if name == "scripts/train_rnn.py"
        or name in {"pyproject.toml", "uv.lock"}
        or (name.startswith("src/wnn/training/") and name.endswith(".py"))
    }
    implementation = provenance.get("trainer_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("trainer implementation provenance is missing")
    encoded = json.dumps(implementation_files, sort_keys=True, separators=(",", ":")).encode()
    _eq(
        (implementation.get("files"), implementation.get("sha256")),
        (implementation_files, hashlib.sha256(encoded).hexdigest()),
        "trainer implementation source",
    )
    runtime = contract.get("runtime")
    runtime_keys = {
        "torch",
        "numpy",
        "python",
        "machine",
        "cublas_workspace_config",
        "devices",
        "cuda",
        "cudnn",
        "threads",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_tf32",
        "cudnn_tf32",
    }
    if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
        raise ValueError("runtime identity is missing")
    _int(runtime.get("threads"), 1, "runtime thread count")
    _eq(
        (
            runtime.get("matmul_tf32"),
            runtime.get("cudnn_tf32"),
            runtime.get("cublas_workspace_config"),
            runtime.get("cudnn_benchmark"),
        ),
        (False, False, ":4096:8", False),
        "strict runtime",
    )
    devices = runtime.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
        or set(devices[0]) != {"name", "capability"}
        or not isinstance(devices[0].get("name"), str)
        or not isinstance(devices[0].get("capability"), list)
        or any(type(value) is not int for value in devices[0]["capability"])
    ):
        raise ValueError("fit did not bind exactly one visible CUDA device")
    for key in ("torch", "numpy", "python", "machine", "cuda"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"runtime {key} identity is malformed")
    if type(runtime.get("cudnn")) is not int or runtime["cudnn"] <= 0:
        raise ValueError("runtime cuDNN version is malformed")
    _check_cuda_cudnn_deterministic(runtime.get("cudnn_deterministic"))
    _eq(
        contract.get("loader"),
        {
            "sampler_type": "torch.utils.data.sampler.RandomSampler",
            "batch_size": 16,
            "drop_last": True,
            "num_workers": 0,
            "dataset_length": 11940,
        },
        "continuation loader identity",
    )


def verify_terminal(
    run_dir: Path,
    source_root: Path,
    source_manifest: Path,
    split_path: Path,
    admission_path: Path,
    launch_path: Path,
    launch_acceptance_path: Path,
    terminal_path: Path,
    cache_root: Path,
    cache_acceptance_path: Path,
    cache_stop_path: Path,
    rng_diagnostic_path: Path,
    rng_diagnostic_acceptance_path: Path,
    rng_diagnostic_stop_path: Path,
) -> dict[str, object]:
    """Check actual exit and bounded saved metadata without training or array reads."""
    _check_paths(source_root, run_dir, split_path, cache_root)
    admission, admission_sha = _json_file(admission_path)
    launch, launch_sha = _json_file(launch_path)
    launch_acceptance, la_sha = _json_file(launch_acceptance_path)
    terminal, terminal_sha = _json_file(terminal_path)
    rng_report, rng_report_sha = _pinned_json(rng_diagnostic_path, RNG_DIAGNOSTIC_SHA)
    rng_acceptance, rng_acceptance_sha = _pinned_json(
        rng_diagnostic_acceptance_path, RNG_DIAGNOSTIC_ACCEPTANCE_SHA
    )
    rng_stop, rng_stop_sha = _pinned_json(rng_diagnostic_stop_path, RNG_DIAGNOSTIC_STOP_SHA)
    _eq(
        (rng_acceptance_sha, rng_stop_sha),
        (RNG_DIAGNOSTIC_ACCEPTANCE_SHA, RNG_DIAGNOSTIC_STOP_SHA),
        "RNG diagnostic receipt digests",
    )
    rng_receipt = _check_rng_diagnostic_receipts(rng_acceptance, rng_stop, rng_report_sha)
    _eq(
        (admission_sha, launch_sha, la_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "launch pins",
    )
    _eq(
        (admission.get("schema"), admission.get("run_name"), admission.get("unit")),
        ("fg19-parent-fit-r2-admission.v1", RUN, UNIT),
        "admission identity",
    )
    _check_worker(launch, admission, source_root)
    _eq(
        (launch.get("schema"), launch.get("unit"), launch.get("submission", {}).get("returncode")),
        ("fg19-parent-fit-r2-launch.v1", UNIT, 0),
        "launch record",
    )
    _eq(
        (
            launch_acceptance.get("schema"),
            launch_acceptance.get("invocation_id"),
            launch_acceptance.get("admission_sha256"),
            launch_acceptance.get("launch_sha256"),
        ),
        ("fg19-parent-fit-r2-launch-acceptance.v1", INVOCATION, admission_sha, launch_sha),
        "launch acceptance",
    )
    _check_terminal(terminal)
    source = _source_identity(source_root, source_manifest, admission)
    split, split_sha = _json_file(split_path)
    _eq((split_sha, admission.get("split_sha256")), (SPLIT_SHA, SPLIT_SHA), "split manifest")
    cache = _check_cache(admission, cache_root, cache_acceptance_path, cache_stop_path, split)
    from teacher_s5_terminal_check_v1 import _canonical_tensor_identity, _launch_cli_args

    launch_args = _launch_cli_args(launch, source_root)
    launch_args.update(
        {
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": split_sha,
        }
    )
    expected_args = _expected_launch_args(source_root, split_path)
    _eq(launch_args, expected_args, "full trainer arguments")
    base = source_root.parent
    _eq(
        launch_args.get("distill_cache_root"),
        str(base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"),
        "cache path",
    )
    _eq(launch_args.get("split_manifest"), str(split_path), "split argument path")
    names = (
        "model.pt",
        "model_ep0.pt",
        "model_ep1.pt",
        "model_ep2.pt",
        "model_ep3.pt",
        "training_state.pt",
        "provenance.json",
        "history.json",
    )
    limits = {name: MAX_MODEL_BYTES for name in names[:5]}
    limits.update(
        {
            "training_state.pt": MAX_STATE_BYTES,
            "provenance.json": MAX_JSON_BYTES,
            "history.json": MAX_JSON_BYTES,
        }
    )
    paths = {name: run_dir / name for name in names}
    files = {name: _hash_file(path, limits[name]) for name, path in paths.items()}
    rng_entries = _validate_rng_diagnostic(
        rng_report, files["training_state.pt"][1], str(torch.__version__)
    )
    if sum(info["bytes"] for _, info in files.values()) > MAX_AGGREGATE_BYTES:
        raise ValueError("terminal artifact set exceeds bounded aggregate size")
    provenance, provenance_sha = _json_file(paths["provenance.json"])
    history_raw, history_sha, _ = _stable_file(paths["history.json"], MAX_JSON_BYTES)
    history = json.loads(history_raw)
    _eq(
        (provenance_sha, history_sha),
        (files["provenance.json"][0], files["history.json"][0]),
        "metadata stability",
    )
    _eq(
        (provenance.get("selected_epoch"), provenance.get("split_manifest_sha256")),
        (3, SPLIT_SHA),
        "selected endpoint",
    )
    _eq(provenance.get("args"), launch_args, "provenance args")
    counts = _check_membership(provenance, split)
    _eq(provenance.get("initial_checkpoint_sha256"), None, "clean initialization")
    _eq(provenance.get("propensity_checkpoint_sha256"), None, "no propensity ancestor")
    _check_cache_provenance(provenance, cache_root)
    tensor_identity = _canonical_tensor_identity(source_root)
    fingerprints = []
    for name in ("model.pt", "model_ep3.pt"):
        payload, digest, info = _load_checkpoint(paths[name], MAX_MODEL_BYTES)
        _eq((digest, info), files[name], f"stable {name}")
        _eq(
            (payload.get("epoch"), payload.get("n_in"), payload.get("args")),
            (3, 211, launch_args),
            f"{name} identity",
        )
        model = payload.get("model")
        fingerprint = _model_fingerprint(model)
        fingerprints.append(fingerprint)
        _eq(
            {
                key: payload.get("provenance", {}).get(key)
                for key in (
                    "args",
                    "split_manifest_sha256",
                    "fit_row_groups",
                    "evaluation_row_groups",
                    "fit_membership_identity",
                    "normalization_identity",
                    "trainer_implementation",
                    "distillation_cache",
                    "update_diagnostics",
                    "cumulative_updates",
                )
            },
            {
                key: provenance.get(key)
                for key in (
                    "args",
                    "split_manifest_sha256",
                    "fit_row_groups",
                    "evaluation_row_groups",
                    "fit_membership_identity",
                    "normalization_identity",
                    "trainer_implementation",
                    "distillation_cache",
                    "update_diagnostics",
                    "cumulative_updates",
                )
            },
            f"{name} provenance",
        )
        actual_norm = {key: tensor_identity(model[key]) for key in ("mean", "std")}
        _eq(actual_norm, provenance.get("normalization_identity"), f"{name} normalizer identity")
    _eq(fingerprints[0], fingerprints[1], "selected model equals epoch 3")
    state, digest, info = _load_checkpoint(paths["training_state.pt"], MAX_STATE_BYTES)
    _eq((digest, info), files["training_state.pt"], "stable training state")
    _eq(
        (state.get("format"), state.get("version"), state.get("boundary"), state.get("next_epoch")),
        ("wnn.epoch-training-state", 1, "epoch", 4),
        "resume boundary",
    )
    _eq(set(state), TRAINING_STATE_KEYS, "training-state schema")
    _eq(
        (state.get("ema"), state.get("swa"), state.get("scaler")),
        (None, None, {}),
        "averaging/scaler state",
    )
    _check_auxiliary(state.get("auxiliary"))
    _eq(_model_fingerprint(state.get("model")), fingerprints[0], "training-state model fingerprint")
    scheduler_lr = _check_scheduler(state.get("scheduler"))
    optimizer = _check_optimizer(state.get("optimizer"), scheduler_lr)
    _check_counters(history, state, provenance)
    _check_loader_and_rng(state, rng_entries)
    manifest, _ = _json_file(source_manifest)
    _check_contract(state, provenance, admission, launch_args, manifest)
    _eq(
        provenance.get("checkpoint_sha256"),
        {name: files[name][0] for name in names[:5]},
        "checkpoint hash map",
    )
    return {
        "schema": "fg19.parent-fit-r2-terminal-check.v3",
        "status": "PASS",
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "terminal_evidence_sha256": terminal_sha,
        "launch_sha256": launch_sha,
        "launch_acceptance_sha256": la_sha,
        "source": source,
        "split_sha256": split_sha,
        "cache": cache,
        "membership_counts": counts,
        "expected_updates_verified": UPDATES,
        "next_epoch": 4,
        "selected_epoch": 3,
        "model_tensor_count": fingerprints[0][1],
        "model_tensor_fingerprint": fingerprints[0][0],
        "optimizer": optimizer,
        "rng_diagnostic": {
            **rng_receipt,
            "schema": "fg19.parent-fit-r2-rng-diagnostic.v2",
            "torch_version": RNG_DIAGNOSTIC_TORCH_VERSION,
            "cuda_rng": EXPECTED_RNG_TENSORS["cuda"],
            "cpu_rng": EXPECTED_RNG_TENSORS["torch"],
        },
        "cuda_resource_policy_source": {
            "path": "src/wnn/resources.py",
            "sha256": CUDA_POLICY_SOURCE_SHA,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "normalizer_source": {
            "path": "src/wnn/training/normalization.py",
            "sha256": _manifest_hash(source_manifest, "src/wnn/training/normalization.py"),
            "scope": "first 2 fit-loader batches (32 sequences); flattened feature rows [::7]; sample std clamp 0.001",
        },
        "files": {
            name: {"sha256": pair[0], "bytes": pair[1]["bytes"]} for name, pair in files.items()
        },
    }


def _manifest_hash(manifest_path: Path, relative: str) -> str:
    """Read the small source manifest to cite the normalizer implementation pin."""
    manifest, _ = _json_file(manifest_path)
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("source manifest file list is missing")
    matches = [
        item.get("sha256")
        for item in rows
        if isinstance(item, dict) and item.get("path") == relative
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("normalization implementation pin is missing")
    return matches[0]


def main() -> int:
    """Run the checker and create a mode-0600 JSON receipt without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-dir",
        "source-root",
        "source-manifest",
        "split",
        "admission",
        "launch",
        "launch-acceptance",
        "terminal-evidence",
        "cache-root",
        "cache-acceptance",
        "cache-exact-stop",
        "rng-diagnostic",
        "rng-diagnostic-acceptance",
        "rng-diagnostic-exact-stop",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = verify_terminal(
        args.run_dir,
        args.source_root,
        args.source_manifest,
        args.split,
        args.admission,
        args.launch,
        args.launch_acceptance,
        args.terminal_evidence,
        args.cache_root,
        args.cache_acceptance,
        args.cache_exact_stop,
        args.rng_diagnostic,
        args.rng_diagnostic_acceptance,
        args.rng_diagnostic_exact_stop,
    )
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
