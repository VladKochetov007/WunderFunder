"""Fail-closed D02 comparison with the completed AdamW producer's artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

CACHE_SHA = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
MEMBERSHIP_SHA = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
ROLES = ("initial_state", "normalization", "sample_orders", "onecycle_trace", "profile_inputs")


def digest(path: Path) -> str:
    """Hash the producer's small shared artifacts without cache materialization."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_controls(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Path]]:
    """Require completed, exactly pinned AdamW controls and relative artifact triples."""
    if digest(path) != expected_sha:
        raise ValueError("D02 AdamW control contract SHA differs")
    contract = json.loads(path.read_text())
    if (
        contract.get("schema") != "wnn.loss-pair.training-contract.v1"
        or contract.get("format_version") != 1
        or contract.get("state") != "complete"
        or contract.get("cache", {}).get("manifest_sha256") != CACHE_SHA
        or contract.get("cache", {}).get("membership_sha256") != MEMBERSHIP_SHA
        or contract.get("fit", {}).get("optimizer") != "AdamW"
    ):
        raise ValueError("D02 control contract is not the completed matched AdamW producer")
    resolved: dict[str, Path] = {}
    for role in ROLES:
        row = contract["artifacts"][role]
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("D02 control artifact escapes its producer output")
        artifact = path.parent / relative
        artifact.resolve(strict=True).relative_to(path.parent.resolve(strict=True))
        if artifact.stat().st_size != row["size_bytes"] or digest(artifact) != row["sha256"]:
            raise ValueError(f"D02 control artifact changed: {role}")
        resolved[role] = artifact
    return contract, resolved


@lru_cache(maxsize=1)
def controls_from_env() -> tuple[dict[str, Any], dict[str, Path]]:
    """Load once in each bounded trainer process, before update zero."""
    return load_controls(
        Path(os.environ["WNN_D02_CONTROL_CONTRACT"]), os.environ["WNN_D02_CONTROL_CONTRACT_SHA256"]
    )


def assert_same_npz(actual: Path, expected: Path) -> None:
    """Compare exact tensor content; NPZ container timestamps are not tensor identity."""
    import numpy as np

    with (
        np.load(actual, allow_pickle=False) as left,
        np.load(expected, allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files):
            raise ValueError("D02 control tensor keys differ")
        for name in left.files:
            a, b = left[name], right[name]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f"D02 control tensor differs: {name}")


def verify_prefit(prefit: Path) -> dict[str, Any]:
    """Verify regenerated fresh initial tensors, normalization and complete epoch order."""
    contract, artifacts = controls_from_env()
    for role in ("initial_state", "normalization"):
        assert_same_npz(prefit / f"{role}.npz", artifacts[role])
    actual = json.loads((prefit / "sample_orders.json").read_text())
    expected = json.loads(artifacts["sample_orders"].read_text())
    for key in (
        "schema",
        "seed",
        "batch_size",
        "drop_last",
        "membership_sha256",
        "epoch_seeds",
        "epochs",
        "evaluation_ids",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"D02 control sequence order differs: {key}")
    return {
        "status": "pass",
        "control_contract_sha256": digest(Path(os.environ["WNN_D02_CONTROL_CONTRACT"])),
        "cache_manifest_sha256": contract["cache"]["manifest_sha256"],
        "tensor_comparison": "exact keys/dtypes/shapes/values; regenerated fresh state",
        "orders_comparison": "normalization epoch -1, fit epochs 0/1 and outer evaluation",
    }


def verify_first_batch(path: Path) -> None:
    """Bind actual pre-update raw inputs/labels/fit mask to the original first batch."""
    _, artifacts = controls_from_env()
    assert_same_npz(path, artifacts["profile_inputs"])


def verify_schedule_row(row: dict[str, Any]) -> None:
    """Both D02 groups must reproduce the AdamW control's pre-update LR and beta1."""
    expected = control_schedule()[row["attempt_index"]]
    for key in ("attempt_index", "epoch_index", "batch_index", "chunk_index", "scheduler_step_pre"):
        if row[key] != expected[key]:
            raise ValueError(f"D02 control schedule location differs: {key}")
    groups = row["optimizer_groups"]
    if len(groups) != 2 or len(expected["optimizer_groups"]) != 1:
        raise ValueError("D02 expects exactly two groups against the one-group AdamW control")
    control = expected["optimizer_groups"][0]
    if any(group["lr"] != control["lr"] or group["beta1"] != control["beta1"] for group in groups):
        raise ValueError("D02 control pre-update LR/beta1 differs")


@lru_cache(maxsize=1)
def control_schedule() -> list[dict[str, Any]]:
    """Buffer the small shared trace instead of per-update filesystem reads."""
    _, artifacts = controls_from_env()
    schedule = json.loads(artifacts["onecycle_trace"].read_text())
    if (
        schedule.get("schema") != "wnn.loss-pair.onecycle-trace.v1"
        or schedule.get("total_steps") != 220
        or len(schedule["rows"]) != 220
    ):
        raise ValueError("D02 requires the complete 220-attempt AdamW trace")
    return schedule["rows"]
