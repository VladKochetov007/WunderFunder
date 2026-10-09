"""Protect recovery grading from changed row order, weights and checkpoint binding."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/grade_recovered_loss_pair.py"
LOADER = importlib.util.spec_from_file_location("loss_grader", SOURCE)
assert LOADER is not None and LOADER.loader is not None
GRADER = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(GRADER)


@pytest.fixture
def bound_inputs() -> tuple[dict, dict, dict, list[int]]:
    """Use actual completed-fit bindings with a synthetic export sidecar."""
    spec = json.loads(
        (
            ROOT
            / "research/campaign_2026-10-06/bandit_continuation"
            / "diversity_stack_20261009/grading_r1/FROZEN_GRADER_R1.json"
        ).read_text()
    )
    split = GRADER.read_pinned(ROOT, spec["split"])
    groups = split["evaluation"]["valid"]
    arm = spec["arms"]["wmse"]
    sidecar = {
        "run": arm["run"],
        "split": "valid",
        "epoch": 1,
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "prediction_sha256": "a" * 64,
        "split_manifest_sha256": spec["split"]["sha256"],
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "parsed_sequence_cache": {
            **spec["cache_identity"],
            "validation_receipt_sha256": "b" * 64,
        },
    }
    return sidecar, arm, spec, groups


@pytest.mark.parametrize("mutation", ["order", "checkpoint", "precision", "payload", "cache"])
def test_changed_export_is_rejected(bound_inputs: tuple, mutation: str) -> None:
    sidecar, arm, spec, groups = bound_inputs
    GRADER.validate_sidecar(sidecar, arm, spec, groups, "a" * 64)
    changed = copy.deepcopy(sidecar)
    if mutation == "order":
        changed["row_groups"] = list(reversed(groups))
    elif mutation == "checkpoint":
        changed["checkpoint_sha256"] = "c" * 64
    elif mutation == "precision":
        changed["inference_precision"] = "fp16"
    elif mutation == "payload":
        changed["prediction_sha256"] = "c" * 64
    else:
        changed["parsed_sequence_cache"]["membership_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="binding differs"):
        GRADER.validate_sidecar(changed, arm, spec, groups, "a" * 64)


def test_artifact_hash_and_missing_input_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"epoch":1}\n')
    pin = {"path": path.name, "sha256": GRADER.sha256(path)}
    path.write_text('{"epoch":0}\n')
    with pytest.raises(ValueError, match="identity changed"):
        GRADER.read_pinned(tmp_path, pin)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        GRADER.read_pinned(tmp_path, pin)
