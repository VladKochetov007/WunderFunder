"""Frozen predictor command and fit/split binding checks for feature-only export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from adapter_common import _NAME, _json, _sha256


def predictor_argv(script: Path, run_name: str, split_manifest: Path) -> list[str]:
    """Select only the frozen fixed-epoch evaluation groups, using strict FP32."""
    return [
        str(script),
        run_name,
        "valid",
        "--epoch",
        "5",
        "--campaign-split-manifest",
        str(split_manifest),
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]


def _load_grader(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("tabdata_r4_grade_endpoint", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the pinned R4 endpoint grader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_fit_binding(
    run_dir: Path,
    split_manifest: Path,
    run_name: str,
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    fit_receipt = _json(run_dir / "producer_receipt.json")
    command = _json(run_dir / "command.json")
    preflight = _json(run_dir / "adapter_preflight.json")
    if (
        fit_receipt.get("schema") != "tabdata-l4-fit-producer.v1"
        or fit_receipt.get("status") != "completed_fixed_epoch5"
        or fit_receipt.get("run_name") != run_name
        or fit_receipt.get("fold") != fold
        or fit_receipt.get("seed") != seed
        or fit_receipt.get("selected_epoch") != 5
    ):
        raise ValueError("export requires the matching completed fixed-epoch-5 fit receipt")
    if (
        command.get("mode") not in {"fit", "resume"}
        or command.get("run_name") != run_name
        or command.get("fold") != fold
        or command.get("seed") != seed
    ):
        raise ValueError("training command receipt does not identify this fit cell")
    if not _NAME.fullmatch(run_name):
        raise ValueError("invalid run name")
    split_sha = _sha256(split_manifest)
    ancestry = fit_receipt.get("input_ancestry", {})
    if (
        split_sha != preflight.get("split_manifest_sha256")
        or split_sha != ancestry.get("split_manifest_sha256")
        or preflight.get("fold") != fold
        or ancestry.get("fold") != fold
    ):
        raise ValueError("export split manifest differs from the fitted workspace ancestry")
    split = _json(split_manifest)
    groups = split.get("evaluation", {}).get("valid")
    if (
        split.get("fold") != fold
        or split.get("n_folds") != 4
        or not isinstance(groups, list)
        or any(type(group) is not int or group < 0 for group in groups)
        or not groups
        or len(set(groups)) != len(groups)
        or groups != ancestry.get("evaluation_groups")
        or groups != preflight.get("evaluation_groups")
    ):
        raise ValueError("evaluation row groups differ from the exact frozen split order")
    checkpoint_relative = f"artifacts/runs/{run_name}/model_ep5.pt"
    recorded = fit_receipt.get("outputs", {}).get(checkpoint_relative)
    checkpoint = run_dir / checkpoint_relative
    if (
        not isinstance(recorded, dict)
        or recorded.get("sha256") != _sha256(checkpoint)
        or not checkpoint.is_file()
    ):
        raise ValueError("fixed epoch-5 checkpoint differs from the completed fit receipt")
    return fit_receipt, command, groups
