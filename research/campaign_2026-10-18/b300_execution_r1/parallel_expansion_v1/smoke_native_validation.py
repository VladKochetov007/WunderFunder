"""Validate epoch-zero smoke artifacts with pinned checkpoint and grader code."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

from feature_common import ARMS, GRADER_SHA256, read, sha  # noqa: E402


def set_option(argv: list[str], flag: str, value: str) -> None:
    """Replace one recipe option while rejecting ambiguous command lines."""
    positions = [index for index, item in enumerate(argv) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"canonical recipe must contain exactly one value for {flag}")
    argv[positions[0] + 1] = value


def options(argv: list[str]) -> dict[str, str | bool]:
    """Parse the small canonical command surface needed by the smoke contract."""
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:]
        if key in parsed:
            raise ValueError(f"canonical argv repeats --{key}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            parsed[key] = argv[index + 1]
            index += 2
        else:
            parsed[key] = True
            index += 1
    return parsed


def load_grader(root: Path, source: Path) -> Any:
    """Load the official scorer after prioritizing the pinned trainer source."""
    sys.path.insert(0, str(source / "src"))
    for name in ("wnn.data", "wnn.evaluation", "wnn.gmetric", "wnn.splits"):
        importlib.import_module(name)
    path = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if path.is_symlink() or sha(path) != GRADER_SHA256:
        raise ValueError("official grader differs from its frozen source hash")
    spec = importlib.util.spec_from_file_location("_smoke_official_grader", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned official endpoint grader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite_tree(value: Any, label: str) -> None:
    """Reject nonfinite numeric values in saved training evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite training evidence at {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


def validate_fit(
    work: Path, job: dict[str, Any], subset: dict[str, Any], split_sha: str, legacy: dict
) -> dict[str, Any]:
    """Check one withheld epoch, exact fit membership, gradient counts and FP32 state."""
    import torch

    run_dir = work / "artifacts/runs" / job["id"]
    paths = {
        name: run_dir / name
        for name in (
            "model_ep0.pt",
            "model.pt",
            "training_state.pt",
            "history.json",
            "provenance.json",
        )
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise FileNotFoundError("trainer omitted an expected epoch-zero smoke artifact")
    provenance = read(paths["provenance.json"])
    history = json.loads(paths["history.json"].read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or len(history) != 1
        or history[0].get("epoch") != 0
        or history[0].get("evaluation") != "withheld"
    ):
        raise ValueError("smoke fit must contain one evaluation-withheld epoch zero")
    args = provenance.get("args")
    feature, control, width = ARMS[job["arm"]][1:]
    expected = {
        "name": job["id"],
        "valid_fold": 0,
        "seed": job["seed"],
        "hidden": 16,
        "layers": 1,
        "epochs": 1,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "precision": "fp32",
        "device": "cuda",
        "research_features": feature,
        "research_feature_control": control,
        "batch": 2,
        "chunk": 2000,
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_train_fraction": 1.0,
        "valid_repeat": 1,
        "no_eval_during_fit": True,
        "save_training_state": True,
        "fit_subset_manifest_sha256": subset["sha256"],
    }
    if not isinstance(args, dict) or any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("saved trainer arguments differ from the engineering smoke contract")
    if (
        provenance.get("split_manifest_sha256") != split_sha
        or provenance.get("fit_subset_manifest_sha256") != subset["sha256"]
        or provenance.get("fit_subset_members_sha256") != subset["selected_members_sha256"]
        or provenance.get("selected_epoch") != 0
        or provenance.get("fit_row_groups") != subset["selected_groups"]
    ):
        raise ValueError("fit provenance does not bind the frozen split, subset and endpoint")
    update_diagnostics = history[0].get("update_diagnostics")
    diagnostics = update_diagnostics.get("epoch") if isinstance(update_diagnostics, dict) else None
    if not isinstance(diagnostics, dict):
        raise ValueError("history lacks per-epoch optimizer counters")
    attempts = diagnostics.get("optimizer_attempts")
    completed = diagnostics.get("optimizer_completed_updates")
    skipped = diagnostics.get("amp_skipped_updates")
    grad = update_diagnostics.get("grad_stats")
    if (
        type(attempts) is not int
        or attempts <= 0
        or completed != attempts
        or skipped != 0
        or not isinstance(grad, dict)
        or grad.get("steps") != attempts
        or grad.get("nonfinite_steps") != 0
    ):
        raise ValueError("optimizer or pre-clip gradient counts are incomplete/nonfinite")
    _finite_tree(history, "history")
    _finite_tree(provenance, "provenance")
    identity = legacy["checkpoint_identity"].verify_checkpoint_pair(
        paths["model_ep0.pt"], paths["model.pt"], expected_epoch=0, expected_input_width=width
    )
    checkpoint = torch.load(paths["model_ep0.pt"], map_location="cpu", weights_only=False)
    embedded = checkpoint.get("provenance")
    recorded = provenance.get("checkpoint_sha256")
    if (
        checkpoint.get("args") != args
        or not isinstance(embedded, dict)
        or embedded.get("args") != args
        or embedded.get("split_manifest_sha256") != split_sha
        or embedded.get("fit_subset_manifest_sha256") != subset["sha256"]
        or embedded.get("fit_subset_members_sha256") != subset["selected_members_sha256"]
        or embedded.get("fit_row_groups") != subset["selected_groups"]
        or not isinstance(recorded, dict)
        or recorded.get("model_ep0.pt") != identity.fixed_file_sha256
        or recorded.get("model.pt") != identity.selected_file_sha256
    ):
        raise ValueError("checkpoint, run provenance and file digests disagree")
    shapes = {
        "mean": (width,),
        "std": (width,),
        "inp.weight": (16, width),
        "inp.bias": (16,),
        "gru.weight_ih_l0": (48, 16),
        "gru.weight_hh_l0": (48, 16),
        "gru.bias_ih_l0": (48,),
        "gru.bias_hh_l0": (48,),
        "head.0.weight": (16, 32),
        "head.0.bias": (16,),
        "head.2.weight": (2, 16),
        "head.2.bias": (2,),
    }
    state = checkpoint.get("model")
    if (
        checkpoint.get("n_in") != width
        or checkpoint.get("epoch") != 0
        or not isinstance(state, dict)
        or set(state) != set(shapes)
    ):
        raise ValueError("checkpoint tensor map, input width or selected epoch differs from H16x1")
    for name, shape in shapes.items():
        tensor = state[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"checkpoint tensor {name} has wrong shape, dtype or values")
    if not torch.all(state["std"] > 0).item():
        raise ValueError("checkpoint input normalization has nonpositive scales")
    return {
        "checkpoint_sha256": identity.fixed_file_sha256,
        "selected_checkpoint_sha256": identity.selected_file_sha256,
        "checkpoint_payload_sha256": identity.payload_sha256,
        "provenance_sha256": sha(paths["provenance.json"]),
        "history_sha256": sha(paths["history.json"]),
        "training_state_sha256": sha(paths["training_state.pt"]),
        "optimizer_attempts": attempts,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": skipped,
        "gradient_stats_steps": grad["steps"],
        "gradient_nonfinite_steps": grad["nonfinite_steps"],
        "gradient_stats": grad,
        "input_width": width,
        "hidden": 16,
        "layers": 1,
    }


def grader_call(
    grader: Any, work: Path, run: str, split_path: Path, output: Path
) -> dict[str, Any]:
    """Invoke and validate one official standalone endpoint grade, with no reference."""
    result = grader.grade_endpoint(
        work,
        run,
        0,
        split_path,
        output,
        reference=None,
        n_boot=20,
        scratch_dir=work / "scratch/grade",
    )
    score = result.get("endpoint", {}).get("score")
    if (
        not isinstance(score, dict)
        or score.get("metric_id") != "global_wp_clip2_v1"
        or type(score.get("selected_rows")) is not int
        or score["selected_rows"] <= 0
        or not math.isfinite(score.get("wp", math.nan))
    ):
        raise ValueError("official grader returned no finite scored-row endpoint")
    return {
        "result": result,
        "score": score,
        "signature": str(inspect.signature(grader.grade_endpoint)),
        "arguments": {
            "root": str(work),
            "run": run,
            "epoch": 0,
            "split_path": str(split_path),
            "reference": None,
            "n_boot": 20,
        },
    }
