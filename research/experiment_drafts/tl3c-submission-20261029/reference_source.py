"""Bind a receiver's canonical Torch implementation to the admitted PR49 sources."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from candidate_contract import PR49_COMMIT, verify_metadata
from checkpoint_export import TIED_FEATURE_SOURCES, export_member
from validation_io import sha256


def git_output(root: Path, *args: str) -> str:
    """Read Git identity without checking out, fetching or modifying anything."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def verify_training_root(root: Path, records: list[dict], split: dict) -> dict:
    """Check exact commit, clean code and only source files recorded by candidate fits."""
    if git_output(root, "rev-parse", "HEAD") != PR49_COMMIT:
        raise ValueError(
            "--training-root must be a separate checkout at the exact admitted PR49 commit"
        )
    tracked = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "utils.py",
        "pyproject.toml",
        "uv.lock",
    )
    if tracked:
        raise ValueError(
            "canonical training checkout contains changed or untracked relevant source"
        )
    required = dict(TIED_FEATURE_SOURCES)
    for record in records:
        provenance = verify_metadata(record, split)
        for name, digest in provenance["trainer_implementation"]["files"].items():
            if name in required and required[name] != digest:
                raise ValueError(f"candidate source records disagree for {name}")
            required[name] = digest
    for name, expected in required.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe admitted source path")
        source = root / name
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"receiver source does not match the fitted implementation: {name}")
    return {
        "commit": PR49_COMMIT,
        "root": str(root),
        "verified_files": required,
        "scorer_sha256": sha256(root / "utils.py"),
    }


def load_reference_models(
    root: Path,
    records: list[dict],
    split: dict,
    receipt_members: list[dict],
    temporary: Path,
    device_name: str,
) -> tuple[list[tuple[Any, Any, Any]], Any, dict[str, Any]]:
    """Reuse canonical construction and already-scored EMA weights, with no fitting."""
    # Canonical Torch/environment types exist only in the pinned receiver tree.
    precision_environment = {
        key: os.environ.get(key)
        for key in (
            "LD_PRELOAD",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "NVIDIA_TF32_OVERRIDE",
            "CUBLAS_EMULATE_SINGLE_PRECISION",
            "CUBLAS_EMULATION_STRATEGY",
            "CUBLAS_WORKSPACE_CONFIG",
        )
    }
    if precision_environment["LD_PRELOAD"]:
        raise RuntimeError("unset LD_PRELOAD; injected B300 libraries can change FP32 semantics")
    for key in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_EMULATE_SINGLE_PRECISION"):
        if precision_environment[key] not in (None, "", "0"):
            raise RuntimeError(f"unset {key} (or set 0) for the strict FP32 reference")
    if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
        raise RuntimeError("reference export requires a fresh Python process")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import torch

    from wnn.resources import configure_strict_fp32
    from wnn.training.environment import InputConfiguration, TrainingEnvironment
    from wnn.training.models import make_model
    from wnn.training.prediction_inputs import configure_prediction_inputs

    if device_name == "cuda:0":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible or torch.cuda.device_count() != 1:
            raise ValueError(
                "CUDA reference export requires one explicitly selected CUDA_VISIBLE_DEVICES"
            )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_strict_fp32()
    device = torch.device(device_name)
    models = []
    for record, packaged in zip(records, receipt_members, strict=True):
        exported = export_member(record, split, temporary, packaged["coef"])
        for key in ("run", "coef", "family", "tensors"):
            if exported[key] != packaged[key]:
                raise ValueError(f"reference export differs from built candidate: {key}")
        if exported["checkpoint"]["sha256"] != packaged["checkpoint"]["sha256"]:
            raise ValueError("reference checkpoint differs from built candidate")
        provenance = verify_metadata(record, split)
        args = provenance["args"]
        width = 263 if args["enc"] == "tiedlead" else 211
        model = make_model(args, width, 2)
        with np.load(temporary / exported["weight_file"], allow_pickle=False) as arrays:
            model.load_state_dict(
                {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}, strict=True
            )
        model = model.to(device).eval()
        model._inference_fp32 = True
        environment = TrainingEnvironment(source_root=root, root=root, inputs=InputConfiguration())
        configure_prediction_inputs(environment, args)
        environment.inputs.raw_only = bool(args.get("raw_only"))
        models.append((model, environment, np.float32(packaged["coef"])))
    runtime = {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "precision_environment": precision_environment,
    }
    return models, device, runtime
