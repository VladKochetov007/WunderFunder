"""Frozen rank-1 profile pair and runtime evidence helpers."""

from __future__ import annotations

import json
import platform
import resource
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "a266b036b02e8e012c6eb31072291dad618a169d"
EXPECTED_RECIPE = {
    "profile_steps": 3,
    "profile_warmup": 0,
    "epochs": 6,
    "fold": 0,
    "seed": 1001,
    "train_source": "train",
    "loss": "wmse",
    "loss_norm": "chunk",
    "precision": "fp32",
    "batch": 8,
    "chunk": 2000,
    "learning_rate": 0.0001,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "warmup_fraction": 0.05,
}
EXPECTED_ARMS = [
    {"name": "computer-vision-homework-rank01-profile-h512l1", "hidden": 512, "layers": 1},
    {"name": "computer-vision-homework-rank01-profile-h384l2", "hidden": 384, "layers": 2},
]
EXPECTED_SUPPLEMENTAL_MEMBERS = {
    "cache/chains.npz",
    "derivedcache/chain_folds.npz",
    "portable-sources.json",
    "splits/fold0.json",
    "splits/fold1.json",
    "splits/fold2.json",
    "splits/fold3.json",
}


def check_runtime(expected: dict[str, Any]) -> dict[str, Any]:
    import attrs
    import numpy
    import pyarrow
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    observed = {
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": numpy.__version__,
        "pyarrow_version": pyarrow.__version__,
        "attrs_version": attrs.__version__,
        "visible_gpu_count": gpu_count,
        "gpu_names": gpu_names,
    }
    for key, got in observed.items():
        if got != expected[key]:
            raise RuntimeError(f"runtime mismatch for {key}: observed {got!r}")
    if gpu_count != 1 or expected["gpu_name_substring"] not in gpu_names[0]:
        raise RuntimeError(f"expected one A100; observed GPU names {gpu_names}")
    return observed


def training_args(arm: dict[str, Any], split_path: Path) -> list[str]:
    recipe = EXPECTED_RECIPE
    return [
        "--name",
        arm["name"],
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        str(arm["hidden"]),
        "--layers",
        str(arm["layers"]),
        "--dropout",
        "0",
        "--batch",
        str(recipe["batch"]),
        "--epochs",
        str(recipe["epochs"]),
        "--train-source",
        "train",
        "--split-manifest",
        str(split_path),
        "--valid-fold",
        "0",
        "--seed",
        "1001",
        "--batch-order",
        "seeded",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--precision",
        "fp32",
        "--lr",
        "0.0001",
        "--wd",
        "0.01",
        "--beta2",
        "0.999",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--chunk",
        "2000",
        "--grad-stats",
        "--fused-adam",
        "--sync-free",
        "--deterministic",
        "--no-eval-during-fit",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "5",
        "--profile-steps",
        "3",
        "--profile-warmup",
        "0",
    ]


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def run_arm(
    arm: dict[str, Any], split_path: Path, source_root: Path, data_root: Path, output: Path
) -> dict[str, Any]:
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.runner import run_training

    log_path = output / f"{arm['name']}.log"
    with log_path.open("w", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            run_training(
                training_args(arm, split_path),
                environment=TrainingEnvironment(source_root=source_root, root=data_root, argv=[]),
            )
    lines = log_path.read_text(encoding="utf-8").splitlines()
    profiles = [
        json.loads(line[len("PROFILE ") :]) for line in lines if line.startswith("PROFILE ")
    ]
    evidence = [
        json.loads(line[len("PROFILE_UPDATE_EVIDENCE ") :])
        for line in lines
        if line.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    if len(profiles) != 1 or len(evidence) != 1 or not evidence[0].get("valid_for_throughput"):
        raise RuntimeError(f"profile evidence missing, duplicated, or invalid for {arm['name']}")
    import torch

    return {
        "name": arm["name"],
        "hidden": arm["hidden"],
        "layers": arm["layers"],
        "profile": profiles[0],
        "update_evidence": evidence[0],
        "log": log_path.name,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
