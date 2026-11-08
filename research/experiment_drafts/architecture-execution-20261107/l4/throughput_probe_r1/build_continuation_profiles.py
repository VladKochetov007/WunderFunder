from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
PYTHON = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"


def write_new(path: Path, value: object) -> str:
    """Write one immutable preparation artifact without replacing earlier evidence."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def build(artifacts: Path, output: Path, source: PurePosixPath) -> dict[str, object]:
    """Prepare two actual-recipe profiles, changing only checkpoint retention."""
    entries = []
    for gpu, seed in enumerate((101001, 101002)):
        metadata_path = artifacts / f"live-dashboard/runs/l4-gru_s{seed}-fit/run_metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        original = metadata["config"]
        if (
            original["model"]["architecture"] != "gru"
            or original["seed"] != seed
            or original["epochs"] != 12
            or original["batch_size"] != 16
            or original["checkpoint_retention"] != "all_epochs"
            or "early_stopping" in original["monitoring"]
        ):
            raise ValueError("actual GRU recipe is outside this bounded continuation")
        config = original | {"checkpoint_retention": "final_only"}
        # Run metadata includes this derived value; the raw parser regenerates it.
        config = config | {
            "model": {k: v for k, v in config["model"].items() if k != "precision_mode"}
        }
        name = f"gru_s{seed}.json"
        sha = write_new(output / "configs" / name, config)
        remote_config = str(RUNTIME / "configs/gru-monitor-continuation-r3" / name)
        remote_output = str(RUNTIME / "runs/gru-monitor-continuation-profiles-r3" / f"gru_s{seed}")
        entries.append(
            {
                "seed": seed,
                "gpu_index": gpu,
                "config_file": name,
                "config_sha256": sha,
                "normalized_config_sha256": hashlib.sha256(
                    json.dumps(
                        original | {"checkpoint_retention": "final_only"},
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
                "old_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "source_root": str(source),
                "output": remote_output,
                "timeout_seconds": 200,
                "argv": [
                    PYTHON,
                    str(source / "scripts/train_sequence.py"),
                    "--config",
                    remote_config,
                    "--data-root",
                    str(RUNTIME / "project"),
                    "--split-manifest",
                    str(RUNTIME / "project/chronological_primary.json"),
                    "--output",
                    remote_output,
                    "--profile-updates",
                    "20",
                ],
                "env_set": {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": str(source / "src"),
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8",
                    "OPENBLAS_NUM_THREADS": "8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "NVIDIA_TF32_OVERRIDE": "0",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                },
                "env_unset": [
                    "CUBLAS_WORKSPACE_CONFIG",
                    "CUBLAS_EMULATE_SINGLE_PRECISION",
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                ],
            }
        )
    plan = {
        "schema": "wnn.gru-monitor-continuation-profiles.v1",
        "scientific_fits": 0,
        "profile_updates_per_cell": 20,
        "profile_cap": 2,
        "config_diff": {"checkpoint_retention": ["all_epochs", "final_only"]},
        "requires_separate_ROOT_approval": True,
        "cells": entries,
    }
    write_new(output / "CONTINUATION_PROFILE_PLAN_R3.json", plan)
    return plan


def main() -> None:
    """Build a portable POSIX command packet locally, without starting profiles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=PurePosixPath, default=RUNTIME / "source-gru-monitor-archive-r1"
    )
    args = parser.parse_args()
    result = build(args.artifacts, args.output, args.source)
    print(json.dumps({"prepared": len(result["cells"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
