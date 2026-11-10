from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

PACKET = Path(__file__).resolve().parents[1]
RUNTIME = PurePosixPath("/data/epanov_s/tabdata/runtime/transformer-optimizer-horizon-20261108-r1")
DATA = PurePosixPath("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project")
SPLIT = PurePosixPath(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json"
)
PYTHON = PurePosixPath("/data/epanov_s/tabdata/.venv/bin/python")
SOURCE_COMMIT = "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CELLS = (
    ("C0_muon256_a4_s203101", "C0", "muon", 203101, 256, 4, "0,1", 1),
    ("C1_adamw256_a4_s203101", "C1", "adamw", 203101, 256, 4, "2,3", 1),
    ("C2_muon512_a2_s203101", "C2", "muon", 203101, 512, 2, "4,5", 1),
    ("C0_muon256_a4_s203102", "C0", "muon", 203102, 256, 4, "6,7", 1),
    ("C1_adamw256_a4_s203102", "C1", "adamw", 203102, 256, 4, "2,3", 2),
    ("C2_muon512_a2_s203102", "C2", "muon", 203102, 512, 2, "4,5", 2),
)


def digest(path: Path) -> str:
    """Hash a frozen config without loading large source or data files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(
    name: str, arm: str, optimizer: str, seed: int, chunk: int, accum: int
) -> dict[str, Any]:
    path = PACKET / "configs" / f"{name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected_roles = (
        {"adamw": 1.0}
        if optimizer == "adamw"
        else {
            "muon_qkv": 1.0,
            "muon_hidden": 1.0,
            "adamw_matrix": 1.0,
            "adamw_no_decay": 1.0,
        }
    )
    if (
        config.get("objective_mode") != "standalone_labels_wmse_v2"
        or config.get("optimizer") != optimizer
        or config.get("seed") != seed
        or config.get("epochs") != 4
        or config.get("batch_size") != 128
        or config.get("accumulate_chunks") != accum
        or config.get("chunk_rows") != chunk
        or config.get("gradient_horizon_rows") != chunk
        or config.get("optimizer_update_diagnostics_every") != 50
        or config.get("lr_head_factor") != 1.0
        or config.get("lr_body_factor") != 1.0
        or config.get("lr_role_factors") != expected_roles
        or config.get("required_world_size") != 2
        or config.get("precision_mode") != "bf16_bf16x9"
        or config.get("training_mask") != "all_after_warmup"
        or config.get("normalization_scope") != "all_fit"
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or "blend_objective_alpha" in config
        or "correlation_prediction_std_floor" in config
        or "teacher_cache" in config
        or "reference_cache" in config
    ):
        raise ValueError(f"configuration does not match frozen arm {arm}: {path.name}")
    return {"path": path, "sha256": digest(path), "config": config}


def build_plan() -> dict[str, Any]:
    """Build the exact six-profile plan; profiles are engineering work, never fits."""
    cells: list[dict[str, Any]] = []
    for name, arm, optimizer, seed, chunk, accum, pair, wave in CELLS:
        frozen = _validate_config(name, arm, optimizer, seed, chunk, accum)
        command = [
            str(PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(RUNTIME / "source-r2/scripts/train_sequence.py"),
            "--config",
            str(RUNTIME / "configs" / f"{name}.json"),
            "--data-root",
            str(DATA),
            "--split-manifest",
            str(SPLIT),
            "--output",
            str(RUNTIME / "profiles-r2" / name),
            "--profile-updates",
            "50",
        ]
        if any(
            flag in command
            for flag in ("--reference-cache", "--teacher-cache", "--monitor-reference-root")
        ):
            raise AssertionError("cache-free v2 profile command gained reference ancestry")
        cells.append(
            {
                "cell": name,
                "arm": arm,
                "optimizer": optimizer,
                "seed": seed,
                "pair": pair,
                "wave": wave,
                "config": f"{name}.json",
                "config_sha256": frozen["sha256"],
                "expected_updates": 3280,
                "updates_per_epoch": 820,
                "epochs": 4,
                "profile_updates": 50,
                "maximum_profile_seconds": 900,
                "maximum_attempts": 1,
                "profile_command": command,
                "profile_receipt_path": str(
                    RUNTIME / "profiles-r2" / name / "profile_receipt.json"
                ),
                "approved_profile_receipt_path": str(RUNTIME / "approvals" / f"{name}.json"),
            }
        )
    return {
        "schema": "wnn.transformer-optimizer-horizon-panel.v1",
        "source_commit": SOURCE_COMMIT,
        "source_root": str(RUNTIME / "source-r2"),
        "source_manifest": str(RUNTIME / "source-r2-manifest.json"),
        "source_manifest_sha256": "a1e16d24ab53032643807edb5471b8bc65b06bdd96da516944bcf36736339937",
        "data_root": str(DATA),
        "split_manifest": str(SPLIT),
        "split_sha256": SPLIT_SHA256,
        "new_scientific_fit_cap": 6,
        "profile_cap": 6,
        "first_profile_wave_seconds": 28800,
        "fit_seconds": 25200,
        "collection_reserve_seconds": 3600,
        "profile_wave_order": [
            [
                "C0_muon256_a4_s203101",
                "C1_adamw256_a4_s203101",
                "C2_muon512_a2_s203101",
                "C0_muon256_a4_s203102",
            ],
            ["C1_adamw256_a4_s203102", "C2_muon512_a2_s203102"],
        ],
        "environment": {
            "PYTHONPATH": str(RUNTIME / "source-r2/src"),
            "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "TORCHINDUCTOR_COMPILE_THREADS": "4",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WNN_MONITOR_HEARTBEAT_SECONDS": "5",
            "TMPDIR": "/data/epanov_s/tdh8",
            "TORCHINDUCTOR_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/inductor",
            "TRITON_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/triton",
        },
        "cells": cells,
    }


def main() -> None:
    """Write a deterministic frozen profile plan or reject conflicting plan bytes."""
    path = PACKET / "profile-plan.json"
    raw = (json.dumps(build_plan(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("preserve changed profile plan; do not overwrite admitted work")
    path.write_bytes(raw)


if __name__ == "__main__":
    main()
