from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data/epanov_s/tabdata/runtime/objective-fix-20261108-r1")
PREVIOUS = PurePosixPath("/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1")
SPLIT = PurePosixPath(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/data/chronological_primary.json"
)


def main() -> None:
    """Freeze exactly four fresh4E recipes and ROOT's bounded direct-canonical profile commands."""
    packet = Path(__file__).resolve().parent
    old = (
        packet.parent
        / "tl3-transformer-complement-execution-20261107/optimized_receiving/prepared-b128/configs"
    )
    output = packet / "configs"
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for seed in (101101, 101102):
        for arm in ("labels", "mixture_corr"):
            original = old / (
                "labels_s101001.json" if arm == "labels" else "complement_s101001.json"
            )
            config = json.loads(original.read_bytes())
            config.update(
                epochs=4,
                seed=seed,
                objective_mode="labels_wmse_v1" if arm == "labels" else "mixture_wmse_pearson_v1",
            )
            if arm == "mixture_corr":
                config["correlation_prediction_std_floor"] = 0.01
            name = f"{arm}_s{seed}"
            path = output / f"{name}.json"
            raw = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
            if path.exists() and path.read_bytes() != raw:
                raise FileExistsError("preserve different existing frozen recipe")
            path.write_bytes(raw)
            pair = len(cells) * 2
            command = [
                "/data/epanov_s/tabdata/.venv/bin/python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                str(RUNTIME / "source-r1/scripts/train_sequence.py"),
                "--config",
                str(RUNTIME / "configs" / path.name),
                "--data-root",
                str(PREVIOUS / "project"),
                "--split-manifest",
                str(SPLIT),
                "--monitor-reference-root",
                str(PREVIOUS / "reference"),
                "--output",
                str(RUNTIME / "profiles" / name),
                "--profile-updates",
                "50",
            ]
            if arm == "mixture_corr":
                command += ["--reference-cache", str(PREVIOUS / "reference/fit")]
            cells.append(
                {
                    "cell": name,
                    "arm": arm,
                    "seed": seed,
                    "pair": f"{pair},{pair + 1}",
                    "config": path.name,
                    "config_sha256": hashlib.sha256(raw).hexdigest(),
                    "epochs": 4,
                    "updates_per_epoch": 820,
                    "expected_updates": 3280,
                    "profile_updates": 50,
                    "maximum_profile_seconds": 180,
                    "maximum_attempts": 2,
                    "profile_command": command,
                }
            )
    manifest = {
        "schema": "wnn.ensemble-objective-panel.v1",
        "new_scientific_fit_cap": 4,
        "first_profile_wave_seconds": 14400,
        "fit_seconds": 10800,
        "collection_reserve_seconds": 3600,
        "data_root": str(PREVIOUS / "project"),
        "reference_root": str(PREVIOUS / "reference"),
        "split_manifest": str(SPLIT),
        "source_root": str(RUNTIME / "source-r1"),
        "source_status": "awaiting final source freeze and native qualification",
        "cells": cells,
        "environment": {
            "PYTHONPATH": str(RUNTIME / "source-r1/src"),
            "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "TORCHINDUCTOR_COMPILE_THREADS": "4",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WNN_MONITOR_HEARTBEAT_SECONDS": "5",
            "TMPDIR": "/data/epanov_s/tdf8/scratch",
            "TORCHINDUCTOR_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/inductor",
            "TRITON_CACHE_DIR": "/data/epanov_s/tdn7run/compiler/triton",
        },
    }
    for cell in cells:
        assert all("\\" not in value for value in cell["profile_command"])
    assert all("\\" not in value for value in manifest["environment"].values())
    path = packet / "profile-plan.json"
    raw = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError("preserve existing different profile allocation")
    path.write_bytes(raw)


if __name__ == "__main__":
    main()
