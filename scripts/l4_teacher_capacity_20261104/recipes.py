from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

RUNTIME = PurePosixPath("/data2/lc-epanov_s/tabdata/runtime/l4-teacher-capacity-20261104-r1")
SOURCE = RUNTIME / "source-r1"
PROJECT = RUNTIME / "project"
PYTHON = PurePosixPath("/data2/lc-epanov_s/tabdata/.venv/bin/python")
SPLIT = PurePosixPath(
    "/data2/lc-epanov_s/tabdata/runtime/launch-20261028/"
    "data-preparation-r1/chronological_primary.json"
)
GPU_UUIDS = (
    "GPU-222adc44-ecd3-a645-cf7e-ddda9d495f7a",
    "GPU-bce838c4-097e-0623-7b67-6a97f3778510",
    "GPU-9fd565e2-ebb2-35be-c628-d971a575aa7e",
    "GPU-02d8270f-e4ee-1363-8232-15c0726fdab1",
    "GPU-9943f86e-f085-2d36-2c12-8d2a22111ef0",
    "GPU-48b463a4-5b0a-e277-9ef1-ee37be8015b1",
)
CONFIGURATIONS = (
    ("gru256x2_control", 256, 2, "control"),
    ("gru384x2", 384, 2, "capacity"),
    ("gru512x1", 512, 1, "capacity"),
)


def train_args(name: str, seed: int, hidden: int, layers: int) -> list[str]:
    """Build the frozen train-only GRU recipe using the legacy canonical CLI."""
    return [
        "--name", name,
        "--device", "cuda",
        "--seed", str(seed),
        "--hidden", str(hidden),
        "--layers", str(layers),
        "--dropout", "0",
        "--cell", "gru",
        "--train-source", "train",
        "--train-seqs", "10325",
        "--eval-source", "dev",
        "--full-valid",
        "--split-manifest", str(SPLIT),
        "--precision", "fp32",
        "--checkpoint-policy", "fixed",
        "--checkpoint-epoch", "3",
        "--loss", "wmse",
        "--loss-norm", "chunk",
        "--aux-next", "0.3",
        "--aux-struct", "0.3",
        "--aux-weight", "0.5",
        "--batch", "32",
        "--chunk", "1000",
        "--epochs", "4",
        "--lr", "0.001",
        "--wd", "0.01",
        "--beta2", "0.999",
        "--clip", "1",
        "--optim", "muon",
        "--muon-scope", "rnn",
        "--sched", "onecycle",
        "--warmup-frac", "0.05",
        "--batch-order", "seeded",
        "--train-eval-seqs", "32",
        "--sync-free",
        "--pin-memory",
        "--grad-stats",
    ]


def build_panel() -> dict[str, object]:
    """Return the six ready GRU cells, each assigned its historical physical UUID."""
    cells: list[dict[str, object]] = []
    slot = 0
    for configuration, hidden, layers, role in CONFIGURATIONS:
        for seed in (1001, 1002):
            name = f"l4tc_{configuration}_s{seed}_20261104"
            cell_id = f"gru{slot}_{configuration}_s{seed}"
            fit = train_args(name, seed, hidden, layers)
            profile = fit.copy()
            profile[profile.index("--name") + 1] += "_profile_r1"
            profile.extend(("--profile-warmup", "65", "--profile-steps", "3"))
            cells.append(
                {
                    "cell_id": cell_id,
                    "configuration_id": configuration,
                    "role": role,
                    "seed": seed,
                    "gpu_slot": slot,
                    "gpu_uuid": GPU_UUIDS[slot],
                    "cpu_threads": 8,
                    "ram_bytes": 48 * 1024**3,
                    "run_name": name,
                    "profile_argv": [str(PYTHON), str(SOURCE / "scripts/train_rnn.py"), *profile],
                    "fit_argv": [str(PYTHON), str(SOURCE / "scripts/train_rnn.py"), *fit],
                    "artifacts": [
                        f"artifacts/runs/{name}/model_ep{epoch}.pt" for epoch in range(4)
                    ]
                    + [f"artifacts/runs/{name}/model.pt", f"artifacts/runs/{name}/history.json", f"artifacts/runs/{name}/provenance.json"],
                }
            )
            slot += 1
    return {
        "schema": "wnn.l4-teacher-capacity-gru.v1",
        "status": "gru_recipes_frozen_waiting_for_source_admission",
        "contract": "research/experiment_drafts/l4-teacher-capacity-20261104/CONTRACT_R1.json",
        "paths": {
            "source_root": str(SOURCE),
            "project_root": str(PROJECT),
            "python": str(PYTHON),
            "split_manifest": str(SPLIT),
            "runtime_root": str(RUNTIME),
            "control_root": str(RUNTIME / "control"),
            "disk_mount": "/data2",
        },
        "identity": {
            "split_sha256": "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c",
            "fit_train_sequences": 10325,
            "fit_valid_sequences": 0,
            "primary_valid_sequences": 1873,
            "primary_scored_rows": 3528273,
            "source_binding_sha256": None,
            "data_binding_sha256": None,
            "fit_members_sha256": None,
        },
        "budget": {
            "max_scientific_fits": 6,
            "max_profiles": 6,
            "max_wave_seconds": 43200,
            "collection_reserve_seconds": 2700,
            "max_output_bytes": 6500000000,
            "minimum_free_bytes": 3000000000,
            "aggregate_cpu_threads": 48,
            "aggregate_memory_bytes": 288 * 1024**3,
        },
        "profile_policy": {
            "warmup_batches": 65,
            "timed_batches": 3,
            "timeout_seconds": 1800,
            "no_full_valid_evaluation": True,
        },
        "cells": cells,
        "execution": {
            "fit_timeout_seconds": 40500,
            "output_policy": "model_ep0..model_ep3, history, provenance; no optimizer-state checkpoint",
            "profile_gate": "all six zero-exit receipts plus independent source/data/profile admission",
            "fixed_endpoint": "model_ep3.pt",
        },
    }


def main() -> None:
    """Emit an exact JSON handoff for the six independent GRU cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(build_panel(), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
