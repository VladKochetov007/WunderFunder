"""Callable actual-shape CUDA probe for the separately admitted E488 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fit_duration as actor
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("E488 CUDA probe needs an admitted CUDA device")
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    actor.check_sources()
    actor.configure_torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E488 probe requires exactly one visible CUDA device")
    torch.set_num_threads(1)
    states = actor.make_initial_states()
    manifest, x, y, need, nodes = actor.load_fit_data()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    parity = actor.e484.check_reference_parity(device, states[actor.SEEDS[0]])
    torch.cuda.synchronize()
    baseline_memory = memory_snapshot()
    cell_records = []
    for seed in actor.SEEDS:
        ids = actor.epoch_order(seed, 0)[:4].tolist()
        for arm in actor.ARMS:
            torch.manual_seed(seed)
            model = actor.make_arm(arm, states[seed]).to(device).float().train()
            optimizer = actor.e484.make_optimizer(model)
            families = actor.e484.parameter_families(model)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            model._e488_state = None
            model._e488_step = 0
            torch.cuda.reset_peak_memory_stats(device)
            loss, state, telemetry = actor.run_chunk_update(
                model, optimizer, families, x, y, need, ids, 0, device, f"{seed}/{arm}/probe"
            )
            torch.cuda.synchronize()
            changed = sum(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if changed == 0 or not torch.isfinite(state).all() or not np.isfinite(loss):
                raise ValueError(f"E488 CUDA actual-shape update failed: {seed}/{arm}")
            if list(state.shape) != [2, 4, 64]:
                raise ValueError(f"E488 unexpected GRU state shape: {tuple(state.shape)}")
            cell_records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "loss": loss,
                    "changed_parameter_tensors": changed,
                    "state_shape": list(state.shape),
                    "telemetry": telemetry,
                    "memory": memory_snapshot(),
                }
            )
            del model, optimizer, families, before, state
            torch.cuda.empty_cache()
    output.mkdir(parents=True)
    report = {
        "kind": "E488 actual-shape CUDA probe",
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "device_count_visible": torch.cuda.device_count(),
        "reference_parity": parity,
        "parity_memory": baseline_memory,
        "cells": cell_records,
        "cells_complete": len(cell_records),
        "fit_components": [node["row_group"] for node in nodes],
        "fit_only_normalization": manifest["normalization"],
        "sources": {
            name: source_sha256(HERE / name)
            for name in ("fit_duration.py", "tests_duration.py", "gpu_probe_duration.py")
        },
        "test_rows_loaded": False,
    }
    (output / "probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
