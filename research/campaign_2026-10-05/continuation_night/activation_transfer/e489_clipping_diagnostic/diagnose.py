"""Non-scoring E489 clipping-coverage diagnostic on the frozen fit-only probe batch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from diagnostic_math import sha256_array, sha256_file, sha256_state
from frozen_runtime import E489, FROZEN, actor, configure_torch, e470, e474
from seed_probe import diagnose_seed

ALLOCATOR_CAP_BYTES = 2 * 1024**3
DEVICE_STOP_BYTES = 3_000_000_000


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically write status so completed seed records survive later failures."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_diagnostic(output: Path, proof_path: Path) -> dict[str, object]:
    """Run the non-scoring diagnostic, never the E489 full fit."""
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    status: dict[str, object] = {
        "study": "E489-clipping-diagnostic-v1",
        "status": "started",
        "fit_only": True,
        "scoring_performed": False,
        "full_fits_admitted": False,
        "seed_records": [],
    }
    _write_json(output / "status.json", status)
    try:
        actor.validate_e487_saved_delta_gate(proof_path)
        parent_sources = actor.parent_source_hashes()
        os.environ["WNN_GPU_MEMORY_BYTES"] = str(ALLOCATOR_CAP_BYTES)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        configure_torch()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("E489 clipping diagnostic requires exactly one visible CUDA device")
        torch.set_num_threads(1)
        device = torch.device("cuda")
        free, total = torch.cuda.mem_get_info()
        if total - free >= DEVICE_STOP_BYTES or torch.cuda.memory_allocated() > ALLOCATOR_CAP_BYTES:
            raise RuntimeError("E489 diagnostic device is already at or above its memory stop")
        states = actor.make_initial_states()
        cohort, x, y, need, _, nodes = e470.load_cohort(include_test=False)
        if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
            raise ValueError("E489 diagnostic may access only the 24 fit-role groups")
        if x.shape != (24, actor.ROWS_PER_GROUP, actor.FEATURES):
            raise ValueError(f"unexpected fit-only feature shape: {x.shape}")
        if y.shape != (24, actor.ROWS_PER_GROUP, 2) or need.shape != (24, actor.ROWS_PER_GROUP):
            raise ValueError("unexpected fit-only label or mask shape")

        frozen_names = ("fit_fp32_duration.py", "gpu_probe_fp32_duration_v2.py")
        frozen_hashes = {name: sha256_file(E489 / name) for name in frozen_names}
        torch.manual_seed(actor.SEEDS[0])
        torch.cuda.reset_peak_memory_stats(device)
        seed_reports: list[dict[str, object]] = []
        with e474.full_ieee_float32():
            for seed in actor.SEEDS:
                ids = actor.canonical_epoch_order(seed, 0)[: actor.BATCH_SIZE].tolist()
                x_batch = np.ascontiguousarray(x[ids, : actor.CHUNK_ROWS])
                y_batch = np.ascontiguousarray(y[ids, : actor.CHUNK_ROWS])
                need_batch = np.ascontiguousarray(need[ids, : actor.CHUNK_ROWS])
                xx = torch.from_numpy(x_batch).to(device=device, dtype=torch.float32)
                yy = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)
                mask = torch.from_numpy(need_batch).to(device=device)
                record, arrays = diagnose_seed(seed, states[seed], xx, yy, mask, device)
                record["canonical_fit_group_indices"] = ids
                record["batch_sha256"] = {
                    "features": sha256_array(x_batch),
                    "targets": sha256_array(y_batch),
                    "need_mask": sha256_array(need_batch),
                }
                array_path = output / f"seed_{seed}_raw_gradients.npz"
                if array_path.exists():
                    raise FileExistsError(array_path)
                np.savez_compressed(array_path, **arrays)
                record["raw_gradient_archive"] = array_path.name
                record["raw_gradient_archive_sha256"] = sha256_file(array_path)
                _write_json(output / f"seed_{seed}.json", record)
                seed_reports.append(record)
                status["seed_records"] = [item["seed"] for item in seed_reports]
                status["last_completed_seed"] = seed
                status["status"] = "running"
                _write_json(output / "status.json", status)
                del xx, yy, mask, arrays

        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        observed_device_bytes = total - free
        peak_allocator_bytes = int(torch.cuda.max_memory_allocated())
        if peak_allocator_bytes > ALLOCATOR_CAP_BYTES or observed_device_bytes > DEVICE_STOP_BYTES:
            raise RuntimeError("E489 diagnostic memory stop exceeded")
        report = {
            **status,
            "status": "completed_diagnostic_only",
            "device_name": torch.cuda.get_device_name(device),
            "device_count_visible": torch.cuda.device_count(),
            "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
            "observed_device_stop_bytes": DEVICE_STOP_BYTES,
            "peak_allocator_bytes": peak_allocator_bytes,
            "observed_device_bytes_after_collection": int(observed_device_bytes),
            "e487_saved_delta_gate_sha256": sha256_file(proof_path),
            "parent_source_sha256": parent_sources,
            "frozen_probe_source_sha256": frozen_hashes,
            "initial_state_sha256": {
                str(seed): sha256_state(state) for seed, state in states.items()
            },
            "fit_row_groups": [node["row_group"] for node in nodes],
            "fit_cohort_sha256": sha256_file(e470.COHORT_PATH),
            "fit_data_sha256": sha256_file(e470.DATA_PATH),
            "fit_normalization": cohort["normalization"],
            "seeds": seed_reports,
            "body_seconds": time.perf_counter() - started,
            "claim_boundary": "does not pass original E489 gate and does not authorize full fits",
        }
        _write_json(output / "report.json", report)
        status.update({"status": "completed_diagnostic_only", "finished_report": "report.json"})
        _write_json(output / "status.json", status)
        return report
    except Exception as error:
        status.update(
            {
                "status": "diagnostic_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "body_seconds": time.perf_counter() - started,
            }
        )
        _write_json(output / "status.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--e487-root-proof",
        type=Path,
        default=FROZEN / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json",
    )
    arguments = parser.parse_args()
    report = run_diagnostic(arguments.output, arguments.e487_root_proof)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
