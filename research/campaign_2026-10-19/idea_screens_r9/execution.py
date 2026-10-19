"""Create-only receipts and the fixed checkpoint/sample execution sequence."""

from __future__ import annotations

import datetime as dt
import importlib
import os
import time
from pathlib import Path


def _architecture_summary(args: dict[str, object]) -> dict[str, object]:
    """Keep checkpoint architecture provenance to fixed scalar arguments."""
    keys = ("arch", "cell", "hidden", "layers", "enc", "dropout", "precision", "loss")
    return {key: args.get(key) for key in keys}


def execute(
    sample_path: Path,
    checkpoint_path: Path,
    output: Path,
    base: Path,
    scratch: Path,
    source_root: Path,
    pins: dict[str, str],
    preparation_sha256: str,
    identities: dict[str, str],
    runtime: dict[str, object],
) -> None:
    """Load the checkpoint before fit arrays, then run the fixed no-update diagnostic."""
    execution_host = importlib.import_module("idea_screens_r6.execution_host")
    execution_host._check_physical_output(output, scratch)
    output.mkdir()
    started = time.monotonic()
    execution_host._write_json(
        output / "START.json",
        {
            "schema": "nf19.r9_history_probe.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "design_sha256": identities["design_sha256"],
            "design_review_sha256": identities["design_review_sha256"],
            "preparation_manifest_sha256": preparation_sha256,
            "sample_sha256": identities["sample_sha256"],
            "membership_sha256": identities["membership_sha256"],
            "checkpoint_sha256": identities["checkpoint_sha256"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "resource_cap": {
                "cpu_cores": 0.5,
                "memory_high_bytes": 1_400_000_000,
                "timeout_s": 600,
            },
        },
    )
    execution_host._append_json_line(
        output / "PROGRESS.jsonl",
        {"event": "initialized", "elapsed_s": time.monotonic() - started},
    )
    try:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        model_contract = importlib.import_module("idea_screens_r6.model_contract")
        sample_panel = importlib.import_module("idea_screens_r6.sample_panel")
        history_probe = importlib.import_module("idea_screens_r9.history_probe")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
            raise RuntimeError("CUDA must be hidden for the R9 CPU-only diagnostic")
        if numpy.__version__ != "2.2.6" or str(torch.__version__) != "2.14.0+cu130":
            raise RuntimeError(
                "R9 requires the reviewed NumPy 2.2.6 and Torch 2.14.0+cu130 runtime"
            )

        checkpoint_started = time.monotonic()
        loaded = model_contract.load_frozen_model(checkpoint_path, source_root, base)
        checkpoint_seconds = time.monotonic() - checkpoint_started
        memory_before_sample = execution_host._guard_memory("after_checkpoint_before_sample_arrays")

        sample_started = time.monotonic()
        sequences, sample_summary = sample_panel.load_first_fit_eight(sample_path)
        canonical_inventory_after_sample = execution_host._verify_runtime_inventory(
            source_root, pins
        )
        sample_seconds = time.monotonic() - sample_started
        memory_before_inputs = execution_host._guard_memory("before_input_preparation_and_forward")

        runtime.update(
            {
                "numpy_version": numpy.__version__,
                "torch_version": str(torch.__version__),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "device": "cpu",
                "model_dtype": "float32",
                "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "canonical_inventory_after_sample_helper": canonical_inventory_after_sample,
                "memory_guard_after_checkpoint": memory_before_sample,
                "memory_guard_before_input_and_forward": memory_before_inputs,
            }
        )
        source = {
            "schema": "nf19.r9_history_probe.source.v1",
            "source_snapshot": identities["source_snapshot"],
            "design_sha256": identities["design_sha256"],
            "design_review_sha256": identities["design_review_sha256"],
            "preparation_manifest_sha256": preparation_sha256,
            "source_file_pins": pins,
            "sample": sample_summary,
            "checkpoint_sha256": identities["checkpoint_sha256"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_model": {
                "n_in": loaded.n_in,
                "state_keys": loaded.state_keys,
                "architecture_args": _architecture_summary(loaded.args),
                "input_width": 211,
                "raw_input_width": 112,
                "causal_event_width": 99,
                "normalization": "unchanged saved checkpoint mean/std buffers",
            },
            "runtime": runtime,
            "phase_seconds": {
                "checkpoint_load_and_reconstruction": checkpoint_seconds,
                "sample_verification_and_mapping": sample_seconds,
            },
            "target_values_accessed": False,
            "target_files_hashed_and_mapped_for_integrity": True,
            "target_file_count_hashed_and_mapped": len(sequences),
            "evaluation_arrays_read": 0,
            "model_archive_written": False,
        }
        execution_host._write_json(output / "SOURCE.json", source)

        def progress(record: dict[str, object]) -> None:
            """Append one compact window completion event."""
            execution_host._append_json_line(output / "PROGRESS.jsonl", record)

        result = history_probe.run_history_probe(
            loaded.model, loaded.environment, sequences, execution_host._guard_memory, progress
        )
        result.update(
            {
                "schema": "nf19.r9_history_probe.result.v1",
                "design_sha256": identities["design_sha256"],
                "sample_sha256": identities["sample_sha256"],
                "membership_sha256": identities["membership_sha256"],
                "checkpoint_sha256": identities["checkpoint_sha256"],
                "selected_components": sample_summary["selected_components"],
                "selected_sequences": sample_summary["selected_sequences"],
                "checkpoint_args": _architecture_summary(loaded.args),
                "input_width": 211,
                "raw_input_width": 112,
                "causal_event_width": 99,
                "normalization": "saved model mean/std only; unchanged from checkpoint",
                "branch_interpretation": "canonical causal input history remains shared and intact",
                "descriptive_reference_scales": {
                    "relative_rms_difference": 0.01,
                    "absolute_rms_difference": 0.001,
                    "purpose": "predeclared descriptive references only; not equivalence or acceptance thresholds",
                },
                "phase_seconds": {
                    **source["phase_seconds"],
                    **result["phase_seconds"],
                    "total": time.monotonic() - started,
                },
                "runtime": runtime,
            }
        )
        result_sha = execution_host._write_json(output / "RESULT.json", result)
        execution_host._write_json(
            output / "DONE.json",
            {
                "schema": "nf19.r9_history_probe.done.v1",
                "status": result["status"],
                "result_sha256": result_sha,
                "result_bytes": (output / "RESULT.json").stat().st_size,
                "elapsed_s": time.monotonic() - started,
                "traversal_counts": result["traversal_counts"],
                "model_state_unchanged": result["model_state_unchanged"],
            },
        )
    except Exception as exc:
        failure = {
            "schema": "nf19.r9_history_probe.failure.v1",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "elapsed_s": time.monotonic() - started,
            "checkpoint_sha256": identities["checkpoint_sha256"],
            "sample_sha256": identities["sample_sha256"],
            "preparation_manifest_sha256": preparation_sha256,
        }
        execution_host._write_json(output / "FAILURE.json", failure)
        execution_host._append_json_line(output / "PROGRESS.jsonl", {"event": "failed", **failure})
        raise
