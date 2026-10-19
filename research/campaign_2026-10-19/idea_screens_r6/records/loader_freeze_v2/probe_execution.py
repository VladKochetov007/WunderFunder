from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

from execution_host import (
    _append_json_line,
    _check_physical_output,
    _guard_memory,
    _runtime_modules,
    _verify_runtime_inventory,
    _write_json,
)


def _initialize_progress_stream(path: Path, initial: dict[str, object]) -> None:
    """Create a JSONL stream once and append its first compact record."""
    with path.open("xb"):
        pass
    _append_json_line(path, initial)


def execute(
    sample_path: Path,
    checkpoint_path: Path,
    output: Path,
    base: Path,
    scratch: Path,
    source_root: Path,
    package_dir: Path,
    pins: dict[str, str],
    preparation_sha256: str,
    identities: dict[str, str],
) -> None:
    """Run the fixed R6 sequence diagnostic after all source and payload hashes pass."""
    started = time.monotonic()
    _check_physical_output(output, scratch)
    output.mkdir()
    _write_json(
        output / "START.json",
        {
            "schema": "nf19.r6_gradient_probe.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "design_sha256": identities["design"],
            "preparation_manifest_sha256": preparation_sha256,
            "sample_sha256": identities["sample"],
            "membership_sha256": identities["membership"],
            "checkpoint_sha256": identities["checkpoint"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        },
    )
    try:
        sys.path.insert(0, str(package_dir))
        torch, runtime = _runtime_modules(source_root, pins)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
            raise RuntimeError("CUDA must be hidden for the R6 CPU-only diagnostic")
        from model_contract import load_frozen_model
        from model_probe import prepare_sequences, run_diagnostic
        from sample_panel import load_first_fit_eight

        sample_started = time.monotonic()
        sequences, sample_summary = load_first_fit_eight(sample_path)
        runtime["canonical_import_inventory_after_sample_helper"] = _verify_runtime_inventory(
            source_root, pins
        )
        sample_seconds = time.monotonic() - sample_started
        checkpoint_started = time.monotonic()
        loaded = load_frozen_model(checkpoint_path, source_root, base)
        checkpoint_seconds = time.monotonic() - checkpoint_started
        input_started = time.monotonic()
        prepared = prepare_sequences(loaded.model, loaded.environment, sequences)
        input_seconds = time.monotonic() - input_started
        runtime.update(
            {
                "torch_threads": torch.get_num_threads(),
                "device": "cpu",
                "model_dtype": "float32",
                "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
                "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
            }
        )
        runtime["memory_guard_initial"] = _guard_memory("initialized_before_first_graph")
        source_record = {
            "schema": "nf19.r6_gradient_probe.source.v1",
            "source_snapshot": identities["source_revision"],
            "preparation_manifest_sha256": preparation_sha256,
            "design_sha256": identities["design"],
            "source_file_pins": pins,
            "sample": sample_summary,
            "checkpoint_sha256": identities["checkpoint"],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_model": {
                "n_in": loaded.n_in,
                "state_keys": loaded.state_keys,
                "architecture_args": {
                    key: loaded.args.get(key)
                    for key in (
                        "arch",
                        "cell",
                        "hidden",
                        "layers",
                        "enc",
                        "dropout",
                        "precision",
                        "loss",
                    )
                },
            },
            "runtime": runtime,
            "phase_seconds": {
                "sample_verification_and_mapping": sample_seconds,
                "checkpoint_load_and_reconstruction": checkpoint_seconds,
                "input_preparation": input_seconds,
            },
        }
        _write_json(output / "SOURCE.json", source_record)
        _initialize_progress_stream(
            output / "PROGRESS.jsonl",
            {"event": "initialized", "elapsed_s": time.monotonic() - started},
        )

        def progress(record: dict[str, object]) -> None:
            """Persist compact progress after each predeclared window."""
            _append_json_line(output / "PROGRESS.jsonl", record)

        gradient_started = time.monotonic()
        result = run_diagnostic(loaded.model, prepared, _guard_memory, progress)
        gradient_seconds = time.monotonic() - gradient_started
        result.update(
            {
                "schema": "nf19.r6_gradient_probe.result.v1",
                "design_sha256": identities["design"],
                "sample_sha256": identities["sample"],
                "membership_sha256": identities["membership"],
                "checkpoint_sha256": identities["checkpoint"],
                "selected_components": sample_summary["selected_components"],
                "selected_sequences": sample_summary["selected_sequences"],
                "checkpoint_args": source_record["checkpoint_model"]["architecture_args"],
                "input_width": 211,
                "raw_input_width": 112,
                "causal_event_width": 99,
                "normalization": "unchanged saved checkpoint mean/std buffers",
                "loss": "canonical pooled WMSE; target contributions share each window's joint weight mass",
                "windows_per_sequence": 2,
                "window_starts": [1024, 4096],
                "window_length": 128,
                "prefix_chunk_max": 512,
                "evaluation_sequences_read": 0,
                "npy_files_hashed_and_mapped": 24,
                "phase_seconds": {
                    **source_record["phase_seconds"],
                    "gradient_and_prefix": gradient_seconds,
                    "total": time.monotonic() - started,
                },
                "runtime": runtime,
            }
        )
        result_sha = _write_json(output / "RESULT.json", result)
        _write_json(
            output / "DONE.json",
            {
                "schema": "nf19.r6_gradient_probe.done.v1",
                "status": "completed",
                "result_sha256": result_sha,
                "result_bytes": (output / "RESULT.json").stat().st_size,
                "windows": 16,
                "model_state_unchanged": result["model_state_unchanged"],
                "parameter_grad_fields_empty": result["parameter_grad_fields_empty"],
                "elapsed_s": time.monotonic() - started,
            },
        )
    except Exception as exc:
        _write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r6_gradient_probe.failure.v1",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "elapsed_s": time.monotonic() - started,
            },
        )
        raise
